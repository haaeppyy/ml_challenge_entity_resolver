import argparse
import csv as _csv
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyarrow.parquet as pq
import numpy as np
import polars as pl
import lightgbm as lgb
import joblib

from config import DATA_DIR, MODEL_DIR, ground_truth_path
from entsplit import load as load_entsplit
from features import side_table, gt_pairs, compute_features

BATCH = 2_000_000
MODEL_DEPLOYED = os.path.join(MODEL_DIR, "lgbm_matcher.txt")
RESULTS_CSV = os.path.join(DATA_DIR, "block_eval_results.csv")
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)
FEATURE_ORDER = ["jw_name", "jw_addr", "tok_jaccard", "country_match",
                 "name_len_diff", "fsig_match",
                 "lev_name", "jaro_name", "jaro_addr",
                 "addr_house_match", "addr_pin_match", "addr_street_jaccard",
                 "src"]


def mem_pct():
    try:
        import psutil
        return float(psutil.virtual_memory().percent)
    except Exception:
        return 0.0


def gt_frame(src):
    g = (pl.read_csv(ground_truth_path("train"), separator="\t",
                     columns=["source1_entity_id", "matched_entity_ids"])
         .filter(pl.col("matched_entity_ids").is_not_null())
         .with_columns(pl.col("matched_entity_ids").str.split(",").alias("ids"))
         .explode("ids")
         .select(pl.col("source1_entity_id").cast(pl.Utf8),
                 pl.col("ids").str.strip_chars().alias("other_entity_id"))
         .filter(pl.col("other_entity_id").str.starts_with(f"S{src}-"))
         .unique())
    return g


def pair_recall(cand, g, src_country):
    """GT pair recall of a candidate set (already restricted to val S1)."""
    hit = (g.join(cand, left_on=["source1_entity_id", "other_entity_id"],
                  right_on=["entity_id_l", "entity_id_r"], how="semi"))
    g2 = g.join(src_country, left_on="source1_entity_id",
                right_on="entity_id", how="left")
    hit2 = hit.join(src_country, left_on="source1_entity_id",
                    right_on="entity_id", how="left")
    per = {}
    for ct in sorted(g2["country_norm"].drop_nulls().unique().to_list()):
        gt_ct = g2.filter(pl.col("country_norm") == ct).height
        hit_ct = hit2.filter(pl.col("country_norm") == ct).height
        per[ct] = {"gt": gt_ct, "hit": hit_ct,
                   "recall": 100 * hit_ct / gt_ct if gt_ct else None}
    overall = 100 * hit.height / g.height
    return g.height, hit.height, overall, per


def candidate_stats(cand, val_entities, n_src_rows):
    """Per-entity candidate distribution incl. zero-candidate S1 entities."""
    cnt = (cand.group_by("entity_id_l").len() if cand.height
           else pl.DataFrame({"entity_id_l": [], "len": []}))
    tab = (val_entities.join(cnt, left_on="entity_id",
                              right_on="entity_id_l", how="left")
           .with_columns(pl.col("len").fill_null(0).cast(pl.Int32)))
    lens = tab["len"]
    zero = int((lens == 0).sum())
    stats = {
        "n_s1_entities": val_entities.height,
        "n_pairs": cand.height,
        "pairs_per_s1_mean": float(lens.mean()),
        "pairs_per_s1_median": float(lens.median()),
        "pairs_per_s1_p90": float(lens.quantile(0.90)),
        "pairs_per_s1_max": int(lens.max()),
        "zero_candidate_entities": zero,
        "n_src_rows": n_src_rows,
        "reduction_ratio": (cand.height / (val_entities.height * n_src_rows)
                            if val_entities.height and n_src_rows else None),
    }
    return stats


def score_candidates(cand_path, t1, tr, gt, mdl, src_feat,
                       best_iteration=None, calibrator=None):
    """Score every candidate row (entity_id_l/r) -> DataFrame(entity_id_l,
    entity_id_r, prob, label). Streaming batches keep peak memory low."""
    keep = []
    pf = pq.ParquetFile(cand_path)
    n_tot = pf.metadata.num_rows
    rows = 0
    for rb in pf.iter_batches(batch_size=BATCH,
                              columns=["entity_id_l", "entity_id_r"]):
        chunk = pl.from_arrow(rb)
        feats = compute_features(chunk, t1, tr, gt)
        feats = feats.with_columns(
            pl.lit(src_feat, dtype=pl.Int8).alias("src"))
        X = np.ascontiguousarray(feats.select(FEATURE_ORDER).to_numpy(),
                                 dtype=np.float32)
        prob = mdl.predict(X, num_iteration=best_iteration) if best_iteration is not None else mdl.predict(X)
        # Apply calibration if available
        if calibrator is not None:
            prob = calibrator.predict_proba(prob.reshape(-1, 1))[:, 1]
        keep.append(pl.DataFrame({"entity_id_l": chunk["entity_id_l"],
                                  "entity_id_r": chunk["entity_id_r"],
                                  "prob": prob,
                                  "label": feats["label"].cast(pl.UInt8)}))
        rows += chunk.height
        if rows % (10 * BATCH) < BATCH:
            print(f"    scored {rows:,}/{n_tot:,} "
                  f"(mem {mem_pct():.0f}%)", flush=True)
        gc.collect()
    print(f"    scored {rows:,}/{n_tot:,} (mem {mem_pct():.0f}%)",
          flush=True)
    return pl.concat(keep) if keep else pl.DataFrame(
        {"entity_id_l": pl.Series([], dtype=pl.Utf8),
         "entity_id_r": pl.Series([], dtype=pl.Utf8),
         "prob": pl.Series([], dtype=pl.Float64),
         "label": pl.Series([], dtype=pl.UInt8)})


def macro_f05_sweep(entity_order, gt_map, prob_rows, label_rows,
                    cand_entity_codes):
    """Per-entity macro F0.5 over EVERY entity in entity_order.

    gt_map:            {entity_id: gt pair count} for all entities (incl 0).
    prob_rows:         model probs for candidate rows (both sources).
    label_rows:        ground-truth match label for each candidate pair.
    cand_entity_codes: index (into entity_order) per candidate row.
    """
    n = len(entity_order)
    gt_cnt = np.array([gt_map.get(e, 0) for e in entity_order],
                      dtype=np.int64)
    results = {}
    for thr in THRESHOLDS:
        pred_rows = prob_rows >= thr
        pred_cnt = np.bincount(cand_entity_codes[pred_rows], minlength=n)
        tp_rows = pred_rows & (label_rows == 1)
        tp_cnt = np.bincount(cand_entity_codes[tp_rows], minlength=n)
        fn_cnt = gt_cnt - tp_cnt
        fp_cnt = pred_cnt - tp_cnt
        with np.errstate(divide="ignore", invalid="ignore"):
            prec = np.where((tp_cnt + fp_cnt) > 0,
                            tp_cnt / (tp_cnt + fp_cnt), 1.0)
            rec = np.where((tp_cnt + fn_cnt) > 0,
                           tp_cnt / (tp_cnt + fn_cnt), 1.0)
        f = np.where((0.25 * prec + rec) > 0,
                     (1.25 * prec * rec) / (0.25 * prec + rec), 0.0)
        results[float(thr)] = (float(f.mean()), int((gt_cnt > 0).sum()),
                               int((pred_cnt > 0).sum()))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-s2", required=True)
    ap.add_argument("--cand-s3", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--restricted", action="store_true",
                    help="candidate files are already restricted to val S1")
    ap.add_argument("--model", default="lgbm_matcher_calibrated.pkl",
                    help="LightGBM model path to evaluate")
    ap.add_argument("--results-csv", default=RESULTS_CSV)
    args = ap.parse_args()

    t0 = time.time()
    print(f"=== block evaluation: {args.label} ===", flush=True)
    ent = load_entsplit()
    val_entities = ent.filter(pl.col("isval") == 1).select("entity_id")
    print(f"  val entities: {val_entities.height:,}", flush=True)
    s1_country = (pl.scan_parquet(
        os.path.join(DATA_DIR, "norm_train_s1.parquet"))
        .select("entity_id", "country_norm").collect())

    rows_full = {}
    recall_tables = {}
    cand_stats = {}
    score_tables = {}
    # Load model (supports both .txt booster and .pkl calibrated model)
    if args.model.endswith('.pkl'):
        import joblib
        model_dict = joblib.load(args.model)
        mdl = model_dict['model']
        best_iteration = model_dict.get('best_iteration', None)
        # Also load calibrator if present
        if 'calibrator' in model_dict:
            calibrator = model_dict['calibrator']
        else:
            # Fit a simple logistic regression calibrator on validation
            from sklearn.linear_model import LogisticRegression
            calibrator = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
            calibrator = None
    else:
        mdl = lgb.Booster(model_file=args.model)
        best_iteration = mdl.best_iteration
        calibrator = None
    t1 = None
    summary = {"label": args.label}
    overall_macro = {}

    for src in (2, 3):
        print(f"\n  ---- S1 x S{src} ----", flush=True)
        g = gt_frame(src).filter(
            pl.col("source1_entity_id").is_in(val_entities["entity_id"]))
        cand = pl.read_parquet(getattr(args, f"cand_s{src}"))
        if not args.restricted:
            cand = cand.filter(
                pl.col("entity_id_l").is_in(val_entities["entity_id"]))
        n_src_rows = pl.scan_parquet(
            os.path.join(DATA_DIR, f"norm_train_s{src}.parquet")
        ).select(pl.len()).collect().item()
        rows_full[f"S{src}"] = cand.height

        tot, hit, overall, per = pair_recall(cand, g, s1_country)
        print(f"    candidate pairs (val S1): {cand.height:,}", flush=True)
        print(f"    GT pairs (val): {tot:,}  captured: {hit:,}  "
              f"recall: {overall:.2f}%", flush=True)
        for ct, d in per.items():
            r = f"{d['recall']:.2f}%" if d["recall"] is not None else "n/a"
            print(f"      {ct}: {d['hit']:,}/{d['gt']:,} = {r}", flush=True)
        recall_tables[f"S{src}"] = {"total_gt": tot, "hit": hit,
                                    "recall": overall, "per_country": per}
        summary[f"recall_S{src}"] = round(overall, 4)

        cs = candidate_stats(cand, val_entities, n_src_rows)
        cand_stats[f"S{src}"] = cs
        summary[f"pairs_S{src}"] = cs["n_pairs"]
        for k in ("pairs_per_s1_mean", "pairs_per_s1_median",
                  "pairs_per_s1_p90", "zero_candidate_entities",
                  "reduction_ratio"):
            v = cs[k]
            summary[f"{k}_S{src}"] = (round(v, 4) if isinstance(v, float)
                                      else v)

        cand_val_path = os.path.join(
            DATA_DIR, f"_valblk_{args.label}_s{src}.parquet")
        cand.select("entity_id_l", "entity_id_r").write_parquet(cand_val_path)

        if t1 is None:
            t1 = side_table(1).rename(
                {c: c + "_l" for c in side_table(1).columns
                 if c != "entity_id"})
        tr = side_table(src).rename(
            {c: c + "_r" for c in side_table(src).columns
             if c != "entity_id"})
        gt_lab = gt_pairs(src)
        scored = score_candidates(cand_val_path, t1, tr, gt_lab, mdl,
                                   0 if src == 2 else 1,
                                   best_iteration=best_iteration,
                                   calibrator=calibrator)
        score_tables[f"S{src}"] = scored
        del tr, gt_lab, cand
        gc.collect()

    # ---- entity-level matching metric over BOTH sources jointly ------------
    print("\n  ---- matching quality: macro F0.5 over ALL held-out entities "
          "----", flush=True)
    s2 = score_tables["S2"].rename({"prob": "p2"})
    s3 = score_tables["S3"].rename({"prob": "p3"})
    rows = (s2.select("entity_id_l", "entity_id_r", "p2",
                      pl.col("label").alias("label2"))
            .join(s3.select("entity_id_l", "entity_id_r", "p3",
                            pl.col("label").alias("label3")),
                  on=["entity_id_l", "entity_id_r"], how="outer", coalesce=True)
            .with_columns(pl.coalesce([pl.col("p2"), pl.col("p3")])
                          .alias("prob"))
            .with_columns(pl.coalesce([pl.col("label2"), pl.col("label3")])
                          .alias("label"))
            .with_columns(pl.when(pl.col("p2").is_not_null()).then(pl.lit(0))
                          .otherwise(pl.lit(1)).alias("src"))
            .select("entity_id_l", "entity_id_r", "prob", "label"))
    del s2, s3
    gc.collect()

    g_all = (gt_frame(2).vstack(gt_frame(3)).unique())
    gt_per = g_all.group_by("source1_entity_id").len()
    gt_map = dict(zip(gt_per["source1_entity_id"].to_list(),
                      gt_per["len"].to_list()))
    del g_all, gt_per
    gc.collect()

    entity_order = val_entities["entity_id"].to_list()
    order_idx = {e: i for i, e in enumerate(entity_order)}
    if rows.height:
        cand_e = rows["entity_id_l"].to_list()
        prob = rows["prob"].to_numpy().astype(np.float64)
        label_rows = rows["label"].to_numpy().astype(np.uint8)
        try:
            codes_full = np.array([order_idx[u] for u in cand_e],
                                  dtype=np.int64)
        except KeyError:
            cand_e = None
            raise SystemExit("Abort: candidate rows reference S1 entities "
                             "outside the val split (use --restricted or "
                             "pre-filter).")
    else:
        codes_full = np.zeros(0, dtype=np.int64)
        prob = np.zeros(0, dtype=np.float64)
        label_rows = np.zeros(0, dtype=np.uint8)

    res = macro_f05_sweep(entity_order, gt_map, prob, label_rows, codes_full)
    print("    threshold  macro-F0.5  entities-with-GT  entities-with-pred")
    best_thr, best_f = None, -1.0
    for thr in THRESHOLDS:
        f, n_gt, n_pred = res[float(thr)]
        mark = ""
        if f > best_f:
            best_f, best_thr = f, thr
            mark = "  <-- best (tuned on held-out)"
        print(f"      {thr:.2f}      {f:.4f}    {n_gt:,}        {n_pred:,}"
              f"{mark}", flush=True)
    # values at fixed thresholds
    for fixed in (0.20, 0.25):
        f, n_gt, n_pred = res[float(fixed)]
        print(f"    @ {fixed:.2f}: macro-F0.5 = {f:.4f}", flush=True)
        summary[f"macroF05@{fixed:.2f}"] = round(f, 4)
    # tuned threshold values
    summary["best_threshold"] = float(best_thr)
    summary["macroF05_tuned"] = round(best_f, 4)

    # pair-level metrics at best threshold
    pred_rows = prob >= best_thr
    tp_rows = pred_rows & (label_rows == 1)
    pair_p = pred_rows.sum()
    tp = tp_rows.sum()
    pair_prec = tp / pair_p if pair_p else 0.0
    pair_rec = tp / label_rows.sum() if label_rows.sum() else 0.0
    from math import isclose
    pair_f05 = (1.25 * pair_prec * pair_rec) / (0.25 * pair_prec + pair_rec) if (pair_prec + pair_rec) else 0.0
    summary["pairP@best"] = round(float(pair_prec), 4)
    summary["pairR@best"] = round(float(pair_rec), 4)
    summary["pairF05@best"] = round(float(pair_f05), 4)
    print(f"    pair-level @ {best_thr:.2f}: P={pair_prec:.4f} R={pair_rec:.4f} F0.5={pair_f05:.4f} "
          f"(pred {pair_p:,}, true {label_rows.sum():,}, tp {tp_rows.sum():,})", flush=True)

    # singleton stats
    # Entities with no true matches in ground truth
    singleton_entities = [e for e in entity_order if gt_map.get(e, 0) == 0]
    n_singletons = len(singleton_entities)
    # For each singleton, check if it has any predicted matches above threshold
    # Get max prob per entity
    max_prob_per_entity = np.full(len(entity_order), 0.0, dtype=np.float64)
    for i, p in zip(codes_full, prob):
        if p > max_prob_per_entity[i]:
            max_prob_per_entity[i] = p
    singleton_correct = sum(1 for e in singleton_entities if max_prob_per_entity[order_idx[e]] < best_thr)
    summary["n_singletons"] = n_singletons
    summary["singleton_correct"] = singleton_correct
    summary["singleton_acc"] = round(100 * singleton_correct / n_singletons, 3)

    print(f"    singletons (no true matches, val): {n_singletons}  "
          f"predicted-empty correctly: {singleton_correct} ({100*singleton_correct/n_singletons:.2f}%)",
          flush=True)

    print(f"\n  SUMMARY {json.dumps(summary, sort_keys=True)}", flush=True)
    # log to CSV
    import csv
    file_exists = os.path.isfile(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=sorted(summary.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(summary)
    print(f"  logged -> {RESULTS_CSV} (peak mem {mem_pct():.0f}%)", flush=True)
    print(f"  elapsed {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()