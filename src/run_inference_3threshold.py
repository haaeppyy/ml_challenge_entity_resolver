import argparse
import gc
import os
import sys
import time
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyarrow.parquet as pq
import polars as pl
import lightgbm as lgb
from config import DATA_DIR, OUTPUT_DIR, source_path
from features import jw_series, first_sig
from rapidfuzz.distance import JaroWinkler, Levenshtein, Jaro
import re

BATCH = 2_000_000
FEATURE_ORDER = ["jw_name", "jw_addr", "tok_jaccard", "country_match",
                 "name_len_diff", "fsig_match",
                 "lev_name", "jaro_name", "jaro_addr",
                 "addr_house_match", "addr_pin_match", "addr_street_jaccard",
                 "src"]

CAND_PREFIX = "candmp_test"


def load_model_and_policy():
    model_path = os.path.join(os.path.dirname(DATA_DIR), "models", "lgbm_matcher_calibrated.pkl")
    policy_path = os.path.join(os.path.dirname(DATA_DIR), "models", "best_3threshold_policy.pkl")
    
    model_dict = joblib.load(model_path)
    mdl = model_dict['model']
    calibrator = model_dict['calibrator']
    best_iteration = model_dict.get('best_iteration', None)
    
    policy = joblib.load(policy_path)
    sc = policy['singleton_cutoff']
    kt = policy['keep_threshold']
    ft = policy['fallback_top1_threshold']
    
    return mdl, calibrator, best_iteration, sc, kt, ft


def mem_pct():
    try:
        import psutil
        return float(psutil.virtual_memory().percent)
    except Exception:
        return 0.0


def side_table(src):
    df = (pl.scan_parquet(os.path.join(DATA_DIR, f"norm_test_s{src}.parquet"))
          .select("entity_id", "name_norm", "addr_norm", "country_norm",
                  "name_tokens")
          .collect())
    df = df.with_columns(pl.col("addr_norm").fill_null(""),
                         pl.col("name_norm").fill_null(""))
    df = df.with_columns(
        pl.col("name_tokens").map_elements(
            first_sig, return_dtype=pl.Utf8, skip_nulls=False).alias("fsig"))
    df = df.with_columns(pl.col("name_norm").str.len_chars().alias("nlen"))
    return df


def score_batch_all_probs(chunk, tl, tr, src, mdl, best_iteration, calibrator):
    """Score a batch and return ALL probabilities (no threshold filtering)."""
    chunk = chunk.join(tl, left_on="entity_id_l", right_on="entity_id")
    chunk = chunk.join(tr, left_on="entity_id_r", right_on="entity_id")
    
    a = chunk["name_tokens_l"].list.set_intersection(chunk["name_tokens_r"])
    u = chunk["name_tokens_l"].list.set_union(chunk["name_tokens_r"])
    jac = ((a.list.len().cast(pl.Float32) / u.list.len().cast(pl.Float32))
           .fill_null(0.0).fill_nan(0.0))
    
    name_l = chunk["name_norm_l"].to_list()
    name_r = chunk["name_norm_r"].to_list()
    lev_name = pl.Series([1.0 - Levenshtein.normalized_distance(x, y) for x, y in zip(name_l, name_r)], dtype=pl.Float32)
    jaro_name = pl.Series([Jaro.similarity(x, y) for x, y in zip(name_l, name_r)], dtype=pl.Float32)
    
    addr_l = chunk["addr_norm_l"].to_list()
    addr_r = chunk["addr_norm_r"].to_list()
    jaro_addr = pl.Series([Jaro.similarity(x, y) for x, y in zip(addr_l, addr_r)], dtype=pl.Float32)
    
    def extract_house_pin(addr):
        house_match = re.search(r'\b(\d+)\b', addr)
        house = house_match.group(1) if house_match else ""
        pin_match = re.search(r'\b(\d{5,6})\b', addr)
        pin = pin_match.group(1) if pin_match else ""
        stop_addr = {"road", "rd", "street", "st", "avenue", "ave", "lane", "ln", 
                     "nagar", "colony", "layout", "extension", "sector", "phase",
                     "block", "ward", "circle", "cross", "main", "market", "area",
                     "near", "opp", "opposite", "behind", "beside", "next"}
        tokens = [t for t in addr.split() if t.isalpha() and t not in stop_addr]
        return house, pin, set(tokens)
    
    addr_comp_l = [extract_house_pin(a) for a in addr_l]
    addr_comp_r = [extract_house_pin(a) for a in addr_r]
    
    house_match = pl.Series([1 if l[0] and l[0] == r[0] else 0 for l, r in zip(addr_comp_l, addr_comp_r)], dtype=pl.Int8)
    pin_match = pl.Series([1 if l[1] and l[1] == r[1] else 0 for l, r in zip(addr_comp_l, addr_comp_r)], dtype=pl.Int8)
    street_jacc = pl.Series([len(l[2] & r[2]) / len(l[2] | r[2]) if (l[2] | r[2]) else 0.0 for l, r in zip(addr_comp_l, addr_comp_r)], dtype=pl.Float32)
    
    jw_name = pl.Series([JaroWinkler.similarity(x, y) for x, y in zip(name_l, name_r)], dtype=pl.Float32)
    jw_addr = pl.Series([JaroWinkler.similarity(x, y) for x, y in zip(addr_l, addr_r)], dtype=pl.Float32)
    
    a = chunk["name_tokens_l"].list.set_intersection(chunk["name_tokens_r"])
    u = chunk["name_tokens_l"].list.set_union(chunk["name_tokens_r"])
    jac = ((a.list.len().cast(pl.Float32) / u.list.len().cast(pl.Float32))
           .fill_null(0.0).fill_nan(0.0))
    
    f = pl.DataFrame({
        "jw_name": jw_name,
        "jw_addr": jw_addr,
        "tok_jaccard": jac,
        "country_match": (chunk["country_norm_l"] == chunk["country_norm_r"]).cast(pl.Int8),
        "name_len_diff": (chunk["nlen_l"].cast(pl.Int64) - chunk["nlen_r"].cast(pl.Int64)).abs(),
        "fsig_match": (chunk["fsig_l"] == chunk["fsig_r"]).cast(pl.Int8),
        "lev_name": lev_name,
        "jaro_name": jaro_name,
        "jaro_addr": jaro_addr,
        "addr_house_match": house_match,
        "addr_pin_match": pin_match,
        "addr_street_jaccard": street_jacc,
        "src": np.full(chunk.height, src, dtype=np.int8),
    })
    X = np.ascontiguousarray(f.select(FEATURE_ORDER).to_numpy(), dtype=np.float32)
    prob = mdl.predict(X, num_iteration=best_iteration) if best_iteration is not None else mdl.predict(X)
    prob = calibrator.predict_proba(prob.reshape(-1, 1))[:, 1]
    
    return pl.DataFrame({
        "entity_id_l": chunk["entity_id_l"],
        "entity_id_r": chunk["entity_id_r"],
        "prob": prob
    })


def apply_3threshold_policy(prob_df, singleton_cutoff, keep_threshold, fallback_top1_threshold):
    """Apply 3-threshold policy per entity using polars expressions."""
    # Add max_prob per entity
    prob_df = prob_df.with_columns(
        pl.col("prob").max().over("entity_id_l").alias("max_prob")
    )
    
    # Determine action per entity
    prob_df = prob_df.with_columns(
        pl.when(pl.col("max_prob") < singleton_cutoff)
        .then(pl.lit(0))
        .when(pl.col("max_prob") >= keep_threshold)
        .then(pl.lit(1))
        .when(pl.col("max_prob") >= fallback_top1_threshold)
        .then(pl.lit(2))
        .otherwise(pl.lit(0))
        .alias("action")
    )
    
    # For action=1 (keep all >= keep_threshold), filter by prob
    # For action=2 (keep top-1), rank and keep rank=1
    prob_df = prob_df.with_columns(
        pl.when(pl.col("action") == 1)
        .then(pl.col("prob") >= keep_threshold)
        .when(pl.col("action") == 2)
        .then(pl.col("prob").rank("dense", descending=True).over("entity_id_l") == 1)
        .otherwise(False)
        .alias("keep")
    )
    
    # Filter and return - ensure entity_id_l is string not list
    result = prob_df.filter(pl.col("keep")).select("entity_id_l", "entity_id_r")
    
    # Fix any list-wrapped entity_id_l
    if result.height > 0 and result["entity_id_l"].dtype == pl.List:
        result = result.with_columns(
            pl.col("entity_id_l").list.first().alias("entity_id_l")
        )
    # Ensure empty result has correct schema
    if result.height == 0:
        result = pl.DataFrame({
            "entity_id_l": pl.Series([], dtype=pl.Utf8),
            "entity_id_r": pl.Series([], dtype=pl.Utf8)
        })
    return result


def apply_3threshold_policy_chunked(prob_iter, singleton_cutoff, keep_threshold, fallback_top1_threshold, chunk_size=2_000_000):
    """Apply 3-threshold policy per entity in chunks."""
    result_parts = []
    buffer = []
    buffer_size = 0
    
    for prob_df in prob_iter:
        buffer.append(prob_df)
        buffer_size += prob_df.height
        
        if buffer_size >= chunk_size:
            probs_cat = pl.concat(buffer)
            pred = apply_3threshold_policy(probs_cat, singleton_cutoff, keep_threshold, fallback_top1_threshold)
            if pred.height > 0:
                result_parts.append(pred)
            buffer = []
            buffer_size = 0
            gc.collect()
    
    if buffer:
        probs_cat = pl.concat(buffer)
        pred = apply_3threshold_policy(probs_cat, singleton_cutoff, keep_threshold, fallback_top1_threshold)
        if pred.height > 0:
            result_parts.append(pred)
    
    if result_parts:
        return pl.concat(result_parts)
    else:
        return pl.DataFrame({"entity_id_l": [], "entity_id_r": []})


def score_phase():
    t0 = time.time()
    print(f"=== Phase 1: Score test candidates with 3-threshold policy ===", flush=True)
    
    mdl, calibrator, best_iteration, sc, kt, ft = load_model_and_policy()
    print(f"Policy: singleton_cutoff={sc}, keep_threshold={kt}, fallback_top1={ft}", flush=True)
    
    tl = side_table(1).rename({c: c + "_l" for c in side_table(1).columns if c != "entity_id"})
    print(f"  Side tables loaded (mem {mem_pct():.0f}%)", flush=True)
    
    for src in [2, 3]:
        print(f"\n  --- test S1 x S{src} ---", flush=True)
        tr = side_table(src).rename({c: c + "_r" for c in side_table(src).columns if c != "entity_id"})
        cand = os.path.join(DATA_DIR, f"{CAND_PREFIX}_test_s{src}.parquet")
        pf = pq.ParquetFile(cand)
        n_tot = pf.metadata.num_rows
        
        all_probs = []
        pred_parts = []
        for rb in pf.iter_batches(batch_size=BATCH, columns=["entity_id_l", "entity_id_r"]):
            chunk = pl.from_arrow(rb)
            probs = score_batch_all_probs(chunk, tl, tr, 0 if src == 2 else 1, mdl, best_iteration, calibrator)
            all_probs.append(probs)
            
            # When we have enough, apply policy and clear
            if sum(p.height for p in all_probs) >= 10_000_000:
                prob_cat = pl.concat(all_probs)
                pred = apply_3threshold_policy(prob_cat, sc, kt, ft)
                if pred.height > 0:
                    pred_parts.append(pred)
                all_probs = []
                gc.collect()
            print(f"    scored {sum(p.height for p in all_probs) + sum(p.height for p in pred_parts)*10 or 0:>12,}/{n_tot:,} (mem {mem_pct():.0f}%)", flush=True)
        
        # Process remaining
        if all_probs:
            prob_cat = pl.concat(all_probs)
            pred = apply_3threshold_policy(prob_cat, sc, kt, ft)
            if pred.height > 0:
                pred_parts.append(pred)
        
        pred = pl.concat(pred_parts) if pred_parts else pl.DataFrame(
            {"entity_id_l": pl.Series([], dtype=pl.Utf8), "entity_id_r": pl.Series([], dtype=pl.Utf8)})
        
        # Fix entity_id_l dtype if it became List
        print(f"  After concat: height={pred.height}, entity_id_l dtype={pred['entity_id_l'].dtype}")
        if pred.height > 0 and pred["entity_id_l"].dtype == pl.List:
            print(f"  Fixing entity_id_l dtype from List to String (height={pred.height})")
            pred = pred.with_columns(pl.col("entity_id_l").list.first().alias("entity_id_l"))
            print(f"  After fix: dtype={pred['entity_id_l'].dtype}")
        
        # Write predicted pairs
        p_path = os.path.join(DATA_DIR, f"_pred_pairs_s{src}.parquet")
        pred.write_parquet(p_path)
        print(f"    predicted matches: {pred.height:,} -> {p_path}", flush=True)
        
        # Candidate aggregate (all candidates, not just predicted)
        c_path = os.path.join(DATA_DIR, f"_cand_agg_s{src}.parquet")
        (pl.scan_parquet(cand)
         .select("entity_id_l", "entity_id_r")
         .group_by("entity_id_l")
         .agg(pl.col("entity_id_r").sort().str.join(","))
         .sink_parquet(c_path))
        print(f"    candidate aggregate -> {c_path} (mem {mem_pct():.0f}%)", flush=True)
        del tr, pred
        gc.collect()
    
    print(f"  Phase 1 done in {time.time()-t0:.0f}s (mem {mem_pct():.0f}%)", flush=True)


def assemble_phase():
    t0 = time.time()
    print("=== Phase 2: Assemble submission files ===", flush=True)
    canon = (pl.read_csv(source_path("test", 1), separator="\t")
             .get_column("entity_id").cast(pl.Utf8))
    print(f"  Canonical test source1 entities: {len(canon):,}", flush=True)

    pred2 = (pl.read_parquet(os.path.join(DATA_DIR, "_pred_pairs_s2.parquet"))
             .group_by("entity_id_l")
             .agg(pl.col("entity_id_r").sort().str.join(","))
             .rename({"entity_id_r": "m2"}))
    pred3 = (pl.read_parquet(os.path.join(DATA_DIR, "_pred_pairs_s3.parquet"))
             .group_by("entity_id_l")
             .agg(pl.col("entity_id_r").sort().str.join(","))
             .rename({"entity_id_r": "m3"}))
    cand2 = (pl.read_parquet(os.path.join(DATA_DIR, "_cand_agg_s2.parquet"))
             .rename({"entity_id_r": "c2"}))
    cand3 = (pl.read_parquet(os.path.join(DATA_DIR, "_cand_agg_s3.parquet"))
             .rename({"entity_id_r": "c3"}))
    print(f"  Intermediates loaded (mem {mem_pct():.0f}%)", flush=True)

    match = pred2.join(pred3, on="entity_id_l", how="full", coalesce=True)
    candm = cand2.join(cand3, on="entity_id_l", how="full", coalesce=True)
    del pred2, pred3, cand2, cand3
    gc.collect()

    def concat_ids(a, b):
        return pl.coalesce([
            pl.when((a != "") & (b != "")).then(a + "," + b),
            pl.when(a != "").then(a),
            pl.when(b != "").then(b),
        ]).fill_null("")

    a, b = match["m2"].fill_null(""), match["m3"].fill_null("")
    matched = (match.with_columns(concat_ids(a, b).alias("ids"))
               .select("entity_id_l", "ids"))
    out_match = (canon.to_frame("source1_entity_id")
                 .join(matched, left_on="source1_entity_id", right_on="entity_id_l", how="left")
                 .with_columns(
                     pl.when((pl.col("ids").is_null()) | (pl.col("ids") == ""))
                     .then(pl.lit(""))
                     .otherwise(pl.col("ids").str.split(",").list.sort().list.join(","))
                     .alias("matched_entity_ids"))
                 .select("source1_entity_id", "matched_entity_ids"))

    a, b = candm["c2"].fill_null(""), candm["c3"].fill_null("")
    cands = (candm.with_columns(concat_ids(a, b).alias("ids"))
             .select("entity_id_l", "ids"))
    out_cand = (canon.to_frame("source1_entity_id")
                .join(cands, left_on="source1_entity_id", right_on="entity_id_l", how="left")
                .with_columns(pl.col("ids").fill_null("").alias("candidate_entity_ids"))
                .select("source1_entity_id", "candidate_entity_ids"))
    del match, candm, matched, cands
    gc.collect()

    for path, df in [(os.path.join(OUTPUT_DIR, "matching_results.tsv"), out_match),
                      (os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"), out_cand)]:
        df.write_csv(path, separator="\t")
        print(f"  Wrote -> {path} (rows {df.height:,} mem {mem_pct():.0f}%)", flush=True)

    def dequote_empty(path):
        tmp = path + ".tmp"
        with open(path, "r", encoding="utf-8", newline="") as fin, \
                open(tmp, "w", encoding="utf-8", newline="") as fout:
            for line in fin:
                fout.write(line.replace('\t""\r\n', "\t\r\n").replace('\t""\n', "\t\n"))
        os.replace(tmp, path)

    dequote_empty(os.path.join(OUTPUT_DIR, "matching_results.tsv"))
    dequote_empty(os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"))
    print("  Emptied quoted-empty fields", flush=True)

    mr = pl.read_csv(os.path.join(OUTPUT_DIR, "matching_results.tsv"), separator="\t")
    cr = pl.read_csv(os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"), separator="\t")
    total = mr.height
    with_match = int((mr["matched_entity_ids"].str.len_chars() > 0).sum())
    print(f"\n  Total test source1 entities: {total:,}", flush=True)
    print(f"  Entities with >=1 predicted match: {with_match:,} ({100*with_match/total:.2f}%)", flush=True)
    print(f"  Entities with zero predicted match: {total-with_match:,} ({100*(total-with_match)/total:.2f}%)", flush=True)
    print(f"  Phase 2 done in {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["1", "2"], default="2")
    args = ap.parse_args()
    if args.phase == "1":
        score_phase()
    else:
        assemble_phase()


if __name__ == "__main__":
    main()