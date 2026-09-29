import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyarrow.parquet as pq
import numpy as np
import polars as pl
import lightgbm as lgb
import joblib
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


def load_best_threshold():
    import joblib
    from config import MODEL_DIR
    threshold_path = os.path.join(MODEL_DIR, "best_threshold.pkl")
    if os.path.exists(threshold_path):
        d = joblib.load(threshold_path)
        return float(d.get('threshold', 0.60))
    return 0.60


def mem_pct():
    try:
        import psutil
        return float(psutil.virtual_memory().percent)
    except Exception:
        return 0.0


def dequote_empty(path):
    tmp = path + ".tmp"
    with open(path, "r", encoding="utf-8", newline="") as fin, \
            open(tmp, "w", encoding="utf-8", newline="") as fout:
        for line in fin:
            fout.write(line.replace('\t""\r\n', "\t\r\n")
                       .replace('\t""\n', "\t\n"))
    os.replace(tmp, path)


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


def score_batch(chunk, tl, tr, src, mdl, best_iteration=None, calibrator=None, threshold=0.60):
    chunk = chunk.join(tl, left_on="entity_id_l", right_on="entity_id")
    chunk = chunk.join(tr, left_on="entity_id_r", right_on="entity_id")
    
    # Token Jaccard
    a = chunk["name_tokens_l"].list.set_intersection(chunk["name_tokens_r"])
    u = chunk["name_tokens_l"].list.set_union(chunk["name_tokens_r"])
    jac = ((a.list.len().cast(pl.Float32) / u.list.len().cast(pl.Float32))
           .fill_null(0.0).fill_nan(0.0))
    
    # Name strings
    name_l = chunk["name_norm_l"].to_list()
    name_r = chunk["name_norm_r"].to_list()
    
    # Levenshtein and Jaro for name
    lev_name = pl.Series([1.0 - Levenshtein.normalized_distance(x, y) for x, y in zip(name_l, name_r)], dtype=pl.Float32)
    jaro_name = pl.Series([Jaro.similarity(x, y) for x, y in zip(name_l, name_r)], dtype=pl.Float32)
    
    # Jaro for address
    addr_l = chunk["addr_norm_l"].to_list()
    addr_r = chunk["addr_norm_r"].to_list()
    jaro_addr = pl.Series([Jaro.similarity(x, y) for x, y in zip(addr_l, addr_r)], dtype=pl.Float32)
    
    # Address component matching
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
    
    # Jaro-Winkler for name and address
    jw_name = pl.Series([JaroWinkler.similarity(x, y) for x, y in zip(name_l, name_r)], dtype=pl.Float32)
    jw_addr = pl.Series([JaroWinkler.similarity(x, y) for x, y in zip(addr_l, addr_r)], dtype=pl.Float32)
    
    # Token Jaccard
    a = chunk["name_tokens_l"].list.set_intersection(chunk["name_tokens_r"])
    u = chunk["name_tokens_l"].list.set_union(chunk["name_tokens_r"])
    jac = ((a.list.len().cast(pl.Float32) / u.list.len().cast(pl.Float32))
           .fill_null(0.0).fill_nan(0.0))
    
    f = pl.DataFrame({
        "jw_name": jw_name,
        "jw_addr": jw_addr,
        "tok_jaccard": jac,
        "country_match": (chunk["country_norm_l"]
                          == chunk["country_norm_r"]).cast(pl.Int8),
        "name_len_diff": (chunk["nlen_l"].cast(pl.Int64)
                          - chunk["nlen_r"].cast(pl.Int64)).abs(),
        "fsig_match": (chunk["fsig_l"] == chunk["fsig_r"]).cast(pl.Int8),
        "lev_name": lev_name,
        "jaro_name": jaro_name,
        "jaro_addr": jaro_addr,
        "addr_house_match": house_match,
        "addr_pin_match": pin_match,
        "addr_street_jaccard": street_jacc,
        "src": np.full(chunk.height, src, dtype=np.int8),
    })
    X = np.ascontiguousarray(f.select(FEATURE_ORDER).to_numpy(),
                             dtype=np.float32)
    prob = mdl.predict(X, num_iteration=best_iteration) if best_iteration is not None else mdl.predict(X)
    # Apply calibration if available
    if calibrator is not None:
        prob = calibrator.predict_proba(prob.reshape(-1, 1))[:, 1]
    return (pl.DataFrame({"entity_id_l": chunk["entity_id_l"],
                          "entity_id_r": chunk["entity_id_r"], "prob": prob})
              .filter(pl.col("prob") >= threshold).drop("prob"))


def score_phase():
    t0 = time.time()
    threshold = load_best_threshold()
    print(f"=== phase 1: score test candidates (thr {threshold:.2f}) ===",
          flush=True)
    import joblib
    model_dict = joblib.load(os.path.join(
        os.path.dirname(DATA_DIR), "models", "lgbm_matcher_calibrated.pkl"))
    mdl = model_dict['model']
    best_iteration = model_dict.get('best_iteration', None)
    calibrator = model_dict.get('calibrator', None)
    tl = side_table(1).rename({c: c + "_l" for c in
                               side_table(1).columns
                               if c != "entity_id"})
    print(f"  side tables loaded (mem {mem_pct():.0f}%)", flush=True)

    threshold = load_best_threshold()
    print(f"  using threshold: {threshold:.2f}", flush=True)
    for src in [2, 3]:
        print(f"\n  --- test S1 x S{src} ---", flush=True)
        tr = side_table(src).rename({c: c + "_r" for c in
                                      side_table(src).columns
                                      if c != "entity_id"})
        cand = os.path.join(DATA_DIR, f"{CAND_PREFIX}_test_s{src}.parquet")
        pf = pq.ParquetFile(cand)
        n_tot = pf.metadata.num_rows
        keep = []
        for rb in pf.iter_batches(batch_size=BATCH,
                                  columns=["entity_id_l", "entity_id_r"]):
            chunk = pl.from_arrow(rb)
            pr = score_batch(chunk, tl, tr, 0 if src == 2 else 1, mdl,
                             best_iteration=model_dict.get('best_iteration', None),
                             calibrator=model_dict.get('calibrator', None),
                             threshold=threshold)
            keep.append(pr)
            if sum(f.height for f in keep) >= 12_000_000:
                out = pl.concat(keep)
                keep = [out]
            print(f"    scored {sum(f.height for f in keep) or 0:>12,}"
                  f"/{n_tot:,} pred-kept (mem {mem_pct():.0f}%)",
                  flush=True)
        pred = pl.concat(keep) if keep else pl.DataFrame(
            {"entity_id_l": [], "entity_id_r": []})
        p_path = os.path.join(DATA_DIR, f"_pred_pairs_s{src}.parquet")
        pred.write_parquet(p_path)
        print(f"    predicted matches >= {threshold:.2f}: {pred.height:,}"
              f" -> {p_path}", flush=True)

        c_path = os.path.join(DATA_DIR, f"_cand_agg_s{src}.parquet")
        (pl.scan_parquet(cand)
         .select("entity_id_l", "entity_id_r")
         .group_by("entity_id_l")
         .agg(pl.col("entity_id_r").sort().str.join(","))
         .sink_parquet(c_path))
        print(f"    candidate aggregate -> {c_path} "
              f"(mem {mem_pct():.0f}%)", flush=True)
        del tr
        gc.collect()
    print(f"  phase 1 done in {time.time()-t0:.0f}s "
          f"(mem {mem_pct():.0f}%)", flush=True)


def assemble_phase():
    t0 = time.time()
    print("=== phase 2: assemble submission files ===", flush=True)
    canon = (pl.read_csv(source_path("test", 1), separator="\t")
             .get_column("entity_id").cast(pl.Utf8))
    print(f"  canonical test source1 entities: {len(canon):,}", flush=True)

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
    print(f"  intermediates loaded (mem {mem_pct():.0f}%)", flush=True)

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

    # matching_results.tsv (small strings -> sortable in-memory)
    a, b = match["m2"].fill_null(""), match["m3"].fill_null("")
    matched = (match.with_columns(concat_ids(a, b).alias("ids"))
               .select("entity_id_l", "ids"))
    out_match = (canon.to_frame("source1_entity_id")
                 .join(matched, left_on="source1_entity_id",
                       right_on="entity_id_l", how="left")
                 .with_columns(
                     pl.when((pl.col("ids").is_null())
                             | (pl.col("ids") == ""))
                     .then(pl.lit(""))
                     .otherwise(pl.col("ids").str.split(",").list.sort()
                                .list.join(","))
                     .alias("matched_entity_ids"))
                 .select("source1_entity_id", "matched_entity_ids"))

    # candidate_pairs.tsv (large per-entity strings -> skip re-sort;
    # s2/s3 segments are internally sorted already)
    a, b = candm["c2"].fill_null(""), candm["c3"].fill_null("")
    cands = (candm.with_columns(concat_ids(a, b).alias("ids"))
             .select("entity_id_l", "ids"))
    out_cand = (canon.to_frame("source1_entity_id")
                .join(cands, left_on="source1_entity_id",
                      right_on="entity_id_l", how="left")
                .with_columns(pl.col("ids").fill_null("")
                              .alias("candidate_entity_ids"))
                .select("source1_entity_id", "candidate_entity_ids"))
    del match, candm, matched, cands
    gc.collect()

    for path, df in [(os.path.join(OUTPUT_DIR, "matching_results.tsv"),
                      out_match),
                     (os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"),
                      out_cand)]:
        df.write_csv(path, separator="\t")
        print(f"  wrote -> {path} (rows {df.height:,} "
              f"mem {mem_pct():.0f}%)", flush=True)

    dequote_empty(os.path.join(OUTPUT_DIR, "matching_results.tsv"))
    dequote_empty(os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"))
    print("  emptied the quoted-empty fields", flush=True)

    mr = pl.read_csv(os.path.join(OUTPUT_DIR, "matching_results.tsv"),
                     separator="\t")
    cr = pl.read_csv(os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"),
                     separator="\t")
    total = mr.height
    with_match = int((mr["matched_entity_ids"].str.len_chars() > 0).sum())
    print(f"\n  total test source1 entities: {total:,}", flush=True)
    print(f"  entities with >=1 predicted match: {with_match:,} "
          f"({100*with_match/total:.2f}%)", flush=True)
    print(f"  entities with zero predicted match: {total-with_match:,} "
          f"({100*(total-with_match)/total:.2f}%)", flush=True)
    mid = set(mr["source1_entity_id"])
    cset = set(cr["source1_entity_id"])
    canon_set = set(canon.to_list())
    print(f"  matching_results: {total} rows, {len(mid)} unique ids, "
          f"== canonical: {mid == canon_set}", flush=True)
    print(f"  candidate_pairs : {total} rows, {len(cset)} unique ids, "
          f"== canonical: {cset == canon_set}", flush=True)
    print(f"  phase 2 done in {time.time()-t0:.0f}s", flush=True)


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