import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyarrow.parquet as pq
import polars as pl
from rapidfuzz.distance import JaroWinkler, Levenshtein, Jaro
import re
from config import DATA_DIR, ground_truth_path

BATCH = 2_000_000
MEM_WARN_PCT = 85.0
STOP = {
    "and", "the", "of", "at", "by", "for", "in", "co", "ltd", "llc", "inc",
    "pvt", "gmbh", "corp", "dr", "mr", "mrs", "ms", "smt", "shri", "sri",
    "prof", "district", "private", "limited",
}

FEATURES = [
    "jw_name", "jw_addr", "tok_jaccard",
    "country_match", "name_len_diff", "fsig_match",
    "lev_name", "jaro_name", "jaro_addr",
    "addr_house_match", "addr_pin_match", "addr_street_jaccard",
    "label",
]


def mem_pct():
    try:
        import psutil
        return float(psutil.virtual_memory().percent)
    except Exception:
        return 0.0


def guard(tag):
    p = mem_pct()
    print(f"    [mem {p:.0f}%] {tag}", flush=True)
    if p >= MEM_WARN_PCT:
        raise SystemExit(f"ABORT: memory at {p:.0f}% while {tag}")


def first_sig(toks):
    toks = toks.to_list()
    for t in toks:
        if t not in STOP:
            return t
    return ""


def side_table(src, split="train"):
    df = (pl.scan_parquet(os.path.join(DATA_DIR, f"norm_{split}_s{src}.parquet"))
          .select("entity_id", "name_norm", "addr_norm", "country_norm",
                  "name_tokens")
          .collect())
    df = df.with_columns(pl.col("addr_norm").fill_null(""))
    df = df.with_columns(
        pl.col("name_tokens").map_elements(
            first_sig, return_dtype=pl.Utf8, skip_nulls=False).alias("fsig"))
    df = df.with_columns(pl.col("name_norm").str.len_chars().alias("nlen"))
    return df


def jw_series(a, b):
    la, lb = a.to_list(), b.to_list()
    return pl.Series([JaroWinkler.similarity(x, y) for x, y in zip(la, lb)],
                     dtype=pl.Float32)


def gt_pairs(src):
    gt_path = ground_truth_path("train")
    return (pl.read_csv(gt_path, separator="\t",
                        columns=["source1_entity_id", "matched_entity_ids"])
            .filter(pl.col("matched_entity_ids").is_not_null())
            .with_columns(pl.col("matched_entity_ids").str.split(",")
                          .alias("ids"))
            .explode("ids")
            .select(pl.col("source1_entity_id").cast(pl.Utf8),
                    pl.col("ids").str.strip_chars().cast(pl.Utf8)
                    .alias("matched"),
                    pl.lit(1, dtype=pl.Int8).alias("gt_row")))


def compute_features(chunk, t1, tr, gt):
    chunk = chunk.join(t1, left_on="entity_id_l", right_on="entity_id")
    chunk = chunk.join(tr, left_on="entity_id_r", right_on="entity_id")

    a = chunk["name_tokens_l"].list.set_intersection(
        chunk["name_tokens_r"])
    u = chunk["name_tokens_l"].list.set_union(
        chunk["name_tokens_r"])
    jac = ((a.list.len().cast(pl.Float32) / u.list.len().cast(pl.Float32))
           .fill_null(0.0).fill_nan(0.0))

    # Levenshtein and Jaro for name
    name_l = chunk["name_norm_l"].to_list()
    name_r = chunk["name_norm_r"].to_list()
    lev_name = pl.Series([1.0 - Levenshtein.normalized_distance(x, y) for x, y in zip(name_l, name_r)], dtype=pl.Float32)
    jaro_name = pl.Series([Jaro.similarity(x, y) for x, y in zip(name_l, name_r)], dtype=pl.Float32)
    
    # Jaro for address
    addr_l = chunk["addr_norm_l"].to_list()
    addr_r = chunk["addr_norm_r"].to_list()
    jaro_addr = pl.Series([Jaro.similarity(x, y) for x, y in zip(addr_l, addr_r)], dtype=pl.Float32)

    # Address component matching
    def extract_house_pin(addr):
        # Extract house number (first digit sequence)
        house_match = re.search(r'\b(\d+)\b', addr)
        house = house_match.group(1) if house_match else ""
        # Extract PIN (5-6 digit sequence)
        pin_match = re.search(r'\b(\d{5,6})\b', addr)
        pin = pin_match.group(1) if pin_match else ""
        # Extract street tokens (non-digit, non-stop words)
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

    labelled = chunk.join(gt, left_on=["entity_id_l", "entity_id_r"],
                          right_on=["source1_entity_id", "matched"],
                          how="left")
    out = pl.DataFrame({
        "entity_id_l": chunk["entity_id_l"],
        "entity_id_r": chunk["entity_id_r"],
        "jw_name": jw_series(chunk["name_norm_l"], chunk["name_norm_r"]),
        "jw_addr": jw_series(chunk["addr_norm_l"], chunk["addr_norm_r"]),
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
        "label": labelled["gt_row"].fill_null(0),
    })
    return out.select("entity_id_l", "entity_id_r", *FEATURES)


def run(src):
    t0 = time.time()
    print(f"=== feature engineering: train S1 x S{src} ===", flush=True)
    cand_path = os.path.join(DATA_DIR, f"candidates_ct2f_train_s{src}.parquet")

    t1 = side_table(1).rename({c: c + "_l" for c in
                               side_table(1).columns
                               if c != "entity_id"})
    guard("s1 table loaded")
    t2_ = side_table(src).rename({c: c + "_r" for c in
                                  side_table(src).columns
                                  if c != "entity_id"})
    guard(f"s{src} table loaded")

    gt = gt_pairs(src)
    guard("ground-truth pairs loaded")

    pf = pq.ParquetFile(cand_path)
    n_total = pf.metadata.num_rows
    print(f"  candidate pairs: {n_total:,}", flush=True)

    files = []
    written = 0
    batch_no = 0
    for rb in pf.iter_batches(batch_size=BATCH,
                              columns=["entity_id_l", "entity_id_r"]):
        tb = time.time()
        chunk = pl.from_arrow(rb)
        feats = compute_features(chunk, t1, t2_, gt)
        batch_no += 1
        p = os.path.join(DATA_DIR, f"_feat_batch_{src}_{batch_no}.parquet")
        feats.write_parquet(p)
        files.append(p)
        written += feats.height
        print(f"  ...{written:,}/{n_total:,} rows featured "
              f"({time.time()-tb:.0f}s/batch, mem {mem_pct():.0f}%)",
              flush=True)
        guard("feature batch")
        chunk = feats = None

    out = os.path.join(DATA_DIR, f"features_train_s{src}.parquet")
    (pl.scan_parquet(files).sink_parquet(out))
    for p in files:
        os.remove(p)
    print(f"  wrote -> {out} ({time.time()-t0:.0f}s)", flush=True)

    df = pl.scan_parquet(out).collect()
    pos = df["label"].sum()
    tot = df.height
    print(f"  rows: {tot:,}", flush=True)
    print(f"  label balance: positives={pos:,} ({100*pos/tot:.3f}%), "
          f"negatives={tot-pos:,} ({100*(tot-pos)/tot:.3f}%)", flush=True)
    samp = (df.sample(10, seed=42)
            .join(t1.select("entity_id", "name_norm_l"),
                  left_on="entity_id_l", right_on="entity_id")
            .join(t2_.select("entity_id", "name_norm_r"),
                  left_on="entity_id_r", right_on="entity_id")
            .select("entity_id_l", "entity_id_r", "name_norm_l",
                    "name_norm_r", *FEATURES))
    pl.Config.set_tbl_cols(16)
    pl.Config.set_tbl_rows(12)
    pl.Config.set_tbl_width_chars(200)
    print(samp)
    print(f"  peak memory observed: {mem_pct():.0f}%", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=int, choices=[2, 3], required=True)
    args = ap.parse_args()
    run(args.src)


if __name__ == "__main__":
    main()