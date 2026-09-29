import argparse
import os
import sys
import time
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyarrow.parquet as pq
import polars as pl
from rapidfuzz.distance import JaroWinkler, Levenshtein, Jaro
import re
from config import DATA_DIR, ground_truth_path

BATCH = 1_000_000  # Reduced batch size
MEM_WARN_PCT = 90.0  # Increased threshold
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


def side_table(src):
    df = (pl.scan_parquet(os.path.join(DATA_DIR, f"norm_train_s{src}.parquet"))
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


def gt_dict(src):
    """Load ground truth as a dict for fast lookup"""
    gt_path = ground_truth_path("train")
    gt = (pl.read_csv(gt_path, separator="\t",
                      columns=["source1_entity_id", "matched_entity_ids"])
          .filter(pl.col("matched_entity_ids").is_not_null())
          .with_columns(pl.col("matched_entity_ids").str.split(",")
                        .alias("ids"))
          .explode("ids")
          .select(pl.col("source1_entity_id").cast(pl.Utf8),
                  pl.col("ids").str.strip_chars().cast(pl.Utf8)
                  .alias("matched"))
          .filter(pl.col("matched").str.starts_with(f"S{src}-")))
    # Convert to dict of sets for O(1) lookup: {entity_id_l: set(entity_id_r)}
    gt_dict = {}
    for l, r in zip(gt["source1_entity_id"].to_list(), gt["matched"].to_list()):
        if l not in gt_dict:
            gt_dict[l] = set()
        gt_dict[l].add(r)
    print(f"  GT dict for S{src}: {len(gt_dict):,} source entities, {sum(len(v) for v in gt_dict.values()):,} pairs", flush=True)
    return gt_dict


def compute_features(chunk, t1, tr, gt_dict):
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

    # Label lookup using dict
    labels = []
    for l_id, r_id in zip(chunk["entity_id_l"].to_list(), chunk["entity_id_r"].to_list()):
        labels.append(1 if l_id in gt_dict and r_id in gt_dict[l_id] else 0)
    label = pl.Series(labels, dtype=pl.Int8)

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
        "label": label,
    })
    return out.select("entity_id_l", "entity_id_r", *FEATURES)


def run(src):
    t0 = time.time()
    print(f"=== feature engineering: train S1 x S{src} (multipass core) ===", flush=True)
    cand_path = os.path.join(DATA_DIR, f"candmp_core_train_s{src}.parquet")

    print("  loading S1 side table...", flush=True)
    s1_base = side_table(1)
    t1 = s1_base.rename({c: c + "_l" for c in s1_base.columns if c != "entity_id"})
    del s1_base
    gc.collect()
    guard("s1 table loaded")

    print(f"  loading S{src} side table...", flush=True)
    s2_base = side_table(src)
    t2_ = s2_base.rename({c: c + "_r" for c in s2_base.columns if c != "entity_id"})
    del s2_base
    gc.collect()
    guard(f"s{src} table loaded")

    print("  loading ground truth...", flush=True)
    gt_dict_ = gt_dict(src)
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
        feats = compute_features(chunk, t1, t2_, gt_dict_)
        batch_no += 1
        p = os.path.join(DATA_DIR, f"_feat_mp_{src}_{batch_no}.parquet")
        feats.write_parquet(p)
        files.append(p)
        written += feats.height
        print(f"  ...{written:,}/{n_total:,} rows featured "
              f"({time.time()-tb:.0f}s/batch, mem {mem_pct():.0f}%)",
              flush=True)
        guard("feature batch")
        del chunk, feats
        gc.collect()

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
    print(f"  peak memory observed: {mem_pct():.0f}%", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=int, choices=[2, 3], required=True)
    args = ap.parse_args()
    run(args.src)


if __name__ == "__main__":
    main()