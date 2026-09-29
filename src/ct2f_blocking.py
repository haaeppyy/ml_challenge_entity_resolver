import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import polars as pl

from config import DATA_DIR, ground_truth_path

KEY_SEP = "|"
MAX_PAIRS = 100_000_000
MEM_WARN_PCT = 85.0
BATCH_ROWS = 4_000_000
BUCKET_CAP = 300

STOP = {
    "and", "the", "of", "at", "by", "for", "in", "co", "ltd", "llc", "inc",
    "pvt", "gmbh", "corp", "dr", "mr", "mrs", "ms", "smt", "shri", "sri",
    "prof", "district", "private", "limited",
}


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


def ct2f(toks):
    toks = toks.to_list()
    sig = [t for t in toks if t not in STOP][:2]
    if not sig:
        return [None]
    return [KEY_SEP.join(sorted(sig))]


def key_table(split, src):
    df = (pl.scan_parquet(os.path.join(DATA_DIR, f"norm_{split}_s{src}.parquet"))
          .select(pl.col("entity_id"), "country_norm", "name_tokens")
          .collect())
    guard("key table loaded")
    df = df.with_columns(
        pl.col("name_tokens").map_elements(
            ct2f, return_dtype=pl.List(pl.Utf8), skip_nulls=False).alias("ctk"))
    df = df.with_columns(
        (pl.col("country_norm") + KEY_SEP + pl.col("ctk").list.first()).alias("bk"))
    df = df.filter(pl.col("bk").is_not_null()).select("entity_id", "bk")
    df = (df.with_columns(pl.col("entity_id").rank("ordinal").over("bk")
                          .alias("rk"))
          .filter(pl.col("rk") <= BUCKET_CAP)
          .drop("rk"))
    return df


def pair_counts(df1, df2):
    c1 = (df1.group_by("bk").agg(pl.col("entity_id").alias("ids1"))
          .to_pandas().set_index("bk"))
    c2 = (df2.group_by("bk").agg(pl.col("entity_id").alias("ids2"))
          .to_pandas().set_index("bk"))
    inter = c1.index.intersection(c2.index)
    n1s = c1.loc[inter, "ids1"].map(len).astype(int)
    n2s = c2.loc[inter, "ids2"].map(len).astype(int)
    n = int((n1s * n2s).sum())
    return c1, c2, inter, n


def generate_pairs(c1, c2, inter, src, out_path):
    t0 = time.time()
    produced = 0
    batch_no = 0
    files = []
    L, R = [], []
    for k in inter:
        ids1 = c1.loc[k, "ids1"]
        ids2 = c2.loc[k, "ids2"]
        ids1 = (ids1 if isinstance(ids1, list)
                else ([ids1] if isinstance(ids1, str) else ids1.tolist()))
        ids2 = (ids2 if isinstance(ids2, list)
                else ([ids2] if isinstance(ids2, str) else ids2.tolist()))
        for a in ids1:
            for b in ids2:
                L.append(a)
                R.append(b)
        produced += len(ids1) * len(ids2)
        if len(L) >= BATCH_ROWS:
            chunk = pl.DataFrame({"entity_id_l": L, "entity_id_r": R})
            L, R = [], []
            batch_no += 1
            p = os.path.join(DATA_DIR, f"_ct2f_batch_{src}_{batch_no}.parquet")
            chunk.write_parquet(p)
            files.append(p)
            print(f"  ...{produced:,} pairs streamed (mem {mem_pct():.0f}%)",
                  flush=True)
            guard("pair streaming")
    if L:
        chunk = pl.DataFrame({"entity_id_l": L, "entity_id_r": R})
        batch_no += 1
        p = os.path.join(DATA_DIR, f"_ct2f_batch_{src}_{batch_no}.parquet")
        chunk.write_parquet(p)
        files.append(p)
    if files:
        (pl.scan_parquet(files)
         .select(pl.col("entity_id_l").cast(pl.Utf8),
                 pl.col("entity_id_r").cast(pl.Utf8))
         .sink_parquet(out_path))
        for p in files:
            os.remove(p)
        produced = pl.scan_parquet(out_path).select(pl.len()).collect().item()
    print(f"  wrote -> {out_path} ({time.time()-t0:.0f}s)", flush=True)
    return produced


def recall(split, src, out_path):
    cand = pl.scan_parquet(out_path).collect()
    gt_path = ground_truth_path(split)
    if not os.path.exists(gt_path):
        return None
    gt = (pl.read_csv(gt_path, separator="\t",
                      columns=["source1_entity_id", "matched_entity_ids"])
          .filter(pl.col("matched_entity_ids").is_not_null())
          .with_columns(pl.col("matched_entity_ids").str.split(",").alias("ids"))
          .explode("ids")
          .select(pl.col("source1_entity_id").cast(pl.Utf8),
                  pl.col("ids").str.strip_chars().cast(pl.Utf8)
                  .alias("other_entity_id"))
          .filter(pl.col("other_entity_id").str.starts_with(f"S{src}-")))
    hit = (gt.join(cand, left_on=["source1_entity_id", "other_entity_id"],
                   right_on=["entity_id_l", "entity_id_r"], how="semi")
           .height)
    return gt.height, hit


def run(split, src):
    print(f"=== ct2f blocking: {split} S1 x S{src} ===", flush=True)
    print("key = country + sorted{first-2 significant name tokens} "
          f"(no soundex) with cap {BUCKET_CAP}/side", flush=True)
    df1 = key_table(split, 1)
    df2 = key_table(split, src)
    print(f"  rows keyed: s1={df1.height:,}  s{src}={df2.height:,}", flush=True)
    guard("keys built")

    c1, c2, inter, n = pair_counts(df1, df2)
    df1 = df2 = None
    print(f"  total capped-300 ct2f pairs (s1 x s{src}): {n:,}", flush=True)

    out = os.path.join(DATA_DIR, f"candidates_ct2f_{split}_s{src}.parquet")
    if n > MAX_PAIRS:
        print(f"ABORTING {split} S1 x S{src}: total pairs {n:,} exceeds 100M "
              f"limit -> not writing candidates", flush=True)
        return

    produced = generate_pairs(c1, c2, inter, src, out)
    print(f"  FINAL candidate pair count ({split} s1xs{src}): {produced:,}",
          flush=True)
    r = recall(split, src, out)
    if r is None:
        print("  recall: no ground truth for this split", flush=True)
    else:
        tot, hit = r
        print(f"  GROUND-TRUTH recall: {hit:,}/{tot:,} = {100*hit/tot:.2f}%",
              flush=True)
    print(f"  peak memory observed: {mem_pct():.0f}%", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--src", type=int, choices=[2, 3], required=True)
    args = ap.parse_args()
    run(args.split, args.src)


if __name__ == "__main__":
    main()