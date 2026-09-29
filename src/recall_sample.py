import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

import polars as pl

from config import DATA_DIR, ground_truth_path
from measure_blocking import build_keys

SAMPLE_N = 50_000
SEED = 42
CUTOFF = 0.003


def norm_path(split: str, source: int) -> str:
    return os.path.join(DATA_DIR, f"norm_{split}_s{source}.parquet")


def compute_dropped(split: str, src: int) -> set:
    """High-frequency tokens (frac > CUTOFF on either side), same rule as
    measure_blocking, but computed with lazy scans so only frequent tokens
    (already filtered) are materialised."""
    out = set()
    for s in (1, src):
        n = (pl.scan_parquet(norm_path(split, s))
             .select(pl.len()).collect().item())
        fr = (pl.scan_parquet(norm_path(split, s))
              .select(pl.col("name_tokens").explode().alias("tok"))
              .drop_nulls()
              .group_by("tok").agg(pl.len().alias("n"))
              .filter(pl.col("n") > n * CUTOFF)
              .select("tok")
              .collect().to_series().to_list())
        out.update(fr)
    return out


def load_pairs(split: str, n: int, seed: int) -> tuple:
    """Sample n (source1_entity_id, other_entity_id) true-match pairs."""
    gt = pl.read_csv(
        ground_truth_path(split), separator="\t",
        columns=["source1_entity_id", "matched_entity_ids"],
    )
    pairs = (
        gt.filter(pl.col("matched_entity_ids").is_not_null())
        .with_columns(pl.col("matched_entity_ids").str.split(",").alias("ids"))
        .explode("ids")
        .select(
            pl.col("source1_entity_id").alias("s1_id"),
            pl.col("ids").str.strip_chars().alias("other_id"),
        )
        .filter(pl.col("other_id").str.starts_with("S2-")
                | pl.col("other_id").str.starts_with("S3-"))
        .unique(maintain_order=True)
    )
    sample = pairs.sample(n=n, seed=seed, shuffle=True)
    del gt, pairs
    return sample


def load_rows(path: str, ids: set) -> pl.DataFrame:
    if not ids:
        return pl.DataFrame(
            schema={"entity_id": pl.Utf8, "name_norm": pl.Utf8,
                    "name_tokens": pl.List(pl.Utf8), "addr_norm": pl.Utf8,
                    "country_norm": pl.Utf8}
        )
    return (pl.scan_parquet(path)
            .filter(pl.col("entity_id").is_in(sorted(ids)))
            .collect())


def to_lookup(keys_df: pl.DataFrame) -> dict:
    out = {}
    for e, ct2f, sdx0, pairf, ak in keys_df.select(
            ["entity_id", "ct2f", "sdx0", "pairf", "ak"]).iter_rows():
        out[e] = (ct2f, sdx0, pairf, ak)
    return out


def str_match(a, b) -> bool:
    return bool(a) and bool(b) and a == b


def list_match(a, b) -> bool:
    if not a or not b:
        return False
    s1 = set(a)
    s1.discard("")
    s2 = set(b)
    s2.discard("")
    return bool(s1 & s2)


def match(k1, k2, scheme: str) -> bool:
    if scheme in ("pairf", "ak"):
        return list_match(k1, k2)
    return str_match(k1, k2)


def run(split: str, n: int, seed: int) -> pl.DataFrame:
    t0 = time.time()
    print(f"sampling {n:,} true-match pairs from {split} gt (seed={seed})...",
          flush=True)
    sample = load_pairs(split, n, seed)

    s2_pairs = sample.filter(pl.col("other_id").str.starts_with("S2-"))
    s3_pairs = sample.filter(pl.col("other_id").str.starts_with("S3-"))

    s1_ids = set(sample["s1_id"].to_list())
    s2_ids = set(s2_pairs["other_id"].to_list())
    s3_ids = set(s3_pairs["other_id"].to_list())
    pt_counts = {"s1-s2": s2_pairs.height, "s1-s3": s3_pairs.height}
    print(f"  n_s1 ids={len(s1_ids):,}  n_s2 ids={len(s2_ids):,}  "
          f"n_s3 ids={len(s3_ids):,}  pair types: {pt_counts}", flush=True)

    print("computing dropped-token lists per src (lazy scan)...", flush=True)
    dropped = {2: compute_dropped(split, 2), 3: compute_dropped(split, 3)}
    print(f"  dropped tokens: s1xs2={len(dropped[2]):,} s1xs3={len(dropped[3]):,}",
          flush=True)

    print("loading only the needed source rows (lazy scan + is_in filter)...",
          flush=True)
    s1_sub = load_rows(norm_path(split, 1), s1_ids)
    s2_sub = load_rows(norm_path(split, 2), s2_ids)
    s3_sub = load_rows(norm_path(split, 3), s3_ids)
    print(f"  rows loaded: s1={s1_sub.height:,} s2={s2_sub.height:,} "
          f"s3={s3_sub.height:,}", flush=True)

    print("building blocking keys on subsets...", flush=True)
    s1k2 = to_lookup(build_keys(s1_sub, dropped[2]))
    s1k3 = to_lookup(build_keys(s1_sub, dropped[3]))
    s2k = to_lookup(build_keys(s2_sub, dropped[2]))
    s3k = to_lookup(build_keys(s3_sub, dropped[3]))
    del s1_sub, s2_sub, s3_sub
    print(f"keys built in {time.time()-t0:.0f}s", flush=True)

    schemes = ["ct2f", "sdx0", "pairf", "ak"]
    matched = {pt: {sc: 0 for sc in schemes} for pt in ("s1-s2", "s1-s3")}
    caught = {pt: 0 for pt in ("s1-s2", "s1-s3")}
    missing = {pt: 0 for pt in ("s1-s2", "s1-s3")}
    tot = {pt: 0 for pt in ("s1-s2", "s1-s3")}

    print("comparing keys per sampled true-match pair (no join)...", flush=True)
    for sid, oid in sample.select(["s1_id", "other_id"]).iter_rows():
        if oid.startswith("S2-"):
            pt, l1, lk = "s1-s2", s1k2, s2k
        else:
            pt, l1, lk = "s1-s3", s1k3, s3k
        tot[pt] += 1
        r1 = l1.get(sid)
        ro = lk.get(oid)
        if r1 is None or ro is None:
            missing[pt] += 1
            continue
        hit_any = False
        for i, sc in enumerate(schemes):
            if match(r1[i], ro[i], sc):
                matched[pt][sc] += 1
                hit_any = True
        if hit_any:
            caught[pt] += 1

    rows = []
    for pt in ("s1-s2", "s1-s3"):
        d = tot[pt]
        row = {"pair_type": pt, "n_pairs": d, "missing_ids": missing[pt]}
        for sc in schemes:
            p = 100 * matched[pt][sc] / max(d, 1)
            ci = 1.96 * (p / 100 * (1 - p / 100) / max(d, 1)) ** 0.5 * 100
            row[f"{sc}_%"] = round(p, 2)
            row[f"{sc}_±"] = round(ci, 2)
            row[f"{sc}_n"] = matched[pt][sc]
        p = 100 * caught[pt] / max(d, 1)
        ci = 1.96 * (p / 100 * (1 - p / 100) / max(d, 1)) ** 0.5 * 100
        row["ANY_%"] = round(p, 2)
        row["ANY_±"] = round(ci, 2)
        row["ANY_n"] = caught[pt]
        rows.append(row)

    return pl.DataFrame(rows)


def report(df: pl.DataFrame) -> None:
    print("\n===== SAMPLED RECALL CEILING (no-cap key equality) =====")
    hdr = (f"  {'pair_type':8} {'n_pairs':>9} {'missing':>7} "
           f"{'ct2f%':>8} {'sdx0%':>8} {'pairf%':>8} {'ak%':>8} {'ANY%':>8}")
    print(hdr)
    for r in df.iter_rows(named=True):
        print(
            f"  {r['pair_type']:8} {r['n_pairs']:>9,} {r['missing_ids']:>7} "
            f"{r['ct2f_%']:>7.2f}% {r['sdx0_%']:>7.2f}% "
            f"{r['pairf_%']:>7.2f}% {r['ak_%']:>7.2f}% "
            f"{r['ANY_%']:>7.2f}%"
        )
    print("\n  (col = % of sampled true-match pairs whose key matches on that")
    print("   scheme, no bucket cap applied; 95% CI ~ +/-1pt at these n)")
    for r in df.iter_rows(named=True):
        print(
            f"  {r['pair_type']}: "
            + ", ".join(f"{sc}={r[f'{sc}_%']}% ({r[f'{sc}_n']:,}/{r['n_pairs']:,})"
                        for sc in ("ct2f", "sdx0", "pairf", "ak"))
            + f" | ANY={r['ANY_%']}% ({r['ANY_n']:,}/{r['n_pairs']:,})"
        )


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else SAMPLE_N
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else SEED

    out_df = run(split, n, seed)
    report(out_df)

    out_path = os.path.join(DATA_DIR, f"recall_sample_{split}_{n}.csv")
    out_df.write_csv(out_path)
    print(f"\nsaved -> {out_path}")