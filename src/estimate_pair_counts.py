import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

import polars as pl

from config import DATA_DIR
from measure_blocking import build_keys
from recall_sample import compute_dropped

CAPS = [None, 300, 100, 50]  # None = uncapped
SCHEMES = ["ct2f", "sdx0", "pairf", "ak", "UNION"]
LIST_KEYS = {"pairf", "ak"}


def norm_path(split: str, source: int) -> str:
    return os.path.join(DATA_DIR, f"norm_{split}_s{source}.parquet")


def load_side(path: str) -> pl.DataFrame:
    return (pl.scan_parquet(path)
            .select(["entity_id", "name_tokens", "addr_norm", "country_norm"])
            .collect(engine="streaming"))


def count_key(kdf: pl.DataFrame, col: str, list_mode: bool) -> pl.DataFrame:
    """Per-key row counts on ONE side: (k, n). List keys are exploded first so
    a record in multiple buckets contributes one row per bucket."""
    if list_mode:
        k = kdf.select(pl.col(col).alias("k")).explode("k")
    else:
        k = kdf.select(pl.col(col).alias("k"))
    return (k.drop_nulls()
            .filter(pl.col("k") != "")
            .group_by("k").len())


def count_union(kdf: pl.DataFrame) -> pl.DataFrame:
    """Per-key counts over the UNION of all 4 schemes' keys, with each record
    contributing each distinct key only once."""
    u = kdf.with_columns(
        pl.concat_list(["ct2f", "sdx0", "pairf", "ak"]).alias("u")
    ).select("u")
    u = u.select(
        pl.col("u").list.eval(
            pl.element().filter((pl.element().is_not_null())
                                & (pl.element() != ""))
        ).list.unique().alias("uu")
    ).explode("uu").drop_nulls()
    return u.group_by("uu").len().rename({"uu": "k"})


def pair_total(n1: pl.DataFrame, n2: pl.DataFrame, cap):
    """From the two small per-key count tables compute the candidate pair count.

    Returns (min_cap, drop_cap):
      min_cap  = sum(min(n1,cap) * min(n2,cap))         [user's formula]
      drop_cap = sum(n1*n2 over keys kept on both sides) [production: buckets
                above cap are dropped entirely on each side]
    For cap=None both are the exact uncapped sum."""
    if cap is None:
        j = n1.join(n2, on="k", how="inner")
        v = int(j.select((pl.col("n") * pl.col("n_right")).sum()).item())
        return v, v
    c1 = n1.with_columns(pl.min_horizontal(pl.col("n"), pl.lit(cap)).alias("n"))
    c2 = n2.with_columns(pl.min_horizontal(pl.col("n"), pl.lit(cap)).alias("n"))
    mm = int(c1.join(c2, on="k", how="inner")
             .select((pl.col("n") * pl.col("n_right")).sum()).item())
    d1 = n1.filter(pl.col("n") <= cap)
    d2 = n2.filter(pl.col("n") <= cap)
    dd = int(d1.join(d2, on="k", how="inner")
             .select((pl.col("n") * pl.col("n_right")).sum()).item())
    return mm, dd


def side_counts(split: str, source: int, dropped: set):
    t = time.time()
    df = load_side(norm_path(split, source))
    print(f"[s{source}] loaded {df.height:,} rows "
          f"({time.time()-t:.0f}s)", flush=True)
    kdf = build_keys(df, dropped)
    del df
    gc.collect()
    print(f"[s{source}] built keys ({time.time()-t:.0f}s)", flush=True)

    counts = {sc: count_key(kdf, sc, sc in LIST_KEYS) for sc in SCHEMES[:4]}
    counts["UNION"] = count_union(kdf)
    del kdf
    gc.collect()
    print(f"[s{source}] per-key counts done ({time.time()-t:.0f}s; "
          f"distinct keys: " + ", ".join(f"{sc}={counts[sc].height:,}"
                                          for sc in SCHEMES) + ")", flush=True)
    return counts


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "train"

    t0 = time.time()
    print("computing dropped-token union (lazy scans)...", flush=True)
    dropped = compute_dropped(split, 2) | compute_dropped(split, 3)
    print(f"  dropped tokens: {len(dropped):,} "
          f"({time.time()-t0:.0f}s)", flush=True)

    s1 = side_counts(split, 1, dropped)

    results = {}
    for src in (2, 3):
        other = side_counts(split, src, dropped)
        results[src] = {}
        for sc in SCHEMES:
            results[src][sc] = {
                cap: pair_total(s1[sc], other[sc], cap) for cap in CAPS
            }
        del other
        gc.collect()

    rows = []
    for sc in SCHEMES:
        for cap in CAPS:
            cap_lbl = "uncap" if cap is None else str(cap)
            m2, d2 = results[2][sc][cap]
            m3, d3 = results[3][sc][cap]
            rows.append({
                "scheme": sc, "cap": cap_lbl,
                "pairs_s1s2_min": m2, "pairs_s1s2_drop": d2,
                "pairs_s1s3_min": m3, "pairs_s1s3_drop": d3,
                "total_min": m2 + m3, "total_drop": d2 + d3,
            })

    out = pl.DataFrame(rows)
    print("\n===== ESTIMATED CANDIDATE PAIR COUNTS (no pair join) =====")
    hdr = (f"  {'scheme':7} {'cap':6} {'s1xs2':>14} {'s1xs3':>14} "
           f"{'combined':>14} {'combined':>14}")
    print(hdr)
    print(f"  {'':7} {'':6} {'drop-cap':>14} {'drop-cap':>14} "
          f"{'drop-cap':>14} {'(min-cap)':>14}")
    for r in out.iter_rows(named=True):
        print(
            f"  {r['scheme']:7} {r['cap']:6} "
            f"{r['pairs_s1s2_drop']:>14,} {r['pairs_s1s3_drop']:>14,} "
            f"{r['total_drop']:>14,} {r['total_min']:>14,}"
        )

    print("\n  drop-cap  = bucket dropped entirely when >cap on EITHER side "
          "(production semantics = real join size).")
    print("  min-cap   = min(n,cap) per side (upper-bound formula).")
    print("  ct2f/sdx0 are exact pair counts (1 key/record); pairf/ak/UNION")
    print("  are UPPER bounds: a pair sharing 2+ keys is counted multiple times.")
    print(f"  keys built with dropped-token union across src2/src3 "
          f"({len(dropped):,} tokens); cap column 'uncap' has drop==min.")

    worst = out["total_drop"].max()
    print(f"\n  largest single-combination estimate: {worst:,} pairs "
          f"(~{worst*16/1e9:.1f} GB as bare 16B/pair; "
          f"~{worst*64/1e9:.1f} GB as 64B/pair rows).")
    print(f"  done in {time.time()-t0:.0f}s")

    out_path = os.path.join(DATA_DIR, f"pair_count_est_{split}.csv")
    out.write_csv(out_path)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()