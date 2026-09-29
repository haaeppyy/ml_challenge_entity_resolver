import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

import polars as pl

from config import DATA_DIR, ground_truth_path
from blocking import _soundex


BUCKET_CAP = 50          # hard cap on rows per bucket (per side, per key)
ABORT_LIMIT = 30_000_000 # refuse to write more than this many pairs in a combo
CHUNK_PAIRS = 1_000_000  # target pair rows per eager cross-product chunk
FLUSH_ROWS = 2_000_000   # write a part file once this many buffered pair rows
MEM_WARN_PCT = 90.0      # abort if virtual-memory usage passes this
MEM_PRINT_EVERY = 5      # print mem stats every N part flushes

PAIR_SCHEMA = {
    "source1_entity_id": pl.Utf8,
    "other_entity_id": pl.Utf8,
}


def norm_path(split: str, src: int) -> str:
    return os.path.join(DATA_DIR, f"norm_{split}_s{src}.parquet")


def mem_pct() -> float:
    try:
        import psutil
        return float(psutil.virtual_memory().percent)
    except Exception:
        return 0.0


def build_buckets(split: str, src: int, cap: int) -> pl.DataFrame:
    """One lazy pass over a normalized parquet: assign sdx0 key, group by key,
    and keep at most `cap` entity_ids per bucket (first 50 in scan order)."""
    lf = (pl.scan_parquet(norm_path(split, src))
          .select(pl.col("entity_id"), pl.col("country_norm"),
                  pl.col("name_tokens"))
          .with_columns(
              pl.col("name_tokens").list.get(0, null_on_oob=True).alias("tok"))
          .with_columns(
              (pl.col("country_norm") + "_"
               + pl.col("tok").map_elements(
                   _soundex, return_dtype=pl.Utf8, skip_nulls=False)
               ).alias("key"))
          .filter(pl.col("key").is_not_null() & (~pl.col("key").str.ends_with("_")))
          .select("entity_id", "key"))
    buckets = (lf.group_by("key")
               .agg(pl.col("entity_id").head(cap).alias("ids")))
    return buckets.collect()


def delete_parts(parts: list) -> None:
    for p in parts:
        try:
            os.remove(p)
        except OSError:
            pass


def consolidate(parts: list, final_path: str) -> None:
    """Memory-safe concat of part files into the single final parquet."""
    if not parts:
        pl.DataFrame(schema=PAIR_SCHEMA).write_parquet(final_path)
        return
    (pl.scan_parquet(parts)
     .select(pl.col("source1_entity_id"), pl.col("other_entity_id"))
     .sink_parquet(final_path, maintain_order=True, row_group_size=1_000_000))
    delete_parts(parts)


def generate_pairs(s1: pl.DataFrame, s2: pl.DataFrame,
                   final_path: str, limit: int) -> int:
    """Cross-product of capped buckets, chunked so no single eager op grows
    beyond CHUNK_PAIRS rows. Runs the real thing with progress + memory guard."""
    t0 = time.time()
    if s1.height == 0 or s2.height == 0:
        pl.DataFrame(schema=PAIR_SCHEMA).write_parquet(final_path)
        return 0

    k1 = s1.with_columns(pl.col("ids").list.len().alias("n1")).select("key", "n1")
    k2 = s2.with_columns(pl.col("ids").list.len().alias("n2")).select("key", "n2")
    sizes = (k1.join(k2, on="key", how="inner")
             .with_columns((pl.col("n1") * pl.col("n2")).alias("pairs")))
    nkeys = sizes.height
    projected = int(sizes["pairs"].sum())
    print(f"    shared keys: {nkeys:,}   projected pairs: {projected:,}",
          flush=True)
    if projected > limit:
        print(f"    ABORT: projected pairs {projected:,} exceed limit "
              f"{limit:,} for {final_path}", flush=True)
        return -1

    keys = sizes.get_column("key").to_list()
    pairs = sizes.get_column("pairs").to_list()

    parts: list = []
    total = 0
    buf: list = []
    buf_rows = 0
    flush_n = 0

    i = 0
    while i < nkeys:
        chunk_keys = []
        chunk_pairs = 0
        while i < nkeys and chunk_pairs < CHUNK_PAIRS:
            chunk_keys.append(keys[i])
            chunk_pairs += pairs[i]
            i += 1
        if chunk_pairs > CHUNK_PAIRS:
            # a single key pair count alone exceeds the chunk target is fine
            # (max 50*50 = 2500), so this branch never triggers; kept for safety.
            pass

        c1 = (s1.filter(pl.col("key").is_in(chunk_keys))
              .select("key", "ids").explode("ids")
              .select(pl.col("key"), pl.col("ids").alias("source1_entity_id")))
        c2 = (s2.filter(pl.col("key").is_in(chunk_keys))
              .select("key", "ids").explode("ids")
              .select(pl.col("key"), pl.col("ids").alias("other_entity_id")))
        chunk = c1.join(c2, on="key", how="inner").drop("key")
        total += chunk.height
        buf.append(chunk)
        buf_rows += chunk.height
        del c1, c2, chunk

        if buf_rows >= FLUSH_ROWS:
            part = f"{final_path}.part{flush_n:04d}"
            (pl.concat(buf).write_parquet(part))
            parts.append(part)
            buf = []
            buf_rows = 0
            flush_n += 1
            mp = mem_pct()
            print(f"    ... {total:,} pairs ({final_path})  "
                  f"[mem {mp:.0f}%]  ({time.time()-t0:.0f}s)", flush=True)
            if flush_n % MEM_PRINT_EVERY == 0 and mp >= MEM_WARN_PCT:
                print(f"    ABORT: memory at {mp:.0f}% while generating "
                      f"{final_path}", flush=True)
                delete_parts(parts)
                return -1
            if total > limit:
                print(f"    ABORT: running pair total {total:,} exceeds "
                      f"{limit:,}", flush=True)
                delete_parts(parts)
                return -1
            gc.collect()

    if buf:
        part = f"{final_path}.part{flush_n:04d}"
        pl.concat(buf).write_parquet(part)
        parts.append(part)

    consolidate(parts, final_path)
    print(f"    WROTE {total:,} pairs -> {final_path} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return total


def recall_capped(split: str, src: int, s1: pl.DataFrame,
                  s2: pl.DataFrame) -> tuple:
    """% of true train matches whose pair landed in one bucket AND both sides
    survived the cap (i.e. both ids are present in the capped buckets)."""
    gt = (pl.read_csv(ground_truth_path(split), separator="\t",
                      columns=["source1_entity_id", "matched_entity_ids"])
          .filter(pl.col("matched_entity_ids").is_not_null())
          .with_columns(pl.col("matched_entity_ids").str.split(",").alias("ids"))
          .explode("ids")
          .select(pl.col("source1_entity_id"),
                  pl.col("ids").str.strip_chars().alias("other_entity_id"))
          .filter(pl.col("other_entity_id").str.starts_with(f"S{src}-"))
          .unique())
    total = gt.height

    m1 = (s1.explode("ids").select(
        pl.col("ids").alias("source1_entity_id"), pl.col("key").alias("key1")))
    m2 = (s2.explode("ids").select(
        pl.col("ids").alias("other_entity_id"), pl.col("key").alias("key2")))

    g = (gt.join(m1, on="source1_entity_id", how="inner")
          .join(m2, on="other_entity_id", how="inner")
          .filter(pl.col("key1") == pl.col("key2")))
    captured = g.height
    del m1, m2, g, gt
    gc.collect()
    return captured, total


def run_combo(split: str, src: int, s1_buckets: pl.DataFrame,
              out_dir: str) -> int:
    final_path = os.path.join(out_dir, f"candidates_{split}_s{src}.parquet")
    print(f"  === {split} S1 x S{src} ===", flush=True)
    s2_buckets = build_buckets(split, src, BUCKET_CAP)
    print(f"  [{split} s{src}] capped buckets: "
          f"{s2_buckets.height:,} keys", flush=True)
    n = generate_pairs(s1_buckets, s2_buckets, final_path, ABORT_LIMIT)
    if n < 0:
        print(f"  {split} S1 x S{src}: ABORTED (limit/memory)", flush=True)
        del s2_buckets
        gc.collect()
        return n
    if split == "train":
        captured, total = recall_capped(split, src, s1_buckets, s2_buckets)
        rec = 100 * captured / max(total, 1)
        print(f"  [{split} s1 x s{src}] capped recall: {captured:,}/{total:,} "
              f"= {rec:.2f}%", flush=True)
    del s2_buckets
    gc.collect()
    return n


def main() -> None:
    out_dir = DATA_DIR
    reported = {}

    for split in ("train", "test"):
        t = time.time()
        print(f"=== {split} ===", flush=True)
        s1_buckets = build_buckets(split, 1, BUCKET_CAP)
        print(f"[{split} s1] capped buckets: {s1_buckets.height:,} keys "
              f"({time.time()-t:.0f}s)", flush=True)
        for src in (2, 3):
            n = run_combo(split, src, s1_buckets, out_dir)
            reported[f"{split}_s{src}"] = n
            if n < 0:
                print("STOPPING: abort triggered — see messages above.",
                      flush=True)
                return
        del s1_buckets
        gc.collect()

    print("\n===== FINAL CANDIDATE PAIR COUNTS =====")
    for split in ("train", "test"):
        for src in (2, 3):
            label = f"{split}_s{src}"
            n = reported[label]
            print(f"  S1 x S{src} ({split}): {n:,} pairs -> "
                  f"{os.path.join(out_dir, 'candidates_' + split + '_s' + str(src) + '.parquet')}",
                  flush=True)
    print("(capped recall numbers for train were printed inline during the run)")


if __name__ == "__main__":
    main()