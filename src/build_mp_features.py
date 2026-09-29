"""Build a training feature set from multipass candidates.

Only training-fold S1 entities are retained. All positive candidates are kept;
negative candidates are sampled deterministically before expensive string
similarity features are calculated. Validation candidates remain untouched.
"""
import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyarrow.parquet as pq
import polars as pl

from config import DATA_DIR, ground_truth_path
from entsplit import load as load_entsplit
from features import side_table, compute_features, BATCH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=int, choices=[2, 3], required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--negative-divisor", type=int, default=5,
                    help="retain one in N negative pairs; all positive pairs stay")
    args = ap.parse_args()
    if args.negative_divisor < 1:
        ap.error("--negative-divisor must be >= 1")

    split = load_entsplit()
    train_ids = split.filter(pl.col("isval") == 0)["entity_id"]
    print(f"training-fold S1 entities: {train_ids.len():,}", flush=True)

    t1 = side_table(1).rename({c: c + "_l" for c in side_table(1).columns
                               if c != "entity_id"})
    tr = side_table(args.src).rename({c: c + "_r" for c in side_table(args.src).columns
                                      if c != "entity_id"})
    gt = (pl.read_csv(ground_truth_path("train"), separator="\t",
                      columns=["source1_entity_id", "matched_entity_ids"])
          .filter(pl.col("matched_entity_ids").is_not_null())
          .with_columns(pl.col("matched_entity_ids").str.split(",").alias("ids"))
          .explode("ids")
          .select(pl.col("source1_entity_id").alias("entity_id_l"),
                  pl.col("ids").str.strip_chars().alias("entity_id_r"))
          .filter(pl.col("entity_id_r").str.starts_with(f"S{args.src}-"))
          .unique()
          .with_columns(pl.lit(1, dtype=pl.Int8).alias("_label")))

    pf = pq.ParquetFile(args.candidate)
    outputs = []
    total_in = total_kept = positives = 0
    for batch_no, rb in enumerate(pf.iter_batches(
            batch_size=BATCH, columns=["entity_id_l", "entity_id_r"]), 1):
        chunk = (pl.from_arrow(rb)
                 .filter(pl.col("entity_id_l").is_in(train_ids)))
        total_in += chunk.height
        if not chunk.height:
            continue
        marked = (chunk.join(gt, on=["entity_id_l", "entity_id_r"],
                             how="left")
                  .with_columns(pl.col("_label").fill_null(0).cast(pl.Int8))
                  .with_columns(pl.struct(["entity_id_l", "entity_id_r"])
                                .hash(seed=42).alias("_h")))
        selected = (marked.filter(
            (pl.col("_label") == 1)
            | ((pl.col("_h") % args.negative_divisor) == 0))
            .select("entity_id_l", "entity_id_r"))
        feats = compute_features(selected, t1, tr, gt.rename(
            {"entity_id_l": "source1_entity_id", "entity_id_r": "matched"})
            .with_columns(pl.lit(1, dtype=pl.Int8).alias("gt_row")))
        feats = feats.with_columns(
            pl.lit(0 if args.src == 2 else 1, dtype=pl.Int8).alias("src"))
        total_kept += feats.height
        positives += int(feats["label"].sum())
        path = os.path.join(DATA_DIR,
                            f"_mpfeat_fit_s{args.src}_{batch_no:04d}.parquet")
        feats.write_parquet(path)
        outputs.append(path)
        print(f"batch {batch_no}: selected {feats.height:,} rows; "
              f"positive labels={int(feats['label'].sum()):,}", flush=True)
        del chunk, marked, selected, feats
        gc.collect()

    if not outputs:
        raise SystemExit("No training candidate rows were found")
    (pl.scan_parquet(outputs)
     .sink_parquet(args.out, maintain_order=True))
    for path in outputs:
        os.remove(path)
    print(f"training candidates seen: {total_in:,}; features retained: "
          f"{total_kept:,}; positives retained: {positives:,} "
          f"({100*positives/max(total_kept, 1):.3f}%)", flush=True)
    print(f"wrote: {args.out}", flush=True)


if __name__ == "__main__":
    main()
