import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyarrow.parquet as pq
import polars as pl

from config import DATA_DIR, ground_truth_path
from entsplit import load as load_entsplit
from features import side_table, compute_features, BATCH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=int, choices=[2, 3], required=True)
    ap.add_argument("--candidate", required=True, help="Path to full multipass candidates (candmpfit3_train)")
    ap.add_argument("--out", required=True, help="Output features parquet path")
    ap.add_argument("--negative-divisor", type=int, default=5,
                    help="retain one in N negative pairs; all positive pairs stay")
    args = ap.parse_args()
    if args.negative_divisor < 1:
        ap.error("--negative-divisor must be >= 1")

    t0 = time.time()
    print(f"=== Generating training features (train entities, sampled): S1 x S{args.src} ===", flush=True)

    # Load entity split
    entsplit = load_entsplit()
    train_ids = entsplit.filter(pl.col("isval") == 0)["entity_id"].to_list()
    print(f"  train entities: {len(train_ids):,}", flush=True)

    t1 = side_table(1).rename({c: c + "_l" for c in side_table(1).columns if c != "entity_id"})
    tr = side_table(args.src).rename({c: c + "_r" for c in side_table(args.src).columns if c != "entity_id"})

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
    n_total = pf.metadata.num_rows
    print(f"  candidate pairs: {n_total:,}", flush=True)

    files = []
    total_in = total_kept = positives = 0
    batch_no = 0
    for rb in pf.iter_batches(batch_size=BATCH, columns=["entity_id_l", "entity_id_r"]):
        chunk = pl.from_arrow(rb)
        total_in += chunk.height
        if not chunk.height:
            continue
        # Filter to train entities only
        chunk = chunk.filter(pl.col("entity_id_l").is_in(train_ids))
        if not chunk.height:
            continue

        # Mark positives and sample negatives deterministically
        marked = (chunk.join(gt, on=["entity_id_l", "entity_id_r"], how="left")
                  .with_columns(pl.col("_label").fill_null(0).cast(pl.Int8))
                  .with_columns(pl.struct(["entity_id_l", "entity_id_r"])
                                .hash(seed=42).alias("_h")))

        selected = (marked.filter(
            (pl.col("_label") == 1)
            | ((pl.col("_h") % args.negative_divisor) == 0))
            .select("entity_id_l", "entity_id_r"))

        # Compute features
        feats = compute_features(selected, t1, tr, gt.rename(
            {"entity_id_l": "source1_entity_id", "entity_id_r": "matched"}
            ).with_columns(pl.lit(1, dtype=pl.Int8).alias("gt_row")))
        feats = feats.with_columns(
            pl.lit(0 if args.src == 2 else 1, dtype=pl.Int8).alias("src"))

        total_kept += feats.height
        positives += int(feats["label"].sum())
        batch_no += 1
        path = os.path.join(DATA_DIR, f"_trainfeat_s{args.src}_{batch_no:04d}.parquet")
        feats.write_parquet(path)
        files.append(path)
        print(f"  batch {batch_no}: selected {feats.height:,} rows; positives={int(feats['label'].sum()):,}", flush=True)
        del chunk, marked, selected, feats
        gc.collect()

    if not files:
        raise SystemExit("No training candidate rows found")

    (pl.scan_parquet(files).sink_parquet(args.out, maintain_order=True))
    for path in files:
        os.remove(path)

    print(f"  training candidates seen: {total_in:,}; features retained: {total_kept:,}; positives: {positives:,} ({100*positives/max(total_kept,1):.3f}%)", flush=True)
    print(f"  wrote: {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()