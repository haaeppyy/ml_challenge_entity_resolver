import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import polars as pl

from config import DATA_DIR, ground_truth_path

SEED = 42
N_SAMPLE = 100_000


def norm_path(split: str, src: int) -> str:
    return os.path.join(DATA_DIR, f"norm_{split}_s{src}.parquet")


def sample_side(split: str, src: int) -> tuple:
    """Build a 100K-row sample of s{src} that contains every ground-truth
    match of a 100K-row sample of s1, plus the pairwise labels table.

    Returns paths to two parquets (s1 sample, s{src} sample) and a labels df.
    """
    s1_ids = (pl.scan_parquet(norm_path(split, 1))
              .select("entity_id").collect()["entity_id"]
              .sample(n=N_SAMPLE, seed=SEED))
    src_all = pl.scan_parquet(norm_path(split, src)).select("entity_id")

    gt = pl.read_csv(ground_truth_path(split), separator="\t",
                     columns=["source1_entity_id", "matched_entity_ids"])
    gt = (gt.filter(pl.col("matched_entity_ids").is_not_null())
          .with_columns(pl.col("matched_entity_ids").str.split(",").alias("ids"))
          .explode("ids")
          .select(pl.col("source1_entity_id"),
                  pl.col("ids").str.strip_chars().alias("other_entity_id"))
          .filter(pl.col("other_entity_id").str.starts_with(f"S{src}-"))
          .unique())

    # true pairs where the s1 side is in our sample
    in_sample = gt.filter(pl.col("source1_entity_id").is_in(s1_ids))
    matched_src = in_sample["other_entity_id"].unique().to_list()

    # build the src sample: matched rows + random fill to N_SAMPLE
    src_frame = pl.scan_parquet(norm_path(split, src))
    matched_df = (src_frame.filter(pl.col("entity_id").is_in(matched_src))
                  .collect())
    n_fill = N_SAMPLE - matched_df.height
    fill_pool = (src_frame.select("entity_id").collect()
                 .filter(~pl.col("entity_id").is_in(matched_src)))
    fill_ids = fill_pool.sample(n=max(n_fill, 0), seed=SEED).to_series()
    fill_df = src_frame.filter(pl.col("entity_id").is_in(fill_ids)).collect()
    src_sample = pl.concat([matched_df, fill_df])

    s1_sample = pl.scan_parquet(norm_path(split, 1)).filter(
        pl.col("entity_id").is_in(s1_ids)).collect()

    out_s1 = os.path.join(DATA_DIR, f"prep_sample_{split}_s1.parquet")
    out_src = os.path.join(DATA_DIR, f"prep_sample_{split}_s{src}.parquet")
    s1_sample.write_parquet(out_s1)
    src_sample.write_parquet(out_src)

    labels = in_sample.select(
        pl.lit("s1").alias("source_dataset_l"),
        pl.col("source1_entity_id").alias("entity_id_l"),
        pl.lit(f"s{src}").alias("source_dataset_r"),
        pl.col("other_entity_id").alias("entity_id_r"),
    ).unique()

    print(f"[sample {split} s1xs{src}] s1={s1_sample.height:,} "
          f"src={src_sample.height:,} (matched rows {matched_df.height:,}) "
          f"true pairs in sample={labels.height:,} "
          f"saved -> {out_s1}, {out_src}")
    return out_s1, out_src, labels


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    for src in (2, 3):
        sample_side(split, src)