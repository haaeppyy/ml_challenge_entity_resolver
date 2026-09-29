import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import polars as pl
import numpy as np
from config import DATA_DIR, ground_truth_path
from datasets import Dataset

def load_normalized_sources(split="train"):
    sources = {}
    for src in [1, 2, 3]:
        df = pl.read_parquet(os.path.join(DATA_DIR, f"norm_{split}_s{src}.parquet"))
        df = df.with_columns(
            (pl.col("name_norm") + " " + pl.col("addr_norm")).alias("text")
        ).select("entity_id", "text")
        sources[src] = df
        print(f"  S{src}: {df.height:,} entities")
    return sources


def load_ground_truth():
    gt = pl.read_csv(ground_truth_path("train"), separator="\t",
                     columns=["source1_entity_id", "matched_entity_ids"])
    gt = gt.filter(pl.col("matched_entity_ids").is_not_null())
    gt = gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("ids"))
    gt = gt.explode("ids")
    gt = gt.with_columns(pl.col("ids").str.strip_chars().alias("matched"))
    gt = gt.select(
        pl.col("source1_entity_id").alias("entity_id_l"),
        pl.col("matched").alias("entity_id_r")
    ).unique()
    print(f"Ground truth pairs: {gt.height:,}")
    return gt


def create_training_pairs_fast(sources, gt, max_neg_per_pos=3):
    """Create positive and negative pairs using efficient vectorized operations."""
    print("Creating positive pairs...")
    t0 = time.time()
    
    # Positive pairs: ground truth matches with texts
    s1_text = sources[1].select("entity_id", "text").rename({"text": "text_l"})
    s2_text = sources[2].select("entity_id", "text").rename({"text": "text_r"})
    s3_text = sources[3].select("entity_id", "text").rename({"text": "text_r"})
    
    # S1-S2 positives
    pos_s2 = gt.filter(pl.col("entity_id_r").str.starts_with("S2-"))
    pos_s2 = pos_s2.join(s1_text, left_on="entity_id_l", right_on="entity_id", how="inner")
    pos_s2 = pos_s2.join(s2_text, left_on="entity_id_r", right_on="entity_id", how="inner")
    pos_s2 = pos_s2.select("text_l", "text_r").with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
    print(f"  Positive S1-S2: {pos_s2.height:,} ({time.time()-t0:.1f}s)")
    
    # S1-S3 positives
    pos_s3 = gt.filter(pl.col("entity_id_r").str.starts_with("S3-"))
    pos_s3 = pos_s3.join(s1_text, left_on="entity_id_l", right_on="entity_id", how="inner")
    pos_s3 = pos_s3.join(s3_text, left_on="entity_id_r", right_on="entity_id", how="inner")
    pos_s3 = pos_s3.select("text_l", "text_r").with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
    print(f"  Positive S1-S3: {pos_s3.height:,} ({time.time()-t0:.1f}s)")
    
    pos_pairs = pl.concat([pos_s2, pos_s3])
    print(f"  Total positives: {pos_pairs.height:,}")
    
    # Negative pairs: efficient random sampling
    print("Creating negative pairs...")
    t0 = time.time()
    
    np.random.seed(42)
    neg_pairs_list = []
    
    # S1-S2 negatives
    s1_entities = sources[1].select("entity_id").to_series().to_list()
    s2_entities = sources[2].select("entity_id").to_series().to_list()
    
    # Build positive set for fast lookup
    pos_s2_keys = gt.filter(pl.col("entity_id_r").str.starts_with("S2-"))
    pos_s2_set = set(zip(pos_s2_keys["entity_id_l"].to_list(), pos_s2_keys["entity_id_r"].to_list()))
    
    # Sample negatives - target ratio
    target_neg_s2 = max_neg_per_pos * len(pos_s2_set)
    print(f"  Target negative S1-S2 pairs: {target_neg_s2:,}")
    
    # Efficient sampling: generate batch and filter
    neg_pairs_s2 = []
    batch_size = 100000
    attempts = 0
    while len(neg_pairs_s2) < target_neg_s2:
        batch_e1 = np.random.choice(s1_entities, size=batch_size)
        batch_e2 = np.random.choice(s2_entities, size=batch_size)
        for e1, e2 in zip(batch_e1, batch_e2):
            if (e1, e2) not in pos_s2_set:
                neg_pairs_s2.append((e1, e2))
                if len(neg_pairs_s2) >= target_neg_s2:
                    break
        attempts += 1
        if attempts % 50 == 0:
            print(f"    S1-S2: {len(neg_pairs_s2):,}/{target_neg_s2:,} ({time.time()-t0:.1f}s)")
    
    print(f"  Sampled {len(neg_pairs_s2):,} negative S1-S2 pairs ({time.time()-t0:.1f}s)")
    
    # S1-S3 negatives
    s3_entities = sources[3].select("entity_id").to_series().to_list()
    pos_s3_keys = gt.filter(pl.col("entity_id_r").str.starts_with("S3-"))
    pos_s3_set = set(zip(pos_s3_keys["entity_id_l"].to_list(), pos_s3_keys["entity_id_r"].to_list()))
    
    target_neg_s3 = max_neg_per_pos * len(pos_s3_set)
    print(f"  Target negative S1-S3 pairs: {target_neg_s3:,}")
    
    neg_pairs_s3 = []
    attempts = 0
    while len(neg_pairs_s3) < target_neg_s3:
        batch_e1 = np.random.choice(s1_entities, size=batch_size)
        batch_e2 = np.random.choice(s3_entities, size=batch_size)
        for e1, e2 in zip(batch_e1, batch_e2):
            if (e1, e2) not in pos_s3_set:
                neg_pairs_s3.append((e1, e2))
                if len(neg_pairs_s3) >= target_neg_s3:
                    break
        attempts += 1
        if attempts % 50 == 0:
            print(f"    S1-S3: {len(neg_pairs_s3):,}/{target_neg_s3:,} ({time.time()-t0:.1f}s)")
    
    print(f"  Sampled {len(neg_pairs_s3):,} negative S1-S3 pairs ({time.time()-t0:.1f}s)")
    
    # Get texts for negatives
    s1_text_dict = dict(sources[1].select("entity_id", "text").to_numpy())
    s2_text_dict = dict(sources[2].select("entity_id", "text").to_numpy())
    s3_text_dict = dict(sources[3].select("entity_id", "text").to_numpy())
    
    neg_s2 = pl.DataFrame({
        "text_l": [s1_text_dict[e1] for e1, e2 in neg_pairs_s2],
        "text_r": [s2_text_dict[e2] for e1, e2 in neg_pairs_s2],
        "label": pl.Series([0] * len(neg_pairs_s2), dtype=pl.Int8)
    })
    neg_s3 = pl.DataFrame({
        "text_l": [s1_text_dict[e1] for e1, e2 in neg_pairs_s3],
        "text_r": [s3_text_dict[e2] for e1, e2 in neg_pairs_s3],
        "label": pl.Series([0] * len(neg_pairs_s3), dtype=pl.Int8)
    })
    
    neg_pairs = pl.concat([neg_s2, neg_s3])
    print(f"Total negatives: {neg_pairs.height:,} ({time.time()-t0:.1f}s)")
    
    # Combine
    all_pairs = pl.concat([pos_pairs, neg_pairs])
    print(f"Total training pairs: {all_pairs.height:,}")
    print(f"  Positive: {pos_pairs.height:,}")
    print(f"  Negative: {neg_pairs.height:,}")
    
    return all_pairs


def main():
    t0 = time.time()
    print("=== Preparing Contrastive Training Data ===")
    
    sources = load_normalized_sources("train")
    gt = load_ground_truth()
    
    pairs = create_training_pairs_fast(sources, gt, max_neg_per_pos=3)
    
    # Save as parquet (avoid Arrow string limit issues)
    print("Saving as parquet...")
    out_path = os.path.join(DATA_DIR, "contrastive_train_pairs.parquet")
    pairs.write_parquet(out_path)
    print(f"Saved pairs parquet to {out_path}")
    print(f"Total time: {time.time()-t0:.1f}s")
    
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()