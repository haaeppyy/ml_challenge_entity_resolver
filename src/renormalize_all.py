import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import polars as pl
from normalize import normalize_business
from config import DATA_DIR

def renormalize_split(split: str, src: int):
    """Re-normalize a source file and overwrite the norm parquet."""
    csv_path = os.path.join(DATA_DIR.replace('data', 'dataset'), split, f"{split}_source{src}.tsv")
    parquet_path = os.path.join(DATA_DIR, f"norm_{split}_s{src}.parquet")
    
    print(f"Reading {csv_path}...")
    df = pl.read_csv(csv_path, separator='\t')
    print(f"  Total rows: {df.height}")
    
    normalized = normalize_business(df)
    print(f"  Writing to {parquet_path}...")
    normalized.write_parquet(parquet_path)
    print(f"  Done.")

if __name__ == "__main__":
    # Renormalize test sources
    for src in [1, 2, 3]:
        renormalize_split("test", src)
    
    # Also renormalize train sources for consistency
    for src in [1, 2, 3]:
        renormalize_split("train", src)
    
    print("All done!")