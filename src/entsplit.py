import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import polars as pl

from config import DATA_DIR

# Deterministic 80/20 entity-level split over ALL training Source-1 entities
# (not just entities that appear in a candidate set).  val = md5(entity_id)
# % 10 < VAL_INT  -> 20% of entities, grouped by Source-1 entity id so no
# entity's pairs ever straddle the train/val boundary.
VAL_INT = 2
ENTSPLIT_PATH = os.path.join(DATA_DIR, "_entsplit_md5.parquet")


def _isval(e: str) -> int:
    h = int(hashlib.md5(e.encode("utf-8")).hexdigest()[:8], 16)
    return 1 if h % 10 < VAL_INT else 0


def build() -> pl.DataFrame:
    ids = (pl.scan_parquet(os.path.join(DATA_DIR, "norm_train_s1.parquet"))
           .select("entity_id").collect()["entity_id"].to_list())
    arr = np.array(sorted(ids), dtype=object)
    isv = np.array([_isval(x) for x in arr], dtype=np.int8)
    out = pl.DataFrame({"entity_id": arr, "isval": isv})
    out.write_parquet(ENTSPLIT_PATH)
    n = out.height
    nval = int(out["isval"].sum())
    print(f"  entity split -> {ENTSPLIT_PATH}: {n:,} S1 entities "
          f"({n - nval:,} train / {nval:,} val)", flush=True)
    return out


def load(force: bool = False) -> pl.DataFrame:
    if force or not os.path.isfile(ENTSPLIT_PATH):
        return build()
    return pl.read_parquet(ENTSPLIT_PATH)


if __name__ == "__main__":
    load(force=False)