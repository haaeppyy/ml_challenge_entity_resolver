import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import polars as pl

from config import DATA_DIR, ground_truth_path
from blocking import (
    load_normalized, build_block_key_columns, _cap_keys, _soundex,
    BUCKET_CAP,
)


def per_pass_recall(split: str, src: int, cap: int = BUCKET_CAP):
    s1 = load_normalized(split, 1)
    other = load_normalized(split, src)
    keys = ["blk_a", "blk_b1", "blk_b2"]

    s1k = build_block_key_columns(s1).select(["entity_id"] + keys)
    ok = build_block_key_columns(other).select(["entity_id"] + keys)

    gt = pl.read_csv(ground_truth_path(split), separator="\t")
    gt = (
        gt.filter(pl.col("matched_entity_ids").is_not_null())
        .with_columns(pl.col("matched_entity_ids").str.split(",").alias("ids"))
        .explode("ids")
        .select(
            pl.col("source1_entity_id"),
            pl.col("ids").str.strip_chars().alias("other_entity_id"),
        )
        .filter(pl.col("other_entity_id").str.starts_with(f"S{src}-"))
        .unique()
    )
    total = gt.height

    g = gt.join(s1k, left_on="source1_entity_id", right_on="entity_id", how="left")
    g = g.join(ok, left_on="other_entity_id", right_on="entity_id", how="left",
               suffix="_o")

    results = {}
    union_cands = 0
    for key in keys:
        keep_s1 = _cap_keys(s1k.select(["entity_id", key]), key, cap)
        keep_o = _cap_keys(ok.select(["entity_id", key]), key, cap)
        cap_s1 = keep_s1.select(key).unique()
        cap_o = keep_o.select(key).unique()
        both = cap_s1.join(cap_o, on=key, how="inner").select(key)
        n1 = keep_s1.group_by(key).len()
        n2 = keep_o.group_by(key).len()
        pair_c = int(both.join(n1, on=key).join(n2, on=key)
                     .select((pl.col("len") * pl.col("len_right")).sum()).item())
        hit = g.filter(pl.col(key) == pl.col(key + "_o"))
        captured = hit.join(both, on=key, how="semi").height
        results[key] = captured
        union_cands += pair_c
        print(f"  pass {key}: cand={pair_c:,} "
              f"recall={captured:,}/{total:,} = {100*captured/total:.2f}%",
              flush=True)

    union = set()
    for key in keys:
        keep_s1 = _cap_keys(s1k.select(["entity_id", key]), key, cap)
        keep_o = _cap_keys(ok.select(["entity_id", key]), key, cap)
        cap_s1 = keep_s1.select(key).unique()
        cap_o = keep_o.select(key).unique()
        both = cap_s1.join(cap_o, on=key, how="inner").select(key)
        hit = g.filter(pl.col(key) == pl.col(key + "_o"))
        hit = hit.join(both, on=key, how="semi")
        union.update(
            hit.select(["source1_entity_id", "other_entity_id"]).unique().rows()
        )
    print(f"  UNION all: {len(union):,}/{total:,} = {100*len(union)/total:.2f}%"
          f"   cand(UB)={union_cands:,}", flush=True)
    return results, len(union), total, union_cands


if __name__ == "__main__":
    for src in (2, 3):
        print(f"=== train S1 x S{src} (cap={BUCKET_CAP})")
        per_pass_recall("train", src)