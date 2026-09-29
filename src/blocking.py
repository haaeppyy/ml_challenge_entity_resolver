import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import polars as pl

from config import DATA_DIR, train_source_path, test_source_path, ground_truth_path
from normalize import normalize_business

# Bucket-size guard: keys shared by more than this many records on EITHER side
# are dropped before the join, otherwise a few generic keys (e.g. "SMITH")
# would generate tens of millions of junk pairs.
BUCKET_CAP = 300


def _soundex(word):
    """Classic us.soundex (e.g. "smith" -> "S530"). Empty for non-ASCII input."""
    if not word:
        return ""
    word = word.lower()
    letters = "".join(c for c in word if c.isascii() and c.isalpha())
    if not letters:
        return ""
    first = letters[0]
    table = {
        "b": "1", "f": "1", "p": "1", "v": "1",
        "c": "2", "g": "2", "j": "2", "k": "2", "q": "2", "s": "2", "x": "2", "z": "2",
        "d": "3", "t": "3",
        "l": "4",
        "m": "5", "n": "5",
        "r": "6",
    }
    code = [first.upper()]
    prev = table.get(first, "")
    for ch in letters[1:]:
        d = table.get(ch, "")
        if not d:
            continue
        if d != prev:
            code.append(d)
        prev = d
        if len(code) == 4:
            break
    return "".join(code).ljust(4, "0")


def build_block_key_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add one blocking-key column per pass.

    Pass A: country + sorted(first 2 significant name tokens).
    Pass B: country + soundex of each of the first 2 tokens (2 key columns).
    """
    toks = pl.col("name_tokens").list.drop_nulls()
    keys = [
        (pl.col("country_norm") + "|" +
         toks.list.slice(0, 2).list.sort().list.join(" ")).alias("blk_a"),
        (pl.col("country_norm") + "|SX|" + toks.list.get(0, null_on_oob=True)
         .map_elements(_soundex, return_dtype=pl.Utf8, skip_nulls=False)).alias("blk_b1"),
        (pl.col("country_norm") + "|SX|" + toks.list.get(1, null_on_oob=True)
         .map_elements(_soundex, return_dtype=pl.Utf8, skip_nulls=False)).alias("blk_b2"),
    ]
    return df.with_columns(keys)


def _cap_keys(df: pl.DataFrame, key: str, cap: int) -> pl.DataFrame:
    df = df.filter(pl.col(key).is_not_null() & (pl.col(key).str.len_chars() > 0))
    keep = df.group_by(key).len().filter(pl.col("len") <= cap).select(key)
    return df.join(keep, on=key, how="inner")


def block_a_partition(s1: pl.DataFrame, other: pl.DataFrame) -> pl.DataFrame:
    """Generate candidate (source1_entity_id, other_entity_id) pairs via the
    union of every blocking pass, deduplicated."""
    s1 = build_block_key_columns(s1).select(["entity_id", "blk_a", "blk_b1", "blk_b2"])
    other = build_block_key_columns(other).select(
        ["entity_id", "blk_a", "blk_b1", "blk_b2"]
    )

    frames = []
    for key in ("blk_a", "blk_b1", "blk_b2"):
        a = s1.select([pl.col("entity_id").alias("e1"), key])
        b = other.select([pl.col("entity_id").alias("e2"), key])
        a = _cap_keys(a, key, BUCKET_CAP)
        b = _cap_keys(b, key, BUCKET_CAP)
        joined = a.join(b, on=key, how="inner")
        joined = joined.filter(pl.col("e1") != pl.col("e2")).drop(key)
        frames.append(joined)

    out = pl.concat(frames).unique()
    return out.select(
        pl.col("e1").alias("source1_entity_id"),
        pl.col("e2").alias("other_entity_id"),
    )


def load_normalized(split: str, source: int) -> pl.DataFrame:
    path = (train_source_path(source) if split == "train"
            else test_source_path(source))
    cache = os.path.join(DATA_DIR, f"norm_{split}_s{source}.parquet")
    if os.path.exists(cache):
        return pl.read_parquet(cache)
    df = pl.read_csv(path, separator="\t")
    out = normalize_business(df).select(
        ["entity_id", "name_norm", "name_tokens", "addr_norm", "country_norm"]
    )
    out.write_parquet(cache)
    del df
    return out


def _gt_explode(split: str) -> pl.DataFrame:
    gt = pl.read_csv(ground_truth_path(split), separator="\t")
    return (
        gt.filter(pl.col("matched_entity_ids").is_not_null())
        .with_columns(pl.col("matched_entity_ids").str.split(",").alias("ids"))
        .explode("ids")
        .select(
            pl.col("source1_entity_id"),
            pl.col("ids").str.strip_chars().alias("other_entity_id"),
        )
        .unique()
    )


def main():
    for split in ("train", "test"):
        s1 = load_normalized(split, 1)
        for src in (2, 3):
            other = load_normalized(split, src)
            cand = block_a_partition(s1, other)
            out_path = os.path.join(DATA_DIR, f"candidates_{split}_s{src}.parquet")
            cand.write_parquet(out_path)
            print(f"[{split}] source1 x source{src}: {cand.height:,} candidate pairs "
                  f"-> {out_path}")

            if split == "train":
                gt = _gt_explode(split)
                total = gt.filter(
                    pl.col("other_entity_id").str.starts_with(f"S{src}-")
                ).height
                captured = gt.join(
                    cand, on=["source1_entity_id", "other_entity_id"], how="inner"
                ).height
                print(f"    recall ceiling vs train ground truth: "
                      f"{captured:,}/{total:,} = {100*captured/max(total,1):.2f}%")
            del cand, other


if __name__ == "__main__":
    main()