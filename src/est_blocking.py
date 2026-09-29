import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "src")

import warnings
warnings.filterwarnings("ignore")

import polars as pl
from blocking import _cap_keys


def combo_keys(toks):
    toks = sorted(set(toks))
    if not toks:
        return []
    return [f"{toks[i]} {toks[j]}"
            for i in range(len(toks)) for j in range(i + 1, len(toks))]


def addr_key(addr):
    if hasattr(addr, "to_list") or not isinstance(addr, str):
        addr = ""
    if not addr:
        return []
    toks = addr.split()
    return [" ".join(sorted(set(toks))[:3]), " ".join(toks[:3])]


def _as_list(x):
    return x.to_list() if hasattr(x, "to_list") else (x if isinstance(x, list) else [])


def add_keys(df):
    df = df.with_columns(
        pl.col("country_norm").alias("cnt"),
        f2=pl.col("name_tokens").map_elements(
            lambda t: " ".join(sorted(set(_as_list(t)))[:2]),
            return_dtype=pl.Utf8, skip_nulls=False,
        ),
        np=pl.col("name_tokens").map_elements(
            lambda t: combo_keys(_as_list(t)),
            return_dtype=pl.List(pl.Utf8), skip_nulls=False,
        ),
        ak=pl.col("addr_norm").map_elements(
            addr_key, return_dtype=pl.List(pl.Utf8), skip_nulls=False,
        ),
    )
    import blocking
    df = df.with_columns(
        f2k=pl.col("cnt") + "|" + pl.col("f2"),
        sdx0=(pl.col("cnt") + "|SX|" + pl.col("name_tokens").list.get(0, null_on_oob=True)
              .map_elements(blocking._soundex, return_dtype=pl.Utf8,
                            skip_nulls=False)),
    )
    return df.select(["entity_id", "f2k", "sdx0", "np", "ak"])


def est_pairs_pairs(a, b, cap):
    a = a.explode("k").drop_nulls().rename({"k": "k"})
    b = b.explode("k").drop_nulls().rename({"k": "k"})
    a = _cap_keys(a, "k", cap)
    b = _cap_keys(b, "k", cap)
    c = (a.group_by("k").len().rename({"len": "n1"})
         .join(b.group_by("k").len().rename({"len": "n2"}), on="k"))
    return int((c["n1"] * c["n2"]).sum())


def est_pairs_single(a, b, cap):
    a = _cap_keys(a.rename({"k": "k"}), "k", cap)
    b = _cap_keys(b.rename({"k": "k"}), "k", cap)
    c = (a.group_by("k").len().rename({"len": "n1"})
         .join(b.group_by("k").len().rename({"len": "n2"}), on="k"))
    return int((c["n1"] * c["n2"]).sum())


if __name__ == "__main__":
    for split in ["train"]:
        s1 = pl.read_parquet(f"data/norm_{split}_s1.parquet")
        s2 = pl.read_parquet(f"data/norm_{split}_s2.parquet")
        s1k = add_keys(s1)
        s2k = add_keys(s2)
        print("built", s1k.height, s2k.height)

        # Explode list keys one time
        s1_np = s1k.select([pl.col("entity_id").alias("e1"),
                            pl.col("np").alias("k")])
        s2_np = s2k.select([pl.col("entity_id").alias("e2"),
                            pl.col("np").alias("k")])
        s1_ak = s1k.select([pl.col("entity_id").alias("e1"),
                            pl.col("ak").alias("k")])
        s2_ak = s2k.select([pl.col("entity_id").alias("e2"),
                            pl.col("ak").alias("k")])

        for cap in (100, 300):
            print(f"--- cap={cap}")
            tot = 0
            for a, b, label in [
                (s1k.select([pl.col("entity_id").alias("e1"),
                             pl.col("f2k").alias("k")]),
                 s2k.select([pl.col("entity_id").alias("e2"),
                             pl.col("f2k").alias("k")]), "first2"),
                (s1k.select([pl.col("entity_id").alias("e1"),
                             pl.col("sdx0").alias("k")]),
                 s2k.select([pl.col("entity_id").alias("e2"),
                             pl.col("sdx0").alias("k")]), "soundex0"),
            ]:
                est = est_pairs_single(a, b, cap)
                tot += est
                print(f"  {label}: est pairs={est:,}")
            for a, b, label in [
                (s1_np, s2_np, "token-pairs"),
                (s1_ak, s2_ak, "addr-keys"),
            ]:
                est = est_pairs_pairs(a, b, cap)
                tot += est
                print(f"  {label}: est pairs={est:,}")
            print(f"  TOTAL est pairs: {tot:,}")