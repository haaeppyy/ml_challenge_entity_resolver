import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "src")

import warnings
warnings.filterwarnings("ignore")

import polars as pl

from config import DATA_DIR, ground_truth_path
from blocking import _soundex


def _as_list(x):
    return x.to_list() if hasattr(x, "to_list") else (x if isinstance(x, list) else [])


def keep_tokens(toks, dropped):
    return [t for t in toks if t not in dropped]


def first2_key(toks):
    return " ".join(sorted(set(toks))[:2])


def addr_keys(addr):
    if hasattr(addr, "to_list"):
        addr = ""
    if not addr:
        return []
    toks = addr.split()
    out = []
    if len(toks) >= 2:
        out.append(" ".join(toks[:2]))
        out.append(" ".join(toks[-2:]))
    return out


def build_keys(df, dropped):
    """Return df with ct2f (str), pairf (list[str], cnt|prefixed), sdx0 (str), ak (list)."""
    toks = _as_list
    fk = keep_tokens
    df = df.with_columns(
        ft=pl.col("name_tokens").map_elements(
            lambda t: fk(toks(t), dropped),
            return_dtype=pl.List(pl.Utf8), skip_nulls=False),
        cnt=pl.col("country_norm"),
        ak=pl.col("addr_norm").map_elements(
            addr_keys, return_dtype=pl.List(pl.Utf8), skip_nulls=False),
    )
    # ct2f and pairf from ft + cnt; do in python via map over rows once
    rows = df.select(["cnt", "ft"]).iter_rows()
    ct2f, pairf = [], []
    for cnt, ft in rows:
        ft_s = sorted(set(ft if isinstance(ft, list) else (ft or [])))
        ct2f.append(cnt + "|" + " ".join(ft_s[:2]))
        if len(ft_s) >= 2:
            pairf.append([f"{cnt}|{ft_s[i]} {ft_s[j]}"
                          for i in range(len(ft_s)) for j in range(i + 1, len(ft_s))])
        else:
            pairf.append([])
    df = df.with_columns(
        pl.Series("ct2f", ct2f),
        pl.Series("pairf", pairf).cast(pl.List(pl.Utf8)),
    ).with_columns(
        sdx0=(pl.col("cnt") + "|SX|" + pl.col("ft").list.get(0, null_on_oob=True)
              .map_elements(_soundex, return_dtype=pl.Utf8, skip_nulls=False)),
    )
    return df.select(["entity_id", "ct2f", "pairf", "sdx0", "ak"])


def capped_key_sets(keys_df, key, cap):
    if key in ("pairf", "ak"):
        k = (keys_df.select(pl.col(key).alias("k")).explode("k"))
    else:
        k = keys_df.select(pl.col(key).alias("k"))
    return (k.drop_nulls().filter(pl.col("k") != "")
            .group_by("k").len().filter(pl.col("len") <= cap)).select("k").to_series()


def explode_n1n2(keys_df, key):
    if key in ("pairf", "ak"):
        k = (keys_df.select(pl.col(key).alias("k")).explode("k"))
    else:
        k = keys_df.select(pl.col(key).alias("k"))
    return (k.drop_nulls().filter(pl.col("k") != "").group_by("k").len())


def measure(split, src, s1k, ok, scheme, cap):
    total = 0
    for key in scheme:
        both = (capped_key_sets(s1k, key, cap).to_frame("k")
                .join(capped_key_sets(ok, key, cap).to_frame("k"),
                      on="k", how="inner"))
        both = both.join(explode_n1n2(s1k, key), on="k").join(
            explode_n1n2(ok, key), on="k")
        total += int((both["len"] * both["len_right"]).sum())

    gt = (pl.read_csv(ground_truth_path(split), separator="\t")
          .filter(pl.col("matched_entity_ids").is_not_null())
          .with_columns(pl.col("matched_entity_ids").str.split(",").alias("ids"))
          .explode("ids")
          .select(pl.col("source1_entity_id"),
                  pl.col("ids").str.strip_chars().alias("other_entity_id"))
          .filter(pl.col("other_entity_id").str.starts_with(f"S{src}-"))
          .unique())
    total_true = gt.height

    g = gt.join(s1k, left_on="source1_entity_id", right_on="entity_id", how="left")
    g = g.join(ok, left_on="other_entity_id", right_on="entity_id", how="left",
               suffix="_o")

    keep_cols = ["source1_entity_id", "other_entity_id"]
    captured = pl.DataFrame(
        schema={"source1_entity_id": pl.Utf8, "other_entity_id": pl.Utf8})
    for key in scheme:
        capped = (capped_key_sets(s1k, key, cap).to_frame("k")
                  .join(capped_key_sets(ok, key, cap).to_frame("k"),
                        on="k", how="inner"))
        if key in ("pairf", "ak"):
            hit = (g.with_columns(
                inter=pl.col(key).list.set_intersection(pl.col(key + "_o"))
            ).filter(pl.col("inter").list.len() > 0)
                .select(keep_cols + [pl.col("inter").alias("hitk")])
                .explode("hitk").drop_nulls())
        else:
            hit = g.filter((pl.col(key).fill_null("") != "") &
                           (pl.col(key) == pl.col(key + "_o")))
            hit = hit.select(keep_cols + [pl.col(key).alias("hitk")])
        hit = hit.join(capped, left_on="hitk", right_on="k", how="inner")
        captured = pl.concat([captured, hit.select(keep_cols).unique()]).unique()

    rec = 100 * captured.height / max(total_true, 1)
    print(f"  [S{src}] scheme={scheme} cap={cap} "
          f"candidates={total:,} recall={rec:.2f}% "
          f"({captured.height:,}/{total_true:,})", flush=True)
    return total, captured.height, total_true


if __name__ == "__main__":
    split = "train"
    for src in (2, 3):
        print(f"===== train S1 x S{src} =====", flush=True)
        s1 = pl.read_parquet(f"{DATA_DIR}/norm_{split}_s1.parquet")
        other = pl.read_parquet(f"{DATA_DIR}/norm_{split}_s{src}.parquet")

        f1 = (s1.select("name_tokens").explode("name_tokens").drop_nulls()
              .group_by("name_tokens").len()
              .with_columns(frac=pl.col("len") / s1.height))
        f2 = (other.select("name_tokens").explode("name_tokens").drop_nulls()
              .group_by("name_tokens").len()
              .with_columns(frac=pl.col("len") / other.height))
        cutoff = 0.003
        dropped = set((pl.concat([f1, f2]).group_by("name_tokens")
                       .agg(pl.col("frac").max())
                       .filter(pl.col("frac") > cutoff)
                       .select("name_tokens").to_series().to_list()))
        print(f"dropped {len(dropped)} high-freq tokens", flush=True)

        t = time.time()
        s1k = build_keys(s1, dropped)
        ok = build_keys(other, dropped)
        print(f"built keys in {time.time()-t:.0f}s", flush=True)
        s1k.write_parquet(f"{DATA_DIR}/keys_{split}_s1.parquet")
        ok.write_parquet(f"{DATA_DIR}/keys_{split}_s{src}.parquet")

        for scheme in (["ct2f"], ["pairf"], ["sdx0"], ["ak"],
                       ["ct2f", "pairf", "sdx0", "ak"]):
            for cap in (50, 100, 300):
                measure(split, src, s1k, ok, scheme, cap)
            print(flush=True)