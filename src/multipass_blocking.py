import argparse
import gc
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import duckdb
import polars as pl

from config import DATA_DIR, ground_truth_path
from blocking import _soundex as _snd
from entsplit import load as load_entsplit
from eval_blocking import gt_frame, mem_pct, pair_recall

KEY_SEP = "|"
MEM_WARN_PCT = 90.0
BATCH_ROWS = 4_000_000

MAX_KEY_FREQ = 4000        # suppress a key when > N records on either FULL side
MAX_PAIRS_PER_KEY = 900    # suppress a shared key whose cross-product exceeds it
MAX_TOTAL_PAIRS = 180_000_000

PASS_OVERRIDES = {
    "postal_ctry": {"max_pairs_per_key": 30_000, "max_key_freq": 40_000},
    "postal_nc":   {"max_pairs_per_key": 30_000, "max_key_freq": 40_000},
    "house_ctry":  {"max_pairs_per_key": 6_000,  "max_key_freq": 10_000},
    "house_nc":    {"max_pairs_per_key": 6_000,  "max_key_freq": 10_000},
    "addr_bigram": {"max_pairs_per_key": 8_000},
    "ph_ctry":     {"max_pairs_per_key": 8_000},
    "ph_nc":       {"max_pairs_per_key": 8_000},
}

ADDR_STOP = {
    "near", "opposite", "opp", "beside", "behind", "next", "adjacent",
    "main", "road", "rd", "street", "st", "area", "locality", "gali",
    "lane", "ln", "cross", "sector", "phase", "block", "ward", "nagar",
    "colony", "layout", "extension", "ext", "circle", "square", "sq",
    "chowk", "market", "bazaar", "complex", "mall", "park", "building",
    "bldg", "tower", "floor", "camp", "post", "badi", "chauraha",
    "crossing", "the", "of", "and", "a", "an",
}

ACTIVE_PASSES = [
    "ct2", "pairs_ctry", "pairs_nc", "tok_ctry", "tok_nc",
    "ph_ctry", "ph_nc", "addr_bigram", "postal_ctry", "postal_nc",
    "house_ctry", "house_nc",
]


def norm_path(split, src):
    return os.path.join(DATA_DIR, f"norm_{split}_s{src}.parquet")


def guard(tag):
    p = mem_pct()
    print(f"    [mem {p:.0f}%] {tag}", flush=True)
    if p >= MEM_WARN_PCT:
        raise SystemExit(f"ABORT: memory at {p:.0f}% while {tag}")


def key_lf(split, src, pass_name, only_ids=None):
    """Lazy frame (entity_id, key) for one pass over a normalized column.
    If only_ids is given, the base scan is filtered to those entity ids first
    (keeps pair-join work and memory proportional to the id set)."""
    lf = pl.scan_parquet(norm_path(split, src))
    if only_ids is not None:
        lf = lf.filter(pl.col("entity_id").is_in(only_ids))
    ground = lf.select("entity_id", "name_tokens", "addr_norm",
                       "country_norm")
    nt = pl.col("name_tokens").list.drop_nulls().list.unique()

    if pass_name == "ct2":
        out = ground.select(
            "entity_id",
            (pl.col("country_norm") + KEY_SEP + "n" + KEY_SEP
             + nt.list.slice(0, 2).list.sort().list.join(KEY_SEP)).alias("key"))

    elif pass_name in ("pairs_ctry", "pairs_nc"):
        ctry = pl.col("c") + KEY_SEP + "np" + KEY_SEP
        key_expr = ((ctry if pass_name == "pairs_ctry" else
                     pl.lit("np" + KEY_SEP)) + pl.col("t") + KEY_SEP
                    + pl.col("t_2"))
        out = self_join_pairs(ground, nt, key_expr)

    elif pass_name in ("tok_ctry", "tok_nc"):
        ex = (ground.select("entity_id", pl.col("country_norm").alias("c"),
                            nt.alias("ntok"))
              .explode("ntok")
              .select("entity_id", "c", pl.col("ntok").alias("tok")))
        out = (ex.with_columns(
            pl.when(pl.lit(pass_name == "tok_ctry"))
            .then(pl.col("c") + KEY_SEP + "nt" + KEY_SEP + pl.col("tok"))
            .otherwise("nt" + KEY_SEP + pl.col("tok")).alias("key"))
            .select("entity_id", "key"))

    elif pass_name in ("ph_ctry", "ph_nc"):
        sdx = nt.list.get(0, null_on_oob=True).map_elements(
            _snd, return_dtype=pl.Utf8, skip_nulls=True)
        out = ground.select(
            "entity_id",
            pl.when(pl.lit(pass_name == "ph_ctry"))
            .then(pl.col("country_norm") + KEY_SEP + "sx" + KEY_SEP + sdx)
            .otherwise("sx" + KEY_SEP + sdx).alias("key"))

    elif pass_name == "addr_bigram":
        out = addr_pairs_filtered(ground)

    elif pass_name in ("postal_ctry", "postal_nc"):
        ex = (ground.select("entity_id", pl.col("country_norm").alias("c"),
                            "addr_norm")
              .with_columns(pl.col("addr_norm").str.split(" ").alias("_a"))
              .explode("_a").filter(
                  pl.col("_a").str.contains(r"^\d+$")
                  & pl.col("_a").str.len_chars().is_in([5, 6])))
        out = (ex.with_columns(
            pl.when(pl.lit(pass_name == "postal_ctry"))
            .then(pl.col("c") + KEY_SEP + "zz" + KEY_SEP + pl.col("_a"))
            .otherwise("zz" + KEY_SEP + pl.col("_a")).alias("key"))
            .select("entity_id", "key"))

    elif pass_name in ("house_ctry", "house_nc"):
        out = house_keys(ground, pass_name)

    else:
        raise ValueError(pass_name)

    return out.filter(pl.col("key").is_not_null()
                      & (pl.col("key").str.len_chars() > 0))


def self_join_pairs(ground, toks_expr, key_expr):
    p = (ground.select("entity_id", pl.col("country_norm").alias("c"),
                       toks_expr.alias("t"))
         .with_row_index("_ri").explode("t")
         .select("_ri", "entity_id", "c", "t"))
    return (p.join(p, on="_ri", how="inner", suffix="_2")
            .filter(pl.col("t") < pl.col("t_2"))
            .with_columns(key_expr.alias("key"))
            .select("entity_id", "key"))


def addr_pairs_filtered(ground):
    p = (ground.select(
        "entity_id",
        pl.col("addr_norm").str.split(" ").list.drop_nulls().alias("t"))
        .with_row_index("_ri").explode("t")
        .filter(~pl.col("t").str.contains(r"^\d+$"))
        .filter(~pl.col("t").is_in(list(ADDR_STOP))))
    return (p.join(p, on="_ri", how="inner", suffix="_2")
            .filter(pl.col("t") < pl.col("t_2"))
            .with_columns((pl.lit("ap" + KEY_SEP, dtype=pl.Utf8)
                           + pl.col("t") + pl.lit(KEY_SEP, dtype=pl.Utf8)
                           + pl.col("t_2")).alias("key"))
            .select("entity_id", "key"))


def addr_pairs(ground, toks_expr):
    p = (ground.select("entity_id", toks_expr.alias("t"))
         .with_row_index("_ri").explode("t")
         .select("_ri", "entity_id", "t"))
    return (p.join(p, on="_ri", how="inner", suffix="_2")
            .filter(pl.col("t") < pl.col("t_2"))
            .with_columns((pl.lit("ap" + KEY_SEP, dtype=pl.Utf8)
                           + pl.col("t") + pl.lit(KEY_SEP, dtype=pl.Utf8)
                           + pl.col("t_2")).alias("key"))
            .select("entity_id", "key"))


def house_keys(ground, pass_name):
    at = (ground.with_columns(
        pl.col("addr_norm").str.split(" ").list.drop_nulls().alias("_a")))
    ex = (at.select("entity_id", "country_norm", "_a")
          .with_row_index("_ri").explode("_a")
          .with_columns(
              pl.col("_a").str.contains(r"^\d+$").alias("isd"),
              pl.col("_a").str.contains(r"^\D+$").alias("isalpha"),
              pl.col("country_norm")))
    dig = (ex.filter(pl.col("isd")).sort("_ri")
           .unique(subset=["_ri"], keep="first")
           .select("_ri", pl.col("_a").alias("hnum")))
    alpha = (ex.filter(pl.col("isalpha")
                       & ~pl.col("_a").is_in(list(ADDR_STOP)))
             .sort("_ri").unique(subset=["_ri"], keep="first")
             .select("_ri", pl.col("_a").alias("stok")))
    meta = ex.select("_ri", "entity_id", pl.col("country_norm")).unique()
    joined = dig.join(alpha, on="_ri", how="inner").join(meta, on="_ri")
    return (joined.with_columns(
        pl.when(pl.lit(pass_name == "house_ctry"))
        .then(pl.col("country_norm") + KEY_SEP + "hn" + KEY_SEP
              + pl.col("hnum") + KEY_SEP + pl.col("stok"))
        .otherwise("hn" + KEY_SEP + pl.col("hnum") + KEY_SEP
                   + pl.col("stok")).alias("key"))
        .select("entity_id", "key"))


# ----------------------------------------------------------------------
# Diagnostics: key frequencies, suppression, projected pairs.
# ----------------------------------------------------------------------
def measure_pass(pass_name, split, src, max_freq, max_pk, verbosity=1):
    t0 = time.time()
    lfl = key_lf(split, 1, pass_name).collect()
    lfr = key_lf(split, src, pass_name).collect()
    c1 = lfl.group_by("key").len()
    c2 = lfr.group_by("key").len()
    comb = (c1.rename({"len": "n1"})
            .join(c2.rename({"len": "n2"}), on="key", how="inner"))
    kept = (comb.filter((pl.col("n1") <= max_freq) & (pl.col("n2") <= max_freq))
            .with_columns((pl.col("n1") * pl.col("n2")).alias("prod")))
    n_drop_pk = int((kept["prod"] > max_pk).sum())
    kept = kept.filter(pl.col("prod") <= max_pk)
    projected = int(kept.select((pl.col("n1") * pl.col("n2")).sum()).item())
    res = {
        "pass": pass_name,
        "keys_s1": int(c1.height), "keys_src": int(c2.height),
        "drop_freq_s1": int((c1["len"] > max_freq).sum()),
        "drop_freq_src": int((c2["len"] > max_freq).sum()),
        "kept_keys": int(kept.height), "drop_prod": n_drop_pk,
        "projected_pairs": projected,
        "seconds": round(time.time() - t0, 1),
    }
    if verbosity:
        print(f"    {pass_name:12s} keys s1={res['keys_s1']:>9,} "
              f"src={res['keys_src']:>9,} dropF=({res['drop_freq_s1']:>7,}," 
              f"{res['drop_freq_src']:>7,}) kept={res['kept_keys']:>7,} "
              f"dropP={res['drop_prod']:>7,} proj={projected:>12,} "
              f"({res['seconds']}s)", flush=True)
    keep = kept.select("key")["key"]
    del lfl, lfr, c1, c2, comb, kept
    gc.collect()
    return res, keep


def pass_limits(pass_name, override=None):
    if override:
        return override
    ov = PASS_OVERRIDES.get(pass_name, {})
    mf = ov.get("max_key_freq", MAX_KEY_FREQ)
    mp = ov.get("max_pairs_per_key", MAX_PAIRS_PER_KEY)
    return mf, mp


def per_pass_recall_gt(split, src, pass_name, gval, kept_keys):
    """GT-recall contribution of a pass (val split, train labels only)."""
    if gval is None or kept_keys is None or kept_keys.is_empty():
        return None
    src1 = gval["source1_entity_id"].unique().to_list()
    others = gval["other_entity_id"].unique().to_list()
    k1 = key_lf(split, 1, pass_name, only_ids=src1).filter(
        pl.col("key").is_in(kept_keys)).collect()
    k2 = (key_lf(split, src, pass_name, only_ids=others)
          .filter(pl.col("key").is_in(kept_keys)).collect()
          .rename({"key": "key2"}))
    g = gval.join(k1, left_on="source1_entity_id", right_on="entity_id",
                  how="inner")
    g = g.join(k2, left_on="other_entity_id", right_on="entity_id",
               how="inner", suffix="_2")
    hit = (g.filter(pl.col("key") == pl.col("key2"))
           .select("source1_entity_id", "other_entity_id").unique().height)
    return hit, gval.height


def generate_pass_pairs(pass_name, split, src, kept_keys, only_val, val_ids,
                        part_files, produced_by_pass):
    """Generate a pass's cross-source pairs in bounded part files."""
    l = key_lf(split, 1, pass_name).filter(
        pl.col("key").is_in(kept_keys)).collect()
    if only_val:
        l = l.filter(pl.col("entity_id").is_in(val_ids))
    r = key_lf(split, src, pass_name).filter(
        pl.col("key").is_in(kept_keys)).collect()
    left = l.group_by("key").agg(pl.col("entity_id").alias("ids1"))
    right = r.group_by("key").agg(pl.col("entity_id").alias("ids2"))
    joined = left.join(right, on="key", how="inner")
    n = 0
    parts = []
    L, R = [], []
    for row in joined.iter_rows(named=True):
        ids1, ids2 = row["ids1"], row["ids2"]
        n += len(ids1) * len(ids2)
        for a in ids1:
            for b in ids2:
                L.append(a)
                R.append(b)
                if len(L) >= BATCH_ROWS:
                    path = os.path.join(DATA_DIR,
                        f"_mp_{pass_name}_{src}_{len(part_files) + len(parts)}.parquet")
                    pl.DataFrame({"entity_id_l": L, "entity_id_r": R}).write_parquet(path)
                    parts.append(path)
                    L, R = [], []
    if L:
        path = os.path.join(DATA_DIR,
            f"_mp_{pass_name}_{src}_{len(part_files) + len(parts)}.parquet")
        pl.DataFrame({"entity_id_l": L, "entity_id_r": R}).write_parquet(path)
        parts.append(path)
    part_files.extend(parts)
    produced_by_pass[pass_name] = n
    print(f"    {pass_name}: generated {n:,} pairs in {len(parts)} part(s)",
          flush=True)


def dedup_parts(part_files, out_path):
    """Deduplicate all pass outputs into the final candidate parquet."""
    if not part_files:
        pl.DataFrame({"entity_id_l": pl.Series([], dtype=pl.Utf8),
                      "entity_id_r": pl.Series([], dtype=pl.Utf8)}).write_parquet(out_path)
        return 0
    con = duckdb.connect()
    files = [p.replace("\\", "/").replace("'", "''") for p in part_files]
    paths = ",".join("'" + p + "'" for p in files)
    con.execute(f"COPY (SELECT DISTINCT entity_id_l, entity_id_r "
                f"FROM read_parquet([{paths}])) TO '{out_path.replace(chr(92), '/')}' "
                "(FORMAT PARQUET)")
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path.replace(chr(92), '/')}' )").fetchone()[0]
    con.close()
    for p in part_files:
        try:
            os.remove(p)
        except OSError:
            pass
    return int(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--src", type=int, choices=[2, 3], required=True)
    ap.add_argument("--mode", choices=["measure", "generate", "both"], default="both")
    ap.add_argument("--only-val", action="store_true")
    ap.add_argument("--freq", type=int, default=MAX_KEY_FREQ)
    ap.add_argument("--pk", type=int, default=MAX_PAIRS_PER_KEY)
    ap.add_argument("--max-total", type=int, default=MAX_TOTAL_PAIRS)
    ap.add_argument("--out-prefix", default="candmp_v2")
    ap.add_argument("--recall", action="store_true")
    ap.add_argument("--pass", dest="passes", default="all")
    args = ap.parse_args()
    active = (ACTIVE_PASSES if args.passes == "all" else
              [p.strip() for p in args.passes.split(",") if p.strip() in ACTIVE_PASSES])
    if not active:
        ap.error("--pass did not select any known blocking pass")
    lim_override = (args.freq, args.pk)
    split, src = args.split, args.src
    print(f"=== multipass blocking: {split} S1 x S{src} "
          f"(freq<={args.freq}, pk<={args.pk}) ===", flush=True)

    val_ids = None
    gval = None
    if args.only_val:
        val_ids = load_entsplit().filter(pl.col("isval") == 1)["entity_id"]
        print(f"  val entities (s1): {val_ids.len():,}", flush=True)
    if split == "train":
        gval = gt_frame(src)
        if val_ids is not None:
            gval = gval.filter(pl.col("source1_entity_id").is_in(val_ids))
        print(f"  GT pairs to capture: {gval.height:,}", flush=True)

    keep_sets, projected_by_pass = {}, {}
    if args.mode in ("measure", "both"):
        print("  ---- per-pass key stats ----", flush=True)
        for ps in active:
            mf, mp = pass_limits(ps, lim_override)
            res, keep = measure_pass(ps, split, src, mf, mp)
            keep_sets[ps] = keep
            projected_by_pass[ps] = res["projected_pairs"]
            if args.recall and gval is not None:
                hit, tot = per_pass_recall_gt(split, src, ps, gval, keep)
                print(f"    {ps}: GT recall {hit:,}/{tot:,} = {100*hit/tot:.2f}%",
                      flush=True)
        if args.mode == "measure":
            return

    kepts = {}
    for ps in active:
        if ps not in keep_sets:
            mf, mp = pass_limits(ps, lim_override)
            res, keep = measure_pass(ps, split, src, mf, mp)
            keep_sets[ps] = keep
            projected_by_pass[ps] = res["projected_pairs"]
        kepts[ps] = keep_sets[ps]

    part_files, produced_by_pass = [], {}
    for ps in active:
        if kepts[ps].is_empty():
            produced_by_pass[ps] = 0
            continue
        generate_pass_pairs(ps, split, src, kepts[ps], args.only_val,
                            val_ids, part_files, produced_by_pass)
        if sum(produced_by_pass.values()) > args.max_total:
            raise SystemExit(f"ABORT: generated pairs exceed {args.max_total:,}")
    n_uniq = dedup_parts(part_files, os.path.join(
        DATA_DIR, f"{args.out_prefix}_{split}_s{src}.parquet"))
    print(f"  pre-dedup generated: {sum(produced_by_pass.values()):,}; "
          f"deduplicated union: {n_uniq:,}", flush=True)
    for ps in active:
        print(f"    {ps}: full-data projected={projected_by_pass[ps]:,}; "
              f"generated={produced_by_pass.get(ps, 0):,}", flush=True)
    print(f"  peak mem {mem_pct():.0f}%", flush=True)


if __name__ == "__main__":
    main()
