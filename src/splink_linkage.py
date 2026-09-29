import argparse
import gc
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import duckdb
import polars as pl

from splink import Linker, SettingsCreator, block_on
from splink.backends.duckdb import DuckDBAPI
from splink.comparison_library import ExactMatch, JaroWinklerAtThresholds

from config import DATA_DIR, ground_truth_path
from blocking import _soundex

BUCKET_CAP = 50
MAX_FEASIBLE_PAIRS = 15_000_000
MEM_WARN_PCT = 90.0
SEED = 42


def norm_path(split: str, src: int) -> str:
    return os.path.join(DATA_DIR, f"norm_{split}_s{src}.parquet")


def prep_path(split: str, src: int, mode: str) -> str:
    return os.path.join(DATA_DIR, f"prep_{mode}_{split}_s{src}.parquet")


def mem_pct() -> float:
    try:
        import psutil
        return float(psutil.virtual_memory().percent)
    except Exception:
        return 0.0


def capped_table(split: str, src: int, cap: int) -> pl.DataFrame:
    """First-`cap` rows per (country_norm, soundex-first-token) bucket."""
    lf = (pl.scan_parquet(norm_path(split, src))
          .select(pl.col("entity_id"), pl.col("country_norm"),
                  pl.col("name_tokens"), pl.col("name_norm"),
                  pl.col("addr_norm"))
          .with_columns(
              pl.col("name_tokens").list.get(0, null_on_oob=True).alias("tok"))
          .with_columns(
              pl.col("tok").map_elements(
                  _soundex, return_dtype=pl.Utf8, skip_nulls=False).alias("sdx1"))
          .with_columns(
              (pl.col("country_norm") + "_" + pl.col("sdx1")).alias("bk"))
          .filter(pl.col("bk").is_not_null() & (~pl.col("bk").str.ends_with("_")))
          .drop("tok"))
    out = (lf.group_by("bk")
           .agg(pl.all().head(cap))
           .explode(pl.all().exclude("bk")))
    return out.unique().select(["entity_id", "name_norm", "name_tokens",
                                "addr_norm", "country_norm", "sdx1",
                                "bk"]).collect()


def build_settings(blocking_rules, prior, **kwargs):
    return SettingsCreator(
        link_type="link_only",
        comparisons=[
            JaroWinklerAtThresholds("name_norm",
                                    score_threshold_or_thresholds=[0.95, 0.9, 0.7]),
            JaroWinklerAtThresholds("addr_norm",
                                    score_threshold_or_thresholds=[0.95, 0.9, 0.7]),
            ExactMatch("country_norm"),
        ],
        blocking_rules_to_generate_predictions=blocking_rules,
        unique_id_column_name="entity_id",
        probability_two_random_records_match=prior,
        **kwargs,
    )


def build_labels(split: str, src: int, ids1: set, ids_src: set):
    """Pairwise labels in splink labels-table schema:
    source_dataset_l, entity_id_l, source_dataset_r, entity_id_r.
    Returns an empty frame when no ground truth file exists for `split`."""
    gt_path = ground_truth_path(split)
    if not os.path.exists(gt_path):
        return pl.DataFrame({
            "source_dataset_l": pl.Series([], dtype=pl.Utf8),
            "entity_id_l": pl.Series([], dtype=pl.Utf8),
            "source_dataset_r": pl.Series([], dtype=pl.Utf8),
            "entity_id_r": pl.Series([], dtype=pl.Utf8),
        })
    gt = pl.read_csv(gt_path, separator="\t",
                     columns=["source1_entity_id", "matched_entity_ids"])
    out = (gt.filter(pl.col("matched_entity_ids").is_not_null())
           .with_columns(pl.col("matched_entity_ids").str.split(",").alias("ids"))
           .explode("ids")
           .select(pl.lit(f"s1").alias("source_dataset_l"),
                   pl.col("source1_entity_id").alias("entity_id_l"),
                   pl.lit(f"s{src}").alias("source_dataset_r"),
                   pl.col("ids").str.strip_chars().alias("entity_id_r"))
           .filter(pl.col("entity_id_r").str.starts_with(f"S{src}-"))
           .unique())
    if ids1:
        out = out.filter(pl.col("entity_id_l").is_in(ids1))
    if ids_src:
        out = out.filter(pl.col("entity_id_r").is_in(ids_src))
    return out


def guard(tag: str) -> None:
    p = mem_pct()
    print(f"    [mem {p:.0f}%] {tag}", flush=True)
    if p >= MEM_WARN_PCT:
        raise SystemExit(f"ABORT: memory at {p:.0f}% while {tag}")


def rule_meas(con, count_sql: str) -> int:
    return int(con.execute(count_sql).fetchone()[0])


def run(mode: str, split: str, src: int, out_dir: str) -> None:
    t0 = time.time()
    con = duckdb.connect()
    con.execute("SET memory_limit='9GB'")
    con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")

    path_s1 = prep_path(split, 1, mode)
    path_src = prep_path(split, src, mode)

    if mode == "sample":
        if not os.path.exists(path_s1) or not os.path.exists(path_src):
            raise SystemExit(
                f"run _splink_sample_prep.py first ({path_s1} / {path_src})")
    else:
        print("building bucket-capped full tables...", flush=True)
        capped_table(split, 1, BUCKET_CAP).write_parquet(path_s1)
        capped_table(split, src, BUCKET_CAP).write_parquet(path_src)
        print(f"  capped tables written ({time.time()-t0:.0f}s)", flush=True)

    print(f"=== [{mode}] {split} S1 x S{src} ===", flush=True)
    con.execute(f"CREATE VIEW t_s1 AS SELECT * FROM read_parquet('{path_s1.replace(chr(92), '/')}')")
    con.execute(f"CREATE VIEW t_s{src} AS SELECT * FROM read_parquet('{path_src.replace(chr(92), '/')}')")
    guard("input views registered")

    n1 = int(con.execute("SELECT COUNT(*) FROM t_s1").fetchone()[0])
    n2 = int(con.execute(f"SELECT COUNT(*) FROM t_s{src}").fetchone()[0])
    print(f"  rows: s1={n1:,}  s{src}={n2:,}", flush=True)

    # ---- blocking rules ----------------------------------------------------
    # every rule is measured first; rules that would generate more than
    # MAX_FEASIBLE_PAIRS are dropped (with their measured count logged),
    # because materialising them exceeds this machine's 16GB.
    if mode == "sample":
        rule_a = (f"SELECT COUNT(*) FROM t_s1 l JOIN t_s{src} r "
                  "ON l.country_norm=r.country_norm "
                  "AND SUBSTR(l.name_norm,1,4)=SUBSTR(r.name_norm,1,4)")
    else:
        rule_a = (f"SELECT COUNT(*) FROM t_s1 l JOIN t_s{src} r "
                  "ON l.country_norm=r.country_norm AND l.sdx1=r.sdx1")
    rule_b = (f"SELECT COUNT(*) FROM "
              f"(SELECT UNNEST(name_tokens) tok, entity_id FROM t_s1) l JOIN "
              f"(SELECT UNNEST(name_tokens) tok, entity_id FROM t_s{src}) r "
              "ON l.tok=r.tok WHERE l.entity_id<>r.entity_id")

    n_a = rule_meas(con, rule_a)
    n_b = rule_meas(con, rule_b)
    print(f"  measured rule A (country+sdx-key): {n_a:,} pairs", flush=True)
    print(f"  measured rule B (token overlap, upper bound): {n_b:,} pairs",
          flush=True)

    rules = []
    keep_a = n_a <= MAX_FEASIBLE_PAIRS
    keep_b = n_b <= MAX_FEASIBLE_PAIRS
    if keep_a:
        rules.append(block_on("country_norm",
                              "sdx1" if mode == "full"
                              else "substr(name_norm,1,4)"))
    if keep_b:
        rules.append(
            block_on("name_tokens", arrays_to_explode=["name_tokens"]))
    if not rules:
        raise SystemExit(
            "ABORT: no blocking rule is feasible under the 16GB budget")
    print(f"  kept rules: {len(rules)} "
          f"(A={'in' if keep_a else 'OUT'}, B={'in' if keep_b else 'OUT'})",
          flush=True)
    gc.collect()

    # ---- labels + prior ----------------------------------------------------
    ids1 = con.execute("SELECT entity_id FROM t_s1").fetchdf()["entity_id"].tolist()
    ids_src = con.execute(f"SELECT entity_id FROM t_s{src}").fetchdf()["entity_id"].tolist()
    labels = build_labels(split, src, set(ids1), set(ids_src))
    n_true = labels.height
    n_true = max(n_true, 1)
    prior = n_true / max(n1 * n2, 1)
    print(f"  labelled true pairs among capped/sampled ids: {labels.height:,} "
          f"prior={prior:.2e}", flush=True)

    # ---- linker ------------------------------------------------------------
    settings = build_settings(rules, float(prior))
    for i, br in enumerate(rules):
        try:
            print(f"  BR{i}: ", br.get_blocking_rule("duckdb").blocking_rule_sql,
                  flush=True)
        except Exception as e:
            print(f"  BR{i}: <render failed: {e}>", flush=True)

    api = DuckDBAPI(connection=con)
    linker = Linker([f"t_s1", f"t_s{src}"], settings, db_api=api,
                    input_table_aliases=[f"s1", f"s{src}"],
                    set_up_basic_logging=False)
    guard("linker created")
    linker.table_management.register_table(labels.to_pandas(), "labels",
                                           overwrite=True)
    guard("labels registered")

    # ---- training ----------------------------------------------------------
    print("training: u via random sampling...", flush=True)
    t = time.time()
    linker.training.estimate_u_using_random_sampling(max_pairs=1_000_000,
                                                     seed=SEED)
    print(f"  u sampling done ({time.time()-t:.0f}s)", flush=True)
    guard("after u")

    use_supervised = labels.height > 0
    if use_supervised:
        trainer = "supervised (m from pairwise labels)"
        t = time.time()
        linker.training.estimate_m_from_pairwise_labels("labels")
        print(f"  supervised m done ({time.time()-t:.0f}s)", flush=True)
    else:
        trainer = ("none (no ground truth; candidate set is "
                   "param-independent, defaults used)")
        print("  no ground truth for this split -> skipping model "
              "training (defaults; candidate set unaffected)", flush=True)
    print(f"  model trained via: {trainer}", flush=True)
    guard("after training")

    # ---- predict (all blocked candidates) ----------------------------------
    print("predicting over all blocked candidates...", flush=True)
    t = time.time()
    df_predict = linker.inference.predict()
    phys = df_predict.physical_name
    print(f"  predict done ({time.time()-t:.0f}s)", flush=True)
    guard("after predict")

    # ---- streamed analysis (no pandas materialisation) --------------------
    n_cand = rule_meas(con, f"SELECT COUNT(*) FROM {phys}")
    n_uniq = rule_meas(con,
        f"SELECT COUNT(*) FROM (SELECT DISTINCT entity_id_l, entity_id_r "
        f"FROM {phys})")

    gt_uni = (build_labels(split, src, set(ids1), set(ids_src))
              .select(pl.col("entity_id_l").alias("source1_entity_id"),
                      pl.col("entity_id_r").alias("other_entity_id"))
              .unique())
    has_gt = gt_uni.height > 0
    if has_gt:
        con.register("gt_view", gt_uni.to_arrow())
        hits = rule_meas(con,
            f"SELECT COUNT(*) FROM gt_view g JOIN "
            f"(SELECT DISTINCT entity_id_l AS lid, entity_id_r AS rid "
            f"FROM {phys}) p ON g.source1_entity_id=p.lid "
            f"AND g.other_entity_id=p.rid")
        recall = 100 * hits / max(gt_uni.height, 1)
    else:
        hits = None
        recall = None

    print("\n===== RESULTS =====")
    print(f"  candidate pairs after blocking: {n_cand:,} "
          f"(unique {n_uniq:,})")
    if has_gt:
        print(f"  ground-truth matches in input universe: {gt_uni.height:,}")
        print(f"  true matches captured in candidates: {hits:,}")
        print(f"  RECALL = {recall:.2f}%")
    else:
        print("  no ground truth available for this split (test): recall N/A")
    print(f"  memory used at peak observed: {mem_pct():.0f}% "
          f"(16GB machine)")

    out_path = os.path.join(out_dir,
                            f"splink_candidates_{split}_s{src}.parquet")
    con.execute(
        f"COPY (SELECT DISTINCT entity_id_l, entity_id_r FROM {phys}) "
        f"TO '{out_path.replace(chr(92), '/')}' (FORMAT PARQUET)")
    print(f"  saved candidate pairs -> {out_path}")
    print(f"  total runtime {time.time()-t0:.0f}s", flush=True)

    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sample", "full"], required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--src", type=int, choices=[2, 3], required=True)
    ap.add_argument("--out-dir", default=DATA_DIR)
    args = ap.parse_args()
    run(args.mode, args.split, args.src, args.out_dir)


if __name__ == "__main__":
    main()