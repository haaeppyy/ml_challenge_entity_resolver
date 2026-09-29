import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import polars as pl
import pandas as pd

from splink import Linker, SettingsCreator, block_on
from splink.backends.duckdb import DuckDBAPI
from splink.comparison_library import ExactMatch, JaroWinklerAtThresholds

n1 = pl.DataFrame({
    "entity_id": ["s1-1", "s1-2", "s1-3", "s1-4"],
    "name_norm": ["acme corp", "smith and sons", "baker street", "delta llc"],
    "name_tokens": [["acme", "corp"], ["smith", "sons"], ["baker", "street"], ["delta", "llc"]],
    "addr_norm": ["1 main street", "2 oak road", "3 bakers lane", "4 pine grove"],
    "country_norm": ["us", "us", "us", "us"],
})
n2 = pl.DataFrame({
    "entity_id": ["s2-10", "s2-11", "s2-12", "s2-13", "s2-14"],
    "name_norm": ["acme corporation", "smith sons", "baker st", "other", "delta llc"],
    "name_tokens": [["acme", "corporation"], ["smith", "sons"], ["baker", "st"], ["other"], ["delta", "llc"]],
    "addr_norm": ["1 main st", "2 oak rd", "3 baker lane", "elsewhere", "4 pine grove"],
    "country_norm": ["us", "us", "us", "us", "us"],
})

# ground truth: s1-1 ~ s2-10, s1-2 ~ s2-11, s1-3 ~ s2-12, s1-4 ~ s2-14
gt = pl.DataFrame({
    "source1_entity_id": ["s1-1", "s1-2", "s1-3", "s1-4"],
    "matched_entity_ids": ["s2-10", "s2-11", "s2-12", "s2-14"],
})

settings = SettingsCreator(
    link_type="link_only",
    comparisons=[
        JaroWinklerAtThresholds("name_norm", score_threshold_or_thresholds=[0.9, 0.7]),
        JaroWinklerAtThresholds("addr_norm", score_threshold_or_thresholds=[0.9, 0.7]),
        ExactMatch("country_norm"),
    ],
    blocking_rules_to_generate_predictions=[
        block_on("country_norm", "substr(name_norm,1,4)"),
        block_on("name_tokens", arrays_to_explode=["name_tokens"]),
    ],
    unique_id_column_name="entity_id",
    probability_two_random_records_match=0.0005,
)

api = DuckDBAPI()
linker = Linker(
    [n1.to_pandas(), n2.to_pandas()],
    settings,
    db_api=api,
    input_table_aliases=["s1", "s2"],
    set_up_basic_logging=False,
)

# show the generated blocking SQL to verify array rule syntax
for i, br in enumerate(settings.blocking_rules_to_generate_predictions):
    try:
        print(f"BR{i}:", br.get_blocking_rule("duckdb").blocking_rule_sql)
    except Exception as e:
        print(f"BR{i}: (render failed: {e})", br.arrays_to_explode)

# register pairwise labels from ground truth
labels = gt.select(
    pl.lit("s1").alias("source_dataset_l"),
    pl.col("source1_entity_id").alias("entity_id_l"),
    pl.lit("s2").alias("source_dataset_r"),
    pl.col("matched_entity_ids").alias("entity_id_r"),
)
print("labels rows:", labels.height, "cols:", labels.columns)
linker.table_management.register_table(labels.to_pandas(), "labels", overwrite=True)

print("--- supervised m from pairwise labels ---")
linker.training.estimate_m_from_pairwise_labels("labels")
print("--- u from random sampling ---")
linker.training.estimate_u_using_random_sampling(max_pairs=1e5)

print("--- trained params check ---")
import splink.internals.settings as SI
print([m for m in dir(linker._settings_obj) if 'trained' in m.lower() or 'param' in m.lower() or 'missing' in m.lower()])

df_predict = linker.inference.predict(threshold_match_probability=None).as_pandas_dataframe()
print("predict cols:", list(df_predict.columns))
print("candidate pairs (predict rows):", len(df_predict))
print(df_predict[["source_dataset_l", "entity_id_l", "source_dataset_r", "entity_id_r"]].head(20))

# recall: fraction of gt pairs in candidates
cand = set(
    (l, r) for l, r in zip(
        df_predict["entity_id_l"].astype(str), df_predict["entity_id_r"].astype(str))
)
rec = sum(1 for row in gt.iter_rows(named=True)
          if (row["source1_entity_id"], row["matched_entity_ids"]) in cand)
print(f"recall: {rec}/{gt.height} = {100*rec/max(gt.height,1):.2f}%")