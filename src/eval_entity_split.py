import gc
import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pyarrow.parquet as pq
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from config import DATA_DIR

MODELS_DIR = os.path.join(os.path.dirname(DATA_DIR), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "lgbm_matcher.txt")

FEATURES = ["jw_name", "jw_addr", "tok_jaccard", "country_match",
            "name_len_diff", "fsig_match", "src"]
TARGET = "label"
BETA = 0.5
RANDOM_STATE = 42
ENT_PATH = os.path.join(DATA_DIR, "_ent_split.parquet")
VAL_FRAC = 0.2
THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.05), 2)
PRIOR_LEAKY_F05 = 0.7889   # row-split val, thr 0.20, from train_lgbm.py

FEAT_FILE = {2: "features_train_s2.parquet", 3: "features_train_s3.parquet"}


def mem_pct():
    try:
        import psutil
        return float(psutil.virtual_memory().percent)
    except Exception:
        return 0.0


def ent_val_mask(entity):
    h = int(hashlib.md5(entity.encode("utf-8")).hexdigest()[:8], 16)
    return h % 10 >= round(VAL_FRAC * 10)


def build_entity_split():
    if os.path.isfile(ENT_PATH):
        print(f"  entity split found: {ENT_PATH}", flush=True)
        return
    print("  building entity-level split (md5 %10: val = 20%)", flush=True)
    ents = set()
    for s in (2, 3):
        pf = pq.ParquetFile(os.path.join(DATA_DIR, FEAT_FILE[s]))
        for rb in pf.iter_batches(batch_size=4_000_000,
                                  columns=["entity_id_l"]):
            ents.update(rb.column("entity_id_l").to_pylist())
    arr = np.array(sorted(ents), dtype=object)
    isval = np.array([ent_val_mask(x) for x in arr], dtype=np.int8)
    (pl.DataFrame({"entity_id": arr, "isval": isval})
     .write_parquet(ENT_PATH))
    print(f"  entities: {len(arr):,}  val entities: {int(isval.sum()):,}  "
          f"(mem {mem_pct():.0f}%)", flush=True)


def load_train():
    Xparts, Yparts = [], []
    for i, s in enumerate([2, 3]):
        lz = (pl.scan_parquet(os.path.join(DATA_DIR, FEAT_FILE[s]))
              .join(pl.scan_parquet(ENT_PATH), left_on="entity_id_l",
                    right_on="entity_id", how="inner")
              .filter(pl.col("isval") == 0)
              .select(FEATURES[:-1] + [TARGET]))
        df = lz.collect()
        df = df.with_columns(pl.lit(i, dtype=pl.Int8).alias("src"))
        Xparts.append(np.ascontiguousarray(df.select(FEATURES).to_numpy(),
                                           dtype=np.float32))
        Yparts.append(df[TARGET].to_numpy().astype(np.uint8))
        print(f"  train rows s{s}: {df.height:,} (mem {mem_pct():.0f}%)",
              flush=True)
        df = None
    X = np.concatenate(Xparts)
    del Xparts
    y = np.concatenate(Yparts)
    del Yparts
    gc.collect()
    return X, y


def load_val():
    parts = []
    for i, s in enumerate([2, 3]):
        lz = (pl.scan_parquet(os.path.join(DATA_DIR, FEAT_FILE[s]))
              .join(pl.scan_parquet(ENT_PATH), left_on="entity_id_l",
                    right_on="entity_id", how="inner")
              .filter(pl.col("isval") == 1)
              .select(["entity_id_l"] + FEATURES[:-1] + [TARGET]))
        df = lz.collect()
        df = df.with_columns(pl.lit(i, dtype=pl.Int8).alias("src"))
        parts.append(df)
        print(f"  val rows s{s}: {df.height:,} (mem {mem_pct():.0f}%)",
              flush=True)
    val = pl.concat(parts)
    del parts
    X = np.ascontiguousarray(val.select(FEATURES).to_numpy(),
                             dtype=np.float32)
    y = val[TARGET].to_numpy().astype(np.uint8)
    eid = val["entity_id_l"].to_list()
    del val
    gc.collect()
    return X, y, eid


def entity_stats(eid, y, prob, thr):
    d = pl.DataFrame({"e": eid, "gt": y,
                      "pr": (prob >= thr).astype(np.int64),
                      "tp": ((y == 1) & (prob >= thr)).astype(np.int64)})
    a = d.group_by("e").agg(pl.col("gt").sum().alias("gt"),
                            pl.col("pr").sum().alias("pr"),
                            pl.col("tp").sum().alias("tp"))
    del d
    gt = a["gt"].to_numpy()
    pr = a["pr"].to_numpy()
    tp = a["tp"].to_numpy()
    fn = gt - tp
    fp = pr - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        rec = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
    f = np.where((0.25 * prec + rec) > 0,
                 (1.25 * prec * rec) / (0.25 * prec + rec), 0.0)
    a = a.with_columns(pl.Series("f", f, dtype=pl.Float64),
                       pl.Series("gt", gt, dtype=pl.Int64))
    return a


def report(name, a, country):
    a = a.join(country, left_on="e", right_on="entity_id", how="left")
    pos = int(a["gt"].sum())
    groups = []
    for ct in ("us", "india"):
        sub = a.filter(pl.col("country_norm") == ct)
        v = sub.filter(pl.col("gt") > 0)
        groups.append(
            f"    {ct:6s}: n_ent_gt={v.height:,}  "
            f"pos_pairs={int(sub['gt'].sum()):,}  "
            f"F0.5={v['f'].mean():.4f}")
    overall = a.filter(pl.col("gt") > 0)["f"].mean()
    print(f"  {name}: entity-macro F0.5 = {overall:.4f} "
          f"({int((a['gt'] > 0).sum()):,} entities with GT, "
          f"{pos:,} pos pairs)", flush=True)
    for g in groups:
        print(g, flush=True)
    return overall


def main():
    t0 = time.time()
    print(f"=== honest entity-level split evaluation ===", flush=True)
    build_entity_split()

    Xtr, ytr = load_train()
    print(f"  train rows total: {len(ytr):,}  "
          f"pos: {int(ytr.sum()):,} ({100*ytr.mean():.3f}%)  "
          f"(mem {mem_pct():.0f}%)", flush=True)

    Xva, yva, eid_va = load_val()
    print(f"  val rows total: {len(yva):,}  "
          f"pos: {int(yva.sum()):,} ({100*yva.mean():.3f}%)  "
          f"(mem {mem_pct():.0f}%)", flush=True)

    country = (pl.scan_parquet(os.path.join(DATA_DIR, "norm_train_s1.parquet"))
               .select("entity_id", "country_norm").collect())

    # ---- retrain on entity-level train split ----
    params = dict(objective="binary",
                  learning_rate=0.05,
                  num_leaves=31,
                  n_estimators=200,
                  is_unbalance=True,
                  max_bin=128,
                  random_state=RANDOM_STATE,
                  n_jobs=8,
                  verbosity=-1)
    m_new = lgb.LGBMClassifier(**params)
    m_new.fit(Xtr, ytr, eval_X=Xva, eval_y=yva, eval_metric="auc",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    prob_new = m_new.predict_proba(Xva)[:, 1]
    bs = m_new.best_score_["valid_0"]["auc"]
    yva_l = yva.astype(np.int64)
    print(f"  [retrained] best_iteration={m_new.best_iteration_}  "
          f"val AUC (early-stop)={bs:.4f}  "
          f"roc_auc_score={roc_auc_score(yva_l, prob_new):.4f}  "
          f"(mem {mem_pct():.0f}%)", flush=True)
    new_path = os.path.join(MODELS_DIR, "lgbm_entity_split.txt")
    m_new.booster_.save_model(new_path)
    print(f"  saved -> {new_path}", flush=True)
    del Xtr, ytr
    gc.collect()

    # ---- existing deployed model for reference ----
    m_old = lgb.Booster(model_file=MODEL_PATH)
    prob_old = m_old.predict(Xva)
    print(f"  [deployed] roc_auc={roc_auc_score(yva_l, prob_old):.4f}",
          flush=True)
    del m_old
    gc.collect()

    print(f"\n  ---- entity-macro F0.5 @ threshold 0.20 ----", flush=True)
    report("RETRAINED (entity-level)", entity_stats(eid_va, yva_l, prob_new,
                                                    0.20), country)
    report("DEPLOYED (row-split)", entity_stats(eid_va, yva_l, prob_old,
                                                0.20), country)

    print(f"\n  --- retrained: full threshold sweep "
          f"(reference; prior leaky best 0.7889 @ 0.20) ---", flush=True)
    best_thr, best_f = None, -1.0
    for thr in THRESHOLDS:
        a = entity_stats(eid_va, yva_l, prob_new, float(thr))
        f = a.filter(pl.col("gt") > 0)["f"].mean()
        mark = ""
        if f > best_f:
            best_f, best_thr = f, thr
            mark = "  <-- best"
        print(f"    {thr:.2f}   {f:.4f}{mark}", flush=True)
    print(f"\n  BEST retrained: threshold={best_thr:.2f}  "
          f"F0.5={best_f:.4f}  (prior leaky row-split best "
          f"{PRIOR_LEAKY_F05})", flush=True)
    print(f"  done in {time.time()-t0:.0f}s (peak mem {mem_pct():.0f}%)",
          flush=True)


if __name__ == "__main__":
    main()