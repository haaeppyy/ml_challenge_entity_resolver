import gc
import os
import sys
import time
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from config import DATA_DIR
from entsplit import load as load_entsplit

FEATURES = ["jw_name", "jw_addr", "tok_jaccard", "country_match",
            "name_len_diff", "fsig_match",
            "lev_name", "jaro_name", "jaro_addr",
            "addr_house_match", "addr_pin_match", "addr_street_jaccard",
            "src"]
TARGET = "label"
RANDOM_STATE = 42
THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.05), 2)
BETA = 0.5


def mem_pct():
    try:
        import psutil
        return float(psutil.virtual_memory().percent)
    except Exception:
        return 0.0


def load_ent_split_features():
    """Load features for train entities (multipass_fit) and val entities (candmp_core)."""
    print("  loading entity split...", flush=True)
    entsplit = load_entsplit()
    train_entities = set(entsplit.filter(pl.col("isval") == 0)["entity_id"].to_list())
    val_entities = set(entsplit.filter(pl.col("isval") == 1)["entity_id"].to_list())
    print(f"    train entities: {len(train_entities):,}")
    print(f"    val entities: {len(val_entities):,}", flush=True)

    # Load TRAIN features (from multipass_fit - train entities only)
    print("  loading training features (multipass_fit)...", flush=True)
    Xtrain_parts, Ytrain_parts = [], []
    for i, src in enumerate([2, 3]):
        df = pl.read_parquet(
            os.path.join(DATA_DIR, f"features_multipass_fit_s{src}.parquet"),
            columns=FEATURES[:-1] + [TARGET])  # exclude 'src' from file
        df = df.with_columns(pl.lit(i, dtype=pl.Int8).alias("src"))
        Xtrain_parts.append(np.ascontiguousarray(
            df.select(FEATURES).to_numpy(), dtype=np.float32))
        Ytrain_parts.append(df[TARGET].to_numpy().astype(np.uint8))
        pos = int(df[TARGET].sum())
        tot = df.height
        print(f"    s{src}: {tot:,} rows, pos={pos:,} ({100*pos/tot:.3f}%) (mem {mem_pct():.0f}%)", flush=True)
        df = None
        gc.collect()

    Xtrain = np.concatenate(Xtrain_parts)
    del Xtrain_parts
    ytrain = np.concatenate(Ytrain_parts)
    del Ytrain_parts
    gc.collect()
    print(f"  total train: {len(ytrain):,} rows (mem {mem_pct():.0f}%)", flush=True)

    # Load VAL features (from candmp_core - val entities only)
    print("  loading validation features (candmp_core)...", flush=True)
    Xval_parts, Yval_parts = [], []
    for i, src in enumerate([2, 3]):
        df = pl.read_parquet(
            os.path.join(DATA_DIR, f"features_train_s{src}.parquet"),
            columns=FEATURES[:-1] + [TARGET])  # exclude 'src' from file
        df = df.with_columns(pl.lit(i, dtype=pl.Int8).alias("src"))
        Xval_parts.append(np.ascontiguousarray(
            df.select(FEATURES).to_numpy(), dtype=np.float32))
        Yval_parts.append(df[TARGET].to_numpy().astype(np.uint8))
        pos = int(df[TARGET].sum())
        tot = df.height
        print(f"    s{src}: {tot:,} rows, pos={pos:,} ({100*pos/tot:.3f}%) (mem {mem_pct():.0f}%)", flush=True)
        df = None
        gc.collect()

    Xval = np.concatenate(Xval_parts)
    del Xval_parts
    yval = np.concatenate(Yval_parts)
    del Yval_parts
    gc.collect()
    print(f"  total val: {len(yval):,} rows (mem {mem_pct():.0f}%)", flush=True)

    return Xtrain, ytrain, Xval, yval


def entity_f05_sweep(entity_codes, label_rows, prob_rows):
    e = entity_codes
    label_b = label_rows.astype(np.int64)
    n_e = int(e.max()) + 1 if len(e) else 0
    gt_cnt = np.bincount(e, weights=label_b, minlength=n_e)
    results = {}
    for thr in THRESHOLDS:
        pred_rows = prob_rows >= thr
        tp_rows = pred_rows & (label_rows == 1)
        pred_cnt = np.bincount(e[pred_rows], minlength=n_e)
        tp_cnt = np.bincount(e[tp_rows], minlength=n_e)
        fn_cnt = gt_cnt - tp_cnt
        fp_cnt = pred_cnt - tp_cnt
        with np.errstate(divide="ignore", invalid="ignore"):
            prec = np.where((tp_cnt + fp_cnt) > 0,
                            tp_cnt / (tp_cnt + fp_cnt), 0.0)
            rec = np.where((tp_cnt + fn_cnt) > 0,
                           tp_cnt / (tp_cnt + fn_cnt), 0.0)
        num = (1 + BETA ** 2) * prec * rec
        den = (BETA ** 2) * prec + rec
        f = np.where(den > 0, num / den, 0.0)
        valid = gt_cnt > 0
        results[float(thr)] = (float(np.mean(f[valid])), int(valid.sum()))
    return results


def verify_split(train_entities, val_entities):
    """Verify no entity appears in both train and val."""
    overlap = train_entities & val_entities
    print(f"  train unique entities: {len(train_entities):,}")
    print(f"  val unique entities: {len(val_entities):,}")
    print(f"  overlap: {len(overlap):,}", flush=True)
    if overlap:
        print(f"  WARNING: {len(overlap)} entities appear in both train and val!", flush=True)
        print(f"  Example overlapping entities: {list(overlap)[:10]}", flush=True)
        return False
    else:
        print("  OK: No entity leakage between train and val", flush=True)
    # Spot check
    sample_train = list(train_entities)[:5]
    sample_val = list(val_entities)[:5]
    print(f"  Sample train entities: {sample_train}", flush=True)
    print(f"  Sample val entities: {sample_val}", flush=True)
    return True


def main():
    t0 = time.time()
    print(f"=== LightGBM matcher training (entity-level split) ===", flush=True)
    
    Xtrain, ytrain, Xval, yval = load_ent_split_features()
    
    # Verify entity split using the feature files' entity_id_l
    print("\n  Verifying entity split...", flush=True)
    # Load entity IDs from feature files to verify
    train_eids = set()
    val_eids = set()
    for src in [2, 3]:
        df_tr = pl.read_parquet(os.path.join(DATA_DIR, f"features_multipass_fit_s{src}.parquet"),
                                columns=["entity_id_l"])
        train_eids.update(df_tr["entity_id_l"].unique().to_list())
        df_va = pl.read_parquet(os.path.join(DATA_DIR, f"features_train_s{src}.parquet"),
                                columns=["entity_id_l"])
        val_eids.update(df_va["entity_id_l"].unique().to_list())
    verify_split(train_eids, val_eids)
    
    pos = int(ytrain.sum())
    neg = len(ytrain) - pos
    scale_pos_weight = neg / pos
    print(f"\n  train positives: {pos:,} ({100*pos/len(ytrain):.3f}%)", flush=True)
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}", flush=True)

    print(f"  arrays ready (mem {mem_pct():.0f}%)", flush=True)

    # Use native LightGBM Dataset API
    dtrain = lgb.Dataset(Xtrain, label=ytrain, free_raw_data=False)
    dval = lgb.Dataset(Xval, label=yval, reference=dtrain, free_raw_data=False)
    del Xtrain, Xval
    gc.collect()

    params = dict(objective="binary",
                  learning_rate=0.03,
                  num_leaves=63,
                  scale_pos_weight=scale_pos_weight,
                  max_bin=255,
                  min_data_in_leaf=50,
                  feature_fraction=0.8,
                  bagging_fraction=0.8,
                  bagging_freq=5,
                  lambda_l2=1.0,
                  metric="auc",
                  random_state=RANDOM_STATE,
                  n_jobs=8,
                  verbosity=-1)

    print("  training with native API...", flush=True)
    model = lgb.train(params,
                      dtrain,
                      num_boost_round=500,
                      valid_sets=[dtrain, dval],
                      valid_names=['train', 'valid'],
                      callbacks=[lgb.early_stopping(100, verbose=True),
                                 lgb.log_evaluation(period=50)])
    
    best_iteration = model.best_iteration
    print(f"  best_iteration={best_iteration}", flush=True)

    # Get validation predictions
    prob_val = model.predict(dval.data, num_iteration=best_iteration)
    yval_arr = dval.get_label()
    
    bs = model.best_score['valid']['auc'] if model.best_score else None
    print(f"  val AUC = {roc_auc_score(yval_arr, prob_val):.4f}  "
          f"val logloss = {log_loss(yval_arr, prob_val):.4f}  "
          f"(mem {mem_pct():.0f}%)", flush=True)

    # Platt calibration on validation set only
    print("  fitting Platt calibration on validation set...", flush=True)
    cal = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
    cal.fit(prob_val.reshape(-1, 1), yval_arr)
    prob_val_cal = cal.predict_proba(prob_val.reshape(-1, 1))[:, 1]
    print(f"  calibrated val AUC = {roc_auc_score(yval_arr, prob_val_cal):.4f}  "
          f"calibrated logloss = {log_loss(yval_arr, prob_val_cal):.4f}", flush=True)
    prob_val = prob_val_cal

    # Save model and calibrator
    out_dir = os.path.join(os.path.dirname(DATA_DIR), "models")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({'model': model, 'calibrator': cal, 'best_iteration': best_iteration},
                os.path.join(out_dir, "lgbm_matcher_calibrated.pkl"))
    model.save_model(os.path.join(out_dir, "lgbm_matcher.txt"))
    print(f"  saved calibrated model -> {out_dir}", flush=True)

    # For entity-macro F0.5, we need entity codes for validation set
    # Load val entity_ids and create codes
    val_eid_list = []
    for src in [2, 3]:
        df = pl.read_parquet(os.path.join(DATA_DIR, f"features_train_s{src}.parquet"),
                             columns=["entity_id_l"])
        val_eid_list.append(df["entity_id_l"].to_numpy())
    val_eid_all = np.concatenate(val_eid_list)
    _, val_entity_codes = np.unique(val_eid_all, return_inverse=True)
    val_entity_codes = val_entity_codes.astype(np.int32)
    del val_eid_list, val_eid_all
    gc.collect()

    # Entity-macro F0.5 sweep on validation set
    res = entity_f05_sweep(val_entity_codes, yval_arr.astype(np.int64), prob_val)
    print(f"\n  threshold  macro-F0.5  entities-with-GT")
    best_thr, best_f = None, -1.0
    for thr in THRESHOLDS:
        f, n_ent = res[float(thr)]
        mark = ""
        if f > best_f:
            best_f, best_thr = f, thr
            mark = "  <-- best"
        print(f"    {thr:.2f}      {f:.4f}    {n_ent}{mark}")
    
    # Flag if implausibly high
    if best_f > 0.90:
        print(f"\n  WARNING: Best F0.5 = {best_f:.4f} at threshold {best_thr:.2f} looks implausibly high!", flush=True)
        print(f"     This may indicate data leakage or overfitting. Please investigate.", flush=True)
    else:
        print(f"\n  OK: Best F0.5 = {best_f:.4f} at threshold {best_thr:.2f} looks plausible.", flush=True)
    
    print(f"\n  BEST: threshold={best_thr:.2f}  "
          f"validation entity-macro F0.5={best_f:.4f}", flush=True)

    # Save best threshold
    joblib.dump({'threshold': best_thr, 'f05': best_f},
                os.path.join(out_dir, "best_threshold.pkl"))
    
    print(f"  done in {time.time()-t0:.0f}s (peak mem {mem_pct():.0f}%)", flush=True)


if __name__ == "__main__":
    main()