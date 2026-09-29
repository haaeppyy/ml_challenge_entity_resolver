import gc
import os
import sys
import time
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss
from config import DATA_DIR

FEATURES = ["jw_name", "jw_addr", "tok_jaccard", "country_match",
            "name_len_diff", "fsig_match",
            "lev_name", "jaro_name", "jaro_addr",
            "addr_house_match", "addr_pin_match", "addr_street_jaccard",
            "src"]
TARGET = "label"
VAL_FRAC = 0.2
RANDOM_STATE = 42
THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.05), 2)
BETA = 0.5


def mem_pct():
    try:
        import psutil
        return float(psutil.virtual_memory().percent)
    except Exception:
        return 0.0


def load():
    # Use a sample for faster training - take every 5th row
    SAMPLE_STEP = 5
    Xparts, Yparts, Eid = [], [], []
    for i, src in enumerate([2, 3]):
        df = pl.read_parquet(
            os.path.join(DATA_DIR, f"features_train_s{src}.parquet"),
            columns=["entity_id_l", "jw_name", "jw_addr", "tok_jaccard",
                     "country_match", "name_len_diff", "fsig_match",
                     "lev_name", "jaro_name", "jaro_addr",
                     "addr_house_match", "addr_pin_match", "addr_street_jaccard",
                     TARGET])
        df = df.with_columns(pl.lit(i, dtype=pl.Int8).alias("src"))
        # Sample every Nth row
        df = df.gather_every(SAMPLE_STEP)
        Xparts.append(np.ascontiguousarray(
            df.select(FEATURES).to_numpy(), dtype=np.float32))
        Yparts.append(df[TARGET].to_numpy().astype(np.uint8))
        Eid.append(df["entity_id_l"])
        print(f"  loaded s{src}: {df.height:,} rows "
              f"(mem {mem_pct():.0f}%)", flush=True)
        df = None
    Xall = np.concatenate(Xparts)
    del Xparts
    yall = np.concatenate(Yparts)
    del Yparts
    eid_all = pl.concat(Eid)
    del Eid
    gc.collect()
    print(f"  total (sampled): {len(yall):,} rows (mem {mem_pct():.0f}%)", flush=True)
    return Xall, yall, eid_all


def stratified_split(y):
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y)
    train_mask = np.zeros(n, dtype=bool)
    for lab in (0, 1):
        idx = np.flatnonzero(y == lab)
        sel = rng.random(len(idx)) < (1 - VAL_FRAC)
        train_mask[idx[sel]] = True
    return train_mask, ~train_mask


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


def main():
    t0 = time.time()
    print(f"=== LightGBM matcher training (unified s2+s3) ===", flush=True)
    Xall, yall, eid_all = load()
    pos = int(yall.sum())
    neg = len(yall) - pos
    scale_pos_weight = neg / pos
    print(f"  positives: {pos:,} ({100*pos/len(yall):.3f}%)", flush=True)
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}", flush=True)

    train_mask, val_mask = stratified_split(yall)
    print(f"  train rows: {train_mask.sum():,}  "
          f"val rows: {val_mask.sum():,}  (mem {mem_pct():.0f}%)",
          flush=True)

    Xtrain, Xval = Xall[train_mask], Xall[val_mask]
    ytrain, yval = yall[train_mask], yall[val_mask]
    del Xall, yall, train_mask
    gc.collect()
    print(f"  arrays ready (mem {mem_pct():.0f}%)", flush=True)

    # Use native LightGBM Dataset API for memory efficiency
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
    # Get predictions
    prob_val = model.predict(dval.data, num_iteration=model.best_iteration)
    yval_arr = dval.get_label()

    # Get predictions
    prob_val = model.predict(dval.data, num_iteration=model.best_iteration)
    yval_arr = dval.get_label()

    bs = model.best_score['valid']['auc'] if model.best_score else None
    print(f"  best_iteration={model.best_iteration}  "
          f"best_val_auc={bs:.4f}", flush=True)
    print(f"  val AUC = {roc_auc_score(yval_arr, prob_val):.4f}  "
          f"val logloss = {log_loss(yval_arr, prob_val):.4f}  "
          f"(mem {mem_pct():.0f}%)", flush=True)

    # Platt calibration
    from sklearn.calibration import CalibratedClassifierCV
    print("  fitting Platt calibration...", flush=True)
    # Use a simple logistic regression on the raw scores
    from sklearn.linear_model import LogisticRegression
    cal = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
    cal.fit(prob_val.reshape(-1, 1), yval_arr)
    prob_val_cal = cal.predict_proba(prob_val.reshape(-1, 1))[:, 1]
    print(f"  calibrated val AUC = {roc_auc_score(yval_arr, prob_val_cal):.4f}  "
          f"calibrated logloss = {log_loss(yval_arr, prob_val_cal):.4f}", flush=True)
    prob_val = prob_val_cal
    # Save calibrator
    calibrator = cal

    # Save model and calibrator
    import joblib
    out_dir = os.path.join(os.path.dirname(DATA_DIR), "models")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({'model': model, 'calibrator': calibrator, 'best_iteration': model.best_iteration},
                os.path.join(out_dir, "lgbm_matcher_calibrated.pkl"))
    # Save raw booster
    model.save_model(os.path.join(out_dir, "lgbm_matcher.txt"))
    print(f"  saved calibrated model -> {out_dir}", flush=True)

    _, entity_codes = np.unique(eid_all, return_inverse=True)
    # Get only validation portion
    val_eid_codes = entity_codes[val_mask]
    del eid_all, val_mask
    entity_codes = val_eid_codes.astype(np.int32)
    gc.collect()

    res = entity_f05_sweep(entity_codes, yval_arr.astype(np.int64), prob_val)
    print(f"\n  threshold  macro-F0.5  entities-with-GT")
    best_thr, best_f = None, -1.0
    for thr in THRESHOLDS:
        f, n_ent = res[float(thr)]
        mark = ""
        if f > best_f:
            best_f, best_thr = f, thr
            mark = "  <-- best"
        print(f"    {thr:.2f}      {f:.4f}    {n_ent}{mark}")
    print(f"\n  BEST: threshold={best_thr:.2f}  "
          f"validation entity-macro F0.5={best_f:.4f}", flush=True)

    # Save model and calibrator
    import joblib
    out_dir = os.path.join(os.path.dirname(DATA_DIR), "models")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({'model': model, 'best_iteration': model.best_iteration},
                os.path.join(out_dir, "lgbm_matcher.pkl"))
    model.save_model(os.path.join(out_dir, "lgbm_matcher.txt"))
    print(f"  saved model -> {out_dir}", flush=True)
    print(f"  done in {time.time()-t0:.0f}s (peak mem {mem_pct():.0f}%)",
          flush=True)


if __name__ == "__main__":
    main()