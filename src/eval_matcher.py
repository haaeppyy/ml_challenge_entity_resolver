import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gc
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss
from config import DATA_DIR

FEATURES = ["jw_name", "jw_addr", "tok_jaccard", "country_match",
            "name_len_diff", "fsig_match", "src"]
TARGET = "label"
VAL_FRAC = 0.2
RANDOM_STATE = 42
THRESHOLDS = np.round(np.arange(0.05, 0.901, 0.05), 2)
BETA = 0.5


def mem_pct():
    try:
        import psutil
        return float(psutil.virtual_memory().percent)
    except Exception:
        return 0.0


def load_all():
    Xparts, Yparts, Eid = [], [], []
    for i, src in enumerate([2, 3]):
        df = pl.read_parquet(
            os.path.join(DATA_DIR, f"features_train_s{src}.parquet"),
            columns=["entity_id_l", "jw_name", "jw_addr", "tok_jaccard",
                     "country_match", "name_len_diff", "fsig_match",
                     TARGET])
        df = df.with_columns(pl.lit(i, dtype=pl.Int8).alias("src"))
        Xparts.append(np.ascontiguousarray(df.select(FEATURES).to_numpy(),
                                           dtype=np.float32))
        Yparts.append(df[TARGET].to_numpy().astype(np.uint8))
        Eid.append(df["entity_id_l"])
        df = None
        print(f"  loaded s{src} (mem {mem_pct():.0f}%)", flush=True)
    return (np.concatenate(Xparts), np.concatenate(Yparts),
            pl.concat(Eid))


def val_mask_of(y):
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y)
    train_mask = np.zeros(n, dtype=bool)
    for lab in (0, 1):
        idx = np.flatnonzero(y == lab)
        sel = rng.random(len(idx)) < (1 - VAL_FRAC)
        train_mask[idx[sel]] = True
    return ~train_mask


def entity_f05_sweep(e, y, p):
    n_e = int(e.max()) + 1 if len(e) else 0
    yb = y.astype(np.int64)
    gt_cnt = np.bincount(e, weights=yb, minlength=n_e)
    res = []
    for thr in THRESHOLDS:
        pr = p >= thr
        tr = pr & (y == 1)
        pred_cnt = np.bincount(e[pr], minlength=n_e).astype(np.float64)
        tp_cnt = np.bincount(e[tr], minlength=n_e).astype(np.float64)
        fn_cnt = gt_cnt - tp_cnt
        fp_cnt = pred_cnt - tp_cnt
        prec = np.zeros(n_e)
        rec = np.zeros(n_e)
        nz = (pred_cnt > 0)
        prec[nz] = tp_cnt[nz] / pred_cnt[nz]
        gr = (gt_cnt > 0)
        rec[gr] = tp_cnt[gr] / gt_cnt[gr]
        num = (1 + BETA ** 2) * prec * rec
        den = (BETA ** 2) * prec + rec
        f = np.zeros(n_e)
        ok = (den > 0) & np.isfinite(num) & np.isfinite(den)
        f[ok] = num[ok] / den[ok]
        valid = gt_cnt > 0
        res.append((float(thr), float(np.mean(f[valid])), int(valid.sum()),
                    int(pred_cnt.sum())))
    return res


def main():
    t0 = time.time()
    X, y, eid = load_all()
    mask = val_mask_of(y)
    print(f"  val rows: {mask.sum():,}  (mem {mem_pct():.0f}%)", flush=True)

    Xval = X[mask]
    yval = y[mask]
    del X, y
    gc.collect()

    mdl = lgb.Booster(model_file=os.path.join(
        os.path.dirname(DATA_DIR), "models", "lgbm_matcher.txt"))
    p = mdl.predict(Xval)
    val_eid = np.array([s for s in eid.filter(
        pl.Series("mask", mask)).to_list()], dtype=object)
    del eid, mask, Xval
    _, e = np.unique(val_eid, return_inverse=True)
    del val_eid
    e = e.astype(np.int32)
    gc.collect()

    print(f"  val AUC = {roc_auc_score(yval, p):.4f}  "
          f"logloss = {log_loss(yval, p):.4f}", flush=True)
    print(f"\n  threshold  macro-F0.5  ent_w/GT  #pred")
    best_thr, best_f = None, -1.0
    for thr, f, n_ent, npred in entity_f05_sweep(e, yval, p):
        mark = ""
        if f > best_f:
            best_f, best_thr = f, thr
            mark = "  <-- best"
        print(f"    {thr:.2f}      {f:.4f}    {n_ent:>8,}  {npred:>12,}{mark}")
    print(f"\n  BEST: threshold={best_thr:.2f}  "
          f"validation entity-macro F0.5={best_f:.4f}", flush=True)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()