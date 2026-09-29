import gc
import os
import sys
import time
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import polars as pl
from config import DATA_DIR
from entsplit import load as load_entsplit

FEATURES = ["jw_name", "jw_addr", "tok_jaccard", "country_match",
            "name_len_diff", "fsig_match",
            "lev_name", "jaro_name", "jaro_addr",
            "addr_house_match", "addr_pin_match", "addr_street_jaccard",
            "src"]
TARGET = "label"
BETA = 0.5


def load_model():
    model_path = os.path.join(os.path.dirname(DATA_DIR), "models", "lgbm_matcher_calibrated.pkl")
    threshold_path = os.path.join(os.path.dirname(DATA_DIR), "models", "best_threshold.pkl")
    
    model_dict = joblib.load(model_path)
    mdl = model_dict['model']
    calibrator = model_dict['calibrator']
    best_iteration = model_dict.get('best_iteration', None)
    
    threshold_dict = joblib.load(threshold_path)
    threshold = float(threshold_dict.get('threshold', 0.55))
    
    return mdl, calibrator, best_iteration, threshold


def load_3threshold_policy():
    policy_path = os.path.join(os.path.dirname(DATA_DIR), "models", "best_3threshold_policy.pkl")
    return joblib.load(policy_path)


def get_full_val_predictions():
    print("Loading FULL validation features...", flush=True)
    
    entsplit = load_entsplit()
    val_ids = entsplit.filter(pl.col("isval") == 1)["entity_id"].to_list()
    print(f"  Val entities: {len(val_ids):,}")
    
    Xval_parts, yval_parts, eid_parts = [], [], []
    
    for src in [2, 3]:
        df = pl.read_parquet(
            os.path.join(DATA_DIR, f"features_val_s{src}.parquet"),
            columns=FEATURES + [TARGET, "entity_id_l", "entity_id_r"]
        )
        df = df.filter(pl.col("entity_id_l").is_in(val_ids))
        print(f"  S{src}: {df.height:,} rows, pos={df[TARGET].sum():,}")
        
        Xval_parts.append(np.ascontiguousarray(df.select(FEATURES).to_numpy(), dtype=np.float32))
        yval_parts.append(df[TARGET].to_numpy().astype(np.uint8))
        eid_parts.append(df["entity_id_l"].to_numpy())
        df = None
        gc.collect()
    
    Xval = np.concatenate(Xval_parts)
    yval = np.concatenate(yval_parts)
    eid_val = np.concatenate(eid_parts)
    del Xval_parts, yval_parts, eid_parts
    gc.collect()
    
    print(f"  Total val: {len(yval):,} rows", flush=True)
    
    mdl, calibrator, best_iteration, threshold = load_model()
    print(f"  Predicting...", flush=True)
    
    prob = mdl.predict(Xval, num_iteration=best_iteration) if best_iteration is not None else mdl.predict(Xval)
    prob = calibrator.predict_proba(prob.reshape(-1, 1))[:, 1]
    
    del Xval
    gc.collect()
    
    return eid_val, yval, prob, threshold


def entity_f05_sweep(eid, label, prob, thresholds):
    unique_eids, inverse = np.unique(eid, return_inverse=True)
    n_e = len(unique_eids)
    
    label_b = label.astype(np.int64)
    gt_cnt = np.bincount(inverse, weights=label_b, minlength=n_e)
    
    results = {}
    for thr in thresholds:
        pred = prob >= thr
        tp = pred & (label == 1)
        
        pred_cnt = np.bincount(inverse[pred], minlength=n_e)
        tp_cnt = np.bincount(inverse[tp], minlength=n_e)
        fn_cnt = gt_cnt - tp_cnt
        fp_cnt = pred_cnt - tp_cnt
        
        with np.errstate(divide="ignore", invalid="ignore"):
            prec = np.where((tp_cnt + fp_cnt) > 0, tp_cnt / (tp_cnt + fp_cnt), 0.0)
            rec = np.where((tp_cnt + fn_cnt) > 0, tp_cnt / (tp_cnt + fn_cnt), 0.0)
        num = (1 + BETA ** 2) * prec * rec
        den = (BETA ** 2) * prec + rec
        f = np.where(den > 0, num / den, 0.0)
        valid = gt_cnt > 0
        results[thr] = float(np.mean(f[valid]))
    
    return results


def three_threshold_policy_vectorized(eid, prob, singleton_cutoff, keep_threshold, fallback_top1_threshold):
    unique_eids, inverse = np.unique(eid, return_inverse=True)
    n_e = len(unique_eids)
    
    max_prob = np.zeros(n_e, dtype=np.float32)
    np.maximum.at(max_prob, inverse, prob)
    
    sort_idx = np.lexsort((-prob, inverse))
    sorted_inv = inverse[sort_idx]
    _, first_idx = np.unique(sorted_inv, return_index=True)
    top1_idx = sort_idx[first_idx]
    
    pred = np.zeros_like(prob, dtype=bool)
    
    keep_entities = max_prob >= keep_threshold
    keep_mask = keep_entities[inverse]
    pred[keep_mask] = prob[keep_mask] >= keep_threshold
    
    fallback_entities = (max_prob >= fallback_top1_threshold) & (max_prob < keep_threshold)
    for idx in top1_idx[fallback_entities]:
        pred[idx] = True
    
    return pred


def evaluate_policy(eid, label, prob, singleton_cutoff, keep_threshold, fallback_top1_threshold):
    pred = three_threshold_policy_vectorized(eid, prob, singleton_cutoff, keep_threshold, fallback_top1_threshold)
    
    unique_eids, inverse = np.unique(eid, return_inverse=True)
    n_e = len(unique_eids)
    
    label_b = label.astype(np.int64)
    gt_cnt = np.bincount(inverse, weights=label_b, minlength=n_e)
    
    tp = pred & (label == 1)
    pred_cnt = np.bincount(inverse[pred], minlength=n_e)
    tp_cnt = np.bincount(inverse[tp], minlength=n_e)
    fn_cnt = gt_cnt - tp_cnt
    fp_cnt = pred_cnt - tp_cnt
    
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where((tp_cnt + fp_cnt) > 0, tp_cnt / (tp_cnt + fp_cnt), 0.0)
        rec = np.where((tp_cnt + fn_cnt) > 0, tp_cnt / (tp_cnt + fn_cnt), 0.0)
    num = (1 + BETA ** 2) * prec * rec
    den = (BETA ** 2) * prec + rec
    f = np.where(den > 0, num / den, 0.0)
    valid = gt_cnt > 0
    
    return float(np.mean(f[valid]))


def main():
    t0 = time.time()
    print("=== Verify 3-Threshold Policy on FULL Validation Set ===", flush=True)
    
    eid_val, yval, prob_val, base_threshold = get_full_val_predictions()
    policy = load_3threshold_policy()
    
    sc = policy['singleton_cutoff']
    kt = policy['keep_threshold']
    ft = policy['fallback_top1_threshold']
    print(f"Policy: singleton_cutoff={sc}, keep_threshold={kt}, fallback_top1={ft}")
    
    print(f"\nBaseline (single threshold {base_threshold:.2f}):")
    base_f05 = entity_f05_sweep(eid_val, yval, prob_val, [base_threshold])[base_threshold]
    print(f"  Entity-macro F0.5: {base_f05:.4f}")
    
    print(f"\n3-Threshold Policy:")
    best_f05 = evaluate_policy(eid_val, yval, prob_val, sc, kt, ft)
    print(f"  Entity-macro F0.5: {best_f05:.4f} (+{best_f05-base_f05:+.4f})")
    
    # Also test a few nearby configs
    print(f"\nNearby configs:")
    for sc_test in [0.05, 0.10, 0.15]:
        for kt_test in [0.55, 0.60, 0.65]:
            for ft_test in [0.15, 0.20, 0.25]:
                if ft_test < kt_test and sc_test < ft_test:
                    f05 = evaluate_policy(eid_val, yval, prob_val, sc_test, kt_test, ft_test)
                    print(f"  sc={sc_test:.2f}, kt={kt_test:.2f}, ft={ft_test:.2f} -> F0.5={f05:.4f}")
    
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()