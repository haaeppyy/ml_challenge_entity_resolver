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

SAMPLE_FRAC = 0.1


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


def get_val_predictions_sampled():
    print("Loading validation features (sampled)...", flush=True)
    
    entsplit = load_entsplit()
    val_ids = entsplit.filter(pl.col("isval") == 1)["entity_id"].to_list()
    
    import random
    random.seed(42)
    sample_size = int(len(val_ids) * SAMPLE_FRAC)
    val_ids_sample = random.sample(val_ids, sample_size)
    print(f"  Sampled entities: {len(val_ids_sample):,}")
    
    Xval_parts, yval_parts, eid_parts = [], [], []
    
    for src in [2, 3]:
        df = pl.read_parquet(
            os.path.join(DATA_DIR, f"features_val_s{src}.parquet"),
            columns=FEATURES + [TARGET, "entity_id_l", "entity_id_r"]
        )
        df = df.filter(pl.col("entity_id_l").is_in(val_ids_sample))
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
    
    print(f"  Total sampled val: {len(yval):,} rows", flush=True)
    
    mdl, calibrator, best_iteration, threshold = load_model()
    
    prob = mdl.predict(Xval, num_iteration=best_iteration) if best_iteration is not None else mdl.predict(Xval)
    prob = calibrator.predict_proba(prob.reshape(-1, 1))[:, 1]
    
    del Xval
    gc.collect()
    
    return eid_val, yval, prob, threshold


def entity_f05_sweep(eid, label, prob, thresholds):
    # Vectorized version using unique entity indices
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


def evaluate_policy_vectorized(eid, label, prob, singleton_cutoff, keep_threshold, fallback_top1_threshold):
    """Fully vectorized 3-threshold policy evaluation."""
    unique_eids, inverse = np.unique(eid, return_inverse=True)
    n_e = len(unique_eids)
    
    label_b = label.astype(np.int64)
    gt_cnt = np.bincount(inverse, weights=label_b, minlength=n_e)
    
    # Max prob per entity
    max_prob = np.zeros(n_e, dtype=np.float32)
    np.maximum.at(max_prob, inverse, prob)
    
    # Top-1 index per entity (argmax of prob within each entity group)
    # Use segment tree approach: sort by entity then prob descending
    sort_idx = np.lexsort((-prob, inverse))
    sorted_inv = inverse[sort_idx]
    _, first_idx = np.unique(sorted_inv, return_index=True)
    top1_idx = sort_idx[first_idx]  # Index of max prob per entity
    
    # Build prediction mask
    pred = np.zeros_like(prob, dtype=bool)
    
    # Entities with max >= keep_threshold: keep all >= keep_threshold
    keep_entities = max_prob >= keep_threshold
    keep_mask = keep_entities[inverse]
    pred[keep_mask] = prob[keep_mask] >= keep_threshold
    
    # Entities with fallback <= max < keep: keep only top-1
    fallback_entities = (max_prob >= fallback_top1_threshold) & (max_prob < keep_threshold)
    fallback_mask = fallback_entities[inverse]
    # Only set top-1 for these entities
    for idx in top1_idx[fallback_entities]:
        pred[idx] = True
    
    # Entities with max < singleton_cutoff: predict empty (already False)
    # Entities with singleton_cutoff <= max < fallback: predict empty
    
    # Compute F0.5
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
    print(f"=== Step 2: 3-Threshold Policy Grid Search (sample={SAMPLE_FRAC*100:.0f}%) ===", flush=True)
    
    eid_val, yval, prob_val, base_threshold = get_val_predictions_sampled()
    
    print(f"\nBaseline (single threshold {base_threshold:.2f}):")
    base_f05 = entity_f05_sweep(eid_val, yval, prob_val, [base_threshold])[base_threshold]
    print(f"  Entity-macro F0.5: {base_f05:.4f}")
    
    # Focused grid around promising regions
    singleton_cutoffs = [0.10, 0.15, 0.20, 0.25, 0.30]
    keep_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    fallback_thresholds = [0.20, 0.25, 0.30, 0.35]
    
    best_f05 = base_f05
    best_params = None
    
    total = 0
    for sc in singleton_cutoffs:
        for kt in keep_thresholds:
            for ft in fallback_thresholds:
                if ft < kt and sc < ft:
                    total += 1
    
    print(f"\nGrid size: {total} combinations", flush=True)
    count = 0
    
    for sc in singleton_cutoffs:
        for kt in keep_thresholds:
            for ft in fallback_thresholds:
                if ft >= kt or sc >= ft:
                    continue
                
                count += 1
                f05 = evaluate_policy_vectorized(eid_val, yval, prob_val, sc, kt, ft)
                
                if f05 > best_f05:
                    best_f05 = f05
                    best_params = (sc, kt, ft)
                    print(f"  NEW BEST [{count}/{total}]: sc={sc:.2f}, kt={kt:.2f}, ft={ft:.2f} -> F0.5={f05:.4f} (+{f05-base_f05:+.4f})")
                elif count % 20 == 0:
                    print(f"  [{count}/{total}] sc={sc:.2f}, kt={kt:.2f}, ft={ft:.2f} -> F0.5={f05:.4f}")
    
    print(f"\n=== RESULTS ===")
    print(f"Baseline F0.5: {base_f05:.4f}")
    if best_params:
        sc, kt, ft = best_params
        print(f"Best 3-threshold: singleton_cutoff={sc:.2f}, keep_threshold={kt:.2f}, fallback_top1={ft:.2f}")
        print(f"Best F0.5: {best_f05:.4f} (+{best_f05-base_f05:+.4f})")
        
        joblib.dump({
            'singleton_cutoff': sc,
            'keep_threshold': kt,
            'fallback_top1_threshold': ft,
            'base_threshold': base_threshold,
            'base_f05': base_f05,
            'best_f05': best_f05
        }, os.path.join(os.path.dirname(DATA_DIR), "models", "best_3threshold_policy.pkl"))
    else:
        print("No improvement over baseline")
    
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()