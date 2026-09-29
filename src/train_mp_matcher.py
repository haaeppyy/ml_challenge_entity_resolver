"""Train a separate matcher on multipass candidate examples."""
import argparse
import gc
import os

import lightgbm as lgb
import numpy as np
import polars as pl

from config import MODEL_DIR

FEATURES = ["jw_name", "jw_addr", "tok_jaccard", "country_match",
            "name_len_diff", "fsig_match", "src"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-s2", required=True)
    ap.add_argument("--features-s3", required=True)
    ap.add_argument("--out", default=os.path.join(
        MODEL_DIR, "lgbm_matcher_multipass.txt"))
    ap.add_argument("--trees", type=int, default=200)
    args = ap.parse_args()

    arrays, labels = [], []
    for src, path in ((2, args.features_s2), (3, args.features_s3)):
        df = pl.read_parquet(path, columns=FEATURES[:-1] + ["label"])
        df = df.with_columns(pl.lit(0 if src == 2 else 1,
                                    dtype=pl.Int8).alias("src"))
        x = np.ascontiguousarray(df.select(FEATURES).to_numpy(),
                                 dtype=np.float32)
        y = df["label"].to_numpy().astype(np.uint8)
        print(f"S{src}: {len(y):,} rows; {int(y.sum()):,} positives; "
              f"positive rate={100*y.mean():.3f}%", flush=True)
        arrays.append(x)
        labels.append(y)
        del df, x, y
        gc.collect()

    X = np.concatenate(arrays)
    y = np.concatenate(labels)
    del arrays, labels
    gc.collect()
    print(f"combined: {len(y):,} rows; {int(y.sum()):,} positives; "
          f"positive rate={100*y.mean():.3f}%", flush=True)

    model = lgb.LGBMClassifier(
        objective="binary", learning_rate=0.05, num_leaves=31,
        n_estimators=args.trees, is_unbalance=True, max_bin=128,
        random_state=42, n_jobs=8, verbosity=-1)
    model.fit(X, y)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    model.booster_.save_model(args.out)
    print(f"saved separate multipass model: {args.out}", flush=True)
    gain = model.booster_.feature_importance(importance_type="gain")
    for name, value in sorted(zip(FEATURES, gain),
                              key=lambda item: item[1], reverse=True):
        print(f"  importance {name}: {value:.3f}", flush=True)


if __name__ == "__main__":
    main()
