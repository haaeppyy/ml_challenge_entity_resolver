import polars as pl
df1 = pl.read_parquet('data/features_multipass_fit_s2.parquet', n_rows=0)
df2 = pl.read_parquet('data/features_train_s2.parquet', n_rows=0)
print('features_multipass_fit_s2 columns:', df1.columns)
print('features_train_s2 columns:', df2.columns)
print()
for src in [2, 3]:
    df_tr = pl.read_parquet(f'data/features_multipass_fit_s{src}.parquet')
    df_va = pl.read_parquet(f'data/features_train_s{src}.parquet')
    print(f'S{src} train (multipass_fit): {df_tr.height:,} rows, pos={df_tr["label"].sum():,}, neg={df_tr.height - df_tr["label"].sum():,}')
    print(f'S{src} val (candmp_core): {df_va.height:,} rows, pos={df_va["label"].sum():,}, neg={df_va.height - df_va["label"].sum():,}')