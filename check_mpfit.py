import polars as pl
for src in [2, 3]:
    df = pl.read_parquet(f'data/features_multipass_fit_s{src}.parquet', columns=['entity_id_l'])
    print(f'features_multipass_fit_s{src}:')
    print(f'  rows: {df.height:,}')
    print(f'  unique entity_id_l: {df["entity_id_l"].n_unique():,}')

split = pl.read_parquet('data/_entsplit_md5.parquet')
val_entities = set(split.filter(pl.col('isval') == 1)['entity_id'].to_list())
train_entities = set(split.filter(pl.col('isval') == 0)['entity_id'].to_list())

for src in [2, 3]:
    df = pl.read_parquet(f'data/features_multipass_fit_s{src}.parquet', columns=['entity_id_l'])
    feat_entities = set(df['entity_id_l'].unique().to_list())
    print(f'  features_multipass_fit_s{src}: in val={len(feat_entities & val_entities):,}, in train={len(feat_entities & train_entities):,}')