import polars as pl
for f in ['candmp_v2_train_s2.parquet', 'candmp_v2_train_s3.parquet', 'candmpfit3_train_s2.parquet', 'candmpfit3_train_s3.parquet']:
    df = pl.read_parquet(f'data/{f}')
    print(f'{f}:')
    print(f'  rows: {df.height:,}')
    print(f'  unique entity_id_l: {df["entity_id_l"].n_unique():,}')

split = pl.read_parquet('data/_entsplit_md5.parquet')
val_entities = set(split.filter(pl.col('isval') == 1)['entity_id'].to_list())
train_entities = set(split.filter(pl.col('isval') == 0)['entity_id'].to_list())

for f in ['candmp_v2_train_s2.parquet', 'candmp_v2_train_s3.parquet', 'candmpfit3_train_s2.parquet', 'candmpfit3_train_s3.parquet']:
    df = pl.read_parquet(f'data/{f}')
    feat_entities = set(df['entity_id_l'].unique().to_list())
    print(f'  {f}: in val={len(feat_entities & val_entities):,}, in train={len(feat_entities & train_entities):,}')