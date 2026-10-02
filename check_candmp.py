import polars as pl
df = pl.read_parquet('data/candmp_train_s2.parquet')
print('candmp_train_s2.parquet:')
print(f'  rows: {df.height:,}')
print(f'  unique entity_id_l: {df["entity_id_l"].n_unique():,}')

split = pl.read_parquet('data/_entsplit_md5.parquet')
val_entities = set(split.filter(pl.col('isval') == 1)['entity_id'].to_list())
train_entities = set(split.filter(pl.col('isval') == 0)['entity_id'].to_list())
feat_entities = set(df['entity_id_l'].unique().to_list())
print(f'  in val: {len(feat_entities & val_entities):,}')
print(f'  in train: {len(feat_entities & train_entities):,}')