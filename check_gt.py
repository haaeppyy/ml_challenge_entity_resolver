import polars as pl
gt = pl.read_csv('dataset/train/train_ground_truth.tsv', separator='\t', columns=['source1_entity_id', 'matched_entity_ids'])
gt = gt.filter(pl.col('matched_entity_ids').is_not_null())
gt = gt.with_columns(pl.col('matched_entity_ids').str.split(',').alias('ids'))
gt = gt.explode('ids')
gt = gt.select(pl.col('source1_entity_id').cast(pl.Utf8), pl.col('ids').str.strip_chars().cast(pl.Utf8).alias('matched'))
print(f'GT pairs: {gt.height:,}')
print(f'Unique source1: {gt["source1_entity_id"].n_unique():,}')
print(f'Unique matched: {gt["matched"].n_unique():,}')

# Filter for S2 and S3 separately
gt_s2 = gt.filter(pl.col('matched').str.starts_with('S2-'))
gt_s3 = gt.filter(pl.col('matched').str.starts_with('S3-'))
print(f'GT pairs S2: {gt_s2.height:,}')
print(f'GT pairs S3: {gt_s3.height:,}')