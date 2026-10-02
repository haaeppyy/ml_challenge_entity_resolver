import polars as pl
import shutil

df = pl.read_csv('E:/ML_challenge/student_resource/output/matching_results.tsv', separator='\t')
df = df.with_columns(
    pl.when(pl.col('matched_entity_ids') == '""')
    .then(pl.lit(''))
    .otherwise(pl.col('matched_entity_ids'))
    .alias('matched_entity_ids')
)
df.write_csv('E:/ML_challenge/student_resource/output/matching_results_fixed.tsv', separator='\t')
shutil.copy2('E:/ML_challenge/student_resource/output/matching_results_fixed.tsv', 
             'E:/ML_challenge/student_resource/output/matching_results.tsv')
print('Fixed matching_results.tsv')
df2 = pl.read_csv('E:/ML_challenge/student_resource/output/matching_results.tsv', separator='\t')
empty = (df2['matched_entity_ids'] == '').sum()
print('Empty string count:', empty)