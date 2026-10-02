import polars as pl
import shutil

df = pl.read_csv('E:/ML_challenge/student_resource/output/candidate_pairs.tsv', separator='\t')
df = df.with_columns(
    pl.when(pl.col('candidate_entity_ids') == '""')
    .then(pl.lit(''))
    .otherwise(pl.col('candidate_entity_ids'))
    .alias('candidate_entity_ids')
)
df.write_csv('E:/ML_challenge/student_resource/output/candidate_pairs_fixed.tsv', separator='\t')
shutil.copy2('E:/ML_challenge/student_resource/output/candidate_pairs_fixed.tsv', 
             'E:/ML_challenge/student_resource/output/candidate_pairs.tsv')
print('Fixed candidate_pairs.tsv')
df2 = pl.read_csv('E:/ML_challenge/student_resource/output/candidate_pairs.tsv', separator='\t')
empty = (df2['candidate_entity_ids'] == '').sum()
print('Empty string count:', empty)