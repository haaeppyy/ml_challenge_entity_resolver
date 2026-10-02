import polars as pl

df = pl.read_csv('E:/ML_challenge/student_resource/output/matching_results.tsv', separator='\t')

# Write without quoting empty fields
with open('E:/ML_challenge/student_resource/output/matching_results_fixed.tsv', 'w', encoding='utf-8', newline='') as f:
    f.write('source1_entity_id\tmatched_entity_ids\n')
    for row in df.iter_rows():
        s1 = row[0]
        matched = row[1] if row[1] else ''
        if matched:
            f.write(f'{s1}\t{matched}\n')
        else:
            f.write(f'{s1}\t\n')

print('Fixed matching_results.tsv')

# Verify
with open('E:/ML_challenge/student_resource/output/matching_results_fixed.tsv', 'r', encoding='utf-8') as f:
    header = f.readline()
    empty_count = 0
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if len(parts) == 2 and parts[1] == '':
            empty_count += 1
print(f'Empty rows: {empty_count}')

# Copy to final location
import shutil
shutil.copy2('E:/ML_challenge/student_resource/output/matching_results_fixed.tsv', 
             'E:/ML_challenge/student_resource/output/matching_results.tsv')
print('Copied to final location')