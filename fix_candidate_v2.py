import polars as pl

df = pl.read_csv('E:/ML_challenge/student_resource/output/candidate_pairs.tsv', separator='\t')

with open('E:/ML_challenge/student_resource/output/candidate_pairs_fixed.tsv', 'w', encoding='utf-8', newline='') as f:
    f.write('source1_entity_id\tcandidate_entity_ids\n')
    for row in df.iter_rows():
        s1 = row[0]
        candidates = row[1] if row[1] else ''
        if candidates:
            f.write(f'{s1}\t{candidates}\n')
        else:
            f.write(f'{s1}\t\n')

print('Fixed candidate_pairs.tsv')

with open('E:/ML_challenge/student_resource/output/candidate_pairs_fixed.tsv', 'r', encoding='utf-8') as f:
    header = f.readline()
    empty_count = 0
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if len(parts) == 2 and parts[1] == '':
            empty_count += 1
print(f'Empty rows: {empty_count}')

import shutil
shutil.copy2('E:/ML_challenge/student_resource/output/candidate_pairs_fixed.tsv', 
             'E:/ML_challenge/student_resource/output/candidate_pairs.tsv')
print('Copied to final location')