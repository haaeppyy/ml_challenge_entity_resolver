import os

path = 'E:/ML_challenge/student_resource/output/candidate_pairs.tsv'
print('Exists:', os.path.isfile(path))
print('Size:', os.path.getsize(path))
try:
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline()
        print('Header:', repr(header[:100]))
        for i, line in enumerate(f):
            if i >= 3:
                break
            parts = line.rstrip('\n').split('\t')
            print(f'Line {i}: parts={len(parts)}, second_len={len(parts[1]) if len(parts)>1 else 0}')
    print('Done')
except Exception as e:
    print('Error:', e)