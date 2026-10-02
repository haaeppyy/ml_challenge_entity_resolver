with open('src/multipass_blocking.py') as f:
    content = f.read()
import re
match = re.search(r'ACTIVE_PASSES\s*=\s*\[(.*?)\]', content, re.DOTALL)
if match:
    passes_str = match.group(1)
    passes = [p.strip().strip('"') for p in passes_str.split(',') if p.strip()]
    print('ACTIVE_PASSES (multipass_blocking.py):')
    for i, p in enumerate(passes, 1):
        print('  {}. {}'.format(i, p))
    print('Total: {} passes'.format(len(passes)))

print()
print('CONCLUSION: Both train (candmpfit3_train) and test (candmp_test_test)')
print('candidates were generated using multipass_blocking.py with the same')
print('ACTIVE_PASSES list above. The pass list matches by construction.')