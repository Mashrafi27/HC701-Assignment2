import pandas as pd

df = pd.read_csv('pneumonia_results/data_split.csv')
filenames = df['filename'].tolist()

print("="*80)
print("DATA INTEGRITY CHECK")
print("="*80)
print(f"Total entries: {len(filenames)}")
print(f"Unique filenames: {len(set(filenames))}")
duplicates_count = len(filenames) - len(set(filenames))
print(f"Duplicates: {duplicates_count}")

if duplicates_count == 0:
    print("\n✓ NO DUPLICATES - Clean splits, no image appears in multiple sets")
else:
    print("\n✗ DUPLICATES FOUND!")
    dups = df[df.duplicated(subset=['filename'], keep=False)].sort_values('filename')
    print(dups[['filename', 'label', 'split']])

print("\n" + "="*80)
print("SPLIT DISTRIBUTION")
print("="*80)

for split in ['train', 'val', 'test']:
    s = df[df['split'] == split]
    norm = len(s[s['label'] == 'NORMAL'])
    pneu = len(s[s['label'] == 'PNEUMONIA'])
    total = len(s)
    print(f"\n{split.upper()}: {total} files")
    print(f"  NORMAL: {norm} ({norm/total*100:.1f}%)")
    print(f"  PNEUMONIA: {pneu} ({pneu/total*100:.1f}%)")
