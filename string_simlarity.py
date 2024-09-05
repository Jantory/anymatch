from difflib import SequenceMatcher
from sklearn.metrics import f1_score
import pandas as pd

def string_similarity(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()

datasets = ['abt', 'amgo', 'beer', 'dbac', 'dbgo', 'foza', 'itam', 'waam', 'wdc']
f1s = []
for d in datasets:
    df = pd.read_csv(f'data/{d}_test_pairs.csv')
    df = df.fillna('nan')
    l_cols = [c for c in df.columns if c.endswith('_l')]
    r_cols = [c for c in df.columns if c.endswith('_r')]
    df['textA'] = df.apply(lambda x: ', '.join([str(x[c]) for c in l_cols]), axis=1)
    df['textB'] = df.apply(lambda x: ', '.join([str(x[c]) for c in r_cols]), axis=1)
    df['prediction'] = df.apply(lambda x: 1 if string_similarity(x['textA'], x['textB'])>0.5 else 0, axis=1)
    f1 = f1_score(df['label'], df['prediction'])
    print(f'{d}: {f1*100:.2f}')
    f1s.append(f1)

print(f'Average F1: {sum(f1s)/len(f1s)*100:.2f}')