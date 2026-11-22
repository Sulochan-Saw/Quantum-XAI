import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.config import CFG
from src.preprocessing.clean_data import load_merged, RAW

OUT = 'results/figures/'
os.makedirs(OUT, exist_ok=True)

# Figure 1: histogram of LN_IC50 (all 712 matched cell lines)
y = np.concatenate([np.load('data/processed/y_' + s + '.npy') for s in ['train', 'val', 'test']])
plt.figure(figsize=(6, 4))
plt.hist(y, bins=30, edgecolor='black')
plt.xlabel('LN_IC50')
plt.ylabel('Number of cell lines')
plt.title('Trametinib: LN_IC50 distribution (n = ' + str(len(y)) + ')')
plt.tight_layout()
plt.savefig(OUT + 'ln_ic50_hist.png', dpi=200)
plt.close()
print('saved ln_ic50_hist.png, n =', len(y))

# Figure 2: cell lines per tissue (lineage from DepMap Model.csv)
df = load_merged(CFG['drug_name'])
print(df.columns[:6].tolist())   # check the ID column name if the next lines fail
model = pd.read_csv(RAW + 'Model.csv', usecols=['SangerModelID', 'OncotreeLineage'])
model = model.drop_duplicates('SangerModelID')
m = df[['SANGER_MODEL_ID']].merge(model, left_on='SANGER_MODEL_ID',
                                  right_on='SangerModelID', how='left')
counts = m['OncotreeLineage'].fillna('Unknown').value_counts()
print(counts)
plt.figure(figsize=(7, 6))
counts.sort_values().plot(kind='barh')
plt.xlabel('Number of cell lines')
plt.title('Trametinib: cell lines per lineage (n = ' + str(len(m)) + ')')
plt.tight_layout()
plt.savefig(OUT + 'tissue_counts.png', dpi=200)
plt.close()
print('saved tissue_counts.png')