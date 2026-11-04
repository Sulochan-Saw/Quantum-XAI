import glob
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.config import CFG

RAW, OUT = 'data/raw/', 'data/processed/'
K = CFG['n_features']


def load_merged(drug):
    resp = pd.read_excel(glob.glob(RAW + 'GDSC2_fitted_dose_response*.xlsx')[0])
    resp = resp[resp['DRUG_NAME'] == drug][['SANGER_MODEL_ID', 'CELL_LINE_NAME', 'LN_IC50']].dropna()
    resp = resp.groupby('SANGER_MODEL_ID', as_index=False).agg(
        CELL_LINE_NAME=('CELL_LINE_NAME', 'first'), LN_IC50=('LN_IC50', 'mean'))

    model = pd.read_csv(RAW + 'Model.csv', usecols=['ModelID', 'SangerModelID']).dropna()
    model = model.drop_duplicates('SangerModelID')

    expr = pd.read_csv(RAW + 'OmicsExpressionProteinCodingGenesTPMLogp1.csv')
    expr = expr.rename(columns={expr.columns[0]: 'ModelID'})
    expr = expr.drop_duplicates('ModelID').set_index('ModelID')
    expr = expr[[c for c in expr.columns if ' (' in str(c)]]          # gene columns only
    expr.columns = [str(c).split(' (')[0] for c in expr.columns]      # "TSPAN6 (7105)" -> "TSPAN6"
    if expr.columns.duplicated().any():                               # merge duplicate symbols
        expr = expr.T.groupby(level=0).mean().T

    df = resp.merge(model, left_on='SANGER_MODEL_ID', right_on='SangerModelID', how='inner')
    df = df.merge(expr, left_on='ModelID', right_index=True, how='inner')
    df = df.drop(columns=['ModelID', 'SangerModelID'])
    df = df.dropna(axis=1)                                            # drop genes with any missing value
    print('matched cell lines:', len(df), '| genes:', df.shape[1] - 3)
    return df.drop_duplicates(subset='SANGER_MODEL_ID')


def main():
    drug = CFG['drug_name']
    assert drug, 'Set drug_name in config.yaml first'
    df = load_merged(drug)
    meta_cols = ['SANGER_MODEL_ID', 'CELL_LINE_NAME', 'LN_IC50']
    genes = [c for c in df.columns if c not in meta_cols]
    G, y = df[genes].values.astype(float), df['LN_IC50'].values

    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=CFG['seed'])
    va, te = train_test_split(tmp, test_size=0.50, random_state=CFG['seed'])
    assert not (set(tr) & set(va) | set(tr) & set(te) | set(va) & set(te))

    # everything below uses TRAIN rows only
    Gtr = G[tr]
    top = np.argsort(Gtr.var(axis=0))[::-1][:2000]                    # variance prefilter
    scores, _ = f_regression(Gtr[:, top], y[tr])
    scores = np.nan_to_num(scores)

    chosen = []
    for j in np.argsort(scores)[::-1]:                                # best score first, skip near-duplicates
        if all(abs(np.corrcoef(Gtr[:, top[j]], Gtr[:, top[c]])[0, 1]) < 0.9 for c in chosen):
            chosen.append(j)
            if len(chosen) == K:
                break
    cols = top[chosen]

    scaler = MinMaxScaler().fit(Gtr[:, cols])

    def prep(i):
        return np.clip(scaler.transform(G[i][:, cols]), 0, 1)

    os.makedirs(OUT, exist_ok=True)
    for name, i in [('train', tr), ('val', va), ('test', te)]:
        np.save(OUT + 'X_' + name + '.npy', prep(i))
        np.save(OUT + 'y_' + name + '.npy', y[i])

    json.dump([genes[c] for c in cols], open(OUT + 'feature_names.json', 'w'))
    json.dump({'drug': drug, 'seed': CFG['seed'], 'n_total': int(len(df)),
               'n_train': int(len(tr)), 'n_val': int(len(va)), 'n_test': int(len(te)),
               'n_genes_available': len(genes), 'n_features': K,
               'y_train_mean': float(y[tr].mean()), 'y_train_std': float(y[tr].std())},
              open(OUT + 'meta.json', 'w'), indent=2)
    joblib.dump(scaler, OUT + 'scaler.joblib')
    df.iloc[te][['SANGER_MODEL_ID', 'CELL_LINE_NAME']].to_csv(OUT + 'test_cell_lines.csv', index=False)
    print('done:', len(tr), len(va), len(te), [genes[c] for c in cols])


if __name__ == '__main__':
    main()