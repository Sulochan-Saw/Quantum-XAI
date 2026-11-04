import glob
import pandas as pd

resp = pd.read_excel(glob.glob('data/raw/GDSC2*.xlsx')[0])
model = pd.read_csv('data/raw/Model.csv', usecols=['ModelID', 'SangerModelID']).dropna()
expr_ids = pd.read_csv('data/raw/OmicsExpressionProteinCodingGenesTPMLogp1.csv', usecols=[0]).iloc[:, 0]
print('expression ids sample:', expr_ids.head(3).tolist(), '| rows:', len(expr_ids))

ok = set(model[model['ModelID'].isin(expr_ids)]['SangerModelID'])
resp = resp[resp['SANGER_MODEL_ID'].isin(ok)]
stats = resp.groupby('DRUG_NAME').agg(n_cells=('SANGER_MODEL_ID', 'nunique'),
                                      std_ic50=('LN_IC50', 'std'))
print(stats.sort_values('n_cells', ascending=False).head(20))