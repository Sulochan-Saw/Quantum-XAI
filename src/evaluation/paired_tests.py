import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from src.data_io import load_data

P = 'results/predictions/'
_, _, _, _, _, yte = load_data()
quantum = np.load(P + 'qnn_ideal_seed0_p0.0.npy')
eq = np.abs(yte - quantum)
rng = np.random.default_rng(0)

rows = []
for name in ['Mean_Baseline', 'Linear_Regression', 'Random_Forest', 'XGBoost', 'MLP']:
    for tag, fname in [('8 genes', name + '_seed_0.npy'), ('4 genes', name + '_4f_seed_0.npy')]:
        if not os.path.exists(P + fname):
            continue
        ec = np.abs(yte - np.load(P + fname))
        d = eq - ec
        boots = [rng.choice(d, len(d), replace=True).mean() for _ in range(5000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({'classical': name + ' (' + tag + ')', 'mean_err_diff': d.mean(),
                     'ci_low': lo, 'ci_high': hi, 'p_wilcoxon': wilcoxon(eq, ec).pvalue})
df = pd.DataFrame(rows)
df['p_holm'] = multipletests(df['p_wilcoxon'], method='holm')[1]
df.to_csv('results/tables/paired_tests_quantum_vs_classical.csv', index=False)
print(df.round(4).to_string(index=False))