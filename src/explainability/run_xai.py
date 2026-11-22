import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from lime.lime_tabular import LimeTabularExplainer
from src.config import CFG
from src.data_io import load_data
from src.quantum.quantum_model import load_model, predict

NQ = CFG['n_qubits']
NAMES = json.load(open('data/processed/feature_names.json'))[:NQ]
Xtr, _, Xte, _, _, _ = load_data(NQ)
XE = Xte[:30]                      # test cell lines to explain
BG = shap.kmeans(Xtr, 10)          # background summary for SHAP
LIME_N, LIME_S = 5, 1000           # LIME: 5 cell lines, 1000 samples each
IDEAL, NA = 'models/qnn_ideal_seed', 'models/qnn_noise_aware_seed'
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)


def lime_explainer(seed):
    return LimeTabularExplainer(Xtr, feature_names=NAMES, mode='regression', random_state=seed)


def explain(path, p, lime_seed=0):
    m, mu, sd = load_model(path, p_dep=p, p_ro=p)
    f = lambda X: predict(m, X, mu, sd)
    ex = shap.KernelExplainer(f, BG)
    sv = np.array(ex.shap_values(XE, nsamples=2 ** NQ, silent=True))   # exact enumeration for 4 features
    le = lime_explainer(lime_seed)
    li = np.zeros(NQ)
    for i in range(LIME_N):
        e = le.explain_instance(XE[i], f, num_features=NQ, num_samples=LIME_S)
        for j, w in e.as_map()[1]:
            li[j] += abs(w) / LIME_N
    return sv, float(np.ravel(ex.expected_value)[0]), li, f, le


def top_overlap(a, b, k=2):
    return len(set(np.argsort(a)[::-1][:k]) & set(np.argsort(b)[::-1][:k])) / k


def compare(sv0, lime0, sv, lime):
    g0, g = np.abs(sv0).mean(0), np.abs(sv).mean(0)
    per = [spearmanr(np.abs(sv0[i]), np.abs(sv[i])).correlation for i in range(len(sv))]
    return {'shap_rho_global': spearmanr(g0, g).correlation,
            'shap_top2_overlap': top_overlap(g0, g),
            'shap_rho_per_sample': float(np.nanmean(per)),
            'lime_rho': spearmanr(lime0, lime).correlation}


sv0, ev0, lime0, f0, le0 = explain(IDEAL + '0.pt', 0.0)

shap.summary_plot(sv0, XE, feature_names=NAMES, show=False)
plt.savefig('results/figures/shap_summary_ideal.png', dpi=200, bbox_inches='tight')
plt.close()
shap.summary_plot(sv0, XE, feature_names=NAMES, plot_type='bar', show=False)
plt.savefig('results/figures/shap_importance_ideal.png', dpi=200, bbox_inches='tight')
plt.close()
shap.plots.waterfall(shap.Explanation(values=sv0[0], base_values=ev0, data=XE[0], feature_names=NAMES), show=False)
plt.savefig('results/figures/shap_waterfall_sample0.png', dpi=200, bbox_inches='tight')
plt.close()
fig = le0.explain_instance(XE[0], f0, num_features=NQ, num_samples=LIME_S).as_pyplot_figure()
fig.savefig('results/figures/lime_sample0.png', dpi=200, bbox_inches='tight')
plt.close('all')

pd.DataFrame({'gene': NAMES, 'mean_abs_shap': np.abs(sv0).mean(0), 'lime_mean_abs_weight': lime0}
             ).to_csv('results/tables/xai_importance_ideal.csv', index=False)
print(pd.read_csv('results/tables/xai_importance_ideal.csv'))

rows = []
for s in CFG['train_seeds'][1:]:           # control: other training seeds, no noise
    sv, _, li, _, _ = explain(IDEAL + str(s) + '.pt', 0.0)
    rows.append({'comparison': 'other_training_seed', 'model': 'qnn_ideal seed ' + str(s), 'p': 0.0,
                 **compare(sv0, lime0, sv, li)})
sv, _, li, _, _ = explain(IDEAL + '0.pt', 0.0, lime_seed=1)    # control: LIME resampling only
rows.append({'comparison': 'lime_resample', 'model': 'qnn_ideal seed 0', 'p': 0.0, **compare(sv0, lime0, sv, li)})
for name, prefix in [('qnn_ideal', IDEAL), ('qnn_noise_aware', NA)]:
    for p in CFG['noise_levels']:
        sv, _, li, _, _ = explain(prefix + '0.pt', p)
        rows.append({'comparison': 'vs_ideal_seed0', 'model': name, 'p': p, **compare(sv0, lime0, sv, li)})
        print(rows[-1])
df = pd.DataFrame(rows)
df.to_csv('results/tables/xai_stability.csv', index=False)

d = df[df['comparison'] == 'vs_ideal_seed0']
plt.figure(figsize=(6, 4))
for name, sub in d.groupby('model'):
    plt.plot(sub['p'], sub['shap_rho_global'], marker='o', label=name)
plt.xlabel('noise probability p')
plt.ylabel('Spearman rho of SHAP importance vs ideal')
plt.legend()
plt.tight_layout()
plt.savefig('results/figures/xai_stability_vs_p.png', dpi=200)
print(df.round(3).to_string(index=False))