import json
import os
import numpy as np

rng = np.random.default_rng(42)
n, k = 600, 8
X = rng.random((n, k))
y = 3 * X[:, 0] - 2 * X[:, 1] ** 2 + np.sin(3 * X[:, 2]) + 0.5 * X[:, 3] + rng.normal(0, 0.3, n) + 2
idx = rng.permutation(n)
splits = {'train': idx[:420], 'val': idx[420:510], 'test': idx[510:]}
os.makedirs('data/processed', exist_ok=True)
for name, ids in splits.items():
    np.save('data/processed/X_' + name + '.npy', X[ids])
    np.save('data/processed/y_' + name + '.npy', y[ids])
json.dump(['GENE_' + str(i) for i in range(k)], open('data/processed/feature_names.json', 'w'))
json.dump({'drug': 'SYNTHETIC', 'seed': 42, 'n_features': k}, open('data/processed/meta.json', 'w'))