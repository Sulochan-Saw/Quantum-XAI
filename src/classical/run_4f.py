import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from src.config import CFG
from src.data_io import load_data
from src.evaluation.metrics import regression_metrics

K = CFG['n_qubits']
Xtr, Xva, Xte, ytr, yva, yte = load_data(K)

MODELS = {
    'Mean_Baseline': (lambda s, **kw: DummyRegressor(strategy='mean'), [{}]),
    'Linear_Regression': (lambda s, **kw: LinearRegression(), [{}]),
    'Random_Forest': (lambda s, **kw: RandomForestRegressor(random_state=s, n_jobs=-1, **kw),
                      [{'n_estimators': 300, 'max_depth': d, 'min_samples_leaf': l}
                       for d in (None, 6) for l in (1, 5)]),
    'XGBoost': (lambda s, **kw: XGBRegressor(random_state=s, **kw),
                [{'n_estimators': n, 'max_depth': d, 'learning_rate': lr}
                 for n in (200, 400) for d in (3, 5) for lr in (0.03, 0.1)]),
    'MLP': (lambda s, **kw: MLPRegressor(random_state=s, max_iter=2000, **kw),
            [{'hidden_layer_sizes': h, 'alpha': a}
             for h in ((16,), (32, 16)) for a in (1e-4, 1e-2)]),
}

rows = []
for name, (make, grid) in MODELS.items():
    for seed in CFG['train_seeds']:
        best, best_rmse = None, float('inf')
        for params in grid:
            m = make(seed, **params).fit(Xtr, ytr)
            rmse = regression_metrics(yva, m.predict(Xva))['RMSE']
            if rmse < best_rmse:
                best, best_rmse = params, rmse
        pred = make(seed, **best).fit(Xtr, ytr).predict(Xte)
        np.save('results/predictions/' + name + '_4f_seed_' + str(seed) + '.npy', pred)
        rows.append({'model': name, 'seed': seed, **regression_metrics(yte, pred)})
df = pd.DataFrame(rows)
df.to_csv('results/tables/classical_metrics_4f.csv', index=False)
print(df.groupby('model')[['MAE', 'RMSE', 'R2']].agg(['mean', 'std']).round(3))