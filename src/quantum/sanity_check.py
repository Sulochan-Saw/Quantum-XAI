"""Sanity check on SYNTHETIC data (debugging only; never goes in a report or slide).

Generates the same synthetic data as scripts/make_synthetic.py IN MEMORY (it does not touch
data/processed), trains the hybrid model with the settings in config.yaml, and compares it with
Linear Regression on the same 4 features and the same test rows.

    python -m src.quantum.sanity_check            # seed 0

What a healthy result looks like (handbook 9.4): validation loss falls clearly below its starting
value within 20 to 30 epochs, and the hybrid model is in the same range as Linear Regression
(the synthetic target has non-linear terms, so it can also beat it).
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from src.config import CFG
from src.evaluation.metrics import regression_metrics
from src.quantum.experiment_log import log_run
from src.quantum.quantum_model import HybridQNN, count_parameters, predict
from src.quantum.train import fit, upsert_row


def make_synthetic(nq):
    rng = np.random.default_rng(42)                       # same formula as scripts/make_synthetic.py
    n, k = 600, 8
    X = rng.random((n, k))
    y = 3 * X[:, 0] - 2 * X[:, 1] ** 2 + np.sin(3 * X[:, 2]) + 0.5 * X[:, 3] + rng.normal(0, 0.3, n) + 2
    idx = rng.permutation(n)
    tr, va, te = idx[:420], idx[420:510], idx[510:]
    return X[tr][:, :nq], X[va][:, :nq], X[te][:, :nq], y[tr], y[va], y[te]


def main(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    nq = CFG['n_qubits']
    Xtr, Xva, Xte, ytr, yva, yte = make_synthetic(nq)
    mu, sd = float(ytr.mean()), float(ytr.std())
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)

    model = HybridQNN(nq, CFG['n_layers'])
    best_state, hist, best_epoch, seconds = fit(model, T(Xtr), T((ytr - mu) / sd), T(Xva), T((yva - mu) / sd),
                                                verbose=False)
    model.load_state_dict(best_state)
    model.eval()
    pred_q = predict(model, Xte, mu, sd)
    lin = LinearRegression().fit(Xtr, ytr)
    pred_l = lin.predict(Xte)

    os.makedirs('results/tables', exist_ok=True)
    pd.DataFrame(hist).to_csv('results/tables/loss_sanity_synthetic_seed' + str(seed) + '.csv', index=False)
    rows = [{'model': 'HybridQNN', 'seed': seed, **regression_metrics(yte, pred_q)},
            {'model': 'LinearRegression', 'seed': seed, **regression_metrics(yte, pred_l)}]
    for r in rows:
        upsert_row('results/tables/sanity_synthetic.csv', r, keys=('model', 'seed'))

    v0 = hist[0]['val_loss']
    v30 = hist[min(29, len(hist) - 1)]['val_loss']
    drop = 100 * (1 - v30 / v0)
    verdict = 'PASS' if v30 < 0.8 * v0 else 'CHECK'
    q, l = rows[0], rows[1]
    text = ('SANITY CHECK on SYNTHETIC data (debug only), seed ' + str(seed) + '\n'
            + '- Validation loss: epoch 0 = ' + format(v0, '.3f') + ', epoch 30 = ' + format(v30, '.3f')
            + ' (' + format(drop, '.0f') + '% lower), best epoch ' + str(best_epoch) + ' -> ' + verdict + '\n'
            + '- Test, same 4 features, same rows: HybridQNN R2 ' + format(q['R2'], '.3f') + ', RMSE '
            + format(q['RMSE'], '.3f') + ' | LinearRegression R2 ' + format(l['R2'], '.3f') + ', RMSE '
            + format(l['RMSE'], '.3f') + '\n'
            + '- Hybrid model has ' + str(count_parameters(model)) + ' trainable parameters; training '
            + format(seconds, '.1f') + ' s. Synthetic results are not reported in the paper.')
    log_run('SANITY-SYN', 'python -m src.quantum.sanity_check --seed ' + str(seed), seed,
            'results/tables/sanity_synthetic.csv',
            'synthetic data, debug only: val loss ' + format(v0, '.3f') + ' -> ' + format(v30, '.3f') + ' at epoch 30 ('
            + verdict + '); hybrid R2 ' + format(q['R2'], '.3f') + ' vs LinearRegression ' + format(l['R2'], '.3f'))
    print(text)
    print('\nsaved results/tables/sanity_synthetic.csv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args().seed)
