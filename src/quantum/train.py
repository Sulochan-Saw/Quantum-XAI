"""Train the hybrid QNN for one seed and write models, tables and log rows.

Usage: python -m src.quantum.train --seed 0 --p_train 0.0 --name qnn_ideal
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from src.config import CFG
from src.data_io import load_data
from src.evaluation.metrics import regression_metrics
from src.quantum.experiment_log import log_run
from src.quantum.quantum_model import HybridQNN, count_parameters, predict


def upsert_row(path, row, keys=('name', 'seed')):
    """Write one row to a CSV, replacing any earlier row with the same keys (re-running never duplicates)."""
    new = pd.DataFrame([row])
    if os.path.exists(path):
        old = pd.read_csv(path)
        same = pd.Series(True, index=old.index)
        for k in keys:
            same &= old[k] == row[k]
        new = pd.concat([old[~same], new], ignore_index=True)
    new.to_csv(path, index=False)


def fit(model, Xtr_t, ytr_t, Xva_t, yva_t, verbose=True):
    """Train with Adam + MSE. Keeps the checkpoint with the lowest VALIDATION loss.
    Returns (best_state, history, best_epoch, seconds)."""
    opt = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
    loss_fn = nn.MSELoss()
    best_val, best_state, best_epoch, hist = float('inf'), None, -1, []
    t0 = time.perf_counter()
    for epoch in range(CFG['epochs']):
        model.train()
        perm = torch.randperm(len(Xtr_t))              # reshuffle every epoch
        total = 0.0
        for i in range(0, len(perm), CFG['batch_size']):
            b = perm[i:i + CFG['batch_size']]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[b]), ytr_t[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)              # weight by batch size (last batch is smaller)
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xva_t), yva_t).item()
        hist.append({'epoch': epoch, 'train_loss': total / len(perm), 'val_loss': val})
        if val < best_val:                             # model selection on validation only, never on test
            best_val, best_epoch = val, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}   # clone: state_dict tensors are live
        if verbose:
            print(epoch, round(total / len(perm), 4), round(val, 4))
    return best_state, hist, best_epoch, time.perf_counter() - t0


def train(seed, p_train, name):
    torch.manual_seed(seed)                            # seeds torch and numpy for reproducible runs
    np.random.seed(seed)
    nq = CFG['n_qubits']
    Xtr, Xva, Xte, ytr, yva, yte = load_data(nq)       # first nq features, one per qubit
    mu, sd = float(ytr.mean()), float(ytr.std())          # standardise target with TRAIN stats
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)   # simulator needs float64
    Xtr_t, Xva_t = T(Xtr), T(Xva)
    ytr_t, yva_t = T((ytr - mu) / sd), T((yva - mu) / sd)

    model = HybridQNN(nq, CFG['n_layers'], p_dep=p_train, p_ro=p_train)
    n_params = count_parameters(model)
    best_state, hist, best_epoch, seconds = fit(model, Xtr_t, ytr_t, Xva_t, yva_t)
    best_val = hist[best_epoch]['val_loss']

    os.makedirs('models', exist_ok=True)
    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/predictions', exist_ok=True)
    tag = name + '_seed' + str(seed)
    # Checkpoint format is read by load_model(); mu/sd are needed to un-standardise predictions.
    torch.save({'state': best_state, 'mu': mu, 'sd': sd, 'n_qubits': nq,
                'n_layers': CFG['n_layers'], 'p_train': p_train}, 'models/' + tag + '.pt')
    pd.DataFrame(hist).to_csv('results/tables/loss_' + tag + '.csv', index=False)

    ideal = HybridQNN(nq, CFG['n_layers'])                 # evaluate on test in the IDEAL simulator
    ideal.load_state_dict(best_state)
    ideal.eval()
    pred = predict(ideal, Xte, mu, sd)                 # test set is touched once, after model selection
    np.save('results/predictions/' + tag + '_p0.0.npy', pred)
    row = {'name': name, 'seed': seed, 'p_train': p_train, **regression_metrics(yte, pred)}
    upsert_row('results/tables/quantum_metrics.csv', row)

    # training time and size of the model, for the paper (results/tables/training_times.csv)
    upsert_row('results/tables/training_times.csv', {
        'name': name, 'seed': seed, 'p_train': p_train, 'epochs': CFG['epochs'],
        'batch_size': CFG['batch_size'], 'lr': CFG['lr'], 'best_epoch': best_epoch,
        'best_val_loss': best_val, 'train_seconds': round(seconds, 1), 'n_params': n_params})
    # E2 = ideal model without noise (handbook experiment table)
    log_run('E2' if p_train == 0 else 'E4',
            'python -m src.quantum.train --seed ' + str(seed) + ' --p_train ' + str(p_train) + ' --name ' + name,
            seed, 'models/' + tag + '.pt; results/tables/quantum_metrics.csv',
            'best epoch ' + str(best_epoch) + ', ' + format(seconds, '.1f') + ' s (' + str(os.cpu_count())
            + ' CPUs), ' + str(n_params) + ' params; test MAE ' + format(row['MAE'], '.3f') + ', RMSE '
            + format(row['RMSE'], '.3f') + ', R2 ' + format(row['R2'], '.3f'))
    print('trainable parameters', n_params, '| training time', round(seconds, 1), 's | best epoch', best_epoch)
    print(row)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--p_train', type=float, default=0.0)
    ap.add_argument('--name', default='qnn_ideal')
    a = ap.parse_args()
    train(a.seed, a.p_train, a.name)
