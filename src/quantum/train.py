import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from src.config import CFG
from src.data_io import load_data
from src.evaluation.metrics import regression_metrics
from src.quantum.quantum_model import HybridQNN, predict

def train(seed, p_train, name):
    torch.manual_seed(seed)
    np.random.seed(seed)
    nq = CFG['n_qubits']
    Xtr, Xva, Xte, ytr, yva, yte = load_data(nq)
    mu, sd = float(ytr.mean()), float(ytr.std())          # standardise target with TRAIN stats
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)
    Xtr_t, Xva_t = T(Xtr), T(Xva)
    ytr_t, yva_t = T((ytr - mu) / sd), T((yva - mu) / sd)

    model = HybridQNN(nq, CFG['n_layers'], p_dep=p_train, p_ro=p_train)
    opt = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
    loss_fn = nn.MSELoss()
    best_val, best_state, hist = float('inf'), None, []

    for epoch in range(CFG['epochs']):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        total = 0.0
        for i in range(0, len(perm), CFG['batch_size']):
            b = perm[i:i + CFG['batch_size']]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[b]), ytr_t[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xva_t), yva_t).item()
        hist.append({'epoch': epoch, 'train_loss': total / len(perm), 'val_loss': val})
        if val < best_val:
            best_val = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(epoch, round(total / len(perm), 4), round(val, 4))

    os.makedirs('models', exist_ok=True)
    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/predictions', exist_ok=True)
    tag = name + '_seed' + str(seed)
    torch.save({'state': best_state, 'mu': mu, 'sd': sd, 'n_qubits': nq,
                'n_layers': CFG['n_layers'], 'p_train': p_train}, 'models/' + tag + '.pt')
    pd.DataFrame(hist).to_csv('results/tables/loss_' + tag + '.csv', index=False)

    ideal = HybridQNN(nq, CFG['n_layers'])                 # evaluate on test in the IDEAL simulator
    ideal.load_state_dict(best_state)
    ideal.eval()
    pred = predict(ideal, Xte, mu, sd)
    np.save('results/predictions/' + tag + '_p0.0.npy', pred)
    row = {'name': name, 'seed': seed, 'p_train': p_train, **regression_metrics(yte, pred)}
    path = 'results/tables/quantum_metrics.csv'
    pd.DataFrame([row]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)
    print(row)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--p_train', type=float, default=0.0)
    ap.add_argument('--name', default='qnn_ideal')
    a = ap.parse_args()
    train(a.seed, a.p_train, a.name)
