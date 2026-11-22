import numpy as np
import pandas as pd
from src.config import CFG
from src.data_io import load_data
from src.evaluation.metrics import regression_metrics
from src.quantum.quantum_model import load_model, predict

NQ = CFG['n_qubits']
_, _, Xte, _, _, yte = load_data(NQ)       # raw features, no extra scaling
LEVELS = CFG['noise_levels']
NAMES = ['qnn_ideal', 'qnn_noise_aware']   # must match the file names in models/
TYPES = {'depolarizing': (1, 0), 'readout': (0, 1), 'combined': (1, 1)}

rows = []
for name in NAMES:
    for seed in CFG['train_seeds']:
        path = 'models/' + name + '_seed' + str(seed) + '.pt'
        for ntype, (a, b) in TYPES.items():
            for p in LEVELS:
                m, mu, sd = load_model(path, p_dep=p * a, p_ro=p * b)
                pred = predict(m, Xte, mu, sd)
                rows.append({'model': name, 'seed': seed, 'noise_type': ntype, 'p': p,
                             **regression_metrics(yte, pred)})
                print(rows[-1])
df = pd.DataFrame(rows)
df.to_csv('results/tables/noise_accuracy_all.csv', index=False)
g = df.groupby(['model', 'noise_type', 'p'])[['MAE', 'RMSE', 'R2']].agg(['mean', 'std'])
g.to_csv('results/tables/noise_accuracy_summary.csv')
print(g.round(4))

stab = []                                  # shot-based variability, seed 0, combined noise
for name in NAMES:
    for p in LEVELS:
        m, mu, sd = load_model('models/' + name + '_seed0.pt', p_dep=p, p_ro=p, shots=CFG['shots_eval'])
        runs = np.stack([predict(m, Xte[:50], mu, sd) for _ in range(CFG['noise_repeats'])])
        stab.append({'model': name, 'p': p, 'pred_std_over_shots': float(runs.std(axis=0).mean())})
        print(stab[-1])
pd.DataFrame(stab).to_csv('results/tables/noise_shot_stability.csv', index=False)