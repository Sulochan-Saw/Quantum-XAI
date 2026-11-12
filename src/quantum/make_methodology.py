"""Write the Quantum Hybrid Model methodology section: docs/methodology_quantum.md

    python -m src.quantum.make_methodology
Every number is read from files (config.yaml, data/processed/meta.json, results/tables/*.csv),
so the text always matches the saved results. Run it after training.
"""
import json
import os
import pandas as pd
from src.config import CFG


def ms(s, f='.2f'):
    return format(s.mean(), f) + ' ± ' + format(s.std(ddof=1), f) if len(s) > 1 else format(s.mean(), f)


def main():
    nq, nl = CFG['n_qubits'], CFG['n_layers']
    meta = json.load(open('data/processed/meta.json'))
    names = json.load(open('data/processed/feature_names.json'))
    n_circ = nl * nq * 3
    n_params = n_circ + nq + 1
    met = pd.read_csv('results/tables/quantum_metrics.csv')
    met = met[met['name'] == 'qnn_ideal'].sort_values('seed')
    seeds = ', '.join(str(s) for s in met['seed'])
    tt_path = 'results/tables/training_times.csv'
    time_txt = ''
    if os.path.exists(tt_path):
        tt = pd.read_csv(tt_path)
        tt = tt[tt['name'] == 'qnn_ideal'].sort_values('seed')
        if len(tt):
            time_txt = (' Training took ' + ms(tt['train_seconds'], '.0f') + ' s per seed (mean ± SD, '
                        + str(len(tt)) + ' seeds, CPU simulation); the selected checkpoint came from epochs '
                        + ', '.join(str(e) for e in tt['best_epoch']) + ' (seeds ' + ', '.join(str(s) for s in tt['seed'])
                        + ').')

    t = []
    t.append('# Quantum Hybrid Model: Methodology\n')
    t.append('## 1. Overview\n')
    t.append('We predict the natural logarithm of the half-maximal inhibitory concentration (LN_IC50) of '
             + str(meta['drug']) + ' from the gene-expression profile of a cancer cell line. The model is a '
             'hybrid quantum-classical network: a small variational quantum circuit, simulated classically, '
             'transforms a few gene-expression features, and a single linear layer maps the circuit '
             'measurements to the predicted LN_IC50. All results are from noise-free simulation; no quantum '
             'hardware was used.\n')
    t.append('## 2. Input features\n')
    t.append('Data are the GDSC2 fitted dose-response values joined with Cell Model Passports RNA-seq '
             '(' + meta.get('expression', 'TPM values') + '), giving ' + str(meta['n_total']) + ' cell lines. They were split once '
             'into ' + str(meta['n_train']) + ' training, ' + str(meta['n_val']) + ' validation and '
             + str(meta['n_test']) + ' test cell lines (70/15/15, seed ' + str(meta['seed']) + '). Feature '
             'selection (variance pre-filter, F-score ranking, correlation filter) and min-max scaling to [0, 1] '
             'were fitted on training rows only. Eight genes were selected; the circuit uses the first ' + str(nq)
             + ' in F-score order, one per qubit: ' + ', '.join(names[:nq]) + '. The remaining genes ('
             + ', '.join(names[nq:]) + ') are used by the classical baselines but not by the circuit.\n')
    t.append('## 3. Circuit\n')
    t.append('**Encoding.** Each scaled feature x_i in [0, 1] is loaded onto qubit i by a Y-rotation, '
             'RY(π x_i), starting from |0⟩. This gives the state U(x)|0⟩  with '
             'U(x) = ⊗_i RY(π x_i).\n')
    t.append('**Variational layers.** The encoded state passes through ' + str(nl) + ' StronglyEntanglingLayers '
             'blocks V(θ). Each block applies a general three-angle rotation to every qubit followed by a '
             'ring of CNOT gates that entangles neighbouring qubits, so the full state is '
             '|ψ(x, θ)⟩ = V(θ) U(x) |0⟩. The circuit has ' + str(nl) + ' × ' + str(nq)
             + ' × 3 = ' + str(n_circ) + ' trainable angles, drawn uniformly from [0, 2\u03c0) '
             '(the default of PennyLane\'s torch layer).\n')
    t.append('**Measurement.** For each qubit we compute the exact expectation value of the Pauli-Z operator, '
             'f_i = ⟨ψ| Z_i |ψ⟩, which lies in [−1, 1]. The ' + str(nq) + ' values form the '
             'quantum feature vector f.\n')
    t.append('**Classical head.** A linear layer gives ŷ = w·f + b (' + str(nq) + ' weights and 1 bias). '
             'Because f is bounded and LN_IC50 is not, targets are standardised with the training mean '
             f"({meta['y_train_mean']:.2f}) and standard deviation ({meta['y_train_std']:.2f}); predictions are "
             'transformed back to LN_IC50 units for evaluation.\n')
    t.append('**Model size.** ' + str(n_circ) + ' circuit parameters + ' + str(nq) + ' head weights + 1 bias = **'
             + str(n_params) + ' trainable parameters**. (The circuit is drawn in `results/figures/quantum_circuit.png`.)\n')
    t.append('## 4. Training\n')
    t.append('The mean squared error on standardised targets is minimised with Adam (learning rate '
             + str(CFG['lr']) + ', batch size ' + str(CFG['batch_size']) + ', ' + str(CFG['epochs']) + ' epochs). '
             'Gradients are computed by back-propagation through the state-vector simulator '
             '(PennyLane `default.qubit`, torch interface, 64-bit floats). After every epoch the loss on the '
             'validation set is computed and the parameters with the lowest validation loss are kept. The test '
             'set is used only once after training. Training was repeated for independent seeds ' + seeds
             + ' (these seeds set the parameter initialisation and the mini-batch order; the data split is fixed).'
             + time_txt + ' Settings were taken from `config.yaml` and were not tuned on the test set.\n')
    t.append('## 5. Evaluation\n')
    t.append('Predictions on the held-out test cell lines are scored with mean absolute error (MAE), root mean '
             'squared error (RMSE) and the coefficient of determination (R²), all in LN_IC50 units except R².\n')
    t.append('| Seed | MAE | RMSE | R² |\n|---|---|---|---|')
    for _, r in met.iterrows():
        t.append('| ' + str(int(r['seed'])) + ' | ' + format(r['MAE'], '.3f') + ' | ' + format(r['RMSE'], '.3f')
                 + ' | ' + format(r['R2'], '.3f') + ' |')
    t.append('| mean ± SD | ' + ms(met['MAE'], '.3f') + ' | ' + ms(met['RMSE'], '.3f') + ' | '
             + ms(met['R2'], '.3f') + ' |\n')
    t.append('Source: `results/tables/quantum_metrics.csv` (noise-free simulation, test set).\n')
    t.append('## 6. Limitations\n')
    t.append('- Simulator only: no hardware execution and, in this section, no noise. Noise is studied separately.\n'
             '- One drug (' + str(meta['drug']) + ') and one fixed split; seeds vary the training, not the data.\n'
             '- The circuit sees ' + str(nq) + ' of the 8 selected genes, so comparisons with classical models '
             'that use all 8 are not like-for-like unless a ' + str(nq) + '-gene baseline is also reported.\n'
             '- We make no claim of quantum advantage; any comparison with classical models rests on the paired '
             'statistical tests in the Results section.\n')
    os.makedirs('docs', exist_ok=True)
    open('docs/methodology_quantum.md', 'w').write('\n'.join(t))
    print('wrote docs/methodology_quantum.md')


if __name__ == '__main__':
    main()
