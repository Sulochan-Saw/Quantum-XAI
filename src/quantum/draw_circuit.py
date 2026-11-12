"""Save the circuit diagram: results/figures/quantum_circuit.png

    python -m src.quantum.draw_circuit
Also writes results/figures/quantum_circuit.txt (text drawing) and, if this PennyLane version
supports it, quantum_circuit_expanded.png (every RY / rotation / CNOT gate drawn out).
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pennylane as qml
import torch
from src.config import CFG
from src.quantum.quantum_model import HybridQNN, count_parameters

OUT = 'results/figures/'


def main():
    os.makedirs(OUT, exist_ok=True)
    nq, nl = CFG['n_qubits'], CFG['n_layers']
    m = HybridQNN(nq, nl)
    x = torch.full((nq,), 0.5, dtype=torch.float64)
    w = m.q.qnode_weights['weights'].detach()
    title = ('Hybrid circuit: ' + str(nq) + ' qubits, RY(pi*x) angle embedding, ' + str(nl)
             + ' StronglyEntanglingLayers, <Z> on every qubit -> linear head ('
             + str(count_parameters(m)) + ' trainable parameters)')

    txt = qml.draw(m.q.qnode)(x, w)
    open(OUT + 'quantum_circuit.txt', 'w').write(title + '\n\n' + txt + '\n')

    def save(name, **kw):
        fig, _ = qml.draw_mpl(m.q.qnode, decimals=None, **kw)(x, w)
        fig.suptitle(title, fontsize=8)
        fig.savefig(OUT + name, dpi=200, bbox_inches='tight')
        plt.close(fig)

    try:
        save('quantum_circuit.png')
    except Exception as e:                      # fallback: render the text drawing as an image
        print('draw_mpl failed (' + repr(e) + '); using the text drawing instead')
        fig = plt.figure(figsize=(10, 3))
        fig.text(0.01, 0.5, txt, family='monospace', fontsize=7, va='center')
        fig.suptitle(title, fontsize=8)
        fig.savefig(OUT + 'quantum_circuit.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
    try:
        save('quantum_circuit_expanded.png', level='device')
    except Exception as e:
        print('expanded drawing skipped:', repr(e))
    print(txt)
    print('saved', OUT + 'quantum_circuit.png')


if __name__ == '__main__':
    main()
