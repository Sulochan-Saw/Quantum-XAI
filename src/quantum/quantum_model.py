import math
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

def make_qlayer(n_qubits=4, n_layers=3, p_dep=0.0, p_ro=0.0, shots=None):
    noisy = (p_dep > 0) or (p_ro > 0)
    dev = qml.device('default.mixed' if noisy else 'default.qubit', wires=n_qubits, shots=shots)
    diff = 'parameter-shift' if shots else 'backprop'

    @qml.qnode(dev, interface='torch', diff_method=diff)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs * math.pi, wires=range(n_qubits), rotation='Y')
        if p_dep > 0:
            for w in range(n_qubits):
                qml.DepolarizingChannel(p_dep, wires=w)
        for l in range(n_layers):
            qml.StronglyEntanglingLayers(weights[l:l + 1], wires=range(n_qubits))
            if p_dep > 0:
                for w in range(n_qubits):
                    qml.DepolarizingChannel(p_dep, wires=w)
        if p_ro > 0:
            for w in range(n_qubits):
                qml.BitFlip(p_ro, wires=w)               # simple readout-error model
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    return qml.qnn.TorchLayer(circuit, {'weights': (n_layers, n_qubits, 3)})

class HybridQNN(nn.Module):
    def __init__(self, n_qubits=4, n_layers=3, p_dep=0.0, p_ro=0.0, shots=None):
        super().__init__()
        self.n_qubits = n_qubits
        self.q = make_qlayer(n_qubits, n_layers, p_dep, p_ro, shots)
        self.head = nn.Linear(n_qubits, 1)
        self.double()

    def forward(self, x):
        return self.head(self.q(x.double())).squeeze(-1)

def load_model(path, p_dep=0.0, p_ro=0.0, shots=None):
    ck = torch.load(path, weights_only=False)
    m = HybridQNN(ck['n_qubits'], ck['n_layers'], p_dep, p_ro, shots)
    m.load_state_dict(ck['state'])
    m.eval()
    return m, ck['mu'], ck['sd']

def predict(model, X, mu, sd):
    X = np.asarray(X)[:, :model.n_qubits]
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float64)).numpy() * sd + mu
