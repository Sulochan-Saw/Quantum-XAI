import math
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

# MODEL = quantum circuit (4 qubits) + one small linear layer.
# Input: 4 gene values. Output: predicted drug response.


def make_qlayer(n_qubits=4, n_layers=3, p_dep=0.0, p_ro=0.0, shots=None):
    # Builds the quantum circuit. p_dep, p_ro and shots are for the noise study (Member 5).
    noisy = (p_dep > 0) or (p_ro > 0)
    dev = qml.device('default.mixed' if noisy else 'default.qubit', wires=n_qubits, shots=shots)
    diff = 'parameter-shift' if shots else 'backprop'

    @qml.qnode(dev, interface='torch', diff_method=diff)
    def circuit(inputs, weights):
        # 1. Put each gene value onto one qubit
        qml.AngleEmbedding(inputs * math.pi, wires=range(n_qubits), rotation='Y')
        if p_dep > 0:
            for w in range(n_qubits):
                qml.DepolarizingChannel(p_dep, wires=w)
        # 2. Learnable layers (the part that is trained)
        for l in range(n_layers):
            qml.StronglyEntanglingLayers(weights[l:l + 1], wires=range(n_qubits))
            if p_dep > 0:
                for w in range(n_qubits):
                    qml.DepolarizingChannel(p_dep, wires=w)
        if p_ro > 0:
            for w in range(n_qubits):
                qml.BitFlip(p_ro, wires=w)               # simple readout-error model
        # 3. Measure every qubit: gives 4 numbers
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    return qml.qnn.TorchLayer(circuit, {'weights': (n_layers, n_qubits, 3)})


class HybridQNN(nn.Module):
    def __init__(self, n_qubits=4, n_layers=3, p_dep=0.0, p_ro=0.0, shots=None):
        super().__init__()
        self.n_qubits = n_qubits
        self.q = make_qlayer(n_qubits, n_layers, p_dep, p_ro, shots)   # quantum part
        self.head = nn.Linear(n_qubits, 1)                             # 4 numbers -> 1 prediction
        self.double()

    def forward(self, x):
        return self.head(self.q(x.double())).squeeze(-1)


def load_model(path, p_dep=0.0, p_ro=0.0, shots=None):
    # Loads a saved .pt file (made by train.py)
    ck = torch.load(path, weights_only=False)
    m = HybridQNN(ck['n_qubits'], ck['n_layers'], p_dep, p_ro, shots)
    m.load_state_dict(ck['state'])
    m.eval()
    return m, ck['mu'], ck['sd']


def predict(model, X, mu, sd):
    # Predicts with the first 4 genes only, then converts back to real units
    X = np.asarray(X)[:, :model.n_qubits]
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float64)).numpy() * sd + mu


def count_parameters(model):
    # Number of trainable values: 3 x 4 x 3 + 4 + 1 = 41
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
