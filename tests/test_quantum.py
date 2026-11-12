import numpy as np
import torch
from src.quantum.quantum_model import HybridQNN, count_parameters, load_model, predict

def test_output_range_and_shape():
    m = HybridQNN(4, 2)
    q = m.q(torch.rand(5, 4, dtype=torch.float64))
    assert q.shape == (5, 4)
    assert (q.abs() <= 1 + 1e-6).all()

def test_gradients_flow():
    m = HybridQNN(4, 2)
    m(torch.rand(3, 4, dtype=torch.float64)).sum().backward()
    assert m.q.qnode_weights['weights'].grad is not None

def test_parameter_count():
    assert count_parameters(HybridQNN(4, 3)) == 3 * 4 * 3 + 4 + 1 == 41

def test_save_load_predict_roundtrip(tmp_path):
    m = HybridQNN(4, 3)
    path = str(tmp_path / 'm.pt')
    torch.save({'state': m.state_dict(), 'mu': 1.5, 'sd': 2.0, 'n_qubits': 4, 'n_layers': 3, 'p_train': 0.0}, path)
    m2, mu, sd = load_model(path)
    X = np.random.rand(6, 8)
    p1, p2 = predict(m, X, mu, sd), predict(m2, X, mu, sd)
    assert p2.shape == (6,) and np.allclose(p1, p2)

def test_noisy_model_runs():
    m = HybridQNN(4, 2, p_dep=0.01, p_ro=0.01)
    out = m(torch.rand(2, 4, dtype=torch.float64))
    assert out.shape == (2,) and torch.isfinite(out).all()
