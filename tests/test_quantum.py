import torch
from src.quantum.quantum_model import HybridQNN

def test_output_range_and_shape():
    m = HybridQNN(4, 2)
    q = m.q(torch.rand(5, 4, dtype=torch.float64))
    assert q.shape == (5, 4)
    assert (q.abs() <= 1 + 1e-6).all()

def test_gradients_flow():
    m = HybridQNN(4, 2)
    m(torch.rand(3, 4, dtype=torch.float64)).sum().backward()
    assert m.q.qnode_weights['weights'].grad is not None
