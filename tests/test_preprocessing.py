import numpy as np

def test_shapes_and_range():
    for s in ['train', 'val', 'test']:
        X = np.load('data/processed/X_' + s + '.npy')
        y = np.load('data/processed/y_' + s + '.npy')
        assert X.shape[0] == y.shape[0] and X.shape[1] >= 4
        assert not np.isnan(X).any() and not np.isnan(y).any()
        assert X.min() >= 0 and X.max() <= 1