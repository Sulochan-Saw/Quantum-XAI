import numpy as np


def load_data(k=None):
    d = 'data/processed/'
    names = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']
    Xtr, Xva, Xte, ytr, yva, yte = [np.load(d + n + '.npy') for n in names]
    if k:
        Xtr, Xva, Xte = Xtr[:, :k], Xva[:, :k], Xte[:, :k]
    return Xtr, Xva, Xte, ytr, yva, yte