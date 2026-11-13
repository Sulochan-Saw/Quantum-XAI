
import numpy as np
from src.classical.models import get_models


def train_models(X_train, y_train, seed=0):
    """
    Train all classical models and return trained models.
    """

    models = get_models(seed)

    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")

        model.fit(X_train, y_train)

        trained_models[name] = model

    return trained_models


if __name__ == "__main__":
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")

    trained_models = train_models(X_train, y_train)

    print("\nAll models trained successfully!")
