
import os
import numpy as np
import pandas as pd

from src.classical.models import get_models
from src.classical.evaluate import evaluate_model


def get_parameter_grids():
    return {
        "Mean Baseline": [{}],

        "Linear Regression": [{}],

        "Random Forest": [
            {"max_depth": depth, "min_samples_leaf": leaf}
            for depth in [None, 6]
            for leaf in [1, 5]
        ],

        "XGBoost": [
            {
                "n_estimators": n,
                "max_depth": depth,
                "learning_rate": lr
            }
            for n in [200, 400]
            for depth in [3, 5]
            for lr in [0.03, 0.1]
        ],

        "MLP": [
            {
                "hidden_layer_sizes": layers,
                "alpha": alpha
            }
            for layers in [(16,), (32, 16)]
            for alpha in [1e-4, 1e-2]
        ]
    }


def main():

    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")

    X_val = np.load("data/processed/X_val.npy")
    y_val = np.load("data/processed/y_val.npy")

    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/predictions", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)

    parameter_grids = get_parameter_grids()
    results = []

    for seed in [0, 1, 2]:

        print(f"\n========== SEED {seed} ==========")

        for name, grid in parameter_grids.items():

            best_params = None
            best_rmse = float("inf")

            print(f"\nTuning {name}...")

            for params in grid:

                model = get_models(seed)[name]

                model.set_params(**params)
                model.fit(X_train, y_train)

                val_predictions = model.predict(X_val)

                from sklearn.metrics import mean_squared_error

                val_rmse = np.sqrt(
                    mean_squared_error(y_val, val_predictions)
                )

                print(f"Params: {params} | Validation RMSE: {val_rmse:.4f}")

                if val_rmse < best_rmse:
                    best_rmse = val_rmse
                    best_params = params

            print(f"Best parameters: {best_params}")

            # Train final model using training data only
            final_model = get_models(seed)[name]
            final_model.set_params(**best_params)
            final_model.fit(X_train, y_train)

            # Evaluate final model on test data
            metrics = evaluate_model(final_model, X_test, y_test)

            print(f"Test metrics: {metrics}")

            predictions = final_model.predict(X_test)

            filename = name.replace(" ", "_")

            np.save(
                f"results/predictions/{filename}_seed_{seed}.npy",
                predictions
            )

            import joblib

            joblib.dump(
                final_model,
                f"results/models/{filename}_seed_{seed}.joblib"
            )

            results.append({
                "Seed": seed,
                "Model": name,
                "Best_Params": str(best_params),
                "Validation_RMSE": best_rmse,
                **metrics
            })

    df = pd.DataFrame(results)

    df.to_csv(
        "results/tables/classical_metrics.csv",
        index=False
    )

    print("\nAll models completed!")
    print("Metrics saved to results/tables/classical_metrics.csv")
    print("Models saved to results/models/")
    print("Predictions saved to results/predictions/")


if __name__ == "__main__":
    main()
