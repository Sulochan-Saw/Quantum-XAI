
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_predictions(y_test, predictions, model_name):

    os.makedirs("results/figures", exist_ok=True)

    plt.figure(figsize=(8, 6))

    plt.scatter(y_test, predictions, alpha=0.7)

    min_value = min(y_test.min(), predictions.min())
    max_value = max(y_test.max(), predictions.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        "r--"
    )

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs Predicted - {model_name}")

    plt.tight_layout()

    filename = model_name.replace(" ", "_")

    plt.savefig(
        f"results/figures/{filename}_actual_vs_pred.png"
    )

    plt.close()


if __name__ == "__main__":

    y_test = np.load("data/processed/y_test.npy")

    for model_name in [
        "Mean_Baseline",
        "Linear_Regression",
        "Random_Forest",
        "XGBoost",
        "MLP"
    ]:

        predictions = np.load(
            f"results/predictions/{model_name}_seed_0.npy"
        )

        plot_predictions(
            y_test,
            predictions,
            model_name
        )

    print("All plots generated successfully!")
