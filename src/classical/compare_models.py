
import pandas as pd
import matplotlib.pyplot as plt
import os


def main():

    # Load averaged metrics
    df = pd.read_csv(
        "results/tables/classical_metrics_summary.csv"
    )

    os.makedirs("results/figures", exist_ok=True)

    metrics = ["MAE_mean", "RMSE_mean", "R2_mean"]

    titles = [
        "MAE Comparison (Lower is Better)",
        "RMSE Comparison (Lower is Better)",
        "R2 Comparison (Higher is Better)"
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, metric, title in zip(axes, metrics, titles):

        ax.bar(df["Model"], df[metric])

        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=45)

        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "results/figures/classical_model_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Comparison visualization saved successfully!")
    print("File: results/figures/classical_model_comparison.png")


if __name__ == "__main__":
    main()
