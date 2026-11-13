
import pandas as pd
import os


def main():

    # Load results from all seeds
    df = pd.read_csv("results/tables/classical_metrics.csv")

    # Calculate mean and standard deviation for each model
    summary = df.groupby("Model")[["MAE", "RMSE", "R2"]].agg(
        ["mean", "std"]
    )

    # Flatten column names
    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    summary = summary.reset_index()

    # Save summary
    os.makedirs("results/tables", exist_ok=True)

    summary.to_csv(
        "results/tables/classical_metrics_summary.csv",
        index=False
    )

    print("\nAverage results across 3 seeds:")
    print(summary.round(4).to_string(index=False))

    print("\nSaved to:")
    print("results/tables/classical_metrics_summary.csv")


if __name__ == "__main__":
    main()
