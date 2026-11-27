from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CASES = {
    "stable_0": Path(
        "results/aneug_flow/stable_0_wss_metrics.csv"
    ),
    "stable_1": Path(
        "results/aneug_flow/stable_1/stable_1_wss_metrics.csv"
    ),
}

OUTPUT_DIR = Path("results/aneug_flow/comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["TAWSS", "OSI", "RRT"]

datasets = {}

for case, path in CASES.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    df = pd.read_csv(path)

    for metric in METRICS:
        if metric not in df.columns:
            raise ValueError(f"{metric} missing in {case}")

        if not df[metric].notna().all():
            raise ValueError(f"Invalid values in {case}: {metric}")

    datasets[case] = df
    print(f"{case}: {len(df)} nodes loaded")


# ==========================================
# Descriptive statistics
# ==========================================

rows = []

for case, df in datasets.items():
    for metric in METRICS:
        values = df[metric]

        rows.append({
            "case": case,
            "metric": metric,
            "nodes": len(values),
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "p05": values.quantile(0.05),
            "p25": values.quantile(0.25),
            "median": values.median(),
            "p75": values.quantile(0.75),
            "p95": values.quantile(0.95),
            "max": values.max(),
        })

summary = pd.DataFrame(rows)

summary_path = OUTPUT_DIR / "hemodynamic_comparison.csv"
summary.to_csv(summary_path, index=False)


# ==========================================
# Percentage difference in means
# ==========================================

comparison_rows = []

for metric in METRICS:
    mean_0 = datasets["stable_0"][metric].mean()
    mean_1 = datasets["stable_1"][metric].mean()

    comparison_rows.append({
        "metric": metric,
        "stable_0_mean": mean_0,
        "stable_1_mean": mean_1,
        "absolute_difference": mean_1 - mean_0,
        "percentage_difference_vs_stable_0": (
            (mean_1 - mean_0) / mean_0 * 100
            if mean_0 != 0 else float("nan")
        ),
    })

comparison = pd.DataFrame(comparison_rows)

comparison_path = OUTPUT_DIR / "mean_differences.csv"
comparison.to_csv(comparison_path, index=False)


# ==========================================
# Distribution plots
# ==========================================

for metric in METRICS:
    fig, ax = plt.subplots(figsize=(8, 6))

    values = [
        datasets["stable_0"][metric].values,
        datasets["stable_1"][metric].values,
    ]

    ax.boxplot(
        values,
        tick_labels=["stable_0", "stable_1"],
        showfliers=False
    )

    ax.set_title(f"{metric} Distribution Across Cases")
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    plot_path = OUTPUT_DIR / f"{metric}_comparison.png"
    fig.savefig(plot_path, dpi=250)
    plt.close(fig)

    print("Saved:", plot_path)


print("\n========== COMPARISON SUMMARY ==========")
print(comparison.to_string(index=False))

print("\nSaved report:", summary_path)
print("Saved mean differences:", comparison_path)
print("\nComparison completed.")
