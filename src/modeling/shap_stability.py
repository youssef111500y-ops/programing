
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

INPUT_PATH = Path(
    "results/shap/fold_level_shap_importance.csv"
)

OUTPUT_DIR = Path("results/shap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 30


# --------------------------------------------------
# 2. Load fold-level SHAP results
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

required_columns = {
    "feature",
    "mean_abs_shap",
    "fold",
}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Missing columns. Expected: {required_columns}"
    )

print("Loaded SHAP records:", len(df))
print("Number of folds:", df["fold"].nunique())
print("Number of unique features:", df["feature"].nunique())


# --------------------------------------------------
# 3. Calculate feature rank inside each fold
# --------------------------------------------------

df["rank"] = (
    df.groupby("fold")["mean_abs_shap"]
    .rank(method="min", ascending=False)
)

df["top_10"] = df["rank"] <= 10
df["top_30"] = df["rank"] <= 30


# --------------------------------------------------
# 4. Aggregate stability metrics
# --------------------------------------------------

stability = (
    df.groupby("feature")
    .agg(
        mean_shap=("mean_abs_shap", "mean"),
        std_shap=("mean_abs_shap", "std"),
        mean_rank=("rank", "mean"),
        std_rank=("rank", "std"),
        folds_present=("fold", "nunique"),
        top_10_count=("top_10", "sum"),
        top_30_count=("top_30", "sum"),
    )
    .reset_index()
)

# Replace missing standard deviations
# (for features appearing in only one fold)
stability["std_shap"] = stability["std_shap"].fillna(0)
stability["std_rank"] = stability["std_rank"].fillna(0)

# Stability frequency as percentages
total_folds = df["fold"].nunique()

stability["top_10_frequency"] = (
    stability["top_10_count"] / total_folds * 100
)

stability["top_30_frequency"] = (
    stability["top_30_count"] / total_folds * 100
)

# Coefficient of variation
# Lower values indicate less relative variation
stability["shap_cv"] = (
    stability["std_shap"]
    / stability["mean_shap"].replace(0, float("nan"))
)


# --------------------------------------------------
# 5. Sort by stability and importance
# --------------------------------------------------

stability = stability.sort_values(
    by=[
        "top_10_count",
        "top_30_count",
        "mean_shap",
    ],
    ascending=[False, False, False],
).reset_index(drop=True)

stability["overall_rank"] = (
    stability.index + 1
)


# --------------------------------------------------
# 6. Save full stability report
# --------------------------------------------------

stability_path = (
    OUTPUT_DIR / "shap_feature_stability.csv"
)

stability.to_csv(stability_path, index=False)


# --------------------------------------------------
# 7. Save top 30 stable features
# --------------------------------------------------

top_30 = stability.head(TOP_N)

top_30_path = (
    OUTPUT_DIR / "shap_top_30_stable_features.csv"
)

top_30.to_csv(top_30_path, index=False)


# --------------------------------------------------
# 8. Plot top 30 stable features
# --------------------------------------------------

plot_data = top_30.sort_values(
    "top_10_count",
    ascending=True,
)

plt.figure(figsize=(12, 10))

plt.barh(
    plot_data["feature"],
    plot_data["top_10_count"],
)

plt.xlabel("Number of Folds in Top 10")
plt.ylabel("Feature")
plt.title("SHAP Feature Stability Across 5 Folds")

plt.xticks(range(0, total_folds + 1))

plt.tight_layout()

plot_path = (
    OUTPUT_DIR / "shap_feature_stability.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# --------------------------------------------------
# 9. Print final report
# --------------------------------------------------

print("\n" + "=" * 70)
print("TOP 30 SHAP STABLE FEATURES")
print("=" * 70)

display_columns = [
    "feature",
    "mean_shap",
    "std_shap",
    "mean_rank",
    "top_10_count",
    "top_30_count",
    "folds_present",
]

print(
    top_30[display_columns]
    .round(4)
    .to_string(index=False)
)

print("\nSaved files:")
print(stability_path)
print(top_30_path)
print(plot_path)

print("\nSHAP stability analysis completed.")
