
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_PATH = Path("data/processed/aneux_dome_dataset.csv")

SHAP_PATH = Path(
    "results/shap/directionality/signed_shap_values.csv"
)

STABILITY_PATH = Path(
    "results/shap/shap_feature_stability.csv"
)

OUTPUT_DIR = Path("results/shap/correlation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 10
MIN_SAMPLES = 10


# --------------------------------------------------
# 2. Load and prepare original dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=["patientID", "status"]
).copy()

# IMPORTANT:
# Reset index to match row_index saved in SHAP file.
df = df.reset_index(drop=True)

df["target"] = df["status"].map({
    "unruptured": 0,
    "ruptured": 1,
})

df = df.dropna(subset=["target"]).copy()

# Reset again after target filtering
df = df.reset_index(drop=True)

df["target"] = df["target"].astype(int)

print("Dataset records:", len(df))
print("Unique patients:", df["patientID"].nunique())


# --------------------------------------------------
# 3. Load SHAP results and stability
# --------------------------------------------------

shap_df = pd.read_csv(SHAP_PATH)
stability_df = pd.read_csv(STABILITY_PATH)

print("SHAP records:", len(shap_df))

# Validate row indices
if shap_df["row_index"].isna().any():
    raise ValueError("SHAP file contains missing row_index values.")

if not shap_df["row_index"].between(
    0, len(df) - 1
).all():
    raise ValueError(
        "SHAP row_index values do not match the prepared dataset."
    )

# Check fold and source coverage
print("\nSources in original dataset:")
print(df["source"].value_counts())

print("\nSources in SHAP row mapping:")
mapped_sources = shap_df["row_index"].map(df["source"])
print(mapped_sources.value_counts())


# --------------------------------------------------
# 4. Select top 10 stable numeric features
# --------------------------------------------------

top_features = (
    stability_df[
        stability_df["feature"].str.startswith("numeric__")
    ]
    .sort_values("mean_shap", ascending=False)
    .head(TOP_N)["feature"]
    .tolist()
)

print("\nSelected features:")
for feature in top_features:
    print(feature)


# --------------------------------------------------
# 5. Attach original values and metadata
# --------------------------------------------------

# Map metadata using the aligned row_index
shap_df["source"] = shap_df["row_index"].map(
    df["source"]
)

shap_df["patientID_original"] = shap_df["row_index"].map(
    df["patientID"].astype(str)
)

shap_df["target_original"] = shap_df["row_index"].map(
    df["target"]
)

# Validate patient and target alignment
if not (
    shap_df["patientID"].astype(str)
    == shap_df["patientID_original"]
).all():
    raise ValueError(
        "Patient ID mismatch between SHAP and original dataset."
    )

if not (
    shap_df["target"].astype(int)
    == shap_df["target_original"].astype(int)
).all():
    raise ValueError(
        "Target mismatch between SHAP and original dataset."
    )

# Attach original feature values
shap_df["original_value"] = np.nan

for feature in top_features:

    original_feature = feature.replace(
        "numeric__", "", 1
    )

    feature_mask = shap_df["feature"] == feature

    shap_df.loc[
        feature_mask, "original_value"
    ] = shap_df.loc[
        feature_mask, "row_index"
    ].map(df[original_feature])


# --------------------------------------------------
# 6. Validate complete feature coverage
# --------------------------------------------------

print("\nFeature sample counts:")

for feature in top_features:

    feature_data = shap_df[
        shap_df["feature"] == feature
    ]

    unique_rows = feature_data["row_index"].nunique()

    unique_sources = feature_data["source"].nunique()

    print(
        f"{feature}: "
        f"samples={len(feature_data)}, "
        f"unique_rows={unique_rows}, "
        f"sources={unique_sources}"
    )

    if unique_rows != len(df):
        raise ValueError(
            f"Incomplete row coverage for {feature}: "
            f"{unique_rows} instead of {len(df)}."
        )

    if unique_sources != df["source"].nunique():
        raise ValueError(
            f"Missing source coverage for {feature}."
        )

print("\nAll feature-to-row mappings validated.")


# --------------------------------------------------
# 7. Correlation helper
# --------------------------------------------------

def calculate_correlations(data):

    data = data[
        ["original_value", "shap_value"]
    ].dropna()

    n = len(data)

    if n < MIN_SAMPLES:
        return {
            "n": n,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }

    x = data["original_value"]
    y = data["shap_value"]

    if x.nunique() < 2 or y.nunique() < 2:
        return {
            "n": n,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }

    pearson_result = pearsonr(x, y)
    spearman_result = spearmanr(x, y)

    return {
        "n": n,
        "pearson_r": pearson_result.statistic,
        "pearson_p": pearson_result.pvalue,
        "spearman_rho": spearman_result.statistic,
        "spearman_p": spearman_result.pvalue,
    }


# --------------------------------------------------
# 8. Overall correlation
# --------------------------------------------------

overall_results = []

for feature in top_features:

    feature_data = shap_df[
        shap_df["feature"] == feature
    ]

    result = calculate_correlations(feature_data)

    overall_results.append({
        "feature": feature,
        **result,
    })

overall_df = pd.DataFrame(overall_results)

overall_path = (
    OUTPUT_DIR / "overall_shap_correlations.csv"
)

overall_df.to_csv(overall_path, index=False)


# --------------------------------------------------
# 9. Correlation per fold
# --------------------------------------------------

fold_results = []

for feature in top_features:

    feature_data = shap_df[
        shap_df["feature"] == feature
    ]

    for fold_number, fold_data in feature_data.groupby("fold"):

        result = calculate_correlations(fold_data)

        fold_results.append({
            "feature": feature,
            "fold": fold_number,
            **result,
        })

fold_df = pd.DataFrame(fold_results)

fold_path = (
    OUTPUT_DIR / "shap_correlations_per_fold.csv"
)

fold_df.to_csv(fold_path, index=False)


# --------------------------------------------------
# 10. Correlation per source
# --------------------------------------------------

source_results = []

for feature in top_features:

    feature_data = shap_df[
        shap_df["feature"] == feature
    ]

    for source_name, source_data in feature_data.groupby("source"):

        result = calculate_correlations(source_data)

        source_results.append({
            "feature": feature,
            "source": source_name,
            **result,
        })

source_df = pd.DataFrame(source_results)

source_path = (
    OUTPUT_DIR / "shap_correlations_per_source.csv"
)

source_df.to_csv(source_path, index=False)


# --------------------------------------------------
# 11. Direction consistency across folds
# --------------------------------------------------

consistency_results = []

for feature in top_features:

    feature_folds = fold_df[
        fold_df["feature"] == feature
    ]

    valid_correlations = feature_folds[
        feature_folds["spearman_rho"].notna()
    ]["spearman_rho"]

    positive_count = (
        valid_correlations > 0
    ).sum()

    negative_count = (
        valid_correlations < 0
    ).sum()

    consistency_results.append({
        "feature": feature,
        "valid_folds": len(valid_correlations),
        "positive_folds": int(positive_count),
        "negative_folds": int(negative_count),
        "mean_spearman_rho": (
            valid_correlations.mean()
            if len(valid_correlations)
            else np.nan
        ),
    })

consistency_df = pd.DataFrame(
    consistency_results
)

consistency_path = (
    OUTPUT_DIR / "shap_direction_consistency.csv"
)

consistency_df.to_csv(
    consistency_path,
    index=False,
)


# --------------------------------------------------
# 12. Print reports
# --------------------------------------------------

print("\n" + "=" * 70)
print("OVERALL SHAP CORRELATIONS")
print("=" * 70)

print(overall_df.round(4).to_string(index=False))


print("\n" + "=" * 70)
print("DIRECTION CONSISTENCY ACROSS FOLDS")
print("=" * 70)

print(
    consistency_df.round(4).to_string(index=False)
)


print("\n" + "=" * 70)
print("CORRELATIONS PER SOURCE")
print("=" * 70)

print(source_df.round(4).to_string(index=False))


print("\n" + "=" * 70)
print("SOURCE SAMPLE COUNTS")
print("=" * 70)

for feature in top_features:

    feature_data = shap_df[
        shap_df["feature"] == feature
    ]

    print(f"\n{feature}")

    print(
        feature_data["source"]
        .value_counts()
        .to_string()
    )


# --------------------------------------------------
# 13. Final output
# --------------------------------------------------

print("\nSaved files:")
print(overall_path)
print(fold_path)
print(source_path)
print(consistency_path)

print("\nSHAP correlation analysis completed.")
