
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. Configuration
# ============================================================

DATA_PATH = Path("data/processed/aneux_dome_dataset.csv")
OUTPUT_DIR = Path("results/regularization_sensitivity")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

# Regularization configurations
MODEL_CONFIGS = [
    {
        "name": "L1_C0.1",
        "penalty": "l1",
        "C": 0.1,
        "l1_ratio": None,
    },
    {
        "name": "L1_C1.0",
        "penalty": "l1",
        "C": 1.0,
        "l1_ratio": None,
    },
    {
        "name": "L2_C0.1",
        "penalty": "l2",
        "C": 0.1,
        "l1_ratio": None,
    },
    {
        "name": "L2_C1.0",
        "penalty": "l2",
        "C": 1.0,
        "l1_ratio": None,
    },
    {
        "name": "ElasticNet_C0.1",
        "penalty": "elasticnet",
        "C": 0.1,
        "l1_ratio": 0.5,
    },
    {
        "name": "ElasticNet_C1.0",
        "penalty": "elasticnet",
        "C": 1.0,
        "l1_ratio": 0.5,
    },
]


# ============================================================
# 2. Load and prepare dataset
# ============================================================

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["patientID", "status"]).copy()
df = df.reset_index(drop=True)

df["target"] = (
    df["status"]
    .astype(str)
    .str.lower()
    .map({"unruptured": 0, "ruptured": 1})
)

df = df.dropna(subset=["target"]).copy()
df = df.reset_index(drop=True)
df["target"] = df["target"].astype(int)

exclude_columns = [
    "status",
    "target",
    "patientID",
    "source",
    "dataset",
    "hospital",
    "vesselFileID",
    "cutType",
]

clinical_features = [
    "age",
    "sex",
    "location",
    "side",
]

morphology_features = [
    col
    for col in df.select_dtypes(include=np.number).columns
    if col not in exclude_columns + clinical_features
]

features = clinical_features + morphology_features

X = df[features].copy()
y = df["target"].copy()
groups = df["patientID"].copy()

numeric_features = [
    col for col in features
    if col in morphology_features or col == "age"
]

categorical_features = [
    col for col in clinical_features
    if col not in numeric_features
]

print("Dataset shape:", df.shape)
print("Patients:", groups.nunique())
print("Features:", len(features))
print("Numeric:", len(numeric_features))
print("Categorical:", len(categorical_features))


# ============================================================
# 3. Patient-level cross-validation
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

splits = list(cv.split(X, y, groups))


# ============================================================
# 4. Build preprocessing pipeline
# ============================================================

def build_preprocessor():

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )


# ============================================================
# 5. Run regularization experiments
# ============================================================

metrics_records = []
coefficients_records = []

for config in MODEL_CONFIGS:

    model_name = config["name"]

    print("\n" + "=" * 65)
    print("MODEL:", model_name)
    print("=" * 65)

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        preprocessor = build_preprocessor()

        classifier = LogisticRegression(
            penalty=config["penalty"],
            C=config["C"],
            l1_ratio=config["l1_ratio"],
            solver="saga",
            class_weight="balanced",
            max_iter=10000,
            random_state=RANDOM_STATE,
            tol=1e-4,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = pipeline.predict(X_test)

        roc = roc_auc_score(y_test, probabilities)
        pr = average_precision_score(y_test, probabilities)
        ba = balanced_accuracy_score(y_test, predictions)

        metrics_records.append({
            "model": model_name,
            "fold": fold,
            "roc_auc": roc,
            "pr_auc": pr,
            "balanced_accuracy": ba,
        })

        fitted_preprocessor = pipeline.named_steps["preprocessor"]
        fitted_classifier = pipeline.named_steps["classifier"]

        feature_names = fitted_preprocessor.get_feature_names_out()
        coefficients = fitted_classifier.coef_[0]

        for feature, coefficient in zip(feature_names, coefficients):

            coefficients_records.append({
                "model": model_name,
                "fold": fold,
                "feature": feature,
                "coefficient": coefficient,
            })

        nonzero_count = np.count_nonzero(np.abs(coefficients) > 1e-8)

        print(
            f"Fold {fold}: "
            f"ROC-AUC={roc:.4f}, "
            f"PR-AUC={pr:.4f}, "
            f"BA={ba:.4f}, "
            f"Non-zero={nonzero_count}/{len(coefficients)}"
        )


# ============================================================
# 6. Save fold-level results
# ============================================================

metrics_df = pd.DataFrame(metrics_records)
coefficients_df = pd.DataFrame(coefficients_records)

metrics_df.to_csv(
    OUTPUT_DIR / "fold_level_metrics.csv",
    index=False,
)

coefficients_df.to_csv(
    OUTPUT_DIR / "fold_level_coefficients.csv",
    index=False,
)


# ============================================================
# 7. Aggregate model performance
# ============================================================

performance_summary = (
    metrics_df
    .groupby("model")
    .agg(
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
    )
    .reset_index()
)

performance_summary = performance_summary.sort_values(
    "roc_auc_mean",
    ascending=False,
)

performance_summary.to_csv(
    OUTPUT_DIR / "performance_summary.csv",
    index=False,
)


# ============================================================
# 8. Coefficient stability
# ============================================================

coefficient_summary = (
    coefficients_df
    .groupby(["model", "feature"])
    .agg(
        mean_coefficient=("coefficient", "mean"),
        std_coefficient=("coefficient", "std"),
        mean_abs_coefficient=(
            "coefficient",
            lambda values: np.mean(np.abs(values)),
        ),
        positive_folds=(
            "coefficient",
            lambda values: int((values > 0).sum()),
        ),
        negative_folds=(
            "coefficient",
            lambda values: int((values < 0).sum()),
        ),
        folds_present=("coefficient", "count"),
        nonzero_folds=(
            "coefficient",
            lambda values: int((np.abs(values) > 1e-8).sum()),
        ),
    )
    .reset_index()
)

coefficient_summary["direction_consistency"] = (
    coefficient_summary[
        ["positive_folds", "negative_folds"]
    ].max(axis=1)
    / coefficient_summary["folds_present"]
)

coefficient_summary.to_csv(
    OUTPUT_DIR / "coefficient_stability_summary.csv",
    index=False,
)


# ============================================================
# 9. Compare selected morphology features
# ============================================================

selected_features = [
    "numeric__zmi_surf.9.7",
    "numeric__zmi_surf.16.12",
    "numeric__zmi_surf.10.10",
    "numeric__zmi_surf.7.7",
    "numeric__curvature.gauss.L2N",
    "numeric__zmi_surf.14.4",
    "numeric__zmi_surf.5.5",
    "numeric__zmi_surf.8.4",
    "numeric__zmi_surf.3.3",
    "numeric__curvature.mean.L2NCH",
]

selected_summary = coefficient_summary[
    coefficient_summary["feature"].isin(selected_features)
].copy()

selected_summary.to_csv(
    OUTPUT_DIR / "selected_features_comparison.csv",
    index=False,
)


# ============================================================
# 10. Print results
# ============================================================

print("\n" + "=" * 75)
print("REGULARIZATION PERFORMANCE SUMMARY")
print("=" * 75)

print(
    performance_summary.round(4).to_string(index=False)
)

print("\n" + "=" * 75)
print("SELECTED FEATURE COEFFICIENT STABILITY")
print("=" * 75)

for model_name in MODEL_CONFIGS:

    name = model_name["name"]

    print(f"\nMODEL: {name}")

    current = selected_summary[
        selected_summary["model"] == name
    ].sort_values("mean_abs_coefficient", ascending=False)

    if current.empty:
        print("No selected features found.")
    else:
        print(
            current[
                [
                    "feature",
                    "mean_coefficient",
                    "std_coefficient",
                    "direction_consistency",
                    "nonzero_folds",
                ]
            ].round(4).to_string(index=False)
        )


print("\nSaved files:")
print(OUTPUT_DIR / "fold_level_metrics.csv")
print(OUTPUT_DIR / "fold_level_coefficients.csv")
print(OUTPUT_DIR / "performance_summary.csv")
print(OUTPUT_DIR / "coefficient_stability_summary.csv")
print(OUTPUT_DIR / "selected_features_comparison.csv")

print("\nRegularization sensitivity analysis completed.")
