
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_PATH = Path("data/processed/aneux_dome_dataset.csv")

OUTPUT_DIR = Path("results/model_stability")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=["patientID", "status"]
).copy()

df = df.reset_index(drop=True)

df["target"] = df["status"].map({
    "unruptured": 0,
    "ruptured": 1,
})

df = df.dropna(subset=["target"]).copy()
df = df.reset_index(drop=True)

df["target"] = df["target"].astype(int)

y = df["target"]
groups = df["patientID"].astype(str)


# --------------------------------------------------
# 3. Define feature sets
# --------------------------------------------------

clinical_features = [
    "age",
    "sex",
    "location",
    "side",
]

excluded_columns = {
    "source",
    "dataset",
    "cutType",
    "hospital",
    "status",
    "patientID",
    "vesselFileID",
    "cutToShow",
    "target",
}

morphology_features = [
    col for col in df.columns
    if col not in excluded_columns
    and col not in clinical_features
    and pd.api.types.is_numeric_dtype(df[col])
]

features = clinical_features + morphology_features

X = df[features].copy()

print("Dataset shape:", df.shape)
print("Patients:", groups.nunique())
print("Clinical features:", len(clinical_features))
print("Morphology features:", len(morphology_features))
print("Total features:", len(features))


# --------------------------------------------------
# 4. Build preprocessing
# --------------------------------------------------

def build_preprocessor(X_train):

    numeric_features = [
        col for col in X_train.columns
        if pd.api.types.is_numeric_dtype(X_train[col])
    ]

    categorical_features = [
        col for col in X_train.columns
        if col not in numeric_features
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


# --------------------------------------------------
# 5. Stratified Group Cross Validation
# --------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

splits = list(cv.split(X, y, groups))


# --------------------------------------------------
# 6. Extract coefficients from every fold
# --------------------------------------------------

all_coefficients = []
fold_metrics = []

for fold_number, (train_idx, test_idx) in enumerate(
    splits, start=1
):

    print(f"\nProcessing Fold {fold_number}/{N_SPLITS}")

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Fit preprocessing on training data only
    preprocessor = build_preprocessor(X_train)

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()

    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    # Train Logistic Regression
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE,
    )

    classifier.fit(X_train_transformed, y_train)

    # Extract coefficients
    coefficients = classifier.coef_[0]

    if len(coefficients) != len(feature_names):
        raise ValueError(
            "Coefficient count does not match feature names."
        )

    for feature_name, coefficient in zip(
        feature_names, coefficients
    ):

        all_coefficients.append({
            "fold": fold_number,
            "feature": feature_name,
            "coefficient": float(coefficient),
            "abs_coefficient": float(abs(coefficient)),
        })

    # Evaluate the fold model
    probabilities = classifier.predict_proba(
        X_test_transformed
    )[:, 1]

    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        balanced_accuracy_score,
    )

    predictions = classifier.predict(X_test_transformed)

    fold_metrics.append({
        "fold": fold_number,
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
    })

    print("Training records:", len(train_idx))
    print("Testing records:", len(test_idx))
    print("Extracted coefficients:", len(coefficients))


# --------------------------------------------------
# 7. Save fold-level coefficients
# --------------------------------------------------

coefficients_df = pd.DataFrame(all_coefficients)

fold_coefficients_path = (
    OUTPUT_DIR / "fold_level_coefficients.csv"
)

coefficients_df.to_csv(
    fold_coefficients_path,
    index=False,
)


# --------------------------------------------------
# 8. Aggregate coefficient stability
# --------------------------------------------------

def positive_count(values):
    return int((values > 0).sum())


def negative_count(values):
    return int((values < 0).sum())


stability = (
    coefficients_df.groupby("feature")
    .agg(
        mean_coefficient=("coefficient", "mean"),
        std_coefficient=("coefficient", "std"),
        mean_abs_coefficient=("abs_coefficient", "mean"),
        positive_folds=("coefficient", positive_count),
        negative_folds=("coefficient", negative_count),
        folds_present=("fold", "nunique"),
    )
    .reset_index()
)

stability["std_coefficient"] = (
    stability["std_coefficient"].fillna(0)
)

stability["direction_consistency"] = (
    stability[
        ["positive_folds", "negative_folds"]
    ].max(axis=1) / N_SPLITS
)

stability["mean_abs_coefficient"] = (
    stability["mean_abs_coefficient"]
)

stability = stability.sort_values(
    "mean_abs_coefficient",
    ascending=False,
).reset_index(drop=True)

stability["rank"] = stability.index + 1


# --------------------------------------------------
# 9. Save coefficient stability report
# --------------------------------------------------

stability_path = (
    OUTPUT_DIR / "coefficient_stability_summary.csv"
)

stability.to_csv(stability_path, index=False)


# --------------------------------------------------
# 10. Save fold performance
# --------------------------------------------------

metrics_df = pd.DataFrame(fold_metrics)

metrics_path = (
    OUTPUT_DIR / "coefficient_model_fold_metrics.csv"
)

metrics_df.to_csv(metrics_path, index=False)


# --------------------------------------------------
# 11. Print top 30 features
# --------------------------------------------------

print("\n" + "=" * 70)
print("TOP 30 COEFFICIENT STABILITY")
print("=" * 70)

display_columns = [
    "feature",
    "mean_coefficient",
    "std_coefficient",
    "mean_abs_coefficient",
    "positive_folds",
    "negative_folds",
    "direction_consistency",
]

print(
    stability.head(30)[display_columns]
    .round(4)
    .to_string(index=False)
)


# --------------------------------------------------
# 12. Final output
# --------------------------------------------------

print("\nFOLD PERFORMANCE")
print(metrics_df.round(4).to_string(index=False))

print("\nSaved files:")
print(fold_coefficients_path)
print(stability_path)
print(metrics_path)

print("\nCoefficient stability analysis completed.")
