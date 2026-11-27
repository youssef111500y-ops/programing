from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score
)

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

DATA_PATH = Path("data/processed/aneux_dome_dataset.csv")
MORPH_PATH = Path(
    "data/raw/aneux/data/per_source/area-005/morpho-per-cut-flat.csv"
)

df = pd.read_csv(DATA_PATH)
morphology = pd.read_csv(MORPH_PATH)

# Primary evaluation: records with known patient IDs
df = df.dropna(subset=["patientID"]).copy()

# Encode target
target_map = {
    "unruptured": 0,
    "ruptured": 1
}

df["target"] = df["status"].map(target_map)

if df["target"].isna().any():
    raise ValueError("Unexpected target labels found.")

y = df["target"].astype(int)
groups = df["patientID"]

# --------------------------------------------------
# 2. Define feature groups
# --------------------------------------------------

clinical_numeric = ["age"]
clinical_categorical = ["sex", "location", "side"]

clinical_features = clinical_numeric + clinical_categorical

# Use morphology columns from the original feature file.
# Exclude identifiers and metadata.
excluded = {
    "source", "dataset", "cutType",
    "status", "patientID", "vesselFileID",
    "hospital", "location", "side", "sex",
    "age", "cutToShow"
}

morphology_features = [
    col for col in morphology.columns
    if col not in excluded
    and col in df.columns
    and pd.api.types.is_numeric_dtype(df[col])
]

if not morphology_features:
    raise ValueError("No numeric morphology features found.")

feature_sets = {
    "clinical_only": clinical_features,
    "morphology_only": morphology_features,
    "combined": clinical_features + morphology_features
}

# --------------------------------------------------
# 3. Preprocessing pipeline
# --------------------------------------------------

def build_pipeline(features):
    numeric_features = [
        col for col in features
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    categorical_features = [
        col for col in features
        if col not in numeric_features
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []

    if numeric_features:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    preprocessing = ColumnTransformer(transformers)

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        ))
    ])

# --------------------------------------------------
# 4. Patient-level cross-validation
# --------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold_results = []

for model_name, features in feature_sets.items():

    print(f"\nRunning: {model_name}")
    print("Number of input features:", len(features))

    X = df[features]

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups), start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])

        assert train_groups.isdisjoint(test_groups), (
            "Patient leakage detected!"
        )

        if y_test.nunique() < 2:
            raise ValueError(
                f"Fold {fold} has only one target class in test data."
            )

        model = build_pipeline(features)
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = model.predict(X_test)

        fold_results.append({
            "model": model_name,
            "fold": fold,
            "train_records": len(train_idx),
            "test_records": len(test_idx),
            "train_patients": len(train_groups),
            "test_patients": len(test_groups),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "pr_auc": average_precision_score(y_test, probabilities),
            "balanced_accuracy": balanced_accuracy_score(
                y_test, predictions
            )
        })

        print(
            f"Fold {fold}: "
            f"ROC-AUC={fold_results[-1]['roc_auc']:.4f}, "
            f"PR-AUC={fold_results[-1]['pr_auc']:.4f}, "
            f"Balanced Accuracy="
            f"{fold_results[-1]['balanced_accuracy']:.4f}"
        )

# --------------------------------------------------
# 5. Save results
# --------------------------------------------------

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

fold_df = pd.DataFrame(fold_results)
fold_df.to_csv(
    results_dir / "baseline_fold_results.csv",
    index=False
)

summary = (
    fold_df.groupby("model")[
        ["roc_auc", "pr_auc", "balanced_accuracy"]
    ]
    .agg(["mean", "std"])
    .round(4)
)

summary.to_csv(results_dir / "baseline_summary.csv")

print("\n" + "=" * 55)
print("BASELINE RESULTS")
print("=" * 55)
print(summary)

print("\nEvaluation records:", len(df))
print("Unique patients:", groups.nunique())
print("Target distribution:")
print(y.value_counts())

print("\nSaved:")
print("results/baseline_fold_results.csv")
print("results/baseline_summary.csv")
