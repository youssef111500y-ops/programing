
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


# --------------------------------------------------
# 1. Load and prepare dataset
# --------------------------------------------------

DATA_PATH = Path("data/processed/aneux_dome_dataset.csv")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Keep records with known patient IDs and status
df = df.dropna(subset=["patientID", "status"]).copy()

df["target"] = df["status"].map({
    "unruptured": 0,
    "ruptured": 1,
})

df = df.dropna(subset=["target"]).copy()
df["target"] = df["target"].astype(int)

y = df["target"]
groups = df["patientID"].astype(str)


# --------------------------------------------------
# 2. Define feature sets
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

feature_sets = {
    "clinical_only": clinical_features,
    "morphology_only": morphology_features,
    "combined": clinical_features + morphology_features,
}

print("Dataset shape:", df.shape)
print("Unique patients:", groups.nunique())
print("Clinical features:", len(clinical_features))
print("Morphology features:", len(morphology_features))


# --------------------------------------------------
# 3. Create identical patient-level CV splits
# --------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

splits = list(cv.split(df, y, groups))


# --------------------------------------------------
# 4. Build model pipeline
# --------------------------------------------------

def build_model(X):

    numeric_features = [
        col for col in X.columns
        if pd.api.types.is_numeric_dtype(X[col])
    ]

    categorical_features = [
        col for col in X.columns
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

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=42,
        )),
    ])


# --------------------------------------------------
# 5. Run Feature Ablation
# --------------------------------------------------

fold_results = []

for model_name, selected_features in feature_sets.items():

    print(f"\n{'=' * 50}")
    print("Running:", model_name)
    print("Number of features:", len(selected_features))
    print(f"{'=' * 50}")

    X = df[selected_features].copy()

    for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = build_model(X_train)

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = model.predict(X_test)

        roc_auc = roc_auc_score(y_test, probabilities)
        pr_auc = average_precision_score(y_test, probabilities)
        balanced_acc = balanced_accuracy_score(
            y_test,
            predictions,
        )

        print(
            f"Fold {fold_number}: "
            f"ROC-AUC={roc_auc:.4f}, "
            f"PR-AUC={pr_auc:.4f}, "
            f"Balanced Accuracy={balanced_acc:.4f}"
        )

        fold_results.append({
            "model": model_name,
            "fold": fold_number,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "balanced_accuracy": balanced_acc,
        })


# --------------------------------------------------
# 6. Save fold-level results
# --------------------------------------------------

fold_results_df = pd.DataFrame(fold_results)

fold_path = OUTPUT_DIR / "feature_ablation_folds.csv"
fold_results_df.to_csv(fold_path, index=False)


# --------------------------------------------------
# 7. Calculate summary
# --------------------------------------------------

summary = (
    fold_results_df
    .groupby("model")[
        ["roc_auc", "pr_auc", "balanced_accuracy"]
    ]
    .agg(["mean", "std"])
)

summary_path = OUTPUT_DIR / "feature_ablation_summary.csv"
summary.to_csv(summary_path)

print("\nFINAL FEATURE ABLATION SUMMARY")
print(summary.round(4))

print("\nSaved files:")
print(fold_path)
print(summary_path)
