from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score
)

# --------------------------------------------------
# 1. Load and prepare data
# --------------------------------------------------

df = pd.read_csv("data/processed/aneux_dome_dataset.csv")
df = df.dropna(subset=["patientID"]).copy().reset_index(drop=True)

target_map = {"unruptured": 0, "ruptured": 1}
y = df["status"].map(target_map).astype(int)
groups = df["patientID"]

excluded = {
    "source", "dataset", "cutType", "status",
    "patientID", "vesselFileID", "hospital",
    "cutToShow"
}

clinical_features = ["age", "sex", "location", "side"]

morphology = pd.read_csv(
    "data/raw/aneux/data/per_source/area-005/morpho-per-cut-flat.csv"
)

morphology_features = [
    c for c in morphology.columns
    if c not in excluded
    and c not in {"age", "sex", "location", "side"}
    and c in df.columns
    and pd.api.types.is_numeric_dtype(df[c])
]

combined_features = clinical_features + morphology_features

feature_sets = {
    "clinical_only": clinical_features,
    "morphology_only": morphology_features,
    "combined_logistic": combined_features,
    "combined_selected_30": combined_features,
    "combined_random_forest": combined_features
}

# --------------------------------------------------
# 2. Build preprocessing
# --------------------------------------------------

def build_preprocessor(features, scale_numeric=True):
    numeric = [
        c for c in features
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    categorical = [c for c in features if c not in numeric]

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []

    if numeric:
        transformers.append(("num", numeric_pipeline, numeric))

    if categorical:
        transformers.append(
            ("cat", categorical_pipeline, categorical)
        )

    return ColumnTransformer(transformers)


def build_model(model_name, features):
    is_rf = model_name == "combined_random_forest"

    preprocessing = build_preprocessor(
        features,
        scale_numeric=not is_rf
    )

    steps = [("preprocessing", preprocessing)]

    if model_name == "combined_selected_30":
        steps.append(
            ("selection", SelectKBest(score_func=f_classif, k=30))
        )

    if is_rf:
        classifier = RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    else:
        classifier = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        )

    steps.append(("classifier", classifier))
    return Pipeline(steps)


# --------------------------------------------------
# 3. Patient-grouped cross-validation
# --------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

for model_name, features in feature_sets.items():
    print(f"\nRunning: {model_name}")

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

        assert train_groups.isdisjoint(test_groups)

        if y_test.nunique() < 2:
            raise ValueError(f"Fold {fold} has one target class.")

        model = build_model(model_name, features)
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = model.predict(X_test)

        row = {
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
        }

        results.append(row)

        print(
            f"Fold {fold}: "
            f"ROC-AUC={row['roc_auc']:.4f}, "
            f"PR-AUC={row['pr_auc']:.4f}, "
            f"Balanced Accuracy={row['balanced_accuracy']:.4f}"
        )

# --------------------------------------------------
# 4. Save comparison
# --------------------------------------------------

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

fold_results = pd.DataFrame(results)
fold_results.to_csv(
    output_dir / "model_comparison_folds.csv",
    index=False
)

summary = (
    fold_results.groupby("model")[
        ["roc_auc", "pr_auc", "balanced_accuracy"]
    ]
    .agg(["mean", "std"])
    .round(4)
)

summary.to_csv(output_dir / "model_comparison_summary.csv")

print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
print(summary)

print("\nSaved:")
print("results/model_comparison_folds.csv")
print("results/model_comparison_summary.csv")
