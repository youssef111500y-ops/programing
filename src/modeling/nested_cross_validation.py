
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = Path("data/processed/aneux_dome_dataset.csv")
OUTPUT_DIR = Path("results/nested_cv")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
OUTER_SPLITS = 5
INNER_SPLITS = 4

PARAM_GRID = {
    "classifier__C": [0.001, 0.01, 0.03, 0.1, 0.3, 1.0],
    "classifier__l1_ratio": [0.0, 0.5, 1.0],
}


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 65)
print("NESTED CROSS-VALIDATION")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

print("\nOriginal dataset shape:", df.shape)


# ============================================================
# 3. PREPARE TARGET
# ============================================================

status_mapping = {
    "unruptured": 0,
    "ruptured": 1,
}

if "status" not in df.columns:
    raise ValueError("Target column 'status' was not found.")

status_text = (
    df["status"]
    .astype("string")
    .str.strip()
    .str.lower()
)

target = status_text.map(status_mapping)

unknown_labels = status_text.notna() & target.isna()

if unknown_labels.any():
    invalid_values = status_text[unknown_labels].unique().tolist()
    raise ValueError(f"Unknown status labels: {invalid_values}")

df = df.copy()
df["target"] = target

df = df.dropna(
    subset=["target", "patientID"]
).copy()

df["target"] = df["target"].astype(int)
df = df.reset_index(drop=True)

if not set(df["target"].unique()).issubset({0, 1}):
    raise ValueError("Target must contain only binary values.")

print("Dataset after filtering:", df.shape)
print("Unique patients:", df["patientID"].nunique())

print("\nTarget distribution:")
print(df["target"].value_counts().sort_index())


# ============================================================
# 4. DEFINE FEATURES
# ============================================================

clinical_numeric = ["age"]

clinical_categorical = [
    "sex",
    "location",
    "side",
]

required_columns = (
    clinical_numeric
    + clinical_categorical
    + ["patientID", "target"]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

excluded_columns = {
    "source",
    "dataset",
    "cutType",
    "hospital",
    "status",
    "location",
    "side",
    "sex",
    "age",
    "patientID",
    "vesselFileID",
    "cutToShow",
    "target",
}

morphology_features = [
    col
    for col in df.columns
    if col not in excluded_columns
    and pd.api.types.is_numeric_dtype(df[col])
]

numeric_features = clinical_numeric + morphology_features
categorical_features = clinical_categorical

feature_columns = numeric_features + categorical_features

X = df[feature_columns].copy()
y = df["target"].copy()
groups = df["patientID"].copy()

print("\nFeature configuration:")
print("Clinical numeric features:", clinical_numeric)
print("Clinical categorical features:", categorical_features)
print("Morphological features:", len(morphology_features))
print("Total features:", len(feature_columns))


# ============================================================
# 5. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ],
    remainder="drop",
)


# ============================================================
# 6. MODEL PIPELINE
# ============================================================

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                solver="saga",
                class_weight="balanced",
                max_iter=10000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)


# ============================================================
# 7. OUTPUT STORAGE
# ============================================================

outer_fold_metrics = []
best_parameters = []
outer_predictions = []
outer_coefficients = []


def save_current_results():
    """Save accumulated results after every completed outer fold."""

    if outer_fold_metrics:
        pd.DataFrame(outer_fold_metrics).to_csv(
            OUTPUT_DIR / "outer_fold_metrics.csv",
            index=False,
        )

    if best_parameters:
        pd.DataFrame(best_parameters).to_csv(
            OUTPUT_DIR / "best_parameters_per_fold.csv",
            index=False,
        )

    if outer_predictions:
        pd.concat(
            outer_predictions,
            ignore_index=True,
        ).to_csv(
            OUTPUT_DIR / "outer_fold_predictions.csv",
            index=False,
        )

    if outer_coefficients:
        pd.DataFrame(outer_coefficients).to_csv(
            OUTPUT_DIR / "outer_fold_coefficients.csv",
            index=False,
        )


# ============================================================
# 8. NESTED CROSS-VALIDATION
# ============================================================

outer_cv = StratifiedGroupKFold(
    n_splits=OUTER_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

print("\nStarting Nested Cross-Validation...")

for outer_fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(X, y, groups),
    start=1,
):

    print("\n" + "=" * 65)
    print(f"OUTER FOLD {outer_fold}/{OUTER_SPLITS}")
    print("=" * 65)

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]

    train_patients = set(groups_train.unique())
    test_patients = set(groups.iloc[test_idx].unique())

    overlap = train_patients.intersection(test_patients)

    if overlap:
        raise RuntimeError(
            f"Patient leakage detected in outer fold {outer_fold}"
        )

    print("Training records:", len(train_idx))
    print("Testing records:", len(test_idx))
    print("Training patients:", len(train_patients))
    print("Testing patients:", len(test_patients))

    # --------------------------------------------------------
    # 8.1 INNER CROSS-VALIDATION
    # --------------------------------------------------------

    inner_cv = StratifiedGroupKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE + outer_fold,
    )

    inner_splits = list(
        inner_cv.split(
            X_train,
            y_train,
            groups_train,
        )
    )

    # --------------------------------------------------------
    # 8.2 GRID SEARCH
    # --------------------------------------------------------

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=PARAM_GRID,
        scoring="roc_auc",
        cv=inner_splits,
        refit=True,
        n_jobs=-1,
        verbose=0,
        error_score="raise",
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Found unknown categories",
        )

        grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print(
        "\nBest inner CV ROC-AUC:",
        round(grid_search.best_score_, 4),
    )
    print("Best parameters:", grid_search.best_params_)

    # --------------------------------------------------------
    # 8.3 OUTER TEST PREDICTIONS
    # --------------------------------------------------------

    y_probability = best_model.predict_proba(X_test)[:, 1]
    y_predicted = (y_probability >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_probability)
    pr_auc = average_precision_score(y_test, y_probability)
    balanced_acc = balanced_accuracy_score(y_test, y_predicted)

    print("\nOuter test metrics:")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    # --------------------------------------------------------
    # 8.4 SAVE FOLD METRICS
    # --------------------------------------------------------

    outer_fold_metrics.append(
        {
            "outer_fold": outer_fold,
            "train_records": len(train_idx),
            "test_records": len(test_idx),
            "train_patients": len(train_patients),
            "test_patients": len(test_patients),
            "inner_best_roc_auc": grid_search.best_score_,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "balanced_accuracy": balanced_acc,
        }
    )

    # Correct attribute: best_params_ (single trailing underscore)
    best_parameters.append(
        {
            "outer_fold": outer_fold,
            **grid_search.best_params_,
            "inner_best_roc_auc": grid_search.best_score_,
        }
    )

    # --------------------------------------------------------
    # 8.5 SAVE OUTER FOLD PREDICTIONS
    # --------------------------------------------------------

    fold_predictions = pd.DataFrame(
        {
            "row_index": test_idx,
            "outer_fold": outer_fold,
            "patientID": groups.iloc[test_idx].values,
            "y_true": y_test.values,
            "y_probability": y_probability,
            "y_predicted": y_predicted,
        }
    )

    outer_predictions.append(fold_predictions)

    # --------------------------------------------------------
    # 8.6 SAVE MODEL COEFFICIENTS
    # --------------------------------------------------------

    fitted_preprocessor = best_model.named_steps["preprocessor"]
    fitted_classifier = best_model.named_steps["classifier"]

    transformed_feature_names = (
        fitted_preprocessor.get_feature_names_out()
    )

    coefficients = fitted_classifier.coef_[0]

    for feature_name, coefficient in zip(
        transformed_feature_names,
        coefficients,
    ):
        outer_coefficients.append(
            {
                "outer_fold": outer_fold,
                "feature": feature_name,
                "coefficient": coefficient,
                "abs_coefficient": abs(coefficient),
            }
        )

    # Save immediately after each completed fold.
    save_current_results()

    print(f"\nFold {outer_fold} results saved successfully.")


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

metrics_df = pd.DataFrame(outer_fold_metrics)
parameters_df = pd.DataFrame(best_parameters)
predictions_df = pd.concat(
    outer_predictions,
    ignore_index=True,
)
coefficients_df = pd.DataFrame(outer_coefficients)

summary_rows = []

for metric in [
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
]:
    summary_rows.append(
        {
            "metric": metric,
            "mean": metrics_df[metric].mean(),
            "std": metrics_df[metric].std(),
        }
    )

summary_df = pd.DataFrame(summary_rows)

print("\n" + "=" * 65)
print("FINAL NESTED CROSS-VALIDATION RESULTS")
print("=" * 65)

print(summary_df.to_string(index=False))

print("\nBest parameters per fold:")
print(parameters_df.to_string(index=False))


# ============================================================
# 10. COEFFICIENT STABILITY
# ============================================================

coefficient_stability = (
    coefficients_df
    .groupby("feature")
    .agg(
        mean_coefficient=("coefficient", "mean"),
        std_coefficient=("coefficient", "std"),
        mean_abs_coefficient=("abs_coefficient", "mean"),
        folds_present=("outer_fold", "nunique"),
    )
    .reset_index()
)

sign_consistency = (
    coefficients_df
    .assign(sign=lambda x: np.sign(x["coefficient"]))
    .groupby("feature")["sign"]
    .agg(
        lambda x: max(
            (x > 0).sum(),
            (x < 0).sum(),
        ) / len(x)
    )
    .rename("sign_consistency")
    .reset_index()
)

coefficient_stability = coefficient_stability.merge(
    sign_consistency,
    on="feature",
    how="left",
)

coefficient_stability = coefficient_stability.sort_values(
    "mean_abs_coefficient",
    ascending=False,
)


# ============================================================
# 11. SAVE FINAL OUTPUTS
# ============================================================

summary_df.to_csv(
    OUTPUT_DIR / "nested_cv_summary.csv",
    index=False,
)

coefficient_stability.to_csv(
    OUTPUT_DIR / "coefficient_stability.csv",
    index=False,
)

save_current_results()


# ============================================================
# 12. FINAL REPORT
# ============================================================

print("\n" + "=" * 65)
print("SAVED OUTPUT FILES")
print("=" * 65)

for file_path in sorted(OUTPUT_DIR.glob("*.csv")):
    print(file_path)

print("\nNested Cross-Validation completed successfully.")
