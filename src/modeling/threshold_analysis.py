
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
)
from sklearn.isotonic import IsotonicRegression


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = Path("data/processed/aneux_dome_dataset.csv")
OUTPUT_DIR = Path("results/threshold_analysis")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
OUTER_SPLITS = 5
INNER_SPLITS = 4

THRESHOLDS = np.arange(0.20, 0.81, 0.01)

PARAM_GRID = {
    "classifier__C": [0.001, 0.01, 0.03, 0.1, 0.3, 1.0],
    "classifier__l1_ratio": [0.0, 0.5, 1.0],
}


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def calculate_ece(y_true, probabilities, n_bins=8):

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    edges = np.quantile(
        probabilities,
        np.linspace(0, 1, n_bins + 1)
    )

    edges[0] = 0.0
    edges[-1] = 1.0

    ece = 0.0

    for i in range(n_bins):

        if i == n_bins - 1:
            mask = (
                (probabilities >= edges[i])
                & (probabilities <= edges[i + 1])
            )
        else:
            mask = (
                (probabilities >= edges[i])
                & (probabilities < edges[i + 1])
            )

        if mask.sum() == 0:
            continue

        confidence = probabilities[mask].mean()
        observed = y_true[mask].mean()

        ece += (
            mask.sum() / len(y_true)
        ) * abs(confidence - observed)

    return ece


def calculate_metrics(y_true, probabilities, threshold):

    predictions = (
        np.asarray(probabilities) >= threshold
    ).astype(int)

    return {
        "threshold": threshold,
        "balanced_accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "roc_auc": roc_auc_score(
            y_true, probabilities
        ),
        "pr_auc": average_precision_score(
            y_true, probabilities
        ),
        "brier_score": brier_score_loss(
            y_true, probabilities
        ),
        "ece": calculate_ece(
            y_true, probabilities
        ),
    }


def choose_threshold(y_true, probabilities):

    results = []

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = balanced_accuracy_score(
            y_true,
            predictions,
        )

        results.append({
            "threshold": threshold,
            "balanced_accuracy": score,
        })

    results_df = pd.DataFrame(results)

    # In a tie, prefer the threshold closest to 0.5.
    best_score = results_df["balanced_accuracy"].max()

    tied = results_df[
        np.isclose(
            results_df["balanced_accuracy"],
            best_score,
        )
    ].copy()

    tied["distance_from_0_5"] = (
        tied["threshold"] - 0.5
    ).abs()

    best_row = tied.sort_values(
        "distance_from_0_5"
    ).iloc[0]

    return float(best_row["threshold"]), results_df


def fit_platt(y_true, probabilities):

    probabilities = np.clip(
        probabilities,
        1e-6,
        1 - 1e-6,
    )

    logits = np.log(
        probabilities / (1 - probabilities)
    ).reshape(-1, 1)

    calibrator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    calibrator.fit(logits, y_true)

    return calibrator


def apply_platt(calibrator, probabilities):

    probabilities = np.clip(
        probabilities,
        1e-6,
        1 - 1e-6,
    )

    logits = np.log(
        probabilities / (1 - probabilities)
    ).reshape(-1, 1)

    return calibrator.predict_proba(logits)[:, 1]


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 65)
print("THRESHOLD ANALYSIS")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

status_mapping = {
    "unruptured": 0,
    "ruptured": 1,
}

status_text = (
    df["status"]
    .astype("string")
    .str.strip()
    .str.lower()
)

target = status_text.map(status_mapping)

unknown = status_text.notna() & target.isna()

if unknown.any():
    raise ValueError(
        f"Unknown target labels: "
        f"{status_text[unknown].unique().tolist()}"
    )

df["target"] = target

df = df.dropna(
    subset=["target", "patientID"]
).copy()

df["target"] = df["target"].astype(int)
df = df.reset_index(drop=True)

print("Dataset:", df.shape)
print("Patients:", df["patientID"].nunique())


# ============================================================
# 4. FEATURES
# ============================================================

clinical_numeric = ["age"]

clinical_categorical = [
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

numeric_features = (
    clinical_numeric + morphology_features
)

categorical_features = clinical_categorical

feature_columns = (
    numeric_features + categorical_features
)

X = df[feature_columns].copy()
y = df["target"].copy()
groups = df["patientID"].copy()

print("Morphology features:", len(morphology_features))
print("Total features:", len(feature_columns))


# ============================================================
# 5. MODEL PIPELINE
# ============================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "encoder",
        OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
        ),
    ),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

model_pipeline = Pipeline([
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
])


# ============================================================
# 6. OUTER CROSS-VALIDATION
# ============================================================

outer_cv = StratifiedGroupKFold(
    n_splits=OUTER_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

fold_results = []
all_predictions = []
all_threshold_curves = []

for fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(X, y, groups),
    start=1,
):

    print("\n" + "=" * 65)
    print(f"OUTER FOLD {fold}/{OUTER_SPLITS}")
    print("=" * 65)

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]

    if set(groups_train).intersection(
        set(groups.iloc[test_idx])
    ):
        raise RuntimeError("Patient leakage detected.")

    # --------------------------------------------------------
    # 6.1 INNER HYPERPARAMETER TUNING
    # --------------------------------------------------------

    inner_cv = StratifiedGroupKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE + fold,
    )

    inner_splits = list(
        inner_cv.split(
            X_train,
            y_train,
            groups_train,
        )
    )

    grid = GridSearchCV(
        estimator=model_pipeline,
        param_grid=PARAM_GRID,
        scoring="roc_auc",
        cv=inner_splits,
        refit=True,
        n_jobs=-1,
        error_score="raise",
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Found unknown categories",
        )
        grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    print("Best parameters:", grid.best_params_)

    # --------------------------------------------------------
    # 6.2 TRAINING OOF PROBABILITIES
    # --------------------------------------------------------

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Found unknown categories",
        )

        train_oof_probabilities = cross_val_predict(
            clone(best_model),
            X_train,
            y_train,
            cv=inner_splits,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

    # --------------------------------------------------------
    # 6.3 FIT PLATT CALIBRATOR ON TRAINING ONLY
    # --------------------------------------------------------

    platt_calibrator = fit_platt(
        y_train.values,
        train_oof_probabilities,
    )

    train_platt_probabilities = apply_platt(
        platt_calibrator,
        train_oof_probabilities,
    )

    # --------------------------------------------------------
    # 6.4 SELECT THRESHOLD ON TRAINING ONLY
    # --------------------------------------------------------

    best_threshold, threshold_curve = choose_threshold(
        y_train.values,
        train_platt_probabilities,
    )

    threshold_curve["outer_fold"] = fold

    all_threshold_curves.append(threshold_curve)

    print(
        "Selected training threshold:",
        round(best_threshold, 3),
    )

    # --------------------------------------------------------
    # 6.5 OUTER TEST PREDICTIONS
    # --------------------------------------------------------

    original_test_probabilities = (
        best_model.predict_proba(X_test)[:, 1]
    )

    platt_test_probabilities = apply_platt(
        platt_calibrator,
        original_test_probabilities,
    )

    # --------------------------------------------------------
    # 6.6 EVALUATE
    # --------------------------------------------------------

    original_metrics = calculate_metrics(
        y_test.values,
        original_test_probabilities,
        0.5,
    )

    platt_05_metrics = calculate_metrics(
        y_test.values,
        platt_test_probabilities,
        0.5,
    )

    platt_tuned_metrics = calculate_metrics(
        y_test.values,
        platt_test_probabilities,
        best_threshold,
    )

    for method, metrics in [
        ("original_0.5", original_metrics),
        ("platt_0.5", platt_05_metrics),
        ("platt_tuned", platt_tuned_metrics),
    ]:

        fold_results.append({
            "outer_fold": fold,
            "method": method,
            **metrics,
        })

        print(f"\n{method}")

        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

    # --------------------------------------------------------
    # 6.7 SAVE PREDICTIONS
    # --------------------------------------------------------

    predictions = pd.DataFrame({
        "row_index": test_idx,
        "outer_fold": fold,
        "patientID": groups.iloc[test_idx].values,
        "y_true": y_test.values,
        "original_probability": original_test_probabilities,
        "platt_probability": platt_test_probabilities,
        "selected_threshold": best_threshold,
        "original_prediction": (
            original_test_probabilities >= 0.5
        ).astype(int),
        "platt_prediction_05": (
            platt_test_probabilities >= 0.5
        ).astype(int),
        "platt_prediction_tuned": (
            platt_test_probabilities >= best_threshold
        ).astype(int),
    })

    all_predictions.append(predictions)

    # Save after each fold.

    pd.DataFrame(fold_results).to_csv(
        OUTPUT_DIR / "threshold_metrics_per_fold.csv",
        index=False,
    )

    pd.concat(
        all_predictions,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR / "threshold_outer_predictions.csv",
        index=False,
    )

    pd.concat(
        all_threshold_curves,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR / "threshold_training_curves.csv",
        index=False,
    )

    print(f"\nFold {fold} saved.")


# ============================================================
# 7. FINAL SUMMARY
# ============================================================

results_df = pd.DataFrame(fold_results)

summary = (
    results_df
    .groupby("method")
    .agg(
        balanced_accuracy_mean=(
            "balanced_accuracy", "mean"
        ),
        balanced_accuracy_std=(
            "balanced_accuracy", "std"
        ),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
        brier_mean=("brier_score", "mean"),
        brier_std=("brier_score", "std"),
        ece_mean=("ece", "mean"),
        ece_std=("ece", "std"),
    )
    .reset_index()
)

summary.to_csv(
    OUTPUT_DIR / "threshold_analysis_summary.csv",
    index=False,
)

print("\n" + "=" * 65)
print("FINAL THRESHOLD ANALYSIS")
print("=" * 65)

print(summary.to_string(index=False))


# ============================================================
# 8. THRESHOLD STABILITY
# ============================================================

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True,
)

threshold_stability = (
    predictions_df
    .groupby("outer_fold")["selected_threshold"]
    .first()
    .reset_index()
)

print("\nSelected threshold per fold:")
print(threshold_stability.to_string(index=False))

print(
    "\nMean selected threshold:",
    round(
        threshold_stability["selected_threshold"].mean(),
        4,
    ),
)

print(
    "Threshold standard deviation:",
    round(
        threshold_stability["selected_threshold"].std(),
        4,
    ),
)

threshold_stability.to_csv(
    OUTPUT_DIR / "threshold_stability.csv",
    index=False,
)

print("\nThreshold analysis completed successfully.")
