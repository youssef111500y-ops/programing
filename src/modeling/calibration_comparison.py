
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from sklearn.model_selection import (
    StratifiedGroupKFold,
    GridSearchCV,
    cross_val_predict,
)

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = Path("data/processed/aneux_dome_dataset.csv")
OUTPUT_DIR = Path("results/calibration_comparison")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
OUTER_SPLITS = 5
INNER_SPLITS = 4
N_BINS = 8

PARAM_GRID = {
    "classifier__C": [0.001, 0.01, 0.03, 0.1, 0.3, 1.0],
    "classifier__l1_ratio": [0.0, 0.5, 1.0],
}


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def expected_calibration_error(y_true, probabilities, n_bins=8):
    """
    Calculate Expected Calibration Error using quantile bins.
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    edges = np.quantile(
        probabilities,
        np.linspace(0, 1, n_bins + 1)
    )

    edges[0] = 0.0
    edges[-1] = 1.0

    ece = 0.0
    bin_details = []

    for i in range(n_bins):

        lower = edges[i]
        upper = edges[i + 1]

        if i == n_bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        if not mask.any():
            continue

        mean_probability = probabilities[mask].mean()
        observed_fraction = y_true[mask].mean()

        count = mask.sum()

        ece += (
            count / len(y_true)
        ) * abs(
            mean_probability - observed_fraction
        )

        bin_details.append({
            "bin": i + 1,
            "count": count,
            "mean_probability": mean_probability,
            "observed_fraction": observed_fraction,
        })

    return ece, bin_details


def calculate_metrics(y_true, probabilities):

    probabilities = np.clip(
        np.asarray(probabilities),
        0.000001,
        0.999999,
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    ece, _ = expected_calibration_error(
        y_true,
        probabilities,
        N_BINS,
    )

    return {
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
        "ece": ece,
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions,
        ),
    }


def fit_platt_scaling(y_train, probabilities):
    """
    Platt scaling using logistic regression on logit probabilities.
    """

    probabilities = np.clip(
        probabilities,
        0.000001,
        0.999999,
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

    calibrator.fit(logits, y_train)

    return calibrator


def apply_platt_scaling(calibrator, probabilities):

    probabilities = np.clip(
        probabilities,
        0.000001,
        0.999999,
    )

    logits = np.log(
        probabilities / (1 - probabilities)
    ).reshape(-1, 1)

    return calibrator.predict_proba(logits)[:, 1]


def plot_calibration_curves(predictions_df):

    methods = [
        "original",
        "platt",
        "isotonic",
    ]

    colors = {
        "original": "tab:blue",
        "platt": "tab:orange",
        "isotonic": "tab:green",
    }

    labels = {
        "original": "Original Model",
        "platt": "Platt Scaling",
        "isotonic": "Isotonic Regression",
    }

    plt.figure(figsize=(8, 8))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        label="Perfect Calibration",
    )

    for method in methods:

        mean_predicted = []
        observed_fraction = []

        for fold in sorted(
            predictions_df["outer_fold"].unique()
        ):

            fold_data = predictions_df[
                predictions_df["outer_fold"] == fold
            ]

            probabilities = fold_data[
                f"{method}_probability"
            ].values

            actual = fold_data["y_true"].values

            edges = np.quantile(
                probabilities,
                np.linspace(0, 1, N_BINS + 1)
            )

            edges[0] = 0.0
            edges[-1] = 1.0

            for i in range(N_BINS):

                if i == N_BINS - 1:
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

                mean_predicted.append(
                    probabilities[mask].mean()
                )

                observed_fraction.append(
                    actual[mask].mean()
                )

        plt.plot(
            mean_predicted,
            observed_fraction,
            marker="o",
            linewidth=1.5,
            color=colors[method],
            label=labels[method],
        )

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Positive Fraction")
    plt.title("Calibration Comparison - Nested Cross-Validation")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "calibration_comparison.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# 3. LOAD AND PREPARE DATA
# ============================================================

print("=" * 65)
print("CALIBRATION COMPARISON")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

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
    raise ValueError(
        f"Unknown status labels: "
        f"{status_text[unknown_labels].unique().tolist()}"
    )

df = df.copy()
df["target"] = target

df = df.dropna(
    subset=["target", "patientID"]
).copy()

df["target"] = df["target"].astype(int)
df = df.reset_index(drop=True)

print("Dataset shape:", df.shape)
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
        f"Missing required columns: {missing_columns}"
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

print("\nMorphology features:", len(morphology_features))
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
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ],
    remainder="drop",
)

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
# 6. OUTER CROSS-VALIDATION
# ============================================================

outer_cv = StratifiedGroupKFold(
    n_splits=OUTER_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

all_predictions = []
all_metrics = []
all_parameters = []

print("\nStarting calibrated nested cross-validation...")


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

    if train_patients.intersection(test_patients):
        raise RuntimeError(
            f"Patient leakage in outer fold {outer_fold}"
        )

    # --------------------------------------------------------
    # 6.1 INNER CV SPLITS
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
    # 6.2 HYPERPARAMETER TUNING
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

    print("Best parameters:", grid_search.best_params_)
    print(
        "Best inner ROC-AUC:",
        round(grid_search.best_score_, 4),
    )

    # --------------------------------------------------------
    # 6.3 INNER OUT-OF-FOLD PROBABILITIES
    # --------------------------------------------------------
    # These predictions are generated without training on
    # the corresponding validation fold.
    #
    # The outer test fold is never used for fitting a calibrator.

    print("Generating inner out-of-fold probabilities...")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Found unknown categories",
        )

        oof_probabilities = cross_val_predict(
            clone(best_model),
            X_train,
            y_train,
            cv=inner_splits,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

    # --------------------------------------------------------
    # 6.4 FIT CALIBRATORS USING TRAINING DATA ONLY
    # --------------------------------------------------------

    platt_calibrator = fit_platt_scaling(
        y_train.values,
        oof_probabilities,
    )

    isotonic_calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    isotonic_calibrator.fit(
        oof_probabilities,
        y_train.values,
    )

    # --------------------------------------------------------
    # 6.5 OUTER TEST PREDICTIONS
    # --------------------------------------------------------

    original_probabilities = best_model.predict_proba(
        X_test
    )[:, 1]

    platt_probabilities = apply_platt_scaling(
        platt_calibrator,
        original_probabilities,
    )

    isotonic_probabilities = isotonic_calibrator.predict(
        original_probabilities
    )

    probability_methods = {
        "original": original_probabilities,
        "platt": platt_probabilities,
        "isotonic": isotonic_probabilities,
    }

    # --------------------------------------------------------
    # 6.6 EVALUATE EACH METHOD
    # --------------------------------------------------------

    for method, probabilities in probability_methods.items():

        metrics = calculate_metrics(
            y_test.values,
            probabilities,
        )

        metrics["outer_fold"] = outer_fold
        metrics["method"] = method

        all_metrics.append(metrics)

        print(f"\n{method.upper()}")

        for metric_name, metric_value in metrics.items():

            if metric_name in {
                "outer_fold",
                "method",
            }:
                continue

            print(
                f"{metric_name}: {metric_value:.4f}"
            )

    # --------------------------------------------------------
    # 6.7 SAVE PREDICTIONS
    # --------------------------------------------------------

    fold_predictions = pd.DataFrame({
        "row_index": test_idx,
        "outer_fold": outer_fold,
        "patientID": groups.iloc[test_idx].values,
        "y_true": y_test.values,
        "original_probability": original_probabilities,
        "platt_probability": platt_probabilities,
        "isotonic_probability": isotonic_probabilities,
    })

    all_predictions.append(fold_predictions)

    all_parameters.append({
        "outer_fold": outer_fold,
        **grid_search.best_params_,
        "inner_best_roc_auc": grid_search.best_score_,
    })

    # Save after every completed fold.

    pd.concat(
        all_predictions,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR / "calibrated_outer_predictions.csv",
        index=False,
    )

    pd.DataFrame(all_metrics).to_csv(
        OUTPUT_DIR / "calibration_metrics_per_fold.csv",
        index=False,
    )

    pd.DataFrame(all_parameters).to_csv(
        OUTPUT_DIR / "calibration_best_parameters.csv",
        index=False,
    )

    print(f"\nFold {outer_fold} saved successfully.")


# ============================================================
# 7. FINAL METRICS SUMMARY
# ============================================================

metrics_df = pd.DataFrame(all_metrics)

summary = (
    metrics_df
    .groupby("method")
    .agg(
        brier_mean=("brier_score", "mean"),
        brier_std=("brier_score", "std"),
        ece_mean=("ece", "mean"),
        ece_std=("ece", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
    )
    .reset_index()
)

summary.to_csv(
    OUTPUT_DIR / "calibration_comparison_summary.csv",
    index=False,
)

print("\n" + "=" * 65)
print("FINAL CALIBRATION COMPARISON")
print("=" * 65)

print(summary.to_string(index=False))


# ============================================================
# 8. CALIBRATION CURVES
# ============================================================

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True,
)

plot_calibration_curves(predictions_df)


# ============================================================
# 9. FINAL OUTPUTS
# ============================================================

print("\n" + "=" * 65)
print("OUTPUT FILES")
print("=" * 65)

for file_path in sorted(OUTPUT_DIR.glob("*")):
    print(file_path)

print("\nCalibration comparison completed successfully.")
