
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

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
STABILITY_PATH = Path("results/shap/shap_feature_stability.csv")

OUTPUT_DIR = Path("results/shap/directionality")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42
TOP_N = 10


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

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
# 3. Define features
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
print("Unique patients:", groups.nunique())
print("Total features:", len(features))
print("SHAP version:", shap.__version__)


# --------------------------------------------------
# 4. CV splits
# --------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

splits = list(cv.split(X, y, groups))


# --------------------------------------------------
# 5. Preprocessing
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
# 6. Calculate signed SHAP values
# --------------------------------------------------

all_shap_records = []
all_feature_values = []

for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):

    print(f"\nProcessing Fold {fold_number}/{N_SPLITS}")

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    preprocessor = build_preprocessor(X_train)

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()

    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    X_train_transformed = np.asarray(X_train_transformed)
    X_test_transformed = np.asarray(X_test_transformed)

    feature_names = preprocessor.get_feature_names_out()

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE,
    )

    classifier.fit(X_train_transformed, y_train)

    # SHAP explanation on the training background
    explainer = shap.LinearExplainer(
        classifier,
        X_train_transformed,
    )

    shap_values = explainer.shap_values(X_test_transformed)
    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    if shap_values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP shape: {shap_values.shape}"
        )

    if shap_values.shape[1] != len(feature_names):
        raise ValueError("SHAP columns do not match feature names.")

    # Save signed SHAP values and transformed feature values
    for row_position, original_index in enumerate(test_idx):

        for feature_position, feature_name in enumerate(feature_names):

            all_shap_records.append({
                "fold": fold_number,
                "row_index": int(original_index),
                "patientID": groups.iloc[original_index],
                "target": int(y.iloc[original_index]),
                "feature": feature_name,
                "shap_value": float(
                    shap_values[row_position, feature_position]
                ),
                "transformed_value": float(
                    X_test_transformed[row_position, feature_position]
                ),
            })

    # Keep arrays for beeswarm plot
    all_feature_values.append({
        "feature_names": feature_names,
        "shap_values": shap_values,
        "X_values": X_test_transformed,
    })

    print("Training records:", len(train_idx))
    print("Testing records:", len(test_idx))
    print("SHAP values calculated:", shap_values.shape)


# --------------------------------------------------
# 7. Save signed SHAP values
# --------------------------------------------------

signed_df = pd.DataFrame(all_shap_records)

signed_path = OUTPUT_DIR / "signed_shap_values.csv"
signed_df.to_csv(signed_path, index=False)


# --------------------------------------------------
# 8. Calculate signed SHAP summary
# --------------------------------------------------

signed_summary = (
    signed_df.groupby("feature")
    .agg(
        mean_signed_shap=("shap_value", "mean"),
        mean_abs_shap=("shap_value", lambda x: np.abs(x).mean()),
        std_signed_shap=("shap_value", "std"),
        sample_count=("shap_value", "count"),
    )
    .reset_index()
    .sort_values("mean_abs_shap", ascending=False)
)

summary_path = OUTPUT_DIR / "signed_shap_summary.csv"
signed_summary.to_csv(summary_path, index=False)


# --------------------------------------------------
# 9. Build aligned data for beeswarm
# --------------------------------------------------

# Feature names may differ between folds because of
# categorical levels. Align all folds by feature name.

all_feature_names = sorted(set(
    name
    for fold_data in all_feature_values
    for name in fold_data["feature_names"]
))

feature_to_position = {
    name: position
    for position, name in enumerate(all_feature_names)
}

shap_rows = []
value_rows = []

for fold_data in all_feature_values:

    fold_names = fold_data["feature_names"]
    fold_shap = fold_data["shap_values"]
    fold_values = fold_data["X_values"]

    aligned_shap = np.zeros(
        (fold_shap.shape[0], len(all_feature_names))
    )

    aligned_values = np.zeros_like(aligned_shap)

    for local_position, feature_name in enumerate(fold_names):

        global_position = feature_to_position[feature_name]

        aligned_shap[:, global_position] = (
            fold_shap[:, local_position]
        )

        aligned_values[:, global_position] = (
            fold_values[:, local_position]
        )

    shap_rows.append(aligned_shap)
    value_rows.append(aligned_values)

combined_shap = np.vstack(shap_rows)
combined_values = np.vstack(value_rows)


# --------------------------------------------------
# 10. SHAP Beeswarm plot
# --------------------------------------------------

explanation = shap.Explanation(
    values=combined_shap,
    data=combined_values,
    feature_names=all_feature_names,
)

plt.figure(figsize=(12, 10))

shap.plots.beeswarm(
    explanation,
    max_display=20,
    show=False,
)

plt.title("SHAP Beeswarm - Combined Model")
plt.tight_layout()

beeswarm_path = OUTPUT_DIR / "shap_beeswarm.png"

plt.savefig(
    beeswarm_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# --------------------------------------------------
# 11. Dependence plots for top 10 stable features
# --------------------------------------------------

stability_df = pd.read_csv(STABILITY_PATH)

top_features = stability_df.head(TOP_N)["feature"].tolist()

dependence_dir = OUTPUT_DIR / "dependence_plots"
dependence_dir.mkdir(exist_ok=True)

for transformed_feature in top_features:

    # Dependence plots here are restricted to
    # original numeric features.
    prefix = "numeric__"

    if not transformed_feature.startswith(prefix):
        print("Skipping non-numeric feature:", transformed_feature)
        continue

    original_feature = transformed_feature[len(prefix):]

    if original_feature not in X.columns:
        print("Feature not found:", original_feature)
        continue

    # Get the OOF SHAP values for this feature
    feature_shap = signed_df[
        signed_df["feature"] == transformed_feature
    ].copy()

    # Attach original, unscaled feature values
    feature_shap["original_value"] = (
        feature_shap["row_index"]
        .map(X[original_feature])
    )

    feature_shap = feature_shap.dropna(
        subset=["original_value", "shap_value"]
    )

    if feature_shap.empty:
        continue

    plt.figure(figsize=(8, 6))

    plt.scatter(
        feature_shap["original_value"],
        feature_shap["shap_value"],
        alpha=0.6,
    )

    plt.axhline(0, linestyle="--")

    plt.xlabel(original_feature)
    plt.ylabel("SHAP value (log-odds)")
    plt.title(f"SHAP Dependence: {original_feature}")

    plt.tight_layout()

    safe_name = original_feature.replace("/", "_")

    plot_path = dependence_dir / f"{safe_name}.png"

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# --------------------------------------------------
# 12. Final report
# --------------------------------------------------

print("\n" + "=" * 65)
print("SIGNED SHAP SUMMARY")
print("=" * 65)

print(
    signed_summary.head(20)
    .round(4)
    .to_string(index=False)
)

print("\nSaved files:")
print(signed_path)
print(summary_path)
print(beeswarm_path)
print(dependence_dir)

print("\nSHAP directionality analysis completed.")
