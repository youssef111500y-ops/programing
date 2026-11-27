
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

DATA_PATH = Path("data/processed/aneux_dome_dataset.csv")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Keep records with known patient IDs and rupture status
df = df.dropna(subset=["patientID", "status"]).copy()

df["target"] = df["status"].map({
    "unruptured": 0,
    "ruptured": 1,
})

df = df.dropna(subset=["target"]).copy()
df["target"] = df["target"].astype(int)


# --------------------------------------------------
# 2. Define features
# --------------------------------------------------

# IMPORTANT: location is intentionally excluded
clinical_features = ["age", "sex", "side"]

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
y = df["target"].copy()
sources = df["source"].copy()
patient_ids = df["patientID"].astype(str)

print("Dataset shape:", df.shape)
print("Number of features:", len(features))
print("Clinical features:", clinical_features)
print("Morphology features:", len(morphology_features))
print("Sources:", sorted(sources.unique()))
print("Unique patients:", patient_ids.nunique())


# --------------------------------------------------
# 3. Preprocessing
# --------------------------------------------------

numeric_features = [
    col for col in features
    if pd.api.types.is_numeric_dtype(X[col])
]

categorical_features = [
    col for col in features
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


# --------------------------------------------------
# 4. Model
# --------------------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=42,
    )),
])


# --------------------------------------------------
# 5. Leave-One-Source-Out Validation
# --------------------------------------------------

results = []

for held_out_source in sorted(sources.unique()):

    print(f"\nTesting on source: {held_out_source}")

    test_mask = sources == held_out_source
    test_indices = df.index[test_mask]

    # Identify patients in the test source
    test_patient_ids = set(patient_ids.loc[test_indices])

    # Test data: all records from the held-out source
    X_test = X.loc[test_indices]
    y_test = y.loc[test_indices]

    # Training data: all other sources
    train_mask = sources != held_out_source

    # Remove any training records belonging to test patients
    train_indices = df.index[
        train_mask & ~patient_ids.isin(test_patient_ids)
    ]

    X_train = X.loc[train_indices]
    y_train = y.loc[train_indices]

    excluded_overlap_records = (
        (sources != held_out_source)
        & patient_ids.isin(test_patient_ids)
    ).sum()

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Overlapping records excluded:", excluded_overlap_records)

    print("Train classes:", y_train.value_counts().to_dict())
    print("Test classes:", y_test.value_counts().to_dict())

    # Check that both classes exist
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        print("Skipped: only one class in train or test.")
        continue

    # Train
    model.fit(X_train, y_train)

    # Predict
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)

    # Evaluate
    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)
    balanced_acc = balanced_accuracy_score(y_test, predictions)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    results.append({
        "held_out_source": held_out_source,
        "train_records": len(X_train),
        "test_records": len(X_test),
        "excluded_overlap_records": excluded_overlap_records,
        "test_ruptured": int(y_test.sum()),
        "test_unruptured": int((y_test == 0).sum()),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "balanced_accuracy": balanced_acc,
    })


# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_path = (
    OUTPUT_DIR / "leave_one_source_out_without_location.csv"
)

results_df.to_csv(results_path, index=False)

print("\nFINAL RESULTS")
print(results_df.to_string(index=False))

if not results_df.empty:
    print("\nMEAN PERFORMANCE")
    print(
        results_df[
            ["roc_auc", "pr_auc", "balanced_accuracy"]
        ].mean().round(4)
    )

print(f"\nSaved to: {results_path}")
