import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

df = pd.read_csv("data/processed/aneux_dome_dataset.csv")
df = df.dropna(subset=["patientID"]).copy().reset_index(drop=True)

morphology = pd.read_csv(
    "data/raw/aneux/data/per_source/area-005/morpho-per-cut-flat.csv"
)

y = df["status"].map({
    "unruptured": 0,
    "ruptured": 1
}).astype(int)

groups = df["patientID"]

excluded = {
    "source", "dataset", "cutType", "status",
    "patientID", "vesselFileID", "hospital",
    "cutToShow", "age", "sex", "location", "side"
}

morph_features = [
    c for c in morphology.columns
    if c not in excluded
    and c in df.columns
    and pd.api.types.is_numeric_dtype(df[c])
]

feature_sets = {
    "combined_with_location": [
        "age", "sex", "location", "side"
    ] + morph_features,

    "combined_without_location": [
        "age", "sex", "side"
    ] + morph_features
}

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

for name, features in feature_sets.items():
    print(f"\nRunning {name}")

    X = df[features]

    numeric = [
        c for c in features
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical = [c for c in features if c not in numeric]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    for fold, (train, test) in enumerate(
        cv.split(X, y, groups), start=1
    ):
        assert set(groups.iloc[train]).isdisjoint(
            set(groups.iloc[test])
        )

        model = Pipeline([
            ("preprocessing", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42
            ))
        ])

        model.fit(X.iloc[train], y.iloc[train])

        probabilities = model.predict_proba(X.iloc[test])[:, 1]

        row = {
            "model": name,
            "fold": fold,
            "roc_auc": roc_auc_score(
                y.iloc[test], probabilities
            ),
            "pr_auc": average_precision_score(
                y.iloc[test], probabilities
            )
        }

        results.append(row)

        print(
            f"Fold {fold}: "
            f"ROC-AUC={row['roc_auc']:.4f}, "
            f"PR-AUC={row['pr_auc']:.4f}"
        )

output = Path("results")
output.mkdir(exist_ok=True)

result_df = pd.DataFrame(results)
result_df.to_csv(
    output / "source_sensitivity_folds.csv",
    index=False
)

summary = result_df.groupby("model")[
    ["roc_auc", "pr_auc"]
].agg(["mean", "std"]).round(4)

summary.to_csv(output / "source_sensitivity_summary.csv")

print("\nSENSITIVITY SUMMARY")
print(summary)
