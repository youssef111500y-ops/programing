import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_csv("data/processed/aneux_dome_dataset.csv")
morphology = pd.read_csv(
    "data/raw/aneux/data/per_source/area-005/morpho-per-cut-flat.csv"
)

# Keep records with known patient IDs for consistency
df = df.dropna(subset=["patientID"]).copy()

excluded = {
    "source", "dataset", "cutType",
    "status", "patientID", "vesselFileID",
    "hospital", "location", "side", "sex",
    "age", "cutToShow"
}

features = [
    c for c in morphology.columns
    if c not in excluded
    and c in df.columns
    and pd.api.types.is_numeric_dtype(df[c])
]

X = df[features]

audit = pd.DataFrame({
    "feature": features,
    "missing_pct": X.isna().mean().values * 100,
    "unique_values": X.nunique(dropna=True).values,
    "std": X.std().values
})

audit["constant"] = audit["unique_values"] <= 1

corr = X.corr().abs()
upper = corr.where(
    np.triu(np.ones(corr.shape), k=1).astype(bool)
)

high_corr = (
    upper.stack()
    .reset_index()
    .rename(columns={
        "level_0": "feature_1",
        "level_1": "feature_2",
        0: "correlation"
    })
    .query("correlation >= 0.95")
    .sort_values("correlation", ascending=False)
)

Path("results").mkdir(exist_ok=True)

audit.to_csv("results/morphology_feature_audit.csv", index=False)
high_corr.to_csv("results/highly_correlated_features.csv", index=False)

print("Total morphology features:", len(features))
print("Constant features:", audit["constant"].sum())
print("Features with missing values:", (audit["missing_pct"] > 0).sum())
print("Features with >20% missing:", (audit["missing_pct"] > 20).sum())
print("Highly correlated pairs (>=0.95):", len(high_corr))

print("\nTop missing features:")
print(audit.sort_values("missing_pct", ascending=False)
      .head(15).to_string(index=False))

print("\nTop correlated pairs:")
print(high_corr.head(20).to_string(index=False))

print("\nSaved audit reports in results/")
