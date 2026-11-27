import pandas as pd
from pathlib import Path
from scipy.stats import chi2_contingency

df = pd.read_csv("data/processed/aneux_dome_dataset.csv")

# Same population used in the primary model evaluation
df = df.dropna(subset=["patientID"]).copy()

df["target"] = df["status"].map({
    "unruptured": 0,
    "ruptured": 1
})

output = Path("results/source_audit")
output.mkdir(parents=True, exist_ok=True)

for col in ["source", "hospital", "location", "sex"]:
    print("\n" + "=" * 60)
    print(f"AUDIT: {col}")
    print("=" * 60)

    if col not in df.columns:
        print("Column not found.")
        continue

    summary = (
        df.groupby(col, dropna=False)
        .agg(
            records=("target", "size"),
            patients=("patientID", "nunique"),
            ruptured=("target", "sum"),
            rupture_rate=("target", "mean")
        )
        .sort_values("records", ascending=False)
    )

    summary["rupture_rate"] = (
        summary["rupture_rate"] * 100
    ).round(2)

    print(summary.to_string())

    # Chi-square association test
    table = pd.crosstab(df[col].fillna("Missing"), df["target"])

    if table.shape[0] > 1 and table.shape[1] > 1:
        chi2, p_value, dof, expected = chi2_contingency(table)

        n = table.to_numpy().sum()
        min_dim = min(table.shape) - 1

        cramers_v = (
            (chi2 / (n * min_dim)) ** 0.5
            if min_dim > 0 else 0
        )

        print(f"\nChi-square p-value: {p_value:.6g}")
        print(f"Cramer's V: {cramers_v:.4f}")

    summary.to_csv(output / f"{col}_summary.csv")

print("\nUnique sources:", df["source"].nunique())
print("Unique hospitals:", df["hospital"].nunique())
print("Unique patients:", df["patientID"].nunique())
print("\nReports saved in results/source_audit/")
