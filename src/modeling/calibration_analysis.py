
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

INPUT_PATH = Path("results/nested_cv/outer_fold_predictions.csv")
OUTPUT_DIR = Path("results/calibration")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_BINS = 8
N_BOOTSTRAPS = 1000
RANDOM_STATE = 42


# --------------------------------------------------
# 2. Load predictions
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("Dataset shape:", df.shape)
print("Available columns:", df.columns.tolist())

# Detect column names
def find_column(candidates, columns):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"Could not find any of these columns: {candidates}"
    )


target_col = find_column(
    ["y_true", "target", "status", "rupture_status"],
    df.columns
)

prob_col = find_column(
    ["y_prob", "y_probability", "probability", "y_pred_proba"],
    df.columns
)

patient_col = find_column(
    ["patientID", "patient_id", "patientId"],
    df.columns
)

print("\nSelected columns:")
print("Target:", target_col)
print("Probability:", prob_col)
print("Patient:", patient_col)


# --------------------------------------------------
# 3. Prepare data
# --------------------------------------------------

data = df[[target_col, prob_col, patient_col]].copy()

data = data.dropna()

data[target_col] = data[target_col].astype(int)
data[prob_col] = data[prob_col].astype(float)

# Keep valid binary targets and probabilities
data = data[
    data[target_col].isin([0, 1])
    & data[prob_col].between(0, 1)
]

y_true = data[target_col].to_numpy()
y_prob = data[prob_col].to_numpy()

print("\nRecords:", len(data))
print("Unique patients:", data[patient_col].nunique())
print("Class distribution:")
print(data[target_col].value_counts())


# --------------------------------------------------
# 4. Overall calibration metrics
# --------------------------------------------------

brier = brier_score_loss(y_true, y_prob)

fraction_positive, mean_predicted = calibration_curve(
    y_true,
    y_prob,
    n_bins=N_BINS,
    strategy="quantile"
)

# Expected Calibration Error
bin_ids = pd.qcut(
    data[prob_col],
    q=N_BINS,
    labels=False,
    duplicates="drop"
)

ece = 0.0

for bin_id in sorted(bin_ids.dropna().unique()):
    mask = bin_ids == bin_id

    observed_rate = data.loc[mask, target_col].mean()
    predicted_rate = data.loc[mask, prob_col].mean()

    ece += (
        mask.sum() / len(data)
    ) * abs(observed_rate - predicted_rate)


print("\nCalibration Metrics")
print("-------------------")
print(f"Brier Score: {brier:.4f}")
print(f"ECE:         {ece:.4f}")


# --------------------------------------------------
# 5. Patient-level bootstrap confidence intervals
# --------------------------------------------------

rng = np.random.default_rng(RANDOM_STATE)

patients = data[patient_col].unique()

bootstrap_brier = []
bootstrap_ece = []

for _ in range(N_BOOTSTRAPS):

    sampled_patients = rng.choice(
        patients,
        size=len(patients),
        replace=True
    )

    # Keep all aneurysms belonging to each sampled patient.
    # Repeated sampled patients are intentionally repeated.
    sampled_frames = [
        data[data[patient_col] == patient]
        for patient in sampled_patients
    ]

    sample = pd.concat(
        sampled_frames,
        ignore_index=True
    )

    if sample[target_col].nunique() < 2:
        continue

    sample_y = sample[target_col].to_numpy()
    sample_prob = sample[prob_col].to_numpy()

    bootstrap_brier.append(
        brier_score_loss(sample_y, sample_prob)
    )

    sample_bins = pd.qcut(
        sample[prob_col],
        q=N_BINS,
        labels=False,
        duplicates="drop"
    )

    sample_ece = 0.0

    for bin_id in sorted(sample_bins.dropna().unique()):
        mask = sample_bins == bin_id

        observed = sample.loc[mask, target_col].mean()
        predicted = sample.loc[mask, prob_col].mean()

        sample_ece += (
            mask.sum() / len(sample)
        ) * abs(observed - predicted)

    bootstrap_ece.append(sample_ece)


def confidence_interval(values):
    return np.percentile(values, [2.5, 97.5])


brier_ci = confidence_interval(bootstrap_brier)
ece_ci = confidence_interval(bootstrap_ece)

print("\nPatient-level Bootstrap 95% CI")
print("--------------------------------")
print(
    f"Brier Score: {brier:.4f} "
    f"({brier_ci[0]:.4f} - {brier_ci[1]:.4f})"
)
print(
    f"ECE:         {ece:.4f} "
    f"({ece_ci[0]:.4f} - {ece_ci[1]:.4f})"
)


# --------------------------------------------------
# 6. Save metrics
# --------------------------------------------------

metrics = pd.DataFrame([
    {
        "metric": "Brier Score",
        "value": brier,
        "ci_lower": brier_ci[0],
        "ci_upper": brier_ci[1]
    },
    {
        "metric": "ECE",
        "value": ece,
        "ci_lower": ece_ci[0],
        "ci_upper": ece_ci[1]
    }
])

metrics.to_csv(
    OUTPUT_DIR / "calibration_metrics.csv",
    index=False
)


# --------------------------------------------------
# 7. Calibration plot
# --------------------------------------------------

plt.figure(figsize=(7, 7))

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

plt.plot(
    mean_predicted,
    fraction_positive,
    marker="o",
    label="Model calibration"
)

plt.xlabel("Mean predicted probability")
plt.ylabel("Observed positive fraction")
plt.title("Calibration Curve - Nested Cross-Validation")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "calibration_curve.png",
    dpi=300
)

plt.close()


print("\nCalibration analysis completed.")
print("Results saved in:", OUTPUT_DIR)
