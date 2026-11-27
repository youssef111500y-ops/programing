from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ==========================================
# Configuration
# ==========================================

CASE_ID = "stable_0"

INPUT_PATH = Path(
    f"data/raw/aneug_flow/{CASE_ID}/"
    f"transient_data/{CASE_ID}/wall_data.pt"
)

OUTPUT_DIR = Path("results/aneug_flow")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / f"{CASE_ID}_wss_metrics.csv"

EPSILON = 1e-12


# ==========================================
# Load data
# ==========================================

print("Loading wall data...")

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"File not found: {INPUT_PATH}")

data = torch.load(
    INPUT_PATH,
    map_location="cpu",
    weights_only=True
)

required_keys = [
    "x_coordinate",
    "y_coordinate",
    "z_coordinate",
    "x_wall_shear",
    "y_wall_shear",
    "z_wall_shear",
    "wall_shear",
]

missing = [key for key in required_keys if key not in data]

if missing:
    raise KeyError(f"Missing variables: {missing}")


def extract_tensor(name):
    return (
        data[name]
        .squeeze(-1)
        .numpy()
        .astype(np.float64)
    )


x = extract_tensor("x_coordinate")
y = extract_tensor("y_coordinate")
z = extract_tensor("z_coordinate")

wss_x = extract_tensor("x_wall_shear")
wss_y = extract_tensor("y_wall_shear")
wss_z = extract_tensor("z_wall_shear")

provided_wss = extract_tensor("wall_shear")


# ==========================================
# Validate data
# ==========================================

shapes = [
    x.shape, y.shape, z.shape,
    wss_x.shape, wss_y.shape, wss_z.shape,
    provided_wss.shape
]

if len(set(shapes)) != 1:
    raise ValueError(f"Tensor shape mismatch: {shapes}")

if x.ndim != 2:
    raise ValueError(f"Expected [time, nodes], got {x.shape}")

num_frames, num_nodes = x.shape

if num_frames != 80:
    raise ValueError(f"Expected 80 frames, got {num_frames}")

for name, array in {
    "x": x,
    "y": y,
    "z": z,
    "wss_x": wss_x,
    "wss_y": wss_y,
    "wss_z": wss_z,
    "provided_wss": provided_wss,
}.items():
    if not np.isfinite(array).all():
        raise ValueError(f"NaN or Inf found in {name}")

# Coordinates should remain fixed across frames.
for name, coordinates in {"x": x, "y": y, "z": z}.items():
    if not np.allclose(coordinates, coordinates[0:1]):
        raise ValueError(f"{name} coordinates change over time")

print(f"Frames: {num_frames}")
print(f"Wall nodes: {num_nodes}")


# ==========================================
# WSS vector magnitude
# ==========================================

wss_magnitude = np.sqrt(
    wss_x**2 + wss_y**2 + wss_z**2
)


# ==========================================
# TAWSS
# ==========================================

tawss = np.mean(wss_magnitude, axis=0)


# ==========================================
# OSI
# ==========================================

mean_wss_x = np.mean(wss_x, axis=0)
mean_wss_y = np.mean(wss_y, axis=0)
mean_wss_z = np.mean(wss_z, axis=0)

magnitude_of_mean_wss = np.sqrt(
    mean_wss_x**2 +
    mean_wss_y**2 +
    mean_wss_z**2
)

mean_magnitude_wss = np.mean(
    wss_magnitude,
    axis=0
)

osi = np.full(num_nodes, np.nan)

valid_osi = mean_magnitude_wss > EPSILON

osi[valid_osi] = 0.5 * (
    1 -
    magnitude_of_mean_wss[valid_osi]
    / mean_magnitude_wss[valid_osi]
)

osi[valid_osi] = np.clip(
    osi[valid_osi],
    0.0,
    0.5
)


# ==========================================
# RRT
# ==========================================

rrt_denominator = (1 - 2 * osi) * tawss

rrt = np.full(num_nodes, np.nan)

valid_rrt = (
    np.isfinite(rrt_denominator)
    & (rrt_denominator > EPSILON)
)

rrt[valid_rrt] = 1.0 / rrt_denominator[valid_rrt]


# ==========================================
# WSS consistency comparison
# ==========================================

wss_difference = np.abs(
    provided_wss - wss_magnitude
)

mean_provided_wss = np.mean(
    provided_wss,
    axis=0
)


# ==========================================
# Results
# ==========================================

results = pd.DataFrame({
    "node_id": np.arange(num_nodes),
    "x": x[0],
    "y": y[0],
    "z": z[0],
    "TAWSS": tawss,
    "OSI": osi,
    "RRT": rrt,
    "mean_wss_x": mean_wss_x,
    "mean_wss_y": mean_wss_y,
    "mean_wss_z": mean_wss_z,
    "mean_vector_wss_magnitude": magnitude_of_mean_wss,
    "mean_provided_wall_shear": mean_provided_wss,
    "mean_wss_magnitude_difference": np.mean(
        wss_difference,
        axis=0
    ),
})

results.to_csv(OUTPUT_PATH, index=False)


# ==========================================
# Summary
# ==========================================

print("\n========== RESULTS ==========")
print("Saved:", OUTPUT_PATH)
print("Rows:", len(results))
print("Columns:", len(results.columns))

for metric in ["TAWSS", "OSI", "RRT"]:
    values = results[metric].dropna()

    print(f"\n{metric}")
    print("Valid:", len(values))
    print("Min:", values.min())
    print("Max:", values.max())
    print("Mean:", values.mean())
    print("Median:", values.median())

print("\nInvalid OSI:", results["OSI"].isna().sum())
print("Invalid RRT:", results["RRT"].isna().sum())

print("\nFirst 5 rows:")
print(results.head().to_string(index=False))

print("\nCalculation completed successfully.")
