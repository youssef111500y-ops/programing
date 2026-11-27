from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri


# ==========================================
# Configuration
# ==========================================

CASE_ID = "stable_0"

BASE_DIR = Path(f"data/raw/aneug_flow/{CASE_ID}/transient_data/{CASE_ID}")

REGISTRATION_PATH = BASE_DIR / "wall_mesh_registration.pt"

METRICS_PATH = Path(
    f"results/aneug_flow/{CASE_ID}_wss_metrics.csv"
)

OUTPUT_DIR = Path(
    f"results/aneug_flow/visualizations/{CASE_ID}"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# Load registration mesh
# ==========================================

print("Loading registration mesh...")

registration = torch.load(
    REGISTRATION_PATH,
    map_location="cpu",
    weights_only=True
)

positions = registration["pos"].numpy().astype(np.float64)
faces = registration["faces"].numpy().astype(np.int64)

print("Mesh vertices:", len(positions))
print("Mesh faces:", len(faces))

# Validate mesh indices
if faces.min() < 0 or faces.max() >= len(positions):
    raise ValueError("Mesh contains invalid face indices.")

if positions.shape[1] != 3 or faces.shape[1] != 3:
    raise ValueError("Expected 3D triangular mesh.")

if not np.isfinite(positions).all():
    raise ValueError("Invalid coordinates found in mesh.")


# ==========================================
# Load calculated metrics
# ==========================================

print("\nLoading hemodynamic metrics...")

metrics = pd.read_csv(METRICS_PATH)

if len(metrics) != len(positions):
    raise ValueError(
        f"Node mismatch: metrics={len(metrics)}, "
        f"mesh={len(positions)}"
    )

expected_node_ids = np.arange(len(positions))

if not np.array_equal(
    metrics["node_id"].to_numpy(),
    expected_node_ids
):
    raise ValueError("Node ID ordering does not match registration mesh.")

for metric in ["TAWSS", "OSI", "RRT"]:
    if not np.isfinite(metrics[metric].to_numpy()).all():
        raise ValueError(f"Invalid values in {metric}.")

# Verify that registration coordinates match wall coordinates.
metric_coordinates = metrics[["x", "y", "z"]].to_numpy()

coordinate_difference = np.linalg.norm(
    positions - metric_coordinates,
    axis=1
)

print("\nCoordinate alignment:")
print("Mean difference:", coordinate_difference.mean())
print("Maximum difference:", coordinate_difference.max())

if coordinate_difference.max() > 1e-5:
    raise ValueError(
        "Registration positions do not match metric coordinates. "
        "Do not generate maps until alignment is verified."
    )


# ==========================================
# Prepare triangulation
# ==========================================

triangulation = mtri.Triangulation(
    positions[:, 0],
    positions[:, 1],
    triangles=faces
)

# Remove invalid/degenerate triangles.
triangles = positions[faces]

cross_products = np.cross(
    triangles[:, 1] - triangles[:, 0],
    triangles[:, 2] - triangles[:, 0]
)

areas = np.linalg.norm(cross_products, axis=1) / 2

valid_faces = (
    np.isfinite(areas)
    & (areas > 1e-15)
)

if not valid_faces.all():
    print(
        "Removing degenerate triangles:",
        int((~valid_faces).sum())
    )

    faces = faces[valid_faces]

    triangulation = mtri.Triangulation(
        positions[:, 0],
        positions[:, 1],
        triangles=faces
    )


# ==========================================
# Plot configuration
# ==========================================

metric_configs = {
    "TAWSS": {
        "title": "Time-Averaged Wall Shear Stress",
        "unit": "Source-reported WSS units (unverified)",
        "cmap": "viridis",
    },
    "OSI": {
        "title": "Oscillatory Shear Index",
        "unit": "OSI (dimensionless)",
        "cmap": "plasma",
    },
    "RRT": {
        "title": "Relative Residence Time",
        "unit": "Derived RRT units (unverified)",
        "cmap": "inferno",
    },
}


# ==========================================
# Generate 3D maps
# ==========================================

for metric_name, config in metric_configs.items():

    print(f"\nGenerating {metric_name} map...")

    values = metrics[metric_name].to_numpy(dtype=np.float64)

    fig = plt.figure(figsize=(12, 9))

    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_trisurf(
        triangulation,
        positions[:, 2],
        cmap=config["cmap"],
        shade=False,
        linewidth=0,
        antialiased=True
    )

    # Assign one scalar value to each triangle.
    face_values = values[faces].mean(axis=1)
    surface.set_array(face_values)
    surface.set_clim(
        float(values.min()),
        float(values.max())
    )

    colorbar = fig.colorbar(
        surface,
        ax=ax,
        shrink=0.65,
        pad=0.1
    )

    colorbar.set_label(config["unit"], fontsize=10)

    ax.set_title(
        f"{CASE_ID} - {config['title']}",
        fontsize=14,
        pad=20
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Use equal spatial scaling.
    ranges = np.ptp(positions, axis=0)
    centers = positions.mean(axis=0)

    radius = max(ranges) / 2

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)

    ax.view_init(elev=25, azim=45)

    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{metric_name}_3D.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", output_path)


# ==========================================
# Export VTK legacy file for ParaView
# ==========================================

vtk_path = OUTPUT_DIR / f"{CASE_ID}_hemodynamics.vtk"

print("\nExporting VTK mesh...")

with vtk_path.open("w") as f:
    f.write("# vtk DataFile Version 3.0\n")
    f.write("AneuG-Flow hemodynamic metrics\n")
    f.write("ASCII\n")
    f.write("DATASET POLYDATA\n")

    f.write(f"POINTS {len(positions)} double\n")

    for point in positions:
        f.write(
            f"{point[0]:.10f} "
            f"{point[1]:.10f} "
            f"{point[2]:.10f}\n"
        )

    f.write(
        f"POLYGONS {len(faces)} {len(faces) * 4}\n"
    )

    for face in faces:
        f.write(
            f"3 {face[0]} {face[1]} {face[2]}\n"
        )

    f.write(f"POINT_DATA {len(positions)}\n")

    for metric_name in ["TAWSS", "OSI", "RRT"]:
        f.write(f"SCALARS {metric_name} double 1\n")
        f.write("LOOKUP_TABLE default\n")

        for value in metrics[metric_name]:
            f.write(f"{value:.12g}\n")


# ==========================================
# Summary
# ==========================================

print("\n========== VISUALIZATION COMPLETE ==========")
print("Output directory:", OUTPUT_DIR)
print("VTK file:", vtk_path)
print("Generated maps:")

for metric_name in metric_configs:
    print(
        OUTPUT_DIR / f"{metric_name}_3D.png"
    )

print("\nAll outputs generated successfully.")
