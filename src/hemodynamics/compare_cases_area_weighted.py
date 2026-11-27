from pathlib import Path

import numpy as np
import pandas as pd
import torch


CASES = {
    "stable_0": {
        "metrics": Path(
            "results/aneug_flow/stable_0_wss_metrics.csv"
        ),
        "registration": Path(
            "data/raw/aneug_flow/stable_0/"
            "transient_data/stable_0/"
            "wall_mesh_registration.pt"
        ),
    },
    "stable_1": {
        "metrics": Path(
            "results/aneug_flow/stable_1/"
            "stable_1_wss_metrics.csv"
        ),
        "registration": Path(
            "data/raw/aneug_flow/transient_data/"
            "stable_1/wall_mesh_registration.pt"
        ),
    },
}

OUTPUT_DIR = Path("results/aneug_flow/comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["TAWSS", "OSI", "RRT"]


def weighted_percentile(values, weights, percentile):
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)
    threshold = percentile / 100 * cumulative[-1]

    index = np.searchsorted(cumulative, threshold)
    index = min(index, len(values) - 1)

    return values[index]


def analyze_case(case_id, paths):
    print(f"\n========== {case_id} ==========")

    df = pd.read_csv(paths["metrics"])

    registration = torch.load(
        paths["registration"],
        map_location="cpu",
        weights_only=True
    )

    positions = registration["pos"].numpy().astype(np.float64)
    faces = registration["faces"].numpy().astype(np.int64)

    if len(df) != len(positions):
        raise ValueError("Node count mismatch.")

    if not np.array_equal(
        df["node_id"].to_numpy(),
        np.arange(len(positions))
    ):
        raise ValueError("Node ordering mismatch.")

    csv_positions = df[["x", "y", "z"]].to_numpy()

    if not np.allclose(positions, csv_positions, atol=1e-5, rtol=0):
        raise ValueError("Coordinate mismatch.")

    if faces.min() < 0 or faces.max() >= len(positions):
        raise ValueError("Invalid face indices.")

    # Calculate triangle areas
    p0 = positions[faces[:, 0]]
    p1 = positions[faces[:, 1]]
    p2 = positions[faces[:, 2]]

    cross = np.cross(p1 - p0, p2 - p0)
    triangle_areas = np.linalg.norm(cross, axis=1) / 2

    if not np.isfinite(triangle_areas).all():
        raise ValueError("Invalid triangle areas.")

    degenerate_count = int(np.sum(triangle_areas <= 0))

    valid_faces = triangle_areas > 0

    if not valid_faces.all():
        print("Degenerate faces:", degenerate_count)

        faces = faces[valid_faces]
        triangle_areas = triangle_areas[valid_faces]

    # Lumped surface area per node
    node_areas = np.zeros(len(positions), dtype=np.float64)

    for corner in range(3):
        np.add.at(
            node_areas,
            faces[:, corner],
            triangle_areas / 3
        )

    # Exclude isolated nodes from area-weighted statistics
    active_nodes = node_areas > 0

    isolated_count = int((~active_nodes).sum())

    print("Total nodes:", len(positions))
    print("Nodes with surface area:", active_nodes.sum())
    print("Nodes with zero area:", isolated_count)
    print("Total surface area:", triangle_areas.sum())

    if active_nodes.sum() == 0:
        raise ValueError("No nodes have positive surface area.")

    weights = node_areas[active_nodes]

    results = {
        "case": case_id,
        "total_nodes": len(positions),
        "active_nodes": int(active_nodes.sum()),
        "isolated_nodes": isolated_count,
        "faces": len(faces),
        "surface_area": triangle_areas.sum(),
    }

    for metric in METRICS:
        values = df[metric].to_numpy(dtype=np.float64)

        if not np.isfinite(values).all():
            raise ValueError(f"Invalid values in {metric}")

        active_values = values[active_nodes]

        weighted_mean = np.average(
            active_values,
            weights=weights
        )

        results[f"{metric}_weighted_mean"] = weighted_mean

        results[f"{metric}_weighted_p05"] = weighted_percentile(
            active_values, weights, 5
        )

        results[f"{metric}_weighted_median"] = weighted_percentile(
            active_values, weights, 50
        )

        results[f"{metric}_weighted_p95"] = weighted_percentile(
            active_values, weights, 95
        )

        print(f"\n{metric}")
        print("Weighted mean:", weighted_mean)
        print("Weighted median:", results[f"{metric}_weighted_median"])

    return results


all_results = []

for case_id, paths in CASES.items():
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    all_results.append(analyze_case(case_id, paths))


summary = pd.DataFrame(all_results)

summary_path = OUTPUT_DIR / "area_weighted_comparison.csv"
summary.to_csv(summary_path, index=False)

print("\n========== FINAL COMPARISON ==========")

for metric in METRICS:
    col = f"{metric}_weighted_mean"

    value_0 = summary.loc[
        summary["case"] == "stable_0", col
    ].iloc[0]

    value_1 = summary.loc[
        summary["case"] == "stable_1", col
    ].iloc[0]

    percentage = (value_1 - value_0) / value_0 * 100

    print(f"\n{metric}")
    print("stable_0:", value_0)
    print("stable_1:", value_1)
    print("Percentage difference:", percentage, "%")

print("\nSaved:", summary_path)
print("\nAnalysis completed.")
