from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


CASES = {
    "stable_0": Path(
        "results/aneug_flow/visualizations/stable_0/"
        "stable_0_hemodynamics.vtk"
    ),
    "stable_1": Path(
        "results/aneug_flow/stable_1/"
        "stable_1_hemodynamics.vtk"
    ),
}

METRICS = ["TAWSS", "OSI", "RRT"]

OUTPUT_DIR = Path("results/aneug_flow/spatial_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_vtk(path):
    """Read the required sections from legacy ASCII VTK POLYDATA."""
    if not path.exists():
        raise FileNotFoundError(f"Missing VTK file: {path}")

    lines = path.read_text().splitlines()

    def find_line(prefix):
        for i, line in enumerate(lines):
            if line.strip().startswith(prefix):
                return i
        raise ValueError(f"Could not find {prefix} in {path}")

    # Points
    i = find_line("POINTS")
    parts = lines[i].split()
    n_points = int(parts[1])

    values = []
    j = i + 1
    while len(values) < n_points * 3:
        values.extend(float(x) for x in lines[j].split())
        j += 1

    points = np.asarray(values, dtype=float).reshape(n_points, 3)

    # Triangular polygons
    i = find_line("POLYGONS")
    parts = lines[i].split()
    n_faces = int(parts[1])

    faces = []
    j = i + 1
    while len(faces) < n_faces:
        row = [int(x) for x in lines[j].split()]
        j += 1
        k = 0
        while k < len(row):
            count = row[k]
            if count != 3:
                raise ValueError(
                    f"Expected triangular faces; found {count} vertices"
                )
            faces.append(row[k + 1:k + 4])
            k += count + 1

    faces = np.asarray(faces, dtype=np.int64)

    # Point data and scalar fields
    i = find_line("POINT_DATA")
    point_count = int(lines[i].split()[1])

    if point_count != n_points:
        raise ValueError("POINT_DATA count does not match POINTS count")

    fields = {}
    j = i + 1

    while j < len(lines):
        line = lines[j].strip()

        if line.startswith("SCALARS"):
            parts = line.split()
            name = parts[1]
            components = int(parts[3]) if len(parts) > 3 else 1

            j += 1
            if not lines[j].strip().startswith("LOOKUP_TABLE"):
                raise ValueError(f"Missing LOOKUP_TABLE for {name}")

            j += 1
            data = []

            while len(data) < point_count * components:
                data.extend(float(x) for x in lines[j].split())
                j += 1

            fields[name] = np.asarray(data).reshape(
                point_count, components
            ).squeeze()

        else:
            j += 1

    for metric in METRICS:
        if metric not in fields:
            raise ValueError(f"{metric} not found in {path}")

    if faces.min() < 0 or faces.max() >= n_points:
        raise ValueError("Face indices are outside the points range")

    return points, faces, fields


def calculate_face_areas(points, faces):
    triangles = points[faces]

    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0]
    )

    areas = 0.5 * np.linalg.norm(cross, axis=1)

    valid = np.isfinite(areas) & (areas > 0)
    return triangles[valid], faces[valid], areas[valid]


def weighted_mean(values, weights):
    return np.sum(values * weights) / np.sum(weights)


def analyze_case(case_name, path):
    points, faces, fields = read_vtk(path)

    triangles, valid_faces, areas = calculate_face_areas(points, faces)

    if len(areas) == 0:
        raise ValueError(f"No valid surface triangles in {case_name}")

    print(f"\n========== {case_name} ==========")
    print(f"Points: {len(points):,}")
    print(f"Faces: {len(faces):,}")
    print(f"Valid faces: {len(valid_faces):,}")
    print(f"Total surface area: {areas.sum():.10f}")

    face_values = {}

    for metric in METRICS:
        values = np.asarray(fields[metric], dtype=float)

        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite values in {case_name}/{metric}")

        # Mean value of the three vertices of each triangle
        values_on_faces = values[valid_faces].mean(axis=1)
        face_values[metric] = values_on_faces

    return {
        "points": points,
        "faces": valid_faces,
        "triangles": triangles,
        "areas": areas,
        "fields": fields,
        "face_values": face_values,
    }


def main():
    data = {}

    for case_name, path in CASES.items():
        data[case_name] = analyze_case(case_name, path)

    summary = []

    # Shared color limits and pooled 90th percentile per metric
    scales = {}
    thresholds = {}

    for metric in METRICS:
        pooled = np.concatenate([
            np.asarray(data[c]["fields"][metric])
            for c in CASES
        ])

        scales[metric] = (
            float(np.percentile(pooled, 1)),
            float(np.percentile(pooled, 99))
        )
        thresholds[metric] = float(np.percentile(pooled, 90))

    # Six spatial maps
    for metric in METRICS:
        vmin, vmax = scales[metric]
        threshold = thresholds[metric]
        norm = Normalize(vmin=vmin, vmax=vmax)

        for case_name in CASES:
            item = data[case_name]

            triangles = item["triangles"]
            areas = item["areas"]
            values = item["face_values"][metric]

            colors = plt.get_cmap("viridis")(
                norm(values)
            )

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            collection = Poly3DCollection(
                triangles,
                facecolors=colors,
                edgecolors="none",
                linewidths=0
            )
            ax.add_collection3d(collection)

            points = item["points"]
            lower = points.min(axis=0)
            upper = points.max(axis=0)

            ax.set_xlim(lower[0], upper[0])
            ax.set_ylim(lower[1], upper[1])
            ax.set_zlim(lower[2], upper[2])

            ax.set_box_aspect(np.maximum(upper - lower, 1e-12))
            ax.view_init(elev=25, azim=35)
            ax.set_axis_off()

            mappable = plt.cm.ScalarMappable(
                norm=norm,
                cmap="viridis"
            )
            mappable.set_array([])

            colorbar = fig.colorbar(
                mappable,
                ax=ax,
                shrink=0.65,
                pad=0.02
            )
            colorbar.set_label(metric)

            ax.set_title(
                f"{case_name} - {metric}\n"
                f"Shared color scale: {vmin:.4g} to {vmax:.4g}",
                pad=15
            )

            output = OUTPUT_DIR / f"{case_name}_{metric}_spatial.png"
            fig.savefig(output, dpi=220, bbox_inches="tight")
            plt.close(fig)

            total_area = areas.sum()
            high_area = areas[values > threshold].sum()

            node_values = np.asarray(
                item["fields"][metric], dtype=float
            )

            row = {
                "case": case_name,
                "metric": metric,
                "node_mean": node_values.mean(),
                "node_median": np.median(node_values),
                "node_p90": np.percentile(node_values, 90),
                "area_weighted_mean": weighted_mean(
                    values, areas
                ),
                "face_mean_median": np.median(values),
                "surface_area": total_area,
                "pooled_node_p90_threshold": threshold,
                "area_fraction_above_pooled_p90_percent":
                    100 * high_area / total_area,
                "valid_faces": len(areas),
            }

            summary.append(row)

            print(
                f"{metric}: area-weighted mean = "
                f"{row['area_weighted_mean']:.6f}, "
                f"area above pooled P90 = "
                f"{row['area_fraction_above_pooled_p90_percent']:.2f}%"
            )

    summary_df = pd.DataFrame(summary)
    output_csv = OUTPUT_DIR / "spatial_distribution_summary.csv"
    summary_df.to_csv(output_csv, index=False)

    print("\n========== OUTPUTS ==========")
    print(f"Maps and summary saved in: {OUTPUT_DIR}")
    print(f"Summary CSV: {output_csv}")
    print("\nSpatial analysis completed successfully.")


if __name__ == "__main__":
    main()
