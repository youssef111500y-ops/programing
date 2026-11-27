from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch


def process_case(case_id):
    base = Path("data/raw/aneug_flow/transient_data") / case_id

    wall_path = base / "wall_data.pt"
    registration_path = base / "wall_mesh_registration.pt"

    print("Looking for files...")
    print("Wall:", wall_path.exists())
    print("Registration:", registration_path.exists())

    if not wall_path.exists() or not registration_path.exists():
        raise FileNotFoundError(
            "Files not found. Check the download location."
        )

    wall = torch.load(wall_path, map_location="cpu", weights_only=True)
    reg = torch.load(registration_path, map_location="cpu", weights_only=True)

    def get(name):
        return wall[name].squeeze(-1).numpy().astype(np.float64)

    x, y, z = get("x_coordinate"), get("y_coordinate"), get("z_coordinate")
    wx, wy, wz = get("x_wall_shear"), get("y_wall_shear"), get("z_wall_shear")

    positions = reg["pos"].numpy().astype(np.float64)
    faces = reg["faces"].numpy().astype(np.int64)

    frames, nodes = x.shape

    if frames != 80 or len(positions) != nodes:
        raise ValueError("Unexpected data dimensions.")

    wall_positions = np.column_stack([x[0], y[0], z[0]])
    if np.max(np.linalg.norm(positions - wall_positions, axis=1)) > 1e-5:
        raise ValueError("Mesh coordinates do not match wall data.")

    magnitude = np.sqrt(wx**2 + wy**2 + wz**2)
    tawss = magnitude.mean(axis=0)

    mean_vector = np.sqrt(
        wx.mean(axis=0)**2 +
        wy.mean(axis=0)**2 +
        wz.mean(axis=0)**2
    )

    osi = np.full(nodes, np.nan)
    valid = magnitude.mean(axis=0) > 1e-12
    osi[valid] = 0.5 * (
        1 - mean_vector[valid] / magnitude.mean(axis=0)[valid]
    )
    osi[valid] = np.clip(osi[valid], 0, 0.5)

    denominator = (1 - 2 * osi) * tawss
    rrt = np.full(nodes, np.nan)
    valid_rrt = np.isfinite(denominator) & (denominator > 1e-12)
    rrt[valid_rrt] = 1 / denominator[valid_rrt]

    output = Path("results/aneug_flow") / case_id
    output.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "node_id": np.arange(nodes),
        "x": positions[:, 0],
        "y": positions[:, 1],
        "z": positions[:, 2],
        "TAWSS": tawss,
        "OSI": osi,
        "RRT": rrt,
    })

    csv_path = output / f"{case_id}_wss_metrics.csv"
    df.to_csv(csv_path, index=False)

    vtk_path = output / f"{case_id}_hemodynamics.vtk"
    with vtk_path.open("w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{case_id} hemodynamics\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {nodes} double\n")
        for p in positions:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
        f.write(f"POLYGONS {len(faces)} {len(faces)*4}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        f.write(f"POINT_DATA {nodes}\n")
        for name in ["TAWSS", "OSI", "RRT"]:
            f.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
            for value in df[name]:
                f.write(f"{value}\n")

    print("\nSUCCESS")
    print("Frames:", frames)
    print("Nodes:", nodes)
    print("Faces:", len(faces))
    print("\n", df[["TAWSS", "OSI", "RRT"]].describe())
    print("\nCSV:", csv_path)
    print("VTK:", vtk_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Example: python src/hemodynamics/process_case.py stable_1")
        sys.exit(1)
    process_case(sys.argv[1])
