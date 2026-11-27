from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class IntrADataset(Dataset):
    """
    IntrA binary classification dataset.

    Input:
        XYZ coordinates + surface normals (6 features).

    Target:
        0 = Not Aneurysm
        1 = Aneurysm
    """

    def __init__(
        self,
        data_root,
        split_file,
        num_points=1024,
        normalize=True,
    ):
        self.data_root = Path(data_root)
        self.num_points = num_points
        self.normalize = normalize
        self.samples = []

        split_file = Path(split_file)

        if not split_file.exists():
            raise FileNotFoundError(split_file)

        with split_file.open() as file:
            for line in file:
                relative_path = line.strip()

                if not relative_path:
                    continue

                label = (
                    1 if "generated/aneurysm/" in relative_path
                    or "annotated/" in relative_path
                    else 0
                )

                sample_path = self.data_root / relative_path

                if not sample_path.exists():
                    raise FileNotFoundError(sample_path)

                self.samples.append((sample_path, label))

        if not self.samples:
            raise ValueError("No samples were loaded.")

        print(f"Loaded {len(self.samples)} samples")
        print(f"Aneurysm: {sum(y == 1 for _, y in self.samples)}")
        print(f"Not Aneurysm: {sum(y == 0 for _, y in self.samples)}")

    @staticmethod
    def _load_ad_file(file_path):
        data = np.loadtxt(file_path, dtype=np.float32)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] != 7:
            raise ValueError(f"Invalid AD format: {file_path}")

        return (
            data[:, :3],
            data[:, 3:6],
            data[:, 6].astype(np.int64),
        )

    @staticmethod
    def _normalize_points(xyz):
        xyz = xyz - np.mean(xyz, axis=0)

        radius = np.max(np.linalg.norm(xyz, axis=1))

        if radius > 0:
            xyz = xyz / radius

        return xyz

    def _sample_points(self, xyz, normals, point_labels):
        indices = np.random.choice(
            len(xyz),
            self.num_points,
            replace=len(xyz) < self.num_points,
        )

        return xyz[indices], normals[indices], point_labels[indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path, class_label = self.samples[index]

        xyz, normals, point_labels = self._load_ad_file(file_path)

        if self.normalize:
            xyz = self._normalize_points(xyz)

        xyz, normals, point_labels = self._sample_points(
            xyz, normals, point_labels
        )

        normal_lengths = np.linalg.norm(
            normals, axis=1, keepdims=True
        )
        normals = normals / np.maximum(normal_lengths, 1e-8)

        features = np.concatenate([xyz, normals], axis=1)

        return {
            "points": torch.tensor(features, dtype=torch.float32),
            "class_label": torch.tensor(class_label, dtype=torch.long),
            "point_labels": torch.tensor(point_labels, dtype=torch.long),
        }
