from pathlib import Path
import re

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

DATA_ROOT = Path("data/raw/IntrA/IntrA")
SPLIT_DIR = Path("data/raw/IntrA/fileSplit/cls")
OUTPUT_DIR = Path("data/processed/patient_splits")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

samples = {}

for fold in range(5):
    split_files = [
        (SPLIT_DIR / f"ann_clsSplit_{fold}.txt", 1),
        (SPLIT_DIR / f"negSplit_{fold}.txt", 0),
    ]

    for split_file, label in split_files:
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with split_file.open() as file:
            for line in file:
                relative_path = line.strip()

                if not relative_path:
                    continue

                match = re.search(r"AN(\d+)", relative_path)
                if not match:
                    raise ValueError(f"Cannot extract patient ID: {relative_path}")

                patient_id = f"AN{match.group(1)}"

                if relative_path in samples:
                    if samples[relative_path]["label"] != label:
                        raise ValueError(f"Conflicting labels: {relative_path}")
                    continue

                full_path = DATA_ROOT / relative_path

                if not full_path.exists():
                    raise FileNotFoundError(f"Sample not found: {full_path}")

                samples[relative_path] = {
                    "label": label,
                    "patient": patient_id,
                }

paths = list(samples)
labels = np.array([samples[p]["label"] for p in paths])
groups = np.array([samples[p]["patient"] for p in paths])

print(f"Total samples: {len(paths)}")
print(f"Unique patients: {len(set(groups))}")
print(f"Aneurysm: {(labels == 1).sum()}")
print(f"Not Aneurysm: {(labels == 0).sum()}")

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

for fold, (train_idx, test_idx) in enumerate(
    splitter.split(np.zeros(len(labels)), labels, groups)
):
    train_patients = set(groups[train_idx])
    test_patients = set(groups[test_idx])
    overlap = train_patients & test_patients

    if overlap:
        raise RuntimeError(f"Patient leakage in fold {fold}: {overlap}")

    fold_dir = OUTPUT_DIR / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    with (fold_dir / "train.txt").open("w") as f:
        for idx in train_idx:
            f.write(paths[idx] + "\n")

    with (fold_dir / "test.txt").open("w") as f:
        for idx in test_idx:
            f.write(paths[idx] + "\n")

    print(f"\nFold {fold}")
    print(f"Train samples: {len(train_idx)}")
    print(f"Test samples: {len(test_idx)}")
    print(f"Train patients: {len(train_patients)}")
    print(f"Test patients: {len(test_patients)}")
    print(f"Train Aneurysm: {(labels[train_idx] == 1).sum()}")
    print(f"Test Aneurysm: {(labels[test_idx] == 1).sum()}")
    print(f"Patient overlap: {len(overlap)}")

print("\nPatient-level splits created successfully.")
