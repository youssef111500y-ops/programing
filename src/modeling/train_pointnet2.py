from pathlib import Path
import argparse
import csv
import json
import re
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.modeling.pointnet_dataset import IntrADataset
from src.modeling.pointnet2 import PointNet2Classifier


DATA_ROOT = Path("data/raw/IntrA/IntrA")
SPLIT_ROOT = Path("data/processed/patient_splits")
OUTPUT_ROOT = Path("results/pointnet2")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_metadata(paths):
    labels, groups = [], []

    for path in paths:
        labels.append(
            1 if "generated/aneurysm/" in path
            or "annotated/" in path else 0
        )

        match = re.search(r"AN(\d+)", path)
        if not match:
            raise ValueError(f"Cannot identify patient: {path}")

        groups.append(f"AN{match.group(1)}")

    return np.array(labels), np.array(groups)


def evaluate(model, loader, criterion):
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(DEVICE)
            labels = batch["class_label"].to(DEVICE)

            outputs = model(points)
            loss = criterion(outputs, labels)

            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)

    return {
        "loss": total_loss / len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
    }


def train_fold(fold, epochs, batch_size):
    print(f"\n{'=' * 55}")
    print(f"STARTING FOLD {fold}")
    print(f"{'=' * 55}")

    fold_dir = SPLIT_ROOT / f"fold_{fold}"
    output_dir = OUTPUT_ROOT / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (fold_dir / "train.txt").open() as f:
        train_paths = [line.strip() for line in f if line.strip()]

    with (fold_dir / "test.txt").open() as f:
        test_paths = [line.strip() for line in f if line.strip()]

    train_labels, train_groups = get_metadata(train_paths)
    test_labels, test_groups = get_metadata(test_paths)

    # Inner patient-level split: training / validation.
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    inner_train_idx, val_idx = next(
        splitter.split(
            np.zeros(len(train_labels)),
            train_labels,
            train_groups,
        )
    )

    inner_train_paths = [train_paths[i] for i in inner_train_idx]
    val_paths = [train_paths[i] for i in val_idx]

    inner_train_groups = set(train_groups[inner_train_idx])
    val_groups = set(train_groups[val_idx])
    test_groups_set = set(test_groups)

    assert not inner_train_groups.intersection(val_groups)
    assert not inner_train_groups.intersection(test_groups_set)
    assert not val_groups.intersection(test_groups_set)

    # Create temporary split files for the Dataset.
    train_file = output_dir / "train_inner.txt"
    val_file = output_dir / "validation.txt"

    train_file.write_text("\n".join(inner_train_paths) + "\n")
    val_file.write_text("\n".join(val_paths) + "\n")

    train_dataset = IntrADataset(
        data_root=DATA_ROOT,
        split_file=train_file,
        num_points=512,
    )

    val_dataset = IntrADataset(
        data_root=DATA_ROOT,
        split_file=val_file,
        num_points=512,
    )

    test_dataset = IntrADataset(
        data_root=DATA_ROOT,
        split_file=fold_dir / "test.txt",
        num_points=512,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Handle class imbalance using training data only.
    counts = np.bincount(
        train_labels[inner_train_idx],
        minlength=2,
    )

    class_weights = len(inner_train_idx) / (
        2.0 * np.maximum(counts, 1)
    )

    weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    model = PointNet2Classifier(
        num_classes=2,
        input_channels=3,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best_auc = -1.0
    best_epoch = 0
    patience = 7
    epochs_without_improvement = 0

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        processed = 0

        for batch in train_loader:
            points = batch["points"].to(DEVICE)
            labels = batch["class_label"].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(points)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            processed += labels.size(0)

        train_loss = running_loss / processed

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step(val_metrics["auc"])

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_metrics["loss"],
            "validation_auc": val_metrics["auc"],
            "validation_f1": val_metrics["f1"],
        })

        print(
            f"Fold {fold} | Epoch {epoch:02d}/{epochs} "
            f"| Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_metrics['loss']:.4f} "
            f"| Val AUC: {val_metrics['auc']:.4f} "
            f"| Val F1: {val_metrics['f1']:.4f}"
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                output_dir / "best_model.pth",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    # Evaluate the selected checkpoint on the untouched test set.
    model.load_state_dict(
        torch.load(
            output_dir / "best_model.pth",
            map_location=DEVICE,
        )
    )

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
    )

    result = {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_validation_auc": best_auc,
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "patient_overlap": 0,
        "test_metrics": test_metrics,
    }

    with (output_dir / "metrics.json").open("w") as f:
        json.dump(result, f, indent=4)

    with (output_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=4)

    print(f"\nFOLD {fold} TEST RESULTS")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    torch.set_num_threads(4)
    set_seed(42)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE)
    print("PyTorch:", torch.__version__)
    print("Folds:", args.folds)
    print("Maximum epochs:", args.epochs)

    results = []

    for fold in args.folds:
        set_seed(42 + fold)

        result = train_fold(
            fold=fold,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        results.append(result)

    summary_file = OUTPUT_ROOT / "cross_validation_summary.csv"

    with summary_file.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold",
                "best_epoch",
                "best_validation_auc",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_auc",
            ],
        )

        writer.writeheader()

        for result in results:
            metrics = result["test_metrics"]

            writer.writerow({
                "fold": result["fold"],
                "best_epoch": result["best_epoch"],
                "best_validation_auc": result["best_validation_auc"],
                "test_accuracy": metrics["accuracy"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_f1": metrics["f1"],
                "test_auc": metrics["auc"],
            })

    print("\nAll requested folds completed.")
    print("Summary:", summary_file)


if __name__ == "__main__":
    main()
