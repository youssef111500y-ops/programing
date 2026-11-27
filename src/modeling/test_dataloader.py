import torch
from torch.utils.data import DataLoader

from src.modeling.pointnet_dataset import IntrADataset

fold = 0

train_dataset = IntrADataset(
    data_root="data/raw/IntrA/IntrA",
    split_file=f"data/processed/patient_splits/fold_{fold}/train.txt",
)

test_dataset = IntrADataset(
    data_root="data/raw/IntrA/IntrA",
    split_file=f"data/processed/patient_splits/fold_{fold}/test.txt",
)

train_loader = DataLoader(
    train_dataset, batch_size=8, shuffle=True
)

test_loader = DataLoader(
    test_dataset, batch_size=8, shuffle=False
)

batch = next(iter(train_loader))

print("\nTrain batch:", batch["points"].shape)
print("Test samples:", len(test_dataset))
print("Train samples:", len(train_dataset))
print("Contains NaN:", torch.isnan(batch["points"]).any().item())
print("Contains Inf:", torch.isinf(batch["points"]).any().item())
