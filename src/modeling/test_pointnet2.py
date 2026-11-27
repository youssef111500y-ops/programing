import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.modeling.pointnet_dataset import IntrADataset
from src.modeling.pointnet2 import PointNet2Classifier

# Reproducibility
torch.manual_seed(42)

# Load real IntrA training samples
dataset = IntrADataset(
    data_root="data/raw/IntrA/IntrA",
    split_file="data/processed/patient_splits/fold_0/train.txt",
    num_points=1024,
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
)

batch = next(iter(loader))

points = batch["points"]
labels = batch["class_label"]

# Initialize model
model = PointNet2Classifier(
    num_classes=2,
    input_channels=3,
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Forward pass
model.train()
outputs = model(points)

# Calculate loss
loss = criterion(outputs, labels)

# Backward pass
optimizer.zero_grad()
loss.backward()

# Verify gradients
gradients_exist = any(
    parameter.grad is not None
    for parameter in model.parameters()
)

# One optimization step
optimizer.step()

print("\nReal-data training test:")
print("Input shape:", points.shape)
print("Labels:", labels.tolist())
print("Output shape:", outputs.shape)
print("Loss:", round(loss.item(), 4))
print("Gradients calculated:", gradients_exist)
print("Backward pass: SUCCESS")
print("Optimizer step: SUCCESS")
