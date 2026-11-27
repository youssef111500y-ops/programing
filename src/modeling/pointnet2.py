import torch
import torch.nn as nn
import torch.nn.functional as F


def farthest_point_sample(xyz, npoint):
    """
    xyz: [B, N, 3]
    Returns sampled point indices: [B, npoint]
    """
    B, N, _ = xyz.shape
    device = xyz.device

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)

    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest

        centroid = xyz[batch_indices, farthest].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)

        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def ball_query(xyz, centers, radius, nsample):
    """
    xyz:     [B, N, 3]
    centers: [B, S, 3]
    Returns neighbor indices: [B, S, nsample]
    """
    distances = torch.cdist(centers, xyz)

    indices = torch.arange(
        xyz.shape[1], device=xyz.device
    ).view(1, 1, -1)

    indices = indices.expand(
        xyz.shape[0], centers.shape[1], -1
    ).clone()

    indices[distances > radius] = xyz.shape[1]

    indices = indices.sort(dim=-1)[0][:, :, :nsample]

    first = indices[:, :, 0].clone()
    first[first == xyz.shape[1]] = 0

    indices[indices == xyz.shape[1]] = first.unsqueeze(-1).expand_as(indices)[
        indices == xyz.shape[1]
    ]

    return indices.long()


def index_points(points, idx):
    """
    points: [B, N, C]
    idx: [B, ...]
    """
    B = points.shape[0]

    batch_indices = torch.arange(
        B, device=points.device
    ).view(B, *([1] * (idx.ndim - 1)))

    batch_indices = batch_indices.expand_as(idx)

    return points[batch_indices, idx]


class SetAbstraction(nn.Module):
    def __init__(
        self,
        npoint,
        radius,
        nsample,
        in_channels,
        mlp_channels,
    ):
        super().__init__()

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        layers = []
        previous_channels = in_channels + 3

        for channels in mlp_channels:
            layers.extend([
                nn.Conv2d(
                    previous_channels,
                    channels,
                    kernel_size=1,
                ),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ])

            previous_channels = channels

        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, features):
        """
        xyz: [B, N, 3]
        features: [B, N, C] or None

        Returns:
            new_xyz: [B, S, 3]
            new_features: [B, S, C_out]
        """
        B, N, _ = xyz.shape

        sample_count = min(self.npoint, N)

        fps_indices = farthest_point_sample(
            xyz, sample_count
        )

        new_xyz = index_points(xyz, fps_indices)

        group_indices = ball_query(
            xyz,
            new_xyz,
            self.radius,
            self.nsample,
        )

        grouped_xyz = index_points(xyz, group_indices)
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)

        if features is not None:
            grouped_features = index_points(
                features, group_indices
            )

            grouped_features = torch.cat(
                [grouped_xyz, grouped_features],
                dim=-1,
            )
        else:
            grouped_features = grouped_xyz

        grouped_features = grouped_features.permute(
            0, 3, 1, 2
        )

        grouped_features = self.mlp(grouped_features)

        new_features = torch.max(
            grouped_features,
            dim=-1,
        )[0]

        return new_xyz, new_features.transpose(1, 2)


class PointNet2Classifier(nn.Module):
    def __init__(self, num_classes=2, input_channels=3):
        super().__init__()

        self.sa1 = SetAbstraction(
            npoint=128,
            radius=0.2,
            nsample=32,
            in_channels=input_channels,
            mlp_channels=[64, 64, 128],
        )

        self.sa2 = SetAbstraction(
            npoint=32,
            radius=0.4,
            nsample=32,
            in_channels=128,
            mlp_channels=[128, 128, 256],
        )

        self.sa3 = SetAbstraction(
            npoint=8,
            radius=0.8,
            nsample=16,
            in_channels=256,
            mlp_channels=[256, 512, 1024],
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(256, num_classes),
        )

    def forward(self, points):
        """
        points: [B, N, 6]
        First 3 channels: XYZ
        Remaining 3 channels: Normals
        """
        xyz = points[:, :, :3]
        features = points[:, :, 3:]

        xyz, features = self.sa1(xyz, features)
        xyz, features = self.sa2(xyz, features)
        xyz, features = self.sa3(xyz, features)

        global_features = torch.max(
            features, dim=1
        )[0]

        return self.classifier(global_features)


if __name__ == "__main__":
    model = PointNet2Classifier(
        num_classes=2,
        input_channels=3,
    )

    dummy_input = torch.randn(2, 1024, 6)

    output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Model parameters:", sum(
        p.numel() for p in model.parameters()
    ))
