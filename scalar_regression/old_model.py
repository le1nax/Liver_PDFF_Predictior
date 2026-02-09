"""
3D regression model that maps a full volume to a single scalar in [0, 1].
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """(Conv3D -> InstanceNorm -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ScalarRegressor3D(nn.Module):
    """
    3D CNN regressor.

    - Input:  (B, 1, D, H, W)
    - Output: (B, 1), values in [0, 1]
    """

    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(base_channels * 8, 1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.pool2(x)
        x = self.enc3(x)
        x = self.pool3(x)
        x = self.bottleneck(x)
        x = self.gap(x).flatten(1)
        x = self.fc(x)
        return torch.sigmoid(x)


def get_scalar_model(**kwargs):
    return ScalarRegressor3D(**kwargs)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScalarRegressor3D().to(device)
    x = torch.randn(2, 1, 33, 97, 101).to(device)
    with torch.no_grad():
        y = model(x)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
    print("Value range :", y.min().item(), y.max().item())