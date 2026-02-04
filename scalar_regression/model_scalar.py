import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """(Conv3D -> GroupNorm -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ScalarRegressor3D(nn.Module):
    """
    Global liver fat regressor from 3D T2 volumes.

    Input:
        x    : (B, 1, D, H, W)
        mask : (B, 1, D, H, W)

    Output:
        (B, 1) unbounded scalar (clamp to [0,1] outside)
    """

    def __init__(self, in_channels=2, base_channels=16):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        # statistical pooling (mean + variance)
        self.fc = nn.Sequential(
            nn.Linear(base_channels * 8 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, t2, mask):
        """
        t2   : (B, 1, D, H, W)
        mask : (B, 1, D, H, W)
        """

        # focus network on liver
        x = torch.cat([t2 * mask, mask], dim=1)

        x = self.enc1(x)
        x = self.pool1(x)

        x = self.enc2(x)
        x = self.pool2(x)

        x = self.enc3(x)
        x = self.pool3(x)

        x = self.bottleneck(x)

        # statistical pooling
        mu = x.mean(dim=(2, 3, 4))
        var = x.var(dim=(2, 3, 4), unbiased=False)
        stats = torch.cat([mu, var], dim=1)

        out = self.fc(stats)
        return out  # raw output, no sigmoid (clamp to [0,1] at inference if needed)


def get_scalar_model(**kwargs):
    return ScalarRegressor3D(**kwargs)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScalarRegressor3D().to(device)

    t2 = torch.randn(2, 1, 33, 97, 101).to(device)
    mask = torch.ones_like(t2)

    with torch.no_grad():
        y = model(t2, mask)

    print("Output:", y.shape, y.min().item(), y.max().item())
