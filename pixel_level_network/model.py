"""
3D U-Net architecture for liver fat fraction prediction from T2 MRI.
Voxel-wise regression with output in [0, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Basic building blocks
# ---------------------------

class DoubleConv(nn.Module):
    """(Conv3D -> InstanceNorm -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Upscaling, concatenation, then double conv"""

    def __init__(self, in_ch_up, in_ch_skip, out_ch, trilinear=True):
        super().__init__()

        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(
                in_ch_up, in_ch_up, kernel_size=2, stride=2
            )

        self.conv = DoubleConv(in_ch_up + in_ch_skip, out_ch)

    def forward(self, x_up, x_skip):
        x_up = self.up(x_up)

        # Padding for odd sizes
        diffZ = x_skip.size(2) - x_up.size(2)
        diffY = x_skip.size(3) - x_up.size(3)
        diffX = x_skip.size(4) - x_up.size(4)

        x_up = F.pad(
            x_up,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2,
             diffZ // 2, diffZ - diffZ // 2]
        )

        x = torch.cat([x_skip, x_up], dim=1)
        return self.conv(x)


# ---------------------------
# Full UNet3D
# ---------------------------

class UNet3D(nn.Module):
    """
    3D U-Net for liver fat fraction regression.

    - Input:  (B, 1, D, H, W)
    - Output: (B, 1, D, H, W), values in [0, 1]
    """

    def __init__(self, n_channels=1, n_outputs=1, base_channels=32, trilinear=True):
        super().__init__()

        # Encoder
        self.inc   = DoubleConv(n_channels, base_channels)        # 32
        self.down1 = Down(base_channels, base_channels * 2)       # 64
        self.down2 = Down(base_channels * 2, base_channels * 4)   # 128
        self.down3 = Down(base_channels * 4, base_channels * 8)   # 256

        # Decoder (explicit channel wiring)
        self.up1 = Up(256, 128, 64, trilinear)
        self.up2 = Up(64, 64, 32, trilinear)
        self.up3 = Up(32, 32, 32, trilinear)

        self.outc = nn.Conv3d(32, n_outputs, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        # Regression output in [0,1]
        return torch.sigmoid(self.outc(x))


# ---------------------------
# Lightweight version
# ---------------------------

class UNet3DLightweight(nn.Module):
    """Smaller U-Net for fast experiments"""

    def __init__(self, n_channels=1, n_outputs=1, base_channels=16, trilinear=True):
        super().__init__()

        self.inc   = DoubleConv(n_channels, base_channels)        # 16
        self.down1 = Down(base_channels, base_channels * 2)       # 32
        self.down2 = Down(base_channels * 2, base_channels * 4)   # 64

        self.up1 = Up(64, 32, 32, trilinear)
        self.up2 = Up(32, 16, 16, trilinear)

        self.outc = nn.Conv3d(16, n_outputs, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.up1(x3, x2)
        x = self.up2(x, x1)

        return torch.sigmoid(self.outc(x))


# ---------------------------
# Factory
# ---------------------------

def get_model(model_type="standard", **kwargs):
    if model_type == "standard":
        return UNet3D(**kwargs)
    elif model_type == "lightweight":
        return UNet3DLightweight(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------
# Sanity check
# ---------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet3D().to(device)
    x = torch.randn(2, 1, 33, 97, 101).to(device)

    with torch.no_grad():
        y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
    print("Value range :", y.min().item(), y.max().item())
