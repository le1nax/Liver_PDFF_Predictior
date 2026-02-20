"""
3D classification model for liver fat fraction steatosis grading.

Identical encoder to ScalarRegressor3D, but FC head outputs class logits
instead of a single scalar. Optional ordinal (CORAL) mode.
"""

import torch
import torch.nn as nn


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


class LiverFatClassifier3D(nn.Module):
    """
    Classify liver fat fraction into steatosis grades from 3D T2 volumes.

    Input:
        t2   : (B, 1, D, H, W)
        mask : (B, 1, D, H, W)

    Output (ordinal=False):
        (B, num_classes) raw logits

    Output (ordinal=True):
        (B, num_classes - 1) cumulative logits (CORAL approach)
    """

    def __init__(self, in_channels=2, base_channels=16, num_classes=4,
                 dropout=0.3, ordinal=False):
        super().__init__()
        self.num_classes = num_classes
        self.ordinal = ordinal

        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        # statistical pooling (mean + variance) -> FC with dropout
        out_logits = num_classes - 1 if ordinal else num_classes
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(base_channels * 8 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, out_logits),
        )

    def forward(self, t2, mask):
        """
        t2   : (B, 1, D, H, W)
        mask : (B, 1, D, H, W)

        Returns logits: (B, num_classes) or (B, num_classes-1) if ordinal.
        """
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

        return self.fc(stats)

    def predict_classes(self, t2, mask):
        """Return predicted class indices (B,)."""
        logits = self.forward(t2, mask)
        if self.ordinal:
            # cumulative probs -> class = number of thresholds exceeded
            probs = torch.sigmoid(logits)
            return (probs > 0.5).sum(dim=1).long()
        return logits.argmax(dim=1)


def get_classification_model(**kwargs):
    return LiverFatClassifier3D(**kwargs)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LiverFatClassifier3D().to(device)

    t2 = torch.randn(2, 1, 33, 97, 101).to(device)
    mask = torch.ones_like(t2)

    with torch.no_grad():
        logits = model(t2, mask)
        classes = model.predict_classes(t2, mask)

    print("Logits:", logits.shape, logits)
    print("Classes:", classes)
