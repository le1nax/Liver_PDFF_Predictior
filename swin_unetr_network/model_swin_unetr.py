"""
Swin-UNETR wrapper for voxel-wise liver fat fraction regression.

Wraps MONAI's SwinUNETR with:
- Automatic spatial padding to multiples of 32
- Output cropping back to original dimensions
- Sigmoid activation for [0, 1] regression output
- Optional loading of MONAI SSL pretrained weights
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad_to_multiple(x, multiple=32):
    """Pad spatial dims of (B, C, D, H, W) to be divisible by `multiple`.

    Returns padded tensor and the original (D, H, W) for later cropping.
    """
    _, _, d, h, w = x.shape
    new_d = ((d + multiple - 1) // multiple) * multiple
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    pad_d = new_d - d
    pad_h = new_h - h
    pad_w = new_w - w
    if pad_d == 0 and pad_h == 0 and pad_w == 0:
        return x, (d, h, w)
    # F.pad order: (left, right, top, bottom, front, back)
    x_padded = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)
    return x_padded, (d, h, w)


def _crop_to_original(x, original_shape):
    """Crop (B, C, D', H', W') back to (B, C, D, H, W)."""
    d, h, w = original_shape
    return x[:, :, :d, :h, :w]


class SwinUNETRWrapper(nn.Module):
    """
    Swin-UNETR for voxel-wise fat fraction regression.

    Handles spatial padding/cropping internally so callers can pass
    arbitrary-sized 3D volumes.

    Input:  (B, n_channels, D, H, W) — arbitrary spatial dims
    Output: (B, n_outputs, D, H, W)  — same spatial dims, values in [0, 1]
    """

    def __init__(
        self,
        n_channels=1,
        n_outputs=1,
        feature_size=48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        use_checkpoint=False,
        pretrained_weights=None,
    ):
        super().__init__()

        try:
            from monai.networks.nets import SwinUNETR
        except ImportError:
            raise ImportError(
                "MONAI is required for Swin-UNETR. "
                "Install it with: pip install monai>=1.3.0"
            )

        # Convert lists from YAML to tuples
        if isinstance(depths, list):
            depths = tuple(depths)
        if isinstance(num_heads, list):
            num_heads = tuple(num_heads)

        self.swin_unetr = SwinUNETR(
            in_channels=n_channels,
            out_channels=n_outputs,
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            spatial_dims=3,
            use_checkpoint=use_checkpoint,
        )

        if pretrained_weights is not None:
            self._load_pretrained(pretrained_weights)

    def _load_pretrained(self, weights_path):
        """Load pretrained weights (SSL encoder weights from MONAI)."""
        if weights_path.startswith("http"):
            state_dict = torch.hub.load_state_dict_from_url(
                weights_path, map_location="cpu"
            )
        elif os.path.isfile(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        else:
            print(f"Warning: pretrained_weights path not found: {weights_path}")
            return

        # Handle different weight formats
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if "model" in state_dict:
            state_dict = state_dict["model"]

        model_dict = self.swin_unetr.state_dict()
        matched = {}
        for k, v in state_dict.items():
            for candidate in [k, f"swinViT.{k}", f"module.{k}"]:
                if candidate in model_dict and model_dict[candidate].shape == v.shape:
                    matched[candidate] = v
                    break

        if matched:
            model_dict.update(matched)
            self.swin_unetr.load_state_dict(model_dict)
            print(f"Loaded {len(matched)}/{len(model_dict)} pretrained weight tensors")
        else:
            print("Warning: No matching pretrained weights found")

    def forward(self, x):
        x_padded, original_shape = _pad_to_multiple(x, multiple=32)
        out = self.swin_unetr(x_padded)
        out = _crop_to_original(out, original_shape)
        return torch.sigmoid(out)


def get_model(**kwargs):
    """Factory function for SwinUNETRWrapper."""
    return SwinUNETRWrapper(**kwargs)


if __name__ == "__main__":
    # Use CPU for sanity check (GPU may be occupied by training)
    device = torch.device("cpu")

    model = SwinUNETRWrapper(n_channels=1, n_outputs=1, feature_size=48).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with cropped liver volume (typical after crop_to_mask)
    x = torch.randn(1, 1, 20, 50, 60).to(device)
    with torch.no_grad():
        y = model(x)
    print(f"Cropped input:  {x.shape} -> output: {y.shape}  range: [{y.min().item():.4f}, {y.max().item():.4f}]")
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"

    # Test with full volume dimensions (not divisible by 32)
    x2 = torch.randn(1, 1, 33, 97, 101).to(device)
    with torch.no_grad():
        y2 = model(x2)
    print(f"Full input:     {x2.shape} -> output: {y2.shape}  range: [{y2.min().item():.4f}, {y2.max().item():.4f}]")
    assert y2.shape == x2.shape, f"Shape mismatch: {y2.shape} != {x2.shape}"

    print("All shape tests: PASSED")
