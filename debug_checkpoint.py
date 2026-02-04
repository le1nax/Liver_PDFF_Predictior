#!/usr/bin/env python3
"""Debug script to verify checkpoint loading."""

import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from scalar_regression.model_scalar import get_scalar_model

def main():
    checkpoint_path = Path("outputs/scalar_regression_run/experiment_20260128_093437/checkpoint_best.pth")
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print("=" * 60)
    print("CHECKPOINT CONTENTS")
    print("=" * 60)
    print(f"Keys in checkpoint: {list(checkpoint.keys())}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")

    # Check model state dict keys
    state_dict = checkpoint.get('model_state_dict', {})
    print(f"\nModel state dict keys ({len(state_dict)} total):")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        print(f"  {key}: shape {state_dict[key].shape}")
    if len(state_dict) > 10:
        print(f"  ... and {len(state_dict) - 10} more")

    # Check for "module." prefix (DDP)
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    print(f"\nHas 'module.' prefix (DDP): {has_module_prefix}")

    # Create fresh model and compare
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    model_cfg = checkpoint.get('config', {}).get('model', {})
    model = get_scalar_model(
        in_channels=model_cfg.get('in_channels', 2),
        base_channels=model_cfg.get('base_channels', 16)
    )

    # Check if keys match
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())

    if model_keys == checkpoint_keys:
        print("Keys match perfectly!")
    else:
        missing = model_keys - checkpoint_keys
        extra = checkpoint_keys - model_keys
        if missing:
            print(f"Missing from checkpoint: {missing}")
        if extra:
            print(f"Extra in checkpoint: {extra}")

    # Load and compare weight statistics
    print("\n" + "=" * 60)
    print("WEIGHT STATISTICS")
    print("=" * 60)

    # Fresh model stats
    fresh_fc_weight = model.fc[0].weight.data
    print(f"\nFresh model fc[0].weight: mean={fresh_fc_weight.mean():.6f}, std={fresh_fc_weight.std():.6f}")

    # Load checkpoint
    try:
        model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully!")

        loaded_fc_weight = model.fc[0].weight.data
        print(f"Loaded model fc[0].weight: mean={loaded_fc_weight.mean():.6f}, std={loaded_fc_weight.std():.6f}")

        # Check if weights are actually different
        if torch.allclose(fresh_fc_weight, loaded_fc_weight, atol=1e-6):
            print("\nWARNING: Weights are the same as initialization!")
        else:
            print("\nWeights are different from initialization (good)")

        # Print a few specific weights
        print("\nFirst 5 values of fc[0].weight[0]:")
        print(f"  {model.fc[0].weight.data[0, :5].tolist()}")

        # Check last layer bias
        print(f"\nfc[-1] (output layer) weight stats:")
        print(f"  Weight: mean={model.fc[-1].weight.data.mean():.6f}, std={model.fc[-1].weight.data.std():.6f}")
        print(f"  Bias: {model.fc[-1].bias.data.item():.6f}")

    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")

    # Test forward pass with dummy data
    print("\n" + "=" * 60)
    print("FORWARD PASS TEST")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        # Create dummy input
        t2 = torch.randn(1, 1, 32, 64, 64) * 100  # Simulate unnormalized T2
        mask = torch.ones(1, 1, 32, 64, 64)

        output = model(t2, mask)
        print(f"Dummy input (random): output = {output.item():.4f}")

        # All zeros input
        t2_zeros = torch.zeros(1, 1, 32, 64, 64)
        output_zeros = model(t2_zeros, mask)
        print(f"Zero input: output = {output_zeros.item():.4f}")

        # Small positive input
        t2_small = torch.ones(1, 1, 32, 64, 64) * 10
        output_small = model(t2_small, mask)
        print(f"Small constant input (10): output = {output_small.item():.4f}")


if __name__ == "__main__":
    main()
