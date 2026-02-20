"""
Utility functions for training, evaluation, and visualization.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import nibabel as nib


def calculate_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for fat fraction prediction.

    Args:
        pred: Predictions [B, C, D, H, W]
        target: Ground truth [B, C, D, H, W]
        mask: Binary mask [B, C, D, H, W] or None

    Returns:
        Dictionary of metrics
    """
    if mask is None:
        mask = torch.ones_like(pred)

    # Convert to numpy and flatten
    pred_np = pred.detach().cpu().numpy().flatten()
    target_np = target.detach().cpu().numpy().flatten()
    mask_np = mask.detach().cpu().numpy().flatten()

    # Apply mask
    valid_idx = mask_np > 0.5
    pred_valid = pred_np[valid_idx]
    target_valid = target_np[valid_idx]

    if len(pred_valid) == 0:
        return {
            'mae': 0.0,
            'rmse': 0.0,
            'correlation': 0.0,
            'bias': 0.0
        }

    # Mean Absolute Error
    mae = np.mean(np.abs(pred_valid - target_valid))

    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((pred_valid - target_valid) ** 2))

    # Pearson correlation
    if len(pred_valid) > 1 and np.std(pred_valid) > 0 and np.std(target_valid) > 0:
        correlation, _ = pearsonr(pred_valid, target_valid)
    else:
        correlation = 0.0

    # Bias (mean error)
    bias = np.mean(pred_valid - target_valid)

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'correlation': float(correlation),
        'bias': float(bias)
    }


def save_checkpoint(
    state: dict,
    is_best: bool = False,
    checkpoint_dir: Path = Path('./checkpoints')
):
    """
    Save model checkpoint.

    Args:
        state: Dictionary containing model state, optimizer state, etc.
        is_best: Whether this is the best model so far
        checkpoint_dir: Directory to save checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save latest checkpoint
    checkpoint_path = checkpoint_dir / 'checkpoint_latest.pth'
    torch.save(state, checkpoint_path)

    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / 'checkpoint_best.pth'
        torch.save(state, best_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> dict:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)

    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        if checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


class EarlyStopping:
    """Early stopping to stop training when a monitored metric stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "min" for metrics where lower is better (loss),
                  "max" for metrics where higher is better (F1, accuracy)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def _is_improvement(self, current: float) -> bool:
        if self.mode == "max":
            return current > self.best_value + self.min_delta
        return current < self.best_value - self.min_delta

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current metric value to monitor

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = val_loss
        elif not self._is_improvement(val_loss):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_value = val_loss
            self.counter = 0

        return self.early_stop


def visualize_prediction(
    t2: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
    save_path: Optional[str] = None
):
    """
    Visualize T2 image, predicted fat fraction, and ground truth.

    Args:
        t2: T2 image [D, H, W]
        pred: Predicted fat fraction [D, H, W]
        target: Ground truth fat fraction [D, H, W]
        mask: Liver mask [D, H, W] (optional)
        slice_idx: Slice index to visualize (middle slice if None)
        save_path: Path to save figure (optional)
    """
    if slice_idx is None:
        slice_idx = t2.shape[0] // 2

    fig, axes = plt.subplots(1, 4 if mask is not None else 3, figsize=(16, 4))

    # T2 image
    axes[0].imshow(t2[slice_idx], cmap='gray')
    axes[0].set_title('T2 MRI')
    axes[0].axis('off')

    # Predicted fat fraction
    im1 = axes[1].imshow(pred[slice_idx], cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Predicted Fat Fraction')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Ground truth fat fraction
    im2 = axes[2].imshow(target[slice_idx], cmap='hot', vmin=0, vmax=1)
    axes[2].set_title('Ground Truth Fat Fraction')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    # Liver mask (if available)
    if mask is not None:
        axes[3].imshow(mask[slice_idx], cmap='gray')
        axes[3].set_title('Liver Mask')
        axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_error_map(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
    save_path: Optional[str] = None
):
    """
    Visualize error map between prediction and ground truth.

    Args:
        pred: Predicted fat fraction [D, H, W]
        target: Ground truth fat fraction [D, H, W]
        mask: Liver mask [D, H, W] (optional)
        slice_idx: Slice index to visualize
        save_path: Path to save figure
    """
    if slice_idx is None:
        slice_idx = pred.shape[0] // 2

    error = pred - target
    if mask is not None:
        error = error * mask

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Predicted
    im1 = axes[0].imshow(pred[slice_idx], cmap='hot', vmin=0, vmax=1)
    axes[0].set_title('Predicted')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # Ground truth
    im2 = axes[1].imshow(target[slice_idx], cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    # Error map
    max_error = np.abs(error).max()
    im3 = axes[2].imshow(error[slice_idx], cmap='RdBu_r', vmin=-max_error, vmax=max_error)
    axes[2].set_title('Error (Pred - GT)')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def bland_altman_plot(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Create Bland-Altman plot for clinical validation.

    Args:
        pred: Predicted fat fraction [D, H, W] or flattened
        target: Ground truth fat fraction [D, H, W] or flattened
        mask: Liver mask [D, H, W] or flattened (optional)
        save_path: Path to save figure
    """
    # Flatten arrays
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    if mask is not None:
        mask_flat = mask.flatten()
        valid_idx = mask_flat > 0.5
        pred_flat = pred_flat[valid_idx]
        target_flat = target_flat[valid_idx]

    # Calculate mean and difference
    mean = (pred_flat + target_flat) / 2
    diff = pred_flat - target_flat

    # Calculate statistics
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(mean, diff, alpha=0.3, s=1)
    ax.axhline(mean_diff, color='red', linestyle='--', label=f'Mean: {mean_diff:.4f}')
    ax.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--',
               label=f'+1.96 SD: {mean_diff + 1.96 * std_diff:.4f}')
    ax.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--',
               label=f'-1.96 SD: {mean_diff - 1.96 * std_diff:.4f}')

    ax.set_xlabel('Mean Fat Fraction')
    ax.set_ylabel('Difference (Predicted - Ground Truth)')
    ax.set_title('Bland-Altman Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def scatter_plot(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Create scatter plot of predictions vs ground truth.

    Args:
        pred: Predicted fat fraction
        target: Ground truth fat fraction
        mask: Liver mask (optional)
        save_path: Path to save figure
    """
    # Flatten arrays
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    if mask is not None:
        mask_flat = mask.flatten()
        valid_idx = mask_flat > 0.5
        pred_flat = pred_flat[valid_idx]
        target_flat = target_flat[valid_idx]

    # Calculate correlation
    correlation, _ = pearsonr(pred_flat, target_flat)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(target_flat, pred_flat, alpha=0.3, s=1)
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect prediction')

    ax.set_xlabel('Ground Truth Fat Fraction')
    ax.set_ylabel('Predicted Fat Fraction')
    ax.set_title(f'Prediction vs Ground Truth (r = {correlation:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_prediction_as_nifti(
    prediction: np.ndarray,
    reference_nifti_path: str,
    output_path: str
):
    """
    Save prediction as NIfTI file using reference image for header info.

    Args:
        prediction: Predicted fat fraction array [D, H, W]
        reference_nifti_path: Path to reference NIfTI file for header
        output_path: Path to save output NIfTI file
    """
    # Load reference to get header and affine
    ref_nii = nib.load(reference_nifti_path)

    # Create new NIfTI image
    pred_nii = nib.Nifti1Image(prediction.astype(np.float32), ref_nii.affine, ref_nii.header)

    # Save
    nib.save(pred_nii, output_path)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    print("Utility functions for fat fraction prediction")
    print("\nAvailable functions:")
    print("  - calculate_metrics: Compute MAE, RMSE, correlation, bias")
    print("  - save_checkpoint / load_checkpoint: Model checkpointing")
    print("  - visualize_prediction: Show T2, prediction, and ground truth")
    print("  - visualize_error_map: Show prediction error")
    print("  - bland_altman_plot: Clinical validation plot")
    print("  - scatter_plot: Prediction vs ground truth scatter")
    print("  - save_prediction_as_nifti: Save predictions in NIfTI format")
