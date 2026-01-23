"""
Loss functions for fat fraction prediction with optional liver masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MaskedLoss(nn.Module):
    """
    Base class for masked losses that only compute loss within liver regions.
    """

    def __init__(self, base_loss: nn.Module, reduction: str = 'mean'):
        """
        Args:
            base_loss: The underlying loss function (e.g., MSELoss, L1Loss)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.base_loss = base_loss
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions [B, C, D, H, W]
            target: Ground truth [B, C, D, H, W]
            mask: Binary mask [B, C, D, H, W] or None

        Returns:
            Loss value
        """
        if mask is None:
            mask = torch.ones_like(pred)

        # Compute element-wise loss
        loss = self.base_loss(pred, target)

        # Apply mask
        masked_loss = loss * mask

        # Reduce
        if self.reduction == 'mean':
            # Average over valid (masked) pixels only
            return masked_loss.sum() / (mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            return masked_loss.sum()
        else:
            return masked_loss


class MaskedMSELoss(nn.Module):
    """Mean Squared Error loss with optional masking."""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(pred)

        squared_error = (pred - target) ** 2
        masked_error = squared_error * mask

        if self.reduction == 'mean':
            return masked_error.sum() / (mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            return masked_error.sum()
        else:
            return masked_error


class MaskedL1Loss(nn.Module):
    """Mean Absolute Error (L1) loss with optional masking."""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(pred)

        abs_error = torch.abs(pred - target)
        masked_error = abs_error * mask

        if self.reduction == 'mean':
            return masked_error.sum() / (mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            return masked_error.sum()
        else:
            return masked_error


class MaskedSmoothL1Loss(nn.Module):
    """Smooth L1 loss (Huber loss) with optional masking."""

    def __init__(self, beta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(pred)

        diff = torch.abs(pred - target)
        # Smooth L1: 0.5 * x^2 / beta if |x| < beta, else |x| - 0.5 * beta
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )

        masked_loss = loss * mask

        if self.reduction == 'mean':
            return masked_loss.sum() / (mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            return masked_loss.sum()
        else:
            return masked_loss


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss.
    SSIM values are in [0, 1], so loss = 1 - SSIM
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.reduction = reduction

        # Create Gaussian window
        self.register_buffer('window', self._create_window(window_size, sigma))

    def _create_window(self, window_size: int, sigma: float) -> torch.Tensor:
        """Create 3D Gaussian window"""
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()

        # Create 3D window
        window_3d = g[:, None, None] * g[None, :, None] * g[None, None, :]
        window_3d = window_3d / window_3d.sum()
        return window_3d.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute SSIM loss.
        Note: This is a simplified version that doesn't use the mask for SSIM calculation.
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        window = self.window.to(pred.device)
        window = window.expand(pred.size(1), 1, -1, -1, -1)

        # Compute statistics
        mu_pred = F.conv3d(pred, window, padding=self.window_size // 2, groups=pred.size(1))
        mu_target = F.conv3d(target, window, padding=self.window_size // 2, groups=target.size(1))

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.conv3d(pred ** 2, window, padding=self.window_size // 2, groups=pred.size(1)) - mu_pred_sq
        sigma_target_sq = F.conv3d(target ** 2, window, padding=self.window_size // 2, groups=target.size(1)) - mu_target_sq
        sigma_pred_target = F.conv3d(pred * target, window, padding=self.window_size // 2, groups=pred.size(1)) - mu_pred_target

        # SSIM formula
        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

        if mask is not None:
            ssim_map = ssim_map * mask
            if self.reduction == 'mean':
                return 1 - (ssim_map.sum() / (mask.sum() + 1e-8))
        else:
            if self.reduction == 'mean':
                return 1 - ssim_map.mean()

        return 1 - ssim_map


class CombinedLoss(nn.Module):
    """
    Combination of multiple losses for fat fraction prediction.
    Typically combines pixel-wise loss (MSE/L1) with structural loss (SSIM).
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        l1_weight: float = 0.0,
        ssim_weight: float = 0.0,
        smooth_l1_weight: float = 0.0,
        mean_ff_weight: float = 0.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            mse_weight: Weight for MSE loss
            l1_weight: Weight for L1 loss
            ssim_weight: Weight for SSIM loss
            smooth_l1_weight: Weight for Smooth L1 loss
            reduction: Reduction method
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.smooth_l1_weight = smooth_l1_weight
        self.mean_ff_weight = mean_ff_weight

        # Initialize loss functions
        self.mse_loss = MaskedMSELoss(reduction=reduction) if mse_weight > 0 else None
        self.l1_loss = MaskedL1Loss(reduction=reduction) if l1_weight > 0 else None
        self.ssim_loss = SSIMLoss(reduction=reduction) if ssim_weight > 0 else None
        self.smooth_l1_loss = MaskedSmoothL1Loss(reduction=reduction) if smooth_l1_weight > 0 else None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            pred: Predictions [B, C, D, H, W]
            target: Ground truth [B, C, D, H, W]
            mask: Binary mask [B, C, D, H, W] or None

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of individual loss components
        """
        total_loss = 0.0
        loss_dict = {}

        if self.mse_loss is not None and self.mse_weight > 0:
            mse = self.mse_loss(pred, target, mask)
            total_loss += self.mse_weight * mse
            loss_dict['mse'] = mse.item()

        if self.l1_loss is not None and self.l1_weight > 0:
            l1 = self.l1_loss(pred, target, mask)
            total_loss += self.l1_weight * l1
            loss_dict['l1'] = l1.item()

        if self.smooth_l1_loss is not None and self.smooth_l1_weight > 0:
            smooth_l1 = self.smooth_l1_loss(pred, target, mask)
            total_loss += self.smooth_l1_weight * smooth_l1
            loss_dict['smooth_l1'] = smooth_l1.item()

        if self.ssim_loss is not None and self.ssim_weight > 0:
            ssim = self.ssim_loss(pred, target, mask)
            total_loss += self.ssim_weight * ssim
            loss_dict['ssim'] = ssim.item()

        if self.mean_ff_weight > 0:
            if mask is None:
                mask = torch.ones_like(pred)
            masked_pred = pred * mask
            masked_target = target * mask
            pred_mean = masked_pred.sum(dim=(1, 2, 3, 4)) / (mask.sum(dim=(1, 2, 3, 4)) + 1e-8)
            target_mean = masked_target.sum(dim=(1, 2, 3, 4)) / (mask.sum(dim=(1, 2, 3, 4)) + 1e-8)
            mean_ff_loss = torch.mean((pred_mean - target_mean) ** 2)
            total_loss += self.mean_ff_weight * mean_ff_loss
            loss_dict['mean_ff'] = mean_ff_loss.item()

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


class WeightedMaskedLoss(nn.Module):
    """
    Masked loss with additional weighting for high fat fraction regions.
    Useful for emphasizing clinically important high fat content areas.
    """

    def __init__(
        self,
        base_loss: str = 'mse',
        high_ff_threshold: float = 0.15,
        high_ff_weight: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            base_loss: 'mse', 'l1', or 'smooth_l1'
            high_ff_threshold: Threshold above which to apply higher weight
            high_ff_weight: Weight multiplier for high FF regions
            reduction: Reduction method
        """
        super().__init__()
        self.high_ff_threshold = high_ff_threshold
        self.high_ff_weight = high_ff_weight
        self.reduction = reduction

        if base_loss == 'mse':
            self.base_loss = MaskedMSELoss(reduction='none')
        elif base_loss == 'l1':
            self.base_loss = MaskedL1Loss(reduction='none')
        elif base_loss == 'smooth_l1':
            self.base_loss = MaskedSmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unknown base loss: {base_loss}")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(pred)

        # Compute base loss
        loss = self.base_loss(pred, target, mask)

        # Create weight map
        weight = torch.ones_like(target)
        high_ff_mask = (target > self.high_ff_threshold).float()
        weight = weight + high_ff_mask * (self.high_ff_weight - 1.0)

        # Apply weights and mask
        weighted_loss = loss * weight * mask

        if self.reduction == 'mean':
            return weighted_loss.sum() / (mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class ContinuousWeightedMaskedLoss(nn.Module):
    """
    Masked loss with continuous weighting based on fat fraction magnitude.
    Useful for emphasizing higher fat fractions without hard thresholds.
    """

    def __init__(
        self,
        base_loss: str = 'mse',
        alpha: float = 4.0,
        gamma: float = 2.0,
        max_weight: Optional[float] = None,
        reduction: str = 'mean'
    ):
        """
        Args:
            base_loss: 'mse', 'l1', or 'smooth_l1'
            alpha: Weight scale applied to target**gamma
            gamma: Exponent for target weighting
            max_weight: Optional cap for weight values
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.max_weight = max_weight
        self.reduction = reduction

        if base_loss == 'mse':
            self.base_loss = MaskedMSELoss(reduction='none')
        elif base_loss == 'l1':
            self.base_loss = MaskedL1Loss(reduction='none')
        elif base_loss == 'smooth_l1':
            self.base_loss = MaskedSmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unknown base loss: {base_loss}")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(pred)

        loss = self.base_loss(pred, target, mask)
        weight = 1.0 + self.alpha * torch.pow(target.clamp_min(0.0), self.gamma)
        if self.max_weight is not None:
            weight = torch.clamp(weight, max=self.max_weight)

        weighted_loss = loss * weight * mask

        if self.reduction == 'mean':
            return weighted_loss.sum() / (mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


def get_loss_function(loss_config: dict) -> nn.Module:
    """
    Factory function to create loss function from config.

    Args:
        loss_config: Dictionary with loss configuration
            Example:
            {
                'type': 'combined',  # 'mse', 'l1', 'combined', 'weighted'
                'mse_weight': 1.0,
                'l1_weight': 0.0,
                'ssim_weight': 0.1,
                'smooth_l1_weight': 0.0,
                'reduction': 'mean'
            }

    Returns:
        Loss function module
    """
    loss_type = loss_config.get('type', 'mse')

    if loss_type == 'mse':
        return MaskedMSELoss(reduction=loss_config.get('reduction', 'mean'))
    elif loss_type == 'l1':
        return MaskedL1Loss(reduction=loss_config.get('reduction', 'mean'))
    elif loss_type == 'smooth_l1':
        return MaskedSmoothL1Loss(
            beta=loss_config.get('beta', 1.0),
            reduction=loss_config.get('reduction', 'mean')
        )
    elif loss_type == 'combined':
        return CombinedLoss(
            mse_weight=loss_config.get('mse_weight', 1.0),
            l1_weight=loss_config.get('l1_weight', 0.0),
            ssim_weight=loss_config.get('ssim_weight', 0.0),
            smooth_l1_weight=loss_config.get('smooth_l1_weight', 0.0),
            mean_ff_weight=loss_config.get('mean_ff_weight', 0.0),
            reduction=loss_config.get('reduction', 'mean')
        )
    elif loss_type == 'weighted':
        return WeightedMaskedLoss(
            base_loss=loss_config.get('base_loss', 'mse'),
            high_ff_threshold=loss_config.get('high_ff_threshold', 0.15),
            high_ff_weight=loss_config.get('high_ff_weight', 2.0),
            reduction=loss_config.get('reduction', 'mean')
        )
    elif loss_type == 'continuous_weighted':
        return ContinuousWeightedMaskedLoss(
            base_loss=loss_config.get('base_loss', 'mse'),
            alpha=loss_config.get('alpha', 4.0),
            gamma=loss_config.get('gamma', 2.0),
            max_weight=loss_config.get('max_weight'),
            reduction=loss_config.get('reduction', 'mean')
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dummy data
    pred = torch.rand(2, 1, 32, 64, 64).to(device)
    target = torch.rand(2, 1, 32, 64, 64).to(device)
    mask = (torch.rand(2, 1, 32, 64, 64) > 0.3).float().to(device)

    print("Testing loss functions...")

    # Test MSE loss
    mse_loss = MaskedMSELoss()
    loss_mse = mse_loss(pred, target, mask)
    print(f"Masked MSE Loss: {loss_mse.item():.6f}")

    # Test L1 loss
    l1_loss = MaskedL1Loss()
    loss_l1 = l1_loss(pred, target, mask)
    print(f"Masked L1 Loss: {loss_l1.item():.6f}")

    # Test combined loss
    combined_loss = CombinedLoss(mse_weight=1.0, l1_weight=0.5, ssim_weight=0.1)
    loss_combined, loss_dict = combined_loss(pred, target, mask)
    print(f"Combined Loss: {loss_combined.item():.6f}")
    print(f"Loss components: {loss_dict}")

    # Test weighted loss
    weighted_loss = WeightedMaskedLoss(base_loss='mse', high_ff_threshold=0.15, high_ff_weight=2.0)
    loss_weighted = weighted_loss(pred, target, mask)
    print(f"Weighted Loss: {loss_weighted.item():.6f}")

    print("\nAll loss functions working correctly!")
