"""
Loss functions for seizure forecasting.

Multi-task loss:
- Binary classification (preictal vs interictal)
- Soft risk regression (continuous risk score)
- Sample weighting for temporal proximity
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeizureForecastingLoss(nn.Module):
    """
    Multi-task loss for seizure forecasting.
    
    L = BCE(y_cls_logit, y_cls) + lambda * MSE(y_soft_pred, y_soft)
    
    With optional sample weighting for preictal windows closer to onset.
    """
    
    def __init__(
        self,
        pos_weight: Optional[float] = None,
        lambda_soft: float = 0.5,
        weight_by_proximity: bool = True,
        preictal_sec: float = 600.0,
    ):
        """
        Args:
            pos_weight: Weight for positive class in BCE
            lambda_soft: Weight for soft risk regression loss
            weight_by_proximity: Whether to weight samples by proximity to onset
            preictal_sec: Preictal horizon in seconds (for proximity weighting)
        """
        super().__init__()
        
        self.lambda_soft = lambda_soft
        self.weight_by_proximity = weight_by_proximity
        self.preictal_sec = preictal_sec
        
        # BCE loss with pos_weight
        if pos_weight is not None:
            self.pos_weight = torch.tensor([pos_weight])
        else:
            self.pos_weight = None
    
    def forward(
        self,
        cls_logit: torch.Tensor,
        soft_pred: torch.Tensor,
        y_cls: torch.Tensor,
        y_soft: torch.Tensor,
        y_tte: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Args:
            cls_logit: Classification logits [B, 1]
            soft_pred: Soft risk predictions [B, 1]
            y_cls: Binary classification targets [B]
            y_soft: Soft risk targets [B]
            y_tte: Time-to-event in seconds [B] (for proximity weighting)
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        batch_size = y_cls.shape[0]
        device = cls_logit.device
        
        # Flatten predictions
        cls_logit = cls_logit.view(-1)
        soft_pred = soft_pred.view(-1)
        
        # Move pos_weight to correct device
        pos_weight = None
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(device)
        
        # Classification loss (BCE)
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_logit,
            y_cls.float(),
            pos_weight=pos_weight,
            reduction="none",
        )
        
        # Soft risk regression loss (MSE)
        soft_loss = F.mse_loss(soft_pred, y_soft, reduction="none")
        
        # Sample weighting by proximity to onset
        if self.weight_by_proximity and y_tte is not None:
            # Compute weights: higher for preictal windows closer to onset
            # w = 1 + (1 - t / preictal_sec) for preictal
            # w = 1 for interictal
            weights = torch.ones(batch_size, device=device)
            preictal_mask = y_cls == 1
            
            if preictal_mask.any():
                t = y_tte[preictal_mask]
                w = 1 + (1 - t / self.preictal_sec).clamp(0, 1)
                weights[preictal_mask] = w
            
            cls_loss = cls_loss * weights
            soft_loss = soft_loss * weights
        
        # Aggregate losses
        cls_loss_mean = cls_loss.mean()
        soft_loss_mean = soft_loss.mean()
        
        # Combined loss
        total_loss = cls_loss_mean + self.lambda_soft * soft_loss_mean
        
        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_cls": cls_loss_mean.item(),
            "loss_soft": soft_loss_mean.item(),
        }
        
        return total_loss, loss_dict


def compute_pos_weight(
    n_pos: int,
    n_neg: int,
) -> float:
    """
    Compute positive class weight for imbalanced data.
    
    Args:
        n_pos: Number of positive samples
        n_neg: Number of negative samples
        
    Returns:
        Positive class weight
    """
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


class FocalLoss(nn.Module):
    """
    Focal loss for hard example mining.
    
    FL(p) = -alpha * (1 - p)^gamma * log(p)
    
    Useful for extremely imbalanced data.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter
            reduction: "mean", "sum", or "none"
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Predictions [B]
            targets: Binary targets [B]
            
        Returns:
            Focal loss
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )
        
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        focal_weight = (1 - pt) ** self.gamma
        
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SeizureForecastingLossFocal(nn.Module):
    """
    Multi-task loss using Focal Loss for classification.
    Better for severe class imbalance in cross-subject training.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        lambda_soft: float = 0.5,
        weight_by_proximity: bool = True,
        preictal_sec: float = 600.0,
    ):
        super().__init__()
        
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, reduction="none")
        self.lambda_soft = lambda_soft
        self.weight_by_proximity = weight_by_proximity
        self.preictal_sec = preictal_sec
    
    def forward(
        self,
        cls_logit: torch.Tensor,
        soft_pred: torch.Tensor,
        y_cls: torch.Tensor,
        y_soft: torch.Tensor,
        y_tte: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute focal-based combined loss."""
        batch_size = y_cls.shape[0]
        device = cls_logit.device
        
        cls_logit = cls_logit.view(-1)
        soft_pred = soft_pred.view(-1)
        
        # Focal classification loss
        cls_loss = self.focal(cls_logit, y_cls)
        cls_loss_per_sample = F.binary_cross_entropy_with_logits(
            cls_logit, y_cls.float(), reduction="none"
        )
        probs = torch.sigmoid(cls_logit)
        pt = torch.where(y_cls == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.focal.gamma
        alpha_t = torch.where(y_cls == 1, self.focal.alpha, 1 - self.focal.alpha)
        cls_loss_weighted = alpha_t * focal_weight * cls_loss_per_sample
        
        # Soft risk regression loss
        soft_loss = F.mse_loss(soft_pred, y_soft, reduction="none")
        
        # Proximity weighting
        if self.weight_by_proximity and y_tte is not None:
            weights = torch.ones(batch_size, device=device)
            preictal_mask = y_cls == 1
            if preictal_mask.any():
                t = y_tte[preictal_mask]
                w = 1 + (1 - t / self.preictal_sec).clamp(0, 1)
                weights[preictal_mask] = w
            cls_loss_weighted = cls_loss_weighted * weights
            soft_loss = soft_loss * weights
        
        cls_loss_mean = cls_loss_weighted.mean()
        soft_loss_mean = soft_loss.mean()
        total_loss = cls_loss_mean + self.lambda_soft * soft_loss_mean
        
        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_cls": cls_loss_mean.item(),
            "loss_soft": soft_loss_mean.item(),
        }


def create_loss_fn(
    loss_type: str = "bce",
    pos_weight: Optional[float] = None,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    lambda_soft: float = 0.5,
    weight_by_proximity: bool = True,
    preictal_sec: float = 600.0,
):
    """
    Factory function to create loss based on config.
    
    Args:
        loss_type: "bce" or "focal"
        pos_weight: Positive class weight for BCE
        focal_alpha: Alpha for focal loss
        focal_gamma: Gamma for focal loss
        lambda_soft: Weight for soft regression loss
        weight_by_proximity: Whether to weight by proximity
        preictal_sec: Preictal window in seconds
        
    Returns:
        Loss module
    """
    if loss_type == "focal":
        return SeizureForecastingLossFocal(
            alpha=focal_alpha,
            gamma=focal_gamma,
            lambda_soft=lambda_soft,
            weight_by_proximity=weight_by_proximity,
            preictal_sec=preictal_sec,
        )
    else:  # bce
        return SeizureForecastingLoss(
            pos_weight=pos_weight,
            lambda_soft=lambda_soft,
            weight_by_proximity=weight_by_proximity,
            preictal_sec=preictal_sec,
        )
