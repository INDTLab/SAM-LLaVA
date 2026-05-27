"""Loss functions and metrics for SAM‑LLaVA training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute the Dice loss between predicted and ground truth masks.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted mask of shape (B,1,H,W) with values in [0,1].
    target : torch.Tensor
        Ground truth mask of shape (B,1,H,W) with binary values.
    eps : float, default=1e-6
        Small constant to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Dice loss averaged over the batch.
    """
    B = pred.shape[0]
    pred_flat = pred.view(B, -1)
    target_flat = target.view(B, -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    loss = 1 - dice
    return loss.mean()


def bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute binary cross entropy loss between predicted and ground truth masks."""
    return F.binary_cross_entropy(pred, target)


def segmentation_loss(
    pred_masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    target_mask: torch.Tensor,
    weights: tuple[float, float, float] = (0.3, 1.0, 0.2),
    lambda_focal: float = 0.0,
) -> torch.Tensor:
    """Compute the multi‑scale segmentation loss.

    The original paper uses a weighted sum of dice and focal/BCE losses【409945085282613†L409-L423】.  Here we
    implement dice + BCE loss for each scale and average them with given
    weights.  Focal loss can be incorporated by setting `lambda_focal` > 0,
    though it is optional in this simplified version.

    Parameters
    ----------
    pred_masks : tuple of tensors
        Tuple containing high, medium and low resolution predicted masks.
    target_mask : torch.Tensor
        Ground truth mask at the input resolution (B,1,H,W).
    weights : tuple of floats, default=(0.3, 1.0, 0.2)
        Weights for high, medium and low resolutions【409945085282613†L409-L423】.
    lambda_focal : float, default=0.0
        Weight for the optional focal loss term.  Set to zero to omit focal loss.

    Returns
    -------
    torch.Tensor
        Total segmentation loss.
    """
    high_pred, mid_pred, low_pred = pred_masks
    B, _, H, W = target_mask.shape
    # Downsample ground truth for medium and low resolutions
    target_mid = F.interpolate(target_mask, size=mid_pred.shape[2:], mode="nearest")
    target_low = F.interpolate(target_mask, size=low_pred.shape[2:], mode="nearest")
    # Compute losses for each scale
    losses = []
    for pred, tgt, w in zip((high_pred, mid_pred, low_pred), (target_mask, target_mid, target_low), weights):
        dice = dice_loss(pred, tgt)
        bce = bce_loss(pred, tgt)
        loss = dice + bce
        losses.append(w * loss)
    return sum(losses)


def language_loss(logits: torch.Tensor, target_tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Compute cross entropy loss for language modelling.

    Parameters
    ----------
    logits : torch.Tensor
        Logits from the language model of shape (B,L,V).
    target_tokens : torch.Tensor
        Target token IDs of shape (B,L).
    pad_id : int
        ID of the padding token, which is ignored in the loss.

    Returns
    -------
    torch.Tensor
        Cross entropy loss averaged over non‑padding tokens.
    """
    B, L, V = logits.shape
    logits_flat = logits.view(-1, V)
    targets_flat = target_tokens.view(-1)
    loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_id)
    return loss