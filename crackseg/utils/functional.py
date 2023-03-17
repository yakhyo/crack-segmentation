from typing import Optional

import torch
from torch.nn import functional as F

__all__ = ["sigmoid_focal_loss", "cross_entropy", "dice_loss"]


def weight_reduce_loss(loss, weight=None, reduction="mean"):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "none":
        return loss

    return loss


def cross_entropy(inputs, targets, weight=None, class_weight=None, reduction="mean", ignore_index=-100):
    loss = F.cross_entropy(inputs, targets, weight=class_weight, reduction="none", ignore_index=ignore_index)

    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction)

    return loss


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "none",
        eps: float = 1e-5,
) -> torch.Tensor:
    inputs = F.softmax(inputs, dim=1)
    targets = F.one_hot(targets, inputs.shape[1]).permute(0, 3, 1, 2)

    if inputs.shape != targets.shape:
        raise AssertionError(f"Ground truth has different shape ({targets.shape}) from input ({inputs.shape})")

    # flatten prediction and label tensors
    inputs = inputs.flatten()
    targets = targets.flatten()

    intersection = torch.sum(inputs * targets)
    denominator = torch.sum(inputs) + torch.sum(targets)

    # calculate the dice loss
    dice_score = (2.0 * intersection + eps) / (denominator + eps)
    loss = 1 - dice_score

    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(inputs)
    loss = weight_reduce_loss(loss, weight, reduction=reduction)

    return loss


def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
) -> torch.Tensor:
    probs = torch.sigmoid(inputs)
    targets = F.one_hot(targets, inputs.shape[1]).permute(0, 3, 1, 2)

    inputs = inputs.float()
    targets = targets.float()

    if inputs.shape != targets.shape:
        raise AssertionError(f"Ground truth has different shape ({targets.shape}) from input ({inputs.shape})")
    pt = (1 - probs) * targets + probs * (1 - targets)
    focal_weight = (alpha * targets + (1 - alpha) * (1 - targets)) * pt.pow(gamma)

    loss = focal_weight * F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    if weight is not None:
        assert weight.dim() == 1, f"Weight dimension must be `weight.dim()=1`, current dimension {weight.dim()}"
        weight = weight.float()
        if inputs.dim() > 1:
            weight = weight.view(-1, 1)

    loss = weight_reduce_loss(loss, weight, reduction=reduction)

    return loss
