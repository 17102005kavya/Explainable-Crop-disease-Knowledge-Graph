"""Owner: Person 3 (Part C). Losses.

Implements link prediction loss (BCE with logits and margin-based),
linear KL warm-up schedule, and composite total loss combining link prediction
and latent KL divergence regularization.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from src.model.latent import VariationalLatent


def link_loss(
    pos_score: torch.Tensor,
    neg_score: torch.Tensor,
    loss_type: str = "bce",
    margin: float = 1.0,
    balanced: bool = True,
) -> torch.Tensor:
    """Compute link prediction loss for positive and negative triple scores.

    pos_score: [B] or [B, 1]
    neg_score: [B, K] or [M]
    loss_type: 'bce' (binary cross entropy with logits) or 'margin'
    balanced: if True, balances positive and negative contributions equally.
    """
    pos_flat = pos_score.reshape(-1)
    neg_flat = neg_score.reshape(-1)

    if loss_type == "bce":
        pos_target = torch.ones_like(pos_flat)
        neg_target = torch.zeros_like(neg_flat)

        if balanced:
            pos_loss = F.binary_cross_entropy_with_logits(pos_flat, pos_target)
            neg_loss = F.binary_cross_entropy_with_logits(neg_flat, neg_target)
            return 0.5 * (pos_loss + neg_loss)
        else:
            all_scores = torch.cat([pos_flat, neg_flat], dim=0)
            all_targets = torch.cat([pos_target, neg_target], dim=0)
            return F.binary_cross_entropy_with_logits(all_scores, all_targets)

    elif loss_type == "margin":
        # Pairwise margin loss: max(0, margin - pos_score + neg_score)
        if neg_score.ndim == 2 and pos_score.ndim == 1:
            pos_expanded = pos_score.unsqueeze(1)  # [B, 1]
            diff = margin - pos_expanded + neg_score
        else:
            diff = margin - pos_flat + neg_flat
        return torch.clamp(diff, min=0.0).mean()

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def kl_weight(epoch: int, warmup_epochs: int, max_weight: float = 1.0) -> float:
    """Linear KL warm-up from 0.0 to max_weight over warmup_epochs.

    epoch: current training epoch (1-indexed or 0-indexed).
    warmup_epochs: number of epochs to ramp up the weight.
    max_weight: maximum KL weight coefficient.
    """
    if warmup_epochs <= 0:
        return float(max_weight)
    if epoch <= 0:
        return 0.0
    if epoch >= warmup_epochs:
        return float(max_weight)
    return float(max_weight * (epoch / warmup_epochs))


def total_loss(
    pos_score: torch.Tensor,
    neg_score: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    epoch: int = 1,
    warmup_epochs: int = 20,
    max_kl_weight: float = 1.0,
    kl_weight_override: Optional[float] = None,
    loss_type: str = "bce",
) -> torch.Tensor:
    """Calculate composite loss: link_loss + kl_weight(epoch) * KL_divergence.

    Returns scalar loss tensor ready for backward().
    """
    l_link = link_loss(pos_score, neg_score, loss_type=loss_type)
    l_kl = VariationalLatent.kl_divergence(mu, logvar)

    beta = kl_weight_override if kl_weight_override is not None else kl_weight(epoch, warmup_epochs, max_kl_weight)
    loss = l_link + beta * l_kl
    return loss


def compute_loss_breakdown(
    pos_score: torch.Tensor,
    neg_score: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    epoch: int = 1,
    warmup_epochs: int = 20,
    max_kl_weight: float = 1.0,
    kl_weight_override: Optional[float] = None,
    loss_type: str = "bce",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute total loss along with a dictionary of individual loss metrics for logging."""
    l_link = link_loss(pos_score, neg_score, loss_type=loss_type)
    l_kl = VariationalLatent.kl_divergence(mu, logvar)

    beta = kl_weight_override if kl_weight_override is not None else kl_weight(epoch, warmup_epochs, max_kl_weight)
    total = l_link + beta * l_kl

    metrics = {
        "loss": total.item(),
        "link_loss": l_link.item(),
        "kl_loss": l_kl.item(),
        "kl_weight": beta,
    }
    return total, metrics
