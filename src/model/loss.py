"""Owner: Person 3 (Part C). Loss functions for VRGAE link prediction.

Combines link prediction loss with KL divergence warm-up:
total_loss = link_loss + kl_weight(epoch) * KL.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.latent import VariationalLatent


def link_loss(pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
    """Compute binary cross-entropy with logits link prediction loss.
    
    pos_score: [B], neg_score: [B, K] or [B * K] -> scalar
    """
    pos_flat = pos_score.reshape(-1)
    neg_flat = neg_score.reshape(-1)

    scores = torch.cat([pos_flat, neg_flat], dim=0)
    labels = torch.cat([
        torch.ones_like(pos_flat),
        torch.zeros_like(neg_flat),
    ], dim=0)

    return F.binary_cross_entropy_with_logits(scores, labels)


def kl_weight(epoch: int, warmup_epochs: int, max_weight: float = 1.0) -> float:
    """Linear KL warm-up from 0 to max_weight over warmup_epochs."""
    if warmup_epochs <= 0:
        return max_weight
    if epoch >= warmup_epochs:
        return max_weight
    return max_weight * (epoch / warmup_epochs)


def total_loss(
    pos_score: torch.Tensor,
    neg_score: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    epoch: int,
    warmup_epochs: int = 20,
    max_kl_weight: float = 0.001,
) -> torch.Tensor:
    """Total VRGAE loss: link_loss + beta(epoch) * KL."""
    recon_loss = link_loss(pos_score, neg_score)
    kl_div = VariationalLatent.kl_divergence(mu, logvar)
    beta = kl_weight(epoch, warmup_epochs, max_weight=max_kl_weight)
    return recon_loss + beta * kl_div
