
"""Link prediction and variational losses for VRGAE."""

import torch
import torch.nn.functional as F

from src.model.latent import VariationalLatent


def link_loss(
    pos_score: torch.Tensor,
    neg_score: torch.Tensor,
    loss_type: str = "bce",
    margin: float = 1.0,
) -> torch.Tensor:
    """Calculate BCE or margin-based link prediction loss."""
    loss_type = loss_type.lower()

    if loss_type == "bce":
        pos_loss = F.softplus(-pos_score).mean()
        neg_loss = F.softplus(neg_score).mean()
        return 0.5 * (pos_loss + neg_loss)

    if loss_type == "margin":
        if neg_score.ndim > pos_score.ndim:
            pos_score = pos_score.unsqueeze(-1)
        return F.relu(margin - pos_score + neg_score).mean()

    raise ValueError(
        f"Unsupported loss_type: {loss_type!r}. Use 'bce' or 'margin'."
    )


def kl_weight(
    epoch: int,
    warmup_epochs: int,
    max_weight: float = 1.0,
) -> float:
    """Linearly warm up the KL weight from zero to max_weight."""
    if warmup_epochs <= 0:
        return float(max_weight)

    progress = min(max(float(epoch), 0.0) / warmup_epochs, 1.0)
    return float(max_weight * progress)


def total_loss(
    pos_score: torch.Tensor,
    neg_score: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    epoch: int,
    warmup_epochs: int,
    max_kl_weight: float = 1.0,
    loss_type: str = "bce",
    margin: float = 1.0,
) -> torch.Tensor:
    """Combine link prediction and weighted KL-divergence losses."""
    link = link_loss(pos_score, neg_score, loss_type, margin)
    kl = VariationalLatent.kl_divergence(mu, logvar)
    weight = kl_weight(epoch, warmup_epochs, max_kl_weight)
    return link + weight * kl


def compute_loss_breakdown(
    pos_score: torch.Tensor,
    neg_score: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    epoch: int,
    warmup_epochs: int,
    max_kl_weight: float = 1.0,
    loss_type: str = "bce",
    margin: float = 1.0,
):
    """Return the differentiable total loss and detached metric values."""
    link = link_loss(pos_score, neg_score, loss_type, margin)
    kl = VariationalLatent.kl_divergence(mu, logvar)
    weight = kl_weight(epoch, warmup_epochs, max_kl_weight)
    loss = link + weight * kl

    metrics = {
        "loss": loss.detach().item(),
        "link_loss": link.detach().item(),
        "kl_loss": kl.detach().item(),
        "kl_weight": weight,
    }
    return loss, metrics
