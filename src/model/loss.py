"""Owner: Person 3 (Part C). Losses.  TODO: implement."""
import torch


def link_loss(pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
    """pos_score [B], neg_score [B, K] -> scalar (e.g. BCE with logits / margin loss)."""
    raise NotImplementedError


def kl_weight(epoch: int, warmup_epochs: int, max_weight: float = 1.0) -> float:
    """Linear KL warm-up from 0 to max_weight."""
    raise NotImplementedError


def total_loss(pos_score, neg_score, mu, logvar, epoch: int, warmup_epochs: int) -> torch.Tensor:
    """link loss + kl_weight(epoch) * KL."""
    raise NotImplementedError
