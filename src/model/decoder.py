"""Owner: Person 3 (Part C). RotatE decoder.  TODO: implement."""
import torch
import torch.nn as nn


class RotatEDecoder(nn.Module):
    """Relations are rotations: score(h, r, t) = gamma - || z_h * rot(r) - z_t ||  (complex space)."""

    def __init__(self, num_base_relations: int = 6, dim: int = 128, gamma: float = 6.0):
        super().__init__()
        raise NotImplementedError

    def score(self, z: torch.Tensor, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """z [N, dim]; h, r, t LongTensor [B] -> scores [B]."""
        raise NotImplementedError

    def score_all_tails(self, z: torch.Tensor, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """-> scores [B, N] for every candidate tail (used by the evaluator)."""
        raise NotImplementedError
