"""Owner: Person 3 (Part C). Decoders for link prediction: ComplEx and RotatE.

Both decoders map complex entity representations and relation parameters into scores,
supporting both batch scoring: score(z, h, r, t) -> [B]
and full candidate scoring: score_all_tails(z, h, r) -> [B, N] for filtered ranking evaluation.

RotatE details:
  Entities are represented as points in complex space C^(d/2).
  Relations are represented as rotations: r = exp(i * theta_r).
  Scoring function: score(h, r, t) = gamma - || z_h * rot(r) - z_t ||.
  Supports inverse relations, batched triple scoring, and score_all_tails for filtered ranking.
"""
import math
from typing import Tuple

import torch
import torch.nn as nn


class ComplExDecoder(nn.Module):
    """ComplEx decoder: Hermitian dot product in complex space.

    score(h, r, t) = Re(<z_h * w_r, conj(z_t)>)
                   = <h_re * r_re - h_im * r_im, t_re> + <h_re * r_im + h_im * r_re, t_im>
    Naturally asymmetric, allowing it to distinguish directed relations like ManifestAs and Damage.
    """

    def __init__(self, num_base_relations: int = 6, dim: int = 128):
        super().__init__()
        self.num_relations = num_base_relations
        self.dim = dim
        self.complex_dim = dim // 2
        assert dim % 2 == 0, f"dim must be even for complex embeddings, got {dim}"

        self.rel_re = nn.Embedding(num_base_relations, self.complex_dim)
        self.rel_im = nn.Embedding(num_base_relations, self.complex_dim)
        nn.init.xavier_uniform_(self.rel_re.weight)
        nn.init.xavier_uniform_(self.rel_im.weight)

    def score(self, z: torch.Tensor, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute ComplEx scores for triples (h, r, t).

        z: [N, dim], h: [B], r: [B], t: [B] -> [B]
        """
        h_emb = z[h]
        t_emb = z[t]

        h_re, h_im = h_emb[..., :self.complex_dim], h_emb[..., self.complex_dim:]
        t_re, t_im = t_emb[..., :self.complex_dim], t_emb[..., self.complex_dim:]
        r_re = self.rel_re(r)
        r_im = self.rel_im(r)

        score_re = (h_re * r_re - h_im * r_im) * t_re
        score_im = (h_re * r_im + h_im * r_re) * t_im
        return (score_re + score_im).sum(dim=-1)

    def score_all_tails(self, z: torch.Tensor, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Compute ComplEx score for all candidate tails given queries (h, r).

        z: [N, dim], h: [B], r: [B] -> [B, N]
        """
        h_emb = z[h]
        h_re, h_im = h_emb[..., :self.complex_dim], h_emb[..., self.complex_dim:]
        r_re = self.rel_re(r)
        r_im = self.rel_im(r)

        q_re = h_re * r_re - h_im * r_im  # [B, d/2]
        q_im = h_re * r_im + h_im * r_re  # [B, d/2]

        t_re = z[:, :self.complex_dim]    # [N, d/2]
        t_im = z[:, self.complex_dim:]    # [N, d/2]

        return (q_re @ t_re.t()) + (q_im @ t_im.t())  # [B, N]


class RotatEDecoder(nn.Module):
    """Relations are rotations: score(h, r, t) = gamma - || z_h * rot(r) - z_t || (complex space).

    Implements complex-valued RotatE (Sun et al., 2019):
      score(h, r, t) = gamma - || z_h o rot(r) - z_t ||_1
    where |rot(r)_j| = 1 via Euler's formula: rot(r) = cos(theta_r) + i * sin(theta_r).
    Supports inverse relations: theta_inv = -theta_base.
    """

    def __init__(self, num_base_relations: int = 6, dim: int = 128, gamma: float = 6.0, epsilon: float = 2.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Embedding dimension {dim} must be even for complex RotatE decoder.")

        self.num_base_relations = num_base_relations
        self.dim = dim
        self.k = dim // 2  # Number of complex dimensions
        self.gamma = gamma
        self.epsilon = epsilon

        # Relation rotation angles in radians: theta_r in [0, 2*pi)
        self.rel_theta = nn.Embedding(num_base_relations, self.k)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize angles uniformly over [0, 2*pi)
        nn.init.uniform_(self.rel_theta.weight, a=0.0, b=2.0 * math.pi)

    def get_relation_rotation(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Obtain real and imaginary rotation factors cos(theta) and sin(theta).

        Handles inverse relations:
        If r >= num_base_relations, the relation is an inverse edge r_inv = r_base + num_base_relations.
        In RotatE, the inverse relation is the inverse rotation: theta_inv = -theta_base.
        """
        is_inverse = (r >= self.num_base_relations)
        base_r = torch.where(is_inverse, r - self.num_base_relations, r)
        theta = self.rel_theta(base_r)

        # Invert angle for inverse relations: theta_inv = -theta_base
        theta = torch.where(is_inverse.unsqueeze(-1), -theta, theta)

        rel_re = torch.cos(theta)
        rel_im = torch.sin(theta)
        return rel_re, rel_im

    def rotate(self, head: torch.Tensor, rel_re: torch.Tensor, rel_im: torch.Tensor) -> torch.Tensor:
        """Perform complex rotation: head * rot(r).

        head: [..., dim]
        rel_re, rel_im: [..., k]
        returns: rotated vector of shape [..., dim]
        """
        re_h = head[..., :self.k]
        im_h = head[..., self.k:]

        # Complex multiplication: (re_h + i*im_h) * (rel_re + i*rel_im)
        re_rot = re_h * rel_re - im_h * rel_im
        im_rot = re_h * rel_im + im_h * rel_re

        return torch.cat([re_rot, im_rot], dim=-1)

    def score(self, z: torch.Tensor, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute RotatE score for triples (h, r, t).

        z: [N, dim] node representations from latent layer.
        h: LongTensor [B]
        r: LongTensor [B]
        t: LongTensor [B] or [B, K]
        returns: FloatTensor [B] or [B, K]
        """
        head = z[h]  # [B, dim]
        rel_re, rel_im = self.get_relation_rotation(r)  # [B, k]

        # Rotate head
        pred = self.rotate(head, rel_re, rel_im)  # [B, dim]

        if t.ndim == 1:
            tail = z[t]  # [B, dim]
            re_diff = pred[:, :self.k] - tail[:, :self.k]
            im_diff = pred[:, self.k:] - tail[:, self.k:]
            # Numerically stable L1 distance in complex space
            dist = torch.sum(torch.sqrt(re_diff.pow(2) + im_diff.pow(2) + 1e-12), dim=-1)
            return self.gamma - dist
        elif t.ndim == 2:
            # Broadcast against multiple tails [B, K, dim]
            tail = z[t]  # [B, K, dim]
            pred = pred.unsqueeze(1)  # [B, 1, dim]
            re_diff = pred[..., :self.k] - tail[..., :self.k]
            im_diff = pred[..., self.k:] - tail[..., self.k:]
            dist = torch.sum(torch.sqrt(re_diff.pow(2) + im_diff.pow(2) + 1e-12), dim=-1)
            return self.gamma - dist
        else:
            raise ValueError(f"Unsupported tail tensor dimension: {t.ndim}")

    def score_all_tails(self, z: torch.Tensor, h: torch.Tensor, r: torch.Tensor, chunk_size: int = 100) -> torch.Tensor:
        """Compute RotatE score for all candidate tails given (h, r).

        z: [N, dim] node representations.
        h: LongTensor [B]
        r: LongTensor [B]
        returns: FloatTensor [B, N] of scores for all entities as candidate tails.
        Chunked evaluation prevents excessive memory consumption.
        """
        B = h.shape[0]
        N = z.shape[0]
        rel_re, rel_im = self.get_relation_rotation(r)  # [B, k]
        head = z[h]  # [B, dim]
        pred = self.rotate(head, rel_re, rel_im)  # [B, dim]

        q_re = pred[:, :self.k]  # [B, k]
        q_im = pred[:, self.k:]  # [B, k]

        t_re = z[:, :self.k]    # [N, k]
        t_im = z[:, self.k:]    # [N, k]

        scores = torch.empty((B, N), dtype=torch.float32, device=z.device)

        for i in range(0, B, chunk_size):
            chunk_q_re = q_re[i:i + chunk_size]  # [b, k]
            chunk_q_im = q_im[i:i + chunk_size]  # [b, k]

            # [b, 1, k] - [1, N, k] -> [b, N, k]
            diff_re = chunk_q_re.unsqueeze(1) - t_re.unsqueeze(0)
            diff_im = chunk_q_im.unsqueeze(1) - t_im.unsqueeze(0)

            chunk_dist = torch.sum(torch.sqrt(diff_re.pow(2) + diff_im.pow(2) + 1e-12), dim=-1)
            scores[i:i + chunk_size] = self.gamma - chunk_dist

        return scores
