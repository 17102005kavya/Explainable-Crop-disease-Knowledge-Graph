"""Owner: Person 3 (Part C). RotatE decoder.

Implements complex-valued RotatE knowledge graph decoder:
Entities are represented as points in complex space C^(d/2).
Relations are represented as rotations: r = exp(i * theta_r).
Scoring function: score(h, r, t) = gamma - || z_h * rot(r) - z_t ||.
Supports batched triple scoring, negative sampling scoring, inverse relations,
and score_all_tails for filtered tail prediction ranking.
"""
import math
from typing import Tuple

import torch
import torch.nn as nn


class RotatEDecoder(nn.Module):
    """Relations are rotations: score(h, r, t) = gamma - || z_h * rot(r) - z_t || (complex space)."""

    def __init__(self, num_base_relations: int = 6, dim: int = 128, gamma: float = 6.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Embedding dimension {dim} must be even for complex RotatE decoder.")

        self.num_base_relations = num_base_relations
        self.dim = dim
        self.k = dim // 2  # Number of complex dimensions
        self.gamma = float(gamma)

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
        # Lookup head and tail representations
        head = z[h]  # [B, dim]
        rel_re, rel_im = self.get_relation_rotation(r)  # [B, k]

        # Rotate head
        pred = self.rotate(head, rel_re, rel_im)  # [B, dim]

        if t.ndim == 1:
            tail = z[t]  # [B, dim]
            re_diff = pred[:, :self.k] - tail[:, :self.k]
            im_diff = pred[:, self.k:] - tail[:, self.k:]
            # Numerically stable Euclidean distance in complex space
            dist_sq = re_diff.pow(2) + im_diff.pow(2)
            dist = torch.sqrt(torch.sum(dist_sq, dim=-1) + 1e-12)
            return self.gamma - dist
        elif t.ndim == 2:
            # Broadcast against multiple tails [B, K, dim]
            tail = z[t]  # [B, K, dim]
            pred = pred.unsqueeze(1)  # [B, 1, dim]
            re_diff = pred[..., :self.k] - tail[..., :self.k]
            im_diff = pred[..., self.k:] - tail[..., self.k:]
            dist_sq = re_diff.pow(2) + im_diff.pow(2)
            dist = torch.sqrt(torch.sum(dist_sq, dim=-1) + 1e-12)
            return self.gamma - dist
        else:
            raise ValueError(f"Unsupported tail tensor dimension: {t.ndim}")

    def score_all_tails(self, z: torch.Tensor, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Compute RotatE score for all candidate tails given (h, r).

        z: [N, dim] node representations.
        h: LongTensor [B]
        r: LongTensor [B]
        returns: FloatTensor [B, N] of scores for all entities as candidate tails.
        """
        B = h.size(0)
        N = z.size(0)

        head = z[h]  # [B, dim]
        rel_re, rel_im = self.get_relation_rotation(r)  # [B, k]
        pred = self.rotate(head, rel_re, rel_im)  # [B, dim]

        # In complex space with unitary rotation: ||pred_b - z_n||^2 = ||pred_b||^2 + ||z_n||^2 - 2 * <pred_b, z_n>
        pred_norm_sq = (pred.pow(2)).sum(dim=-1, keepdim=True)  # [B, 1]
        tail_norm_sq = (z.pow(2)).sum(dim=-1, keepdim=True).t()  # [1, N]
        dot = torch.mm(pred, z.t())  # [B, N]

        dist_sq = torch.clamp(pred_norm_sq + tail_norm_sq - 2.0 * dot, min=0.0)
        dist = torch.sqrt(dist_sq + 1e-12)
        return self.gamma - dist
