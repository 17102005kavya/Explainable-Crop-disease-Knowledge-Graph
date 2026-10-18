"""Owner: Person 3 (Part C). Decoders for link prediction: ComplEx and RotatE.

Both decoders map complex entity representations and relation parameters into scores,
supporting both batch scoring: score(z, h, r, t) -> [B]
and full candidate scoring: score_all_tails(z, h, r) -> [B, N] for filtered ranking evaluation.
"""
import math
from typing import Optional
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
    """RotatE decoder: Relations are rotations in complex vector space (Sun et al., 2019).
    
    score(h, r, t) = gamma - || z_h o rot(r) - z_t ||_1
    where |rot(r)_j| = 1 via Euler's formula rot(r) = cos(theta_r) + i * sin(theta_r).
    """

    def __init__(self, num_base_relations: int = 6, dim: int = 128, gamma: float = 6.0, epsilon: float = 2.0):
        super().__init__()
        self.num_relations = num_base_relations
        self.dim = dim
        self.complex_dim = dim // 2
        self.gamma = gamma
        self.epsilon = epsilon
        assert dim % 2 == 0, f"dim must be even for complex embeddings, got {dim}"

        self.rel_dim = self.complex_dim
        # Phase embedding in radians [-pi, pi]
        self.rel_phase = nn.Embedding(num_base_relations, self.rel_dim)
        phase_range = (gamma + epsilon) / self.rel_dim
        nn.init.uniform_(self.rel_phase.weight, -phase_range, phase_range)

    def _rotate(self, h_emb: torch.Tensor, r: torch.Tensor):
        h_re, h_im = h_emb[..., :self.complex_dim], h_emb[..., self.complex_dim:]
        theta = self.rel_phase(r)
        r_re = torch.cos(theta)
        r_im = torch.sin(theta)
        q_re = h_re * r_re - h_im * r_im
        q_im = h_re * r_im + h_im * r_re
        return q_re, q_im

    def score(self, z: torch.Tensor, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute RotatE score for triples (h, r, t).
        
        z: [N, dim], h: [B], r: [B], t: [B] -> [B]
        """
        h_emb = z[h]
        t_emb = z[t]
        t_re, t_im = t_emb[..., :self.complex_dim], t_emb[..., self.complex_dim:]

        q_re, q_im = self._rotate(h_emb, r)
        diff_re = q_re - t_re
        diff_im = q_im - t_im

        # L1 norm of complex difference
        dist = torch.sum(torch.sqrt(diff_re.pow(2) + diff_im.pow(2) + 1e-12), dim=-1)
        return self.gamma - dist

    def score_all_tails(self, z: torch.Tensor, h: torch.Tensor, r: torch.Tensor, chunk_size: int = 100) -> torch.Tensor:
        """Compute RotatE score for all candidate tails given queries (h, r).
        
        z: [N, dim], h: [B], r: [B] -> [B, N]
        Chunked evaluation prevents excessive memory consumption.
        """
        B = h.shape[0]
        N = z.shape[0]
        q_re, q_im = self._rotate(z[h], r)  # [B, d/2]

        t_re = z[:, :self.complex_dim]      # [N, d/2]
        t_im = z[:, self.complex_dim:]      # [N, d/2]

        scores = torch.empty((B, N), dtype=torch.float32, device=z.device)

        for i in range(0, B, chunk_size):
            chunk_q_re = q_re[i:i + chunk_size]  # [b, d/2]
            chunk_q_im = q_im[i:i + chunk_size]  # [b, d/2]

            # [b, 1, d/2] - [1, N, d/2] -> [b, N, d/2]
            diff_re = chunk_q_re.unsqueeze(1) - t_re.unsqueeze(0)
            diff_im = chunk_q_im.unsqueeze(1) - t_im.unsqueeze(0)

            chunk_dist = torch.sum(torch.sqrt(diff_re.pow(2) + diff_im.pow(2) + 1e-12), dim=-1)
            scores[i:i + chunk_size] = self.gamma - chunk_dist

        return scores
