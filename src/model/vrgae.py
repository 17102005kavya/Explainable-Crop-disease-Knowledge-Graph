"""Owner: Person 2 + Person 3 (integration, day 4).
Full VRGAE-lite = NodeFeatures + RGCNEncoder + VariationalLatent + RotatEDecoder."""
import torch.nn as nn


class VRGAE(nn.Module):
    def __init__(self, node_types, cfg: dict):
        super().__init__()
        raise NotImplementedError   # TODO: build the four parts from cfg

    def forward(self, edge_index, edge_type):
        """-> z, mu, logvar"""
        raise NotImplementedError

    def make_score_all_tails(self, edge_index, edge_type):
        """Return the shared scorer: score_all_tails(h[B], r[B]) -> [B, N], in eval mode with z = mu."""
        raise NotImplementedError
