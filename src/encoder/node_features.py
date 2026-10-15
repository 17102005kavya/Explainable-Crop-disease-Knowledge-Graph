"""Owner: Person 2 (Part B). Node features = fixed entity-type vector + learnable embedding."""
import torch
import torch.nn as nn


class NodeFeatures(nn.Module):
    def __init__(self, node_types: torch.Tensor, emb_dim: int = 64):
        super().__init__()
        self.register_buffer("types", node_types.float())              # [N, 8], not trained
        self.emb = nn.Embedding(node_types.size(0), emb_dim)           # [N, emb_dim], trained
        self.out_dim = node_types.size(1) + emb_dim

    def forward(self) -> torch.Tensor:                                  # -> [N, 8 + emb_dim]
        return torch.cat([self.types, self.emb.weight], dim=-1)
