"""Owner: Person 2 (Part B). R-GCN encoder (see README interface contract)."""
import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv


class RGCNEncoder(nn.Module):
    """Two torch_geometric.nn.RGCNConv layers with ReLU + dropout between layers.

    forward(x, edge_index, edge_type) -> H of shape [N, out_dim]   (N = 11417, out_dim = 128)
    x: [N, in_dim] from NodeFeatures; edge_index [2, E]; edge_type [E] with ids 0..11 (inverse edges included).
    """

    def __init__(self, in_dim: int = 72, hidden_dim: int = 128, out_dim: int = 128,
                 num_relations: int = 12, num_bases: int = 4, dropout: float = 0.2):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations=num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_dim, out_dim, num_relations=num_relations, num_bases=num_bases)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index, edge_type)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_type)
        return h

