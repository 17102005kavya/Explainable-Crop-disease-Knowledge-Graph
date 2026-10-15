"""Owner: Person 2 (Part B). R-GCN encoder.  TODO: implement (see README interface contract)."""
import torch
import torch.nn as nn


class RGCNEncoder(nn.Module):
    """Two torch_geometric.nn.RGCNConv layers with ReLU + dropout.

    forward(x, edge_index, edge_type) -> H of shape [N, out_dim]   (N = 11417, out_dim = 128)
    x: [N, in_dim] from NodeFeatures; edge_index [2, E]; edge_type [E] with ids 0..11 (inverse edges included).
    """

    def __init__(self, in_dim: int = 72, hidden_dim: int = 128, out_dim: int = 128,
                 num_relations: int = 12, num_bases: int = 4, dropout: float = 0.2):
        super().__init__()
        # TODO: self.conv1 = RGCNConv(...), self.conv2 = RGCNConv(...), self.dropout = dropout
        raise NotImplementedError

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError
