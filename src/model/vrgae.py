"""Owner: Person 2 + Person 3 (integration).
Full VRGAE architecture: NodeFeatures + RGCNEncoder + VariationalLatent + RotatEDecoder.
"""
from typing import Dict, Any, Callable
import torch
import torch.nn as nn

from src.encoder.node_features import NodeFeatures
from src.encoder.rgcn import RGCNEncoder
from src.model.latent import VariationalLatent
from src.model.decoder import RotatEDecoder, ComplExDecoder


class VRGAE(nn.Module):
    """Variational Relational Graph Autoencoder (VRGAE).
    
    Architecture:
    1. NodeFeatures: Combines fixed 8-dim entity-type multi-hot buffer with learnable embedding.
    2. RGCNEncoder: 2-layer relational graph convolutional network with basis decomposition.
    3. VariationalLatent: mu and logvar projection heads + reparameterization trick.
    4. RotatEDecoder: Complex rotation scoring gamma - ||z_h o rot(r) - z_t||_1.
    """

    def __init__(self, node_types: torch.Tensor, cfg: Dict[str, Any]):
        super().__init__()
        enc_cfg = cfg.get("encoder", {})
        dec_cfg = cfg.get("decoder", {})
        model_cfg = cfg.get("model", {})

        emb_dim = enc_cfg.get("emb_dim", 64)
        hidden_dim = enc_cfg.get("hidden_dim", 128)
        out_dim = enc_cfg.get("out_dim", 128)
        num_relations = enc_cfg.get("num_relations", 12)
        num_bases = enc_cfg.get("num_bases", 4)
        dropout = enc_cfg.get("dropout", 0.2)

        num_base_relations = dec_cfg.get("num_base_relations", 6)
        gamma = float(dec_cfg.get("gamma", 6.0))
        decoder_type = model_cfg.get("decoder", "rotate").lower()

        self.node_features = NodeFeatures(node_types, emb_dim=emb_dim)
        self.encoder = RGCNEncoder(
            in_dim=self.node_features.out_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_relations=num_relations,
            num_bases=num_bases,
            dropout=dropout,
        )
        self.latent = VariationalLatent(in_dim=out_dim, latent_dim=out_dim)

        if decoder_type == "complex":
            self.decoder = ComplExDecoder(num_base_relations=num_base_relations, dim=out_dim)
        else:
            self.decoder = RotatEDecoder(num_base_relations=num_base_relations, dim=out_dim, gamma=gamma)

    def forward(self, edge_index: torch.Tensor, edge_type: torch.Tensor):
        """Forward pass through node features, RGCN encoder, and variational latent layer.
        
        -> z: [N, latent_dim], mu: [N, latent_dim], logvar: [N, latent_dim]
        """
        x = self.node_features()
        H = self.encoder(x, edge_index, edge_type)
        z, mu, logvar = self.latent(H)
        return z, mu, logvar

    def score(self, z: torch.Tensor, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute decoder score for triples (h, r, t)."""
        return self.decoder.score(z, h, r, t)

    def make_score_all_tails(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return evaluation score function: score_all_tails(h[B], r[B]) -> [B, N] in eval mode (z = mu)."""
        self.eval()
        with torch.no_grad():
            x = self.node_features()
            H = self.encoder(x, edge_index, edge_type)
            z, _, _ = self.latent(H)  # in eval mode, z = mu deterministically

        def score_fn(h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
            return self.decoder.score_all_tails(z, h, r)

        return score_fn
