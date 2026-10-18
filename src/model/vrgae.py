"""Owner: Person 2 + Person 3 (integration, day 4).
Full VRGAE-lite = NodeFeatures + RGCNEncoder + VariationalLatent + RotatEDecoder (or ComplExDecoder).
"""
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.encoder.node_features import NodeFeatures
from src.encoder.rgcn import RGCNEncoder
from src.model.decoder import ComplExDecoder, RotatEDecoder
from src.model.latent import VariationalLatent
from src.model.loss import compute_loss_breakdown, total_loss


class VRGAE(nn.Module):
    """Variational Relational Graph Autoencoder (VRGAE).

    Integrates:
      1. NodeFeatures (type one-hot + learnable embedding)
      2. RGCNEncoder (2-layer relation-aware GCN with basis decomposition)
      3. VariationalLatent (mu & logvar heads with reparameterization trick)
      4. RotatEDecoder or ComplExDecoder (configurable via cfg['model']['decoder'])
    """

    def __init__(self, node_types: torch.Tensor, cfg: dict):
        super().__init__()
        self.cfg = cfg

        enc_cfg = cfg.get("encoder", {})
        lat_cfg = cfg.get("latent", {})
        dec_cfg = cfg.get("decoder", {})
        model_cfg = cfg.get("model", {})

        # 1. Node features (type features + entity embedding)
        emb_dim = enc_cfg.get("emb_dim", 64)
        self.node_features = NodeFeatures(node_types, emb_dim=emb_dim)

        # 2. R-GCN Encoder
        hidden_dim = enc_cfg.get("hidden_dim", 128)
        out_dim = enc_cfg.get("out_dim", 128)
        num_relations = enc_cfg.get("num_relations", 12)
        num_bases = enc_cfg.get("num_bases", 4)
        dropout = enc_cfg.get("dropout", 0.2)

        self.encoder = RGCNEncoder(
            in_dim=self.node_features.out_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_relations=num_relations,
            num_bases=num_bases,
            dropout=dropout,
        )

        # 3. Variational Latent Layer
        latent_dim = lat_cfg.get("latent_dim", out_dim)
        self.latent = VariationalLatent(in_dim=out_dim, latent_dim=latent_dim)

        # 4. Decoder (RotatE or ComplEx, configurable)
        num_base_relations = dec_cfg.get("num_base_relations", 6)
        gamma = float(dec_cfg.get("gamma", 6.0))
        decoder_type = model_cfg.get("decoder", "rotate").lower()

        if decoder_type == "complex":
            self.decoder = ComplExDecoder(num_base_relations=num_base_relations, dim=latent_dim)
        else:
            self.decoder = RotatEDecoder(num_base_relations=num_base_relations, dim=latent_dim, gamma=gamma)

    def encode(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through encoder and variational latent layer -> (z, mu, logvar)."""
        x = self.node_features()
        H = self.encoder(x, edge_index, edge_type)
        z, mu, logvar = self.latent(H)
        return z, mu, logvar

    def forward(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Alias for encode: returns (z, mu, logvar)."""
        return self.encode(edge_index, edge_type)

    def score(self, z: torch.Tensor, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute decoder score for triples (h, r, t) using latent representations z."""
        return self.decoder.score(z, h, r, t)

    def score_all_tails(self, z: torch.Tensor, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Compute decoder score for all candidate tails given (h, r) using latent representations z."""
        return self.decoder.score_all_tails(z, h, r)

    def compute_loss(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        pos_triples: torch.Tensor,
        neg_triples: torch.Tensor,
        epoch: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute training loss on positive and negative triples."""
        pos_scores = self.decoder.score(
            z, pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]
        )
        neg_scores = self.decoder.score(
            z, neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]
        )

        train_cfg = self.cfg.get("training", {})
        warmup_epochs = train_cfg.get("kl_warmup_epochs", 20)
        max_kl_weight = train_cfg.get("kl_weight", 1.0)

        return compute_loss_breakdown(
            pos_score=pos_scores,
            neg_score=neg_scores,
            mu=mu,
            logvar=logvar,
            epoch=epoch,
            warmup_epochs=warmup_epochs,
            max_kl_weight=max_kl_weight,
        )

    def make_score_all_tails(
        self, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return the shared scorer: score_all_tails(h[B], r[B]) -> [B, N], in eval mode with z = mu."""
        self.eval()
        with torch.no_grad():
            x = self.node_features()
            H = self.encoder(x, edge_index, edge_type)
            _, mu, _ = self.latent(H)
            z_eval = mu

        def scorer_fn(h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return self.decoder.score_all_tails(z_eval, h, r)

        return scorer_fn

