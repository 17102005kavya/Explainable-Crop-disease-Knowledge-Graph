"""Owner: Person 3 (Part C). Variational latent space.

Projects R-GCN representations H into a latent Gaussian distribution:
mu and logvar heads + reparameterization trick.
forward(H) -> (z, mu, logvar), each [N, latent_dim].
"""
from typing import Tuple
import torch
import torch.nn as nn


class VariationalLatent(nn.Module):
    """Variational latent layer for graph autoencoder.
    
    Generates mean (mu) and log-variance (logvar) vectors for each node.
    At training time: samples z = mu + eps * exp(0.5 * logvar) via reparameterization.
    At evaluation time: deterministic z = mu.
    """

    def __init__(self, in_dim: int = 128, latent_dim: int = 128):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim

        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

        # Initialize logvar near zero so initial variance is close to 1
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute latent representations.
        
        H: [N, in_dim] -> z: [N, latent_dim], mu: [N, latent_dim], logvar: [N, latent_dim]
        """
        mu = self.fc_mu(H)
        logvar = self.fc_logvar(H)

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        return z, mu, logvar

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute mean analytical KL divergence to standard normal N(0, I) across all nodes.
        
        KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) / N
        """
        kl_per_node = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - torch.exp(logvar), dim=-1)
        return torch.mean(kl_per_node)
