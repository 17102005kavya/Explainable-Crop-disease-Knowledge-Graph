"""Owner: Person 3 (Part C). Variational latent space.

Implements linear heads for mean (mu) and log-variance (logvar),
reparameterization trick with numerical clamping, and analytical KL divergence to N(0, I).
"""
from typing import Tuple

import torch
import torch.nn as nn


class VariationalLatent(nn.Module):
    """mu and logvar heads + reparameterization.
    
    forward(H) -> (z, mu, logvar), each [N, latent_dim].
    """

    def __init__(self, in_dim: int = 128, latent_dim: int = 128):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        # Initialize logvar to small weights so initial variance is close to 1
        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=0.01)
        nn.init.zeros_(self.fc_logvar.bias)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = mu + eps * exp(0.5 * logvar) during training, or return mu in eval mode."""
        if self.training:
            clamped_logvar = torch.clamp(logvar, min=-10.0, max=10.0)
            std = torch.exp(0.5 * clamped_logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert encoder representations H [N, in_dim] -> (z, mu, logvar) [N, latent_dim]."""
        mu = self.fc_mu(H)
        logvar = torch.clamp(self.fc_logvar(H), min=-10.0, max=10.0)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Mean KL divergence of q(z|x) = N(mu, diag(exp(logvar))) from prior p(z) = N(0, I).

        KL per node: -0.5 * sum(1 + logvar - mu^2 - exp(logvar), dim=-1).
        Averaged over all nodes N to produce a numerically stable scalar.
        """
        clamped_logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        kl_per_node = -0.5 * torch.sum(1.0 + clamped_logvar - mu.pow(2) - torch.exp(clamped_logvar), dim=-1)
        return kl_per_node.mean()
