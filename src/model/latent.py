
"""Variational latent layer for VRGAE."""

import torch
import torch.nn as nn


class VariationalLatent(nn.Module):
    """Map encoder representations to a Gaussian latent space."""

    def __init__(self, in_dim: int = 128, latent_dim: int = 128):
        super().__init__()
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

    def forward(self, H: torch.Tensor):
        """Return latent samples, means, and log variances."""
        mu = self.fc_mu(H)
        logvar = self.fc_logvar(H)

        # Keep the exponential in a numerically stable range.
        logvar = logvar.clamp(min=-20.0, max=20.0)

        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            # Deterministic inference.
            z = mu

        return z, mu, logvar

    @staticmethod
    def kl_divergence(
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean KL divergence to a standard normal prior."""
        logvar = logvar.clamp(min=-20.0, max=20.0)

        kl_per_node = -0.5 * (
            1.0 + logvar - mu.pow(2) - logvar.exp()
        ).sum(dim=-1)

        return kl_per_node.mean()
