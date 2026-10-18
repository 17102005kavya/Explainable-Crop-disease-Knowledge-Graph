"""Owner: Person 3 (Part C). Variational latent space.  TODO: implement."""
import torch
import torch.nn as nn


class VariationalLatent(nn.Module):
    """mu and logvar heads + reparameterization.  forward(H) -> (z, mu, logvar), each [N, latent_dim]."""

    def __init__(self, in_dim: int = 128, latent_dim: int = 128):
        super().__init__()
        raise NotImplementedError   # TODO: Linear heads for mu and logvar

    def forward(self, H: torch.Tensor):
        raise NotImplementedError   # TODO: z = mu + eps * exp(0.5 * logvar); at eval time use z = mu

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError   # TODO: mean KL to N(0, I)
