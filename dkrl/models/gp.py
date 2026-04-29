"""
Gaussian Process Layers for NIGP-DKL.

Contains the low-rank GP implementation for O(n) exact inference.

References:
    - Low-Rank DKL: arXiv:2505.18526
"""

import math

import torch
import torch.nn as nn
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import DiagLinearOperator

from dkrl.config._constants import VARIANCE_FLOOR


class LowRankGPLayer(nn.Module):
    """
    Low-Rank GP Layer (O(N) inference) via basis decomposition.
    """

    def __init__(
        self,
        input_dim: int,
        rank: int = 64,
        *,
        noise_init: float = 1e-2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.rank = rank

        hidden_dim = 2 * rank
        self.basis_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, rank),
        )

        self.log_noise = nn.Parameter(torch.tensor(math.log(noise_init)))

        self.register_buffer("alpha", None)
        self.register_buffer("A_inv", None)
        self.register_buffer("y_mean", torch.zeros(1))

    @property
    def noise(self) -> torch.Tensor:
        """Observation noise variance."""
        return self.log_noise.exp()

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        """Compute basis function features."""
        return self.basis_net(x)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Fit GP using closed-form conjugate Gaussian solution."""
        self.y_mean = y.mean()
        y_centered = y - self.y_mean

        Phi = self.phi(X)

        noise_var = self.noise
        A = Phi.T @ Phi + noise_var * torch.eye(self.rank, device=X.device, dtype=X.dtype)

        L = torch.linalg.cholesky(A)
        Phi_y = Phi.T @ y_centered
        z = torch.linalg.solve_triangular(L, Phi_y.unsqueeze(-1), upper=False)
        alpha = torch.linalg.solve_triangular(L.mT, z, upper=True).squeeze(-1)

        self.alpha = alpha
        L_inv = torch.linalg.solve_triangular(
            L, torch.eye(self.rank, device=X.device, dtype=X.dtype), upper=False
        )
        self.A_inv = L_inv.mT @ L_inv

    def compute_trace_regularization(self, x: torch.Tensor) -> torch.Tensor:
        """
        Anti-collapse regularization via basis function trace (arXiv:2505.18526).

        Penalizes variance in basis norms to prevent rank-1 collapse.
        """
        phi = self.phi(x)
        phi_norms_sq = (phi ** 2).sum(dim=-1)  # [N]
        max_norm = phi_norms_sq.max().detach()
        return ((max_norm - phi_norms_sq) / (2 * self.noise + 1e-8)).mean()

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        phi_star = self.phi(x)

        mean = phi_star @ self.alpha + self.y_mean

        noise_var = self.noise
        quad_form = (phi_star @ self.A_inv * phi_star).sum(dim=-1)
        var = noise_var * (1.0 + quad_form)

        var = var.clamp(min=VARIANCE_FLOOR)

        return MultivariateNormal(mean, DiagLinearOperator(var))


__all__ = ["LowRankGPLayer"]
