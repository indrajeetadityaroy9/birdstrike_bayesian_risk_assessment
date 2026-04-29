"""
Deep Sigma Point Processes (arXiv:2002.09112).
Unscented Transform for uncertainty propagation (no Hessian needed).
"""

import torch
import torch.nn as nn

from dkrl.config._constants import VARIANCE_FLOOR
from dkrl.models.covariance import build_cov_6x6
from dkrl.models.nigp import NIGPDeepKernelGP, UncertaintyDecomposition


class SigmaPointNIGP(nn.Module):
    """
    Sigma-Point NIGP (arXiv:2002.09112).
    Propagates uncertainty via Unscented Transform (O(2D+1) forward passes).
    """

    def __init__(
        self,
        model: NIGPDeepKernelGP,
        *,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        """
        Args:
            model: NIGP-DKL model to wrap
            alpha: Spread of sigma points (1e-4 <= alpha <= 1, typically 1e-3)
            beta: Prior knowledge about distribution (beta=2 optimal for Gaussian)
            kappa: Secondary scaling parameter (typically 0 or 3-n)
        """
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def _compute_ut_weights(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Unscented Transform weights for mean and covariance reconstruction."""
        lambda_ = self.alpha**2 * (n + self.kappa) - n

        w_m = torch.zeros(2 * n + 1, device=device, dtype=dtype)
        w_m[0] = lambda_ / (n + lambda_)
        w_m[1:] = 1.0 / (2 * (n + lambda_))

        w_c = torch.zeros(2 * n + 1, device=device, dtype=dtype)
        w_c[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        w_c[1:] = 1.0 / (2 * (n + lambda_))

        return w_m, w_c

    def _generate_sigma_points(
        self, mean: torch.Tensor, cov: torch.Tensor
    ) -> torch.Tensor:
        """Generate 2n+1 sigma points from Gaussian N(mean, cov)."""
        batch_size, n = mean.shape
        device, dtype = mean.device, mean.dtype

        lambda_ = self.alpha**2 * (n + self.kappa) - n
        cov_scaled = (n + lambda_) * cov + torch.eye(n, device=device, dtype=dtype) * 1e-8
        L = torch.linalg.cholesky(cov_scaled)

        sigma_points = torch.zeros(batch_size, 2 * n + 1, n, device=device, dtype=dtype)
        sigma_points[:, 0, :] = mean
        for i in range(n):
            sigma_points[:, i + 1, :] = mean + L[:, :, i]
            sigma_points[:, n + i + 1, :] = mean - L[:, :, i]

        return sigma_points

    def _generate_reduced_sigma_points(
        self,
        x_norm: torch.Tensor,
        cov_10: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate sigma points using only the 6D non-zero covariance block.

        Since P_19 is rank-6 (only position/velocity have covariance), generating
        sigma points in the full 19D space wastes computation. Instead, generate
        2*6+1 = 13 sigma points by perturbing only the first 6 features.
        """
        batch_size, full_dim = x_norm.shape
        device, dtype = x_norm.device, x_norm.dtype
        reduced_dim = 6

        # Build 6x6 covariance from compressed format using canonical implementation
        P_6x6 = build_cov_6x6(cov_10)

        # Normalize covariance to match normalized input space
        # P_norm[i,j] = P[i,j] / (sigma_i * sigma_j)
        s = self.model.input_std[:reduced_dim]
        scale = s.unsqueeze(0) * s.unsqueeze(1)  # [6, 6]
        P_6x6 = P_6x6 / scale.unsqueeze(0)

        mean_6 = x_norm[:, :reduced_dim]
        weights_m, weights_c = self._compute_ut_weights(reduced_dim, device, dtype)
        sigma_6 = self._generate_sigma_points(mean_6, P_6x6)

        num_sigma = 2 * reduced_dim + 1
        sigma_full = x_norm.unsqueeze(1).expand(batch_size, num_sigma, full_dim).clone()
        sigma_full[:, :, :reduced_dim] = sigma_6

        return sigma_full, weights_m, weights_c

    def predict(
        self,
        x: torch.Tensor,
        cov_10: torch.Tensor,
        *,
        return_decomposition: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with sigma-point uncertainty propagation.

        Args:
            x: Input features [N, D]
            cov_10: Compressed input covariances [N, 10]
            return_decomposition: If True, return UncertaintyDecomposition

        Returns:
            (mean, variance) or (mean, UncertaintyDecomposition) if return_decomposition
        """
        with torch.no_grad():
            batch_size = x.shape[0]
            n = self.model.input_dim

            x_norm = self.model.normalize(x)

            sigma_points, weights_m, weights_c = self._generate_reduced_sigma_points(
                x_norm, cov_10
            )
            num_sigma = sigma_points.shape[1]

            # Propagate through feature extractor and GP
            sigma_flat = sigma_points.view(-1, n)
            h_flat = self.model.feature_forward(sigma_flat)
            gp_dist = self.model.gp_layer(h_flat)
            gp_mean_sigma = gp_dist.mean.view(batch_size, num_sigma)
            gp_var_sigma = gp_dist.variance.view(batch_size, num_sigma)

            # Reconstruct statistics from sigma points
            pred_mean_norm = torch.einsum("k,bk->b", weights_m, gp_mean_sigma)
            diff = gp_mean_sigma - pred_mean_norm.unsqueeze(1)
            input_var_contrib = torch.einsum("k,bk->b", weights_c, diff**2)
            gp_var_mean = torch.einsum("k,bk->b", weights_m, gp_var_sigma)

            obs_noise = self.model.gp_layer.noise

            total_var_norm = torch.clamp(
                input_var_contrib + gp_var_mean + obs_noise, min=VARIANCE_FLOOR
            )

            mean = pred_mean_norm * self.model.target_std + self.model.target_mean
            variance = total_var_norm * (self.model.target_std**2)

            if return_decomposition:
                decomp = UncertaintyDecomposition(
                    observation_noise=obs_noise
                    * (self.model.target_std**2)
                    * torch.ones_like(mean),
                    input_noise=input_var_contrib * (self.model.target_std**2),
                    gp_variance=gp_var_mean * (self.model.target_std**2),
                    nigp_2nd_aleatoric=torch.zeros_like(mean),
                    nigp_2nd_epistemic=torch.zeros_like(mean),
                    aleatoric=(obs_noise + input_var_contrib) * (self.model.target_std**2),
                    epistemic=gp_var_mean * (self.model.target_std**2),
                    total=variance,
                    mean_correction=torch.zeros_like(mean),
                )
                return mean, decomp

            return mean, variance
