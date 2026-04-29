"""
Conformal Prediction: LCMQR (Localized Quantile) and Gauss-Newton Full CP.
"""

import math
from dataclasses import dataclass

import torch

from dkrl.models.nigp import NIGPDeepKernelGP
from dkrl.models.predict_utils import batched_predict


@dataclass
class ConformalMetrics:
    coverage: float
    efficiency: float
    median_width: float
    width_std: float
    target_coverage: float
    num_samples: int


def _compute_conformal_metrics(
    y_test: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, confidence: float
) -> ConformalMetrics:
    """Shared evaluation logic for conformal predictors."""
    covered = (y_test >= lower) & (y_test <= upper)
    widths = upper - lower
    return ConformalMetrics(
        coverage=covered.float().mean().item(),
        efficiency=widths.mean().item(),
        median_width=torch.median(widths).item(),
        width_std=widths.std().item(),
        target_coverage=confidence,
        num_samples=len(y_test),
    )


class KernelConformalizedNIGP:
    """LCMQR via RBF kernel localization."""

    def __init__(self, model: NIGPDeepKernelGP, bandwidth: float = None):
        """Args: model, bandwidth (auto-computed if None)."""
        self.model = model
        self.bandwidth = bandwidth
        self._cal_sigmas: torch.Tensor
        self._cal_residuals: torch.Tensor

    def _kernel_weighted_quantiles(
        self, y_hat: torch.Tensor, sigmas: torch.Tensor, confidence: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute LCMQR prediction intervals via RBF-weighted quantiles."""
        sigmas = sigmas.view(-1)
        y_hat = y_hat.view(-1)

        n_cal = len(self._cal_residuals)

        # RBF kernel weights: [M, N_cal]
        sq_dist = (sigmas.unsqueeze(1) - self._cal_sigmas.unsqueeze(0)) ** 2
        weights = torch.exp(-sq_dist / (self.bandwidth**2 + 1e-8))
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted quantile over sorted calibration residuals
        sorted_residuals, sorted_idx = torch.sort(self._cal_residuals)

        # Finite-sample correction: append +inf sentinel with weight 1/(n+1)
        # and scale calibration weights to sum to n/(n+1) per LCMQR specification
        sentinel = torch.tensor(
            [float("inf")], device=sorted_residuals.device, dtype=sorted_residuals.dtype
        )
        sorted_residuals = torch.cat([sorted_residuals, sentinel])

        sorted_weights = weights[:, sorted_idx] * (n_cal / (n_cal + 1))
        sentinel_weight = torch.full(
            (sorted_weights.shape[0], 1),
            1.0 / (n_cal + 1),
            device=sorted_weights.device,
            dtype=sorted_weights.dtype,
        )
        sorted_weights = torch.cat([sorted_weights, sentinel_weight], dim=1)

        cum_weights = torch.cumsum(sorted_weights, dim=1)
        q_idx = (cum_weights >= confidence).long().argmax(dim=1)
        half_widths = sorted_residuals[q_idx]

        return y_hat - half_widths, y_hat + half_widths

    def calibrate(
        self,
        X_cal: torch.Tensor,
        y_cal: torch.Tensor,
        P_cal: torch.Tensor,
        batch_size: int = 512,
    ):
        self.model.eval()
        y_hat, var = batched_predict(self.model.predict_with_nigp, X_cal, P_cal, batch_size)
        sigmas = torch.sqrt(var)
        self._cal_sigmas = sigmas.view(-1)
        self._cal_residuals = torch.abs(y_cal - y_hat).view(-1)

        # Auto-bandwidth via Silverman's rule if not provided
        if self.bandwidth is None:
            n = len(self._cal_sigmas)
            self.bandwidth = 1.06 * self._cal_sigmas.std().item() * (n ** (-0.2))

        return self

    def predict(
        self,
        X: torch.Tensor,
        P: torch.Tensor,
        confidence: float = 0.95,
        batch_size: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.model.eval()
        y_hat, var = batched_predict(self.model.predict_with_nigp, X, P, batch_size)
        sigmas = torch.sqrt(var)
        lower, upper = self._kernel_weighted_quantiles(y_hat, sigmas, confidence)
        return torch.clamp(lower, min=0, max=1), torch.clamp(upper, min=0, max=1), y_hat

    def evaluate(
        self,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        P_test: torch.Tensor,
        confidence: float = 0.95,
    ) -> ConformalMetrics:
        lower, upper, _ = self.predict(X_test, P_test, confidence)
        return _compute_conformal_metrics(y_test, lower, upper, confidence)


class LeverageSplitCP:
    """
    LOO-calibrated Split CP using GP leverage scores (arXiv:2507.20272).
    Approximates LOO residuals: r_i^LOO = r_i / (1 - h_ii).
    """

    def __init__(self, model: NIGPDeepKernelGP, *, noise_floor: float = 1e-6):
        """
        Args:
            model: Trained NIGP-DKL model
            noise_floor: Minimum noise variance for numerical stability
        """
        self.model = model
        self.noise_floor = noise_floor

        # Cached quantities
        self._train_labels = None
        self._leverage_scores = None
        self._loo_abs_residuals = None

    def _compute_leverage_batch(
        self,
        x: torch.Tensor,
        P: torch.Tensor,
        batch_size: int = 512,
    ) -> torch.Tensor:
        """
        Compute leverage scores (hat matrix diagonals) for GP predictions.

        For the low-rank GP with K_XX = Phi @ Phi^T, the hat matrix is:
            H = Phi @ A^{-1} @ Phi^T, where A = Phi^T @ Phi + sigma^2 I
            h_ii = phi_i^T @ A^{-1} @ phi_i

        Args:
            x: Input features [N, D]
            P: Input covariances [N, 10]
            batch_size: Batch size for computation

        Returns:
            Leverage scores [N]
        """
        N = x.shape[0]
        leverages = torch.zeros(N, device=x.device, dtype=x.dtype)

        with torch.no_grad():
            for i in range(0, N, batch_size):
                x_batch = x[i : i + batch_size]

                # Get latent features
                x_norm = self.model.normalize(x_batch)
                h = self.model.feature_forward(x_norm)

                # Low-rank GP hat matrix diagonal: h_ii = phi^T A^{-1} phi
                # where A = Phi^T Phi + sigma^2 I (noise already in A_inv)
                gp = self.model.gp_layer
                phi = gp.phi(h)  # [batch, rank]
                leverage_batch = (phi @ gp.A_inv * phi).sum(dim=-1)

                # Clamp leverage to [0, 1)
                leverages[i : i + batch_size] = torch.clamp(
                    leverage_batch, min=0, max=0.999
                )

        return leverages

    def calibrate(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        P_train: torch.Tensor,
        batch_size: int = 512,
    ) -> LeverageSplitCP:
        """
        Calibrate the conformal predictor by computing LOO residuals.

        Args:
            X_train: Training features [N, D]
            y_train: Training labels [N]
            P_train: Training input covariances [N, 10]
            batch_size: Batch size for computation

        Returns:
            self for method chaining
        """
        self._train_labels = y_train

        # Get model predictions
        y_hat, _ = batched_predict(self.model.predict_with_nigp, X_train, P_train, batch_size)

        # Compute leverage scores
        self._leverage_scores = self._compute_leverage_batch(X_train, P_train, batch_size)

        # Compute LOO residuals: r_i^LOO = r_i / (1 - h_ii)
        residuals = y_train - y_hat
        scale = torch.clamp(1 - self._leverage_scores, min=0.001)
        loo_residuals = residuals / scale

        self._loo_abs_residuals = torch.abs(loo_residuals)

        return self

    def predict_intervals(
        self,
        X_test: torch.Tensor,
        P_test: torch.Tensor,
        confidence: float = 0.95,
        batch_size: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute conformalized prediction intervals.

        Args:
            X_test: Test features [N*, D]
            P_test: Test input covariances [N*, 10]
            confidence: Desired coverage probability
            batch_size: Batch size for computation

        Returns:
            (lower, upper, point_estimate) tuple
        """
        # Get point predictions
        y_hat, _ = batched_predict(self.model.predict_with_nigp, X_test, P_test, batch_size)

        # Jackknife+ quantile computation (Barber et al. 2021, arXiv:2507.20272)
        # Q = ceil((1-alpha)(n+1))-th smallest of {|r_i^LOO|} ∪ {+inf}
        n = len(self._loo_abs_residuals)
        alpha = 1.0 - confidence
        k = math.ceil((1 - alpha) * (n + 1))  # 1-indexed rank

        if k > n:
            # Quantile is +infinity — too few calibration points
            quantile = float('inf')
        else:
            sorted_residuals, _ = torch.sort(self._loo_abs_residuals)
            quantile = sorted_residuals[k - 1]  # k-1 for 0-indexed

        lower = y_hat - quantile
        upper = y_hat + quantile

        return lower, upper, y_hat

    def evaluate(
        self,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        P_test: torch.Tensor,
        confidence: float = 0.95,
    ) -> ConformalMetrics:
        """Evaluate conformal prediction performance."""
        lower, upper, _ = self.predict_intervals(X_test, P_test, confidence)
        return _compute_conformal_metrics(y_test, lower, upper, confidence)
