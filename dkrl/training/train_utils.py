"""Training utilities: convergence detection, Lipschitz bounds, GP fitting."""

import torch

from dkrl.config._constants import EPSILON
from dkrl.models.nigp import NIGPDeepKernelGP


def check_convergence(
    history: list[float],
    *,
    n_train: int,
    epoch: int,
    min_window: int = 10,
) -> bool:
    """
    Check if training has converged via linear regression slope.

    Adaptive parameters:
        - window: max(min_window, epoch // 10) -- grows with training
        - threshold: 1/n_train -- statistical detectability limit

    Args:
        history: List of validation losses (most recent last)
        n_train: Number of training samples (sets threshold)
        epoch: Current epoch number (sets window)
        min_window: Minimum window size (default 10)

    Returns:
        True if converged (slope indicates no meaningful improvement)
    """
    window = max(min_window, epoch // 10)
    threshold = 1.0 / max(n_train, 1)

    if len(history) < window:
        return False
    recent = history[-window:]
    w = len(recent)

    # Closed-form OLS slope
    sum_x = w * (w - 1) / 2.0
    sum_x2 = w * (w - 1) * (2 * w - 1) / 6.0
    sum_y = sum(recent)
    sum_xy = sum(i * y for i, y in enumerate(recent))

    slope = (w * sum_xy - sum_x * sum_y) / (w * sum_x2 - sum_x ** 2 + 1e-12)

    # Normalize by mean value to make threshold scale-invariant
    mean_y = sum_y / w
    normalized_slope = slope / (abs(mean_y) + EPSILON)

    # Converged if slope is not meaningfully negative (no improvement)
    return normalized_slope > -threshold


def compute_lipschitz_bounds(X: torch.Tensor) -> tuple[float, float]:
    """
    Derive Lipschitz TARGET CONSTRAINTS from data geometry via SVD.

    Lower bound: 1/condition_number(X) ensures bi-Lipschitz property.
    Upper bound: 1.0 is inherent from spectral normalization.

    Args:
        X: Input data tensor [N, D]

    Returns:
        (min_lipschitz, target_lipschitz) tuple for training constraints
    """
    n_sample = min(1000, len(X))
    with torch.no_grad():
        _, S, _ = torch.linalg.svd(X[:n_sample], full_matrices=False)
        condition = (S[0] / (S[-1] + EPSILON)).item()
    min_lip = 1.0 / max(condition, 1.0)
    return min_lip, 1.0


def fit_low_rank_gp(
    model: NIGPDeepKernelGP,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
) -> None:
    """Fit low-rank GP layer via closed-form conjugate Gaussian solution."""
    with torch.no_grad():
        x_norm = model.normalize(train_x)
        h = model.feature_forward(x_norm)

        # Normalize targets
        y_norm = (train_y - model.target_mean) / model.target_std

        # Fit the low-rank GP layer
        model.gp_layer.fit(h, y_norm)
