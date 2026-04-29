"""
JAX-accelerated scoring rules: CRPS, DSS, AUSE, Interval Score, Tail-CRPS.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scoringrules as sr
import torch
from jax import jit
from jax.dlpack import from_dlpack
from jax.scipy.special import ndtr
from jax.scipy.stats import norm as jax_norm
from torch.utils.dlpack import to_dlpack

from dkrl.config._constants import Z_90, Z_95

jax.config.update("jax_enable_x64", True)

Array = torch.Tensor | np.ndarray | jnp.ndarray


def _to_jax(x: Array) -> jnp.ndarray:
    """Zero-copy transfer from GPU tensors via DLPack."""
    if isinstance(x, torch.Tensor):
        return from_dlpack(to_dlpack(x))
    return jnp.asarray(x)


@dataclass
class ScoringResults:
    crps: float
    scaled_crps: float
    dss: float
    interval_score_90: float
    interval_score_95: float
    ause: float
    ause_oracle: float
    tail_crps_upper: float = 0.0
    tail_crps_lower: float = 0.0


@jit
def _dss(y_true: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """
    Dawid-Sebastiani Score.
    DSS = log(σ²) + (y-μ)²/σ²
    Depends only on first two moments.
    """
    sigma = jnp.maximum(sigma, 1e-8)
    var = sigma ** 2
    return jnp.log(var) + (y_true - mu) ** 2 / var


# PyTorch training counterpart: dkrl.training.losses._crps_rectified_gaussian_torch
@partial(jax.jit, static_argnums=(4,))
def _crps_rectified_gaussian(
    mu: jnp.ndarray, sigma: jnp.ndarray, y_val: float, threshold: float, tail: str
) -> jnp.ndarray:
    """
    Exact CRPS for Rectified Gaussian via Brier Score integral (arXiv:2407.00650).

    CRPS = ∫₀^∞ (F_Y(z) - 𝟙{z≥y})² dz where Y = max(Z, 0), Z ~ N(μ_z, σ)
    """
    sigma = jnp.maximum(sigma, 1e-8)

    # Transform to rectified normal: Y = max(Z, 0)
    mu_z = (mu - threshold) if tail == "upper" else (threshold - mu)
    y_trans = jnp.maximum(y_val - threshold, 0.0) if tail == "upper" else jnp.maximum(threshold - y_val, 0.0)

    # Primitives for ∫Φ(x)² dx and ∫(1-Φ(x))² dx (closed-form)
    def primitive_phi_sq(x):
        return x * ndtr(x)**2 + 2 * ndtr(x) * jax_norm.pdf(x) - 1/jnp.sqrt(jnp.pi) * ndtr(x * jnp.sqrt(2))

    def primitive_one_minus_phi_sq(x):
        return x - 2 * (x * ndtr(x) + jax_norm.pdf(x)) + primitive_phi_sq(x)

    val_0 = -mu_z / sigma
    val_y = (y_trans - mu_z) / sigma

    # CRPS = σ * [G(val_y) - G(val_0)] + σ * [lim_inf - H(val_y)]
    part1 = sigma * (primitive_phi_sq(val_y) - primitive_phi_sq(val_0))
    part2 = sigma * (-1/jnp.sqrt(jnp.pi) - primitive_one_minus_phi_sq(val_y))

    return part1 + part2


@partial(jax.jit, static_argnums=(2,))
def _compute_ause_curve(
    errors: jnp.ndarray,
    uncertainties: jnp.ndarray,
    n_bins: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute sparsification error curve.

    Sorts by uncertainty (descending) and computes RMSE at each sparsification level.
    Returns (fractions, rmse_values) for area computation.
    """
    n = len(errors)

    # Sort by uncertainty (highest first - remove most uncertain)
    sorted_idx = jnp.argsort(-uncertainties)
    sorted_errors = errors[sorted_idx]

    # Compute cumulative squared errors (from end, keeping low-uncertainty samples)
    cumsum_sq = jnp.cumsum(sorted_errors[::-1] ** 2)[::-1]

    # Sample at n_bins points
    fractions = jnp.linspace(0, 1, n_bins + 1)[:-1]
    indices = (fractions * n).astype(jnp.int32)
    indices = jnp.clip(indices, 0, n - 1)

    # Compute RMSE at each sparsification level
    remaining_counts = n - indices
    remaining_sq_sums = cumsum_sq[indices]
    rmse_values = jnp.sqrt(remaining_sq_sums / jnp.maximum(remaining_counts, 1))

    return fractions, rmse_values


def compute_ause(
    y_true: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    n_bins: int = 100,
) -> tuple[float, dict]:
    """
    Compute Area Under Sparsification Error (AUSE) per arXiv:2405.04278.

    AUSE measures how well predicted uncertainties correlate with actual errors.
    Lower is better; 0 = perfect uncertainty ordering.

    Args:
        y_true: Ground truth values [N]
        mu: Predicted means [N]
        sigma: Predicted standard deviations [N]
        n_bins: Number of sparsification levels (default 100)

    Returns:
        ause: Normalized AUSE score (0 = perfect, 1 = random)
        details: Dict with ause_oracle, ause_random, ause_model
    """
    errors = jnp.abs(y_true - mu)

    # Model curve: sort by predicted uncertainty
    fractions, rmse_model = _compute_ause_curve(errors, sigma, n_bins)

    # Oracle curve: sort by actual error (best possible ordering)
    _, rmse_oracle = _compute_ause_curve(errors, errors, n_bins)

    # Random baseline: constant RMSE (no sorting benefit)
    rmse_random = jnp.full_like(fractions, jnp.sqrt(jnp.mean(errors ** 2)))

    # Compute areas using trapezoidal rule
    area_model = float(jnp.trapezoid(rmse_model, fractions))
    area_oracle = float(jnp.trapezoid(rmse_oracle, fractions))
    area_random = float(jnp.trapezoid(rmse_random, fractions))

    # Normalized AUSE: (model - oracle) / (random - oracle)
    # Range: 0 (perfect) to 1 (random), can exceed 1 if worse than random
    denom = max(area_random - area_oracle, 1e-8)
    ause = (area_model - area_oracle) / denom

    details = {
        "ause_oracle": area_oracle,
        "ause_random": area_random,
        "ause_model": area_model,
    }

    return float(ause), details


@jit
def _crps_gaussian_per_sample(mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """CRPS self-score for Gaussian: sigma / sqrt(pi)."""
    return jnp.maximum(sigma, 1e-8) / jnp.sqrt(jnp.pi)


@jit
def _cross_crps_gaussian(
    mu_i: jnp.ndarray, sigma_i: jnp.ndarray,
    mu_j: jnp.ndarray, sigma_j: jnp.ndarray,
) -> jnp.ndarray:
    """Cross-CRPS between two Gaussian predictives (arXiv:2404.12215)."""
    sigma_c = jnp.sqrt(sigma_i ** 2 + sigma_j ** 2)
    z_c = (mu_j - mu_i) / jnp.maximum(sigma_c, 1e-8)
    return sigma_c * (
        z_c * (2 * ndtr(z_c) - 1)
        + 2 * jax_norm.pdf(z_c)
        - 1.0 / jnp.sqrt(jnp.pi)
    )


# PyTorch inference counterpart: dkrl.models.nigp.NIGPDeepKernelGP.compute_scoring_rule_decomposition
def compute_epistemic_crps(
    mu_ensemble: Array, sigma_ensemble: Array,
) -> dict[str, float]:
    """
    Scoring-rule uncertainty decomposition (arXiv:2404.12215).

    Args:
        mu_ensemble: [N, K] per-member means
        sigma_ensemble: [N, K] per-member stds

    Returns:
        Dict with au_crps, eu_crps, tu_crps
    """
    mu_e = _to_jax(mu_ensemble)
    sig_e = _to_jax(sigma_ensemble)
    K = mu_e.shape[1]

    # AU = (1/K) sum_k sigma_k / sqrt(pi)
    au = jnp.mean(_crps_gaussian_per_sample(mu_e, sig_e), axis=1)

    # TU = (1/K^2) sum_{i,j} cross-CRPS(i,j)
    tu = jnp.zeros(mu_e.shape[0])
    for i in range(K):
        for j in range(K):
            tu = tu + _cross_crps_gaussian(mu_e[:, i], sig_e[:, i], mu_e[:, j], sig_e[:, j])
    tu = tu / (K * K)

    eu = tu - au

    return {
        "au_crps": float(jnp.mean(au)),
        "eu_crps": float(jnp.mean(eu)),
        "tu_crps": float(jnp.mean(tu)),
    }


def compute_all_scores(
    y_true: Array, mu: Array, sigma: Array, *, n_ause_bins: int = 100, tail_quantile: float = 0.9
) -> ScoringResults:
    """Compute all scoring rules in one pass."""
    y_true = _to_jax(y_true)
    mu = _to_jax(mu)
    sigma = jnp.maximum(_to_jax(sigma), 1e-8)

    # CRPS & Scaled CRPS (scoringrules: peer-reviewed implementation)
    crps_all = sr.crps_normal(y_true, mu, sigma, backend="jax")
    crps_val = float(jnp.mean(crps_all))
    y_std = float(jnp.std(y_true))
    scrps = crps_val / max(y_std, 1e-8)

    # DSS (no parametric Gaussian DSS in scoringrules; only ensemble-based dssuv_ensemble)
    dss_all = _dss(y_true, mu, sigma)
    dss = float(jnp.mean(dss_all))

    # Interval Scores (scoringrules: peer-reviewed implementation)
    lower_90 = mu - Z_90 * sigma
    upper_90 = mu + Z_90 * sigma
    lower_95 = mu - Z_95 * sigma
    upper_95 = mu + Z_95 * sigma

    is_90_all = sr.interval_score(y_true, lower_90, upper_90, 0.10, backend="jax")
    is_90 = float(jnp.mean(is_90_all))
    is_95_all = sr.interval_score(y_true, lower_95, upper_95, 0.05, backend="jax")
    is_95 = float(jnp.mean(is_95_all))

    # AUSE
    ause, ause_details = compute_ause(y_true, mu, sigma, n_ause_bins)

    # Tail-Weighted CRPS
    # Upper
    threshold_up = float(jnp.percentile(y_true, tail_quantile * 100))
    tail_upper = float(jnp.mean(_crps_rectified_gaussian(mu, sigma, y_true, threshold_up, "upper")))

    # Lower
    threshold_lo = float(jnp.percentile(y_true, (1 - tail_quantile) * 100))
    tail_lower = float(jnp.mean(_crps_rectified_gaussian(mu, sigma, y_true, threshold_lo, "lower")))

    return ScoringResults(
        crps=crps_val,
        scaled_crps=scrps,
        dss=dss,
        interval_score_90=is_90,
        interval_score_95=is_95,
        ause=ause,
        ause_oracle=ause_details["ause_oracle"],
        tail_crps_upper=tail_upper,
        tail_crps_lower=tail_lower,
    )


