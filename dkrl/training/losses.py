"""
Loss Functions for NIGP-DKL Training

DB-MTL (arXiv:2308.12029) unified loss: log-transform all losses + gradient magnitude balancing.
"""

import math

import scoringrules as sr
import torch
import torch.nn as nn
from gpytorch.distributions import MultivariateNormal

from dkrl.config._constants import VARIANCE_FLOOR
from dkrl.models.layers import (
    compute_lower_lipschitz_penalty,
    compute_network_lipschitz_bound,
)

_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)


# JAX evaluation counterpart: dkrl.evaluation.metrics._crps_rectified_gaussian
def _crps_rectified_gaussian_torch(
    pred_mean: torch.Tensor,
    pred_std: torch.Tensor,
    target: torch.Tensor,
    threshold: torch.Tensor,
    tail: str = "upper",
) -> torch.Tensor:
    """Exact CRPS for Rectified Gaussian (arXiv:2407.00650)."""
    sigma = torch.clamp(pred_std, min=1e-8)
    normal = torch.distributions.Normal(0, 1)

    if tail == "upper":
        mu_z = pred_mean - threshold
        y_trans = torch.clamp(target - threshold, min=0.0)
    else:
        mu_z = threshold - pred_mean
        y_trans = torch.clamp(threshold - target, min=0.0)

    def phi(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * x**2) * _INV_SQRT_2PI

    def Phi(x: torch.Tensor) -> torch.Tensor:
        return normal.cdf(x)

    def primitive_phi_sq(x: torch.Tensor) -> torch.Tensor:
        cdf_x = Phi(x)
        return x * cdf_x**2 + 2 * cdf_x * phi(x) - _INV_SQRT_PI * Phi(x * _SQRT_2)

    def primitive_one_minus_phi_sq(x: torch.Tensor) -> torch.Tensor:
        cdf_x = Phi(x)
        return x - 2 * (x * cdf_x + phi(x)) + primitive_phi_sq(x)

    val_0 = -mu_z / sigma
    val_y = (y_trans - mu_z) / sigma

    part1 = sigma * (primitive_phi_sq(val_y) - primitive_phi_sq(val_0))
    part2 = sigma * (-_INV_SQRT_PI - primitive_one_minus_phi_sq(val_y))

    return part1 + part2


def differentiable_tmcb(
    pred_mean: torch.Tensor,
    pred_var: torch.Tensor,
    target: torch.Tensor,
    tail_quantile: float = 0.9,
) -> torch.Tensor:
    """
    Differentiable Tail MisCaliBration via Wasserstein-1 distance (arXiv:2506.13687).

    Falls back to a wider tail (median) with reduced weight when too few tail samples,
    avoiding dead gradient paths.
    """
    pred_std = torch.sqrt(torch.clamp(pred_var, min=1e-8))
    z = (target - pred_mean) / pred_std

    pit = torch.distributions.Normal(0, 1).cdf(z)

    tail_mask = pit > tail_quantile
    n_tail = tail_mask.sum()

    # Batch-scaled thresholds for statistical evidence
    batch_size = len(target)
    expected_tail_count = batch_size * (1.0 - tail_quantile)
    min_tail_samples = max(10, int(0.02 * batch_size))
    min_fallback_samples = max(5, int(0.01 * batch_size))

    # Smooth fallback: use wider tail with evidence-proportional weight
    if n_tail < min_tail_samples:
        fallback_quantile = 0.5
        tail_mask = pit > fallback_quantile
        n_tail = tail_mask.sum()
        if n_tail < min_fallback_samples:
            # Weight proportional to statistical evidence
            return (pit - 0.5).abs().mean() * (n_tail / max(expected_tail_count, 1.0))
        effective_quantile = fallback_quantile
        weight = n_tail / max(expected_tail_count, 1.0)
    else:
        effective_quantile = tail_quantile
        weight = 1.0

    tail_pit = (pit[tail_mask] - effective_quantile) / (1.0 - effective_quantile)
    tail_pit = torch.clamp(tail_pit, min=0.0, max=1.0)

    sorted_pit, _ = torch.sort(tail_pit)
    n = len(sorted_pit)

    uniform_ref = torch.linspace(0.5 / n, 1.0 - 0.5 / n, n, device=target.device)

    return weight * (sorted_pit - uniform_ref).abs().mean()


class LowRankNIGPLoss(nn.Module):
    """
    DB-MTL unified loss (arXiv:2308.12029): log-transform + max-grad-norm scaling.

    Two balancing mechanisms:
    1. Loss-scale balancing: log(1 + |loss_i|) for scale invariance
    2. Gradient magnitude balancing: scale losses so gradient norms are comparable
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        *,
        min_lipschitz: float = 0.1,
        target_lipschitz: float = 1.0,
        tail_quantile: float = 0.9,
    ):
        super().__init__()
        self.min_lipschitz = min_lipschitz
        self.target_lipschitz = target_lipschitz
        self.feature_extractor = feature_extractor
        self.tail_quantile = tail_quantile

        # Exponential gradient balance schedule: starts frequent, doubles every 10 computations
        self._grad_balance_interval = 1
        self._grad_balance_count = 0

        # Cached gradient balance scales
        self._grad_scales: dict[str, float] | None = None
        self._step_count = 0

    def compute_individual_losses(
        self,
        dist: MultivariateNormal,
        target: torch.Tensor,
        *,
        jacobian_feat: torch.Tensor = None,
        cov_10: torch.Tensor,
        sample_weights: torch.Tensor,
        nigp_1st_precomputed: torch.Tensor = None,
        nigp_2nd_precomputed: torch.Tensor = None,
        trace_reg: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all individual losses."""
        weights_norm = sample_weights / sample_weights.sum()
        pred_mean = dist.mean
        pred_var = dist.variance

        # NLL
        nll = -dist.log_prob(target).mean()

        # 1st-order NIGP
        nigp_1st = (nigp_1st_precomputed * weights_norm).sum()
        nigp_1st = torch.clamp(nigp_1st, min=VARIANCE_FLOOR)

        # 2nd-order NIGP
        nigp_2nd = (nigp_2nd_precomputed * weights_norm).sum()
        nigp_2nd = torch.clamp(nigp_2nd, min=VARIANCE_FLOOR)

        # CRPS (scoringrules.crps_normal, PyTorch backend; JAX evaluation counterpart: scoringrules.crps_normal)
        pred_std = torch.sqrt(torch.clamp(pred_var, min=1e-8))
        crps_per_sample = sr.crps_normal(target, pred_mean, pred_std, backend="torch")
        crps = (crps_per_sample * weights_norm).sum()

        # Tail CRPS
        upper_threshold = torch.quantile(target, self.tail_quantile)
        lower_threshold = torch.quantile(target, 1.0 - self.tail_quantile)

        tail_upper = (_crps_rectified_gaussian_torch(
            pred_mean, pred_std, target, upper_threshold, tail="upper"
        ) * weights_norm).sum()

        tail_lower = (_crps_rectified_gaussian_torch(
            pred_mean, pred_std, target, lower_threshold, tail="lower"
        ) * weights_norm).sum()

        # Bi-Lipschitz penalties
        if jacobian_feat is not None:
            lower_lip = compute_lower_lipschitz_penalty(
                jacobian_feat, min_sigma=self.min_lipschitz
            )
            lip_bound = compute_network_lipschitz_bound(self.feature_extractor)
            eclipse = torch.clamp(lip_bound - self.target_lipschitz, min=0.0) ** 2
        else:
            lower_lip = torch.tensor(0.0, device=target.device)
            eclipse = torch.tensor(0.0, device=target.device)

        # TMCB
        tmcb = differentiable_tmcb(pred_mean, pred_var, target, self.tail_quantile)

        losses = {
            "nll": nll,
            "nigp_1st": nigp_1st,
            "nigp_2nd": nigp_2nd,
            "crps": crps,
            "tail_upper": tail_upper,
            "tail_lower": tail_lower,
            "lower_lip": lower_lip,
            "eclipse": eclipse,
            "tmcb": tmcb,
        }

        # GP basis trace regularization (anti-collapse, arXiv:2505.18526)
        if trace_reg is not None:
            losses["trace_reg"] = trace_reg

        return losses

    def _compute_grad_scales(
        self, losses: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        """Compute gradient magnitude balancing scales (arXiv:2308.12029 Sec 3.2)."""
        shared_params = [p for p in self.feature_extractor.parameters() if p.requires_grad]
        if not shared_params:
            return {k: 1.0 for k in losses}

        grad_norms = {}
        for name, loss in losses.items():
            if loss.requires_grad and loss.abs().item() > 1e-10:
                grads = torch.autograd.grad(
                    torch.log1p(loss.abs()), shared_params,
                    retain_graph=True, allow_unused=True,
                )
                norm_sq = sum(
                    g.norm().pow(2) for g in grads if g is not None
                )
                grad_norms[name] = norm_sq.sqrt().item()
            else:
                grad_norms[name] = 0.0

        max_norm = max(grad_norms.values()) if grad_norms else 1.0
        if max_norm < 1e-10:
            return {k: 1.0 for k in losses}

        scales = {}
        for name in losses:
            gn = grad_norms.get(name, 0.0)
            scales[name] = (max_norm / (gn + 1e-10)) if gn > 1e-10 else 1.0

        return scales

    def forward(
        self,
        dist: MultivariateNormal,
        target: torch.Tensor,
        *,
        jacobian_feat: torch.Tensor = None,
        cov_10: torch.Tensor,
        sample_weights: torch.Tensor,
        nigp_1st_precomputed: torch.Tensor = None,
        nigp_2nd_precomputed: torch.Tensor = None,
        trace_reg: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        DB-MTL combined loss: log-transform + gradient magnitude balancing.
        """
        losses = self.compute_individual_losses(
            dist, target,
            jacobian_feat=jacobian_feat,
            cov_10=cov_10,
            sample_weights=sample_weights,
            nigp_1st_precomputed=nigp_1st_precomputed,
            nigp_2nd_precomputed=nigp_2nd_precomputed,
            trace_reg=trace_reg,
        )

        # Exponential gradient balance schedule: frequent early, amortized late
        self._step_count += 1
        if self._step_count % self._grad_balance_interval == 0 or self._grad_scales is None:
            self._grad_scales = self._compute_grad_scales(losses)
            self._grad_balance_count += 1
            if self._grad_balance_count % 10 == 0:
                self._grad_balance_interval = min(self._grad_balance_interval * 2, 50)

        # DB-MTL: log-transform + gradient magnitude balancing
        total = torch.tensor(0.0, device=target.device)
        for name, loss in losses.items():
            scale = self._grad_scales.get(name, 1.0)
            total = total + scale * torch.log1p(loss.abs())

        return total
