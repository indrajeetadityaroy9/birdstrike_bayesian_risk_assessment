"""
NIGP Deep Kernel GP Model.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from gpytorch.distributions import MultivariateNormal

from dkrl.config._constants import VARIANCE_FLOOR
from dkrl.config.defaults import HessianMode
from dkrl.models.covariance import build_cov_19x19
from dkrl.models.covariance import normalize_cov_10 as _normalize_cov_10
from dkrl.models.gp import LowRankGPLayer
from dkrl.models.kernels import nigp_trace
from dkrl.models.layers import DissipativeBatchEnsembleLinear, ResidualDissipativeBlock
from dkrl.models.normalization import NormalizationMixin


@dataclass
class UncertaintyDecomposition:
    """
    Decomposed uncertainty: aleatoric (obs+input+2nd_aleatoric) vs epistemic (gp+2nd_epistemic).
    """

    observation_noise: torch.Tensor  # GP observation noise [N]
    input_noise: torch.Tensor  # NIGP 1st-order: J @ P @ J.T [N]
    nigp_2nd_aleatoric: torch.Tensor  # Aleatoric portion of 2nd-order [N]

    gp_variance: torch.Tensor  # GP posterior variance [N]
    nigp_2nd_epistemic: torch.Tensor  # Epistemic portion of 2nd-order [N]

    aleatoric: torch.Tensor  # observation_noise + input_noise + nigp_2nd_aleatoric [N]
    epistemic: torch.Tensor  # gp_variance + nigp_2nd_epistemic [N]
    total: torch.Tensor  # aleatoric + epistemic [N]

    mean_correction: torch.Tensor  # 0.5 * tr(H @ P) [N]

    @property
    def aleatoric_ratio(self) -> torch.Tensor:
        """Fraction of total uncertainty that is aleatoric."""
        return self.aleatoric / (self.total + 1e-8)

    @property
    def epistemic_ratio(self) -> torch.Tensor:
        """Fraction of total uncertainty that is epistemic."""
        return self.epistemic / (self.total + 1e-8)


class NIGPDeepKernelGP(NormalizationMixin, nn.Module):
    """
    Second-order NIGP-DKL: low-rank GP + dissipative deep kernel + Hutchinson trace.
    """

    def __init__(
        self,
        input_dim: int = 19,
        *,
        hidden_dims: list[int] | None = None,
        num_ensemble: int = 4,
        low_rank_dim: int = 64,
        hutchinson_samples: int = 20,
        lipschitz_gamma: float = 1.0,
        hessian_mode: HessianMode = HessianMode.HUTCHPP,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dim = 4 * input_dim
            hidden_dims = [hidden_dim, hidden_dim]
        self.hidden_dims = hidden_dims
        self.latent_dim = input_dim
        self.input_dim = input_dim
        self.num_ensemble = num_ensemble

        self.low_rank_dim = low_rank_dim
        self.lipschitz_gamma = lipschitz_gamma
        self.hutchinson_samples = hutchinson_samples
        self.hessian_mode = hessian_mode

        layers = []
        in_dim = input_dim
        for h_dim in self.hidden_dims:
            layers.append(ResidualDissipativeBlock(
                in_dim, h_dim, num_ensemble=num_ensemble, gamma=lipschitz_gamma,
            ))
            in_dim = h_dim

        layers.append(DissipativeBatchEnsembleLinear(
            in_dim, self.latent_dim, num_ensemble=num_ensemble, gamma=lipschitz_gamma,
        ))
        self.feature_extractor = nn.Sequential(*layers)

        self.gp_layer = LowRankGPLayer(self.latent_dim, rank=low_rank_dim)

        self.register_normalization_buffers(input_dim)

    def feature_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extractor with ensemble averaging."""
        return self.feature_extractor(x).mean(dim=1)

    def normalize_cov_10(self, cov_10: torch.Tensor) -> torch.Tensor:
        """Normalize compressed covariance to match normalized input space."""
        return _normalize_cov_10(cov_10, self.input_std)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        x_norm = self.normalize(x)
        h = self.feature_forward(x_norm)
        return self.gp_layer(h)

    def forward_with_jacobian_and_hutchinson_trace(
        self,
        x: torch.Tensor,
        cov_10: torch.Tensor,
        *,
        hutchinson_samples: int | None = None,
        skip_feature_jacobian: bool = False,
    ) -> tuple[MultivariateNormal, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training-path: vmap J_feat + Hutchinson NIGP corrections.
        """
        x_norm = self.normalize(x)
        cov_10_norm = self.normalize_cov_10(cov_10)

        J_feat = None if skip_feature_jacobian else self._compute_jacobian(x_norm)

        J_gp = self._compute_gp_mean_jacobian(x_norm)
        nigp_1st = nigp_trace(J_gp, cov_10_norm, reduce="none").detach()

        P_19 = build_cov_19x19(cov_10_norm)

        def gp_mean_fn(x_in):
            return self.gp_layer(self.feature_forward(x_in)).mean

        n = hutchinson_samples or self.hutchinson_samples
        nigp_mean_corr, nigp_2nd = self._hutchinson_nigp_correction(x_norm, P_19, gp_mean_fn, n)
        nigp_2nd = nigp_2nd.detach()
        nigp_mean_corr = nigp_mean_corr.detach()

        h = self.feature_forward(x_norm)
        f = self.gp_layer(h)

        return f, J_feat, nigp_1st, nigp_2nd, nigp_mean_corr

    # JAX evaluation counterpart: dkrl.evaluation.metrics.compute_epistemic_crps
    def compute_scoring_rule_decomposition(
        self, x_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        CRPS-based aleatoric/epistemic decomposition via ensemble cross-scores.

        Uses the scoring-rule framework (arXiv:2404.12215):
            AU = (1/K) sum_k CRPS_self(k)
            TU = (1/K^2) sum_{i,j} cross-CRPS(i, j)
            EU = TU - AU

        Returns:
            aleatoric_frac: Per-sample aleatoric fraction [N]
            au: Per-sample aleatoric uncertainty [N]
            eu: Per-sample epistemic uncertainty [N]
        """
        h_ensemble = self.feature_extractor(x_norm)  # [N, K, latent]
        K = h_ensemble.shape[1]

        mu_k = []
        sigma_k = []
        for k in range(K):
            dist_k = self.gp_layer(h_ensemble[:, k, :])
            mu_k.append(dist_k.mean)
            sigma_k.append(dist_k.variance.sqrt())

        inv_sqrt_pi = 1.0 / math.sqrt(math.pi)

        # AU = (1/K) sum_k sigma_k / sqrt(pi)
        au = sum(s * inv_sqrt_pi for s in sigma_k) / K

        # Cross-CRPS: E_{Y~N(mu_j,sigma_j^2)}[CRPS(N(mu_i,sigma_i^2), Y)]
        normal = torch.distributions.Normal(0, 1)
        tu = torch.zeros_like(au)
        for i in range(K):
            for j in range(K):
                sigma_c = (sigma_k[i] ** 2 + sigma_k[j] ** 2).sqrt()
                z_c = (mu_k[j] - mu_k[i]) / (sigma_c + 1e-8)
                cross = sigma_c * (
                    z_c * (2 * normal.cdf(z_c) - 1)
                    + 2 * normal.log_prob(z_c).exp()
                    - inv_sqrt_pi
                )
                tu = tu + cross
        tu = tu / (K * K)

        eu = tu - au
        aleatoric_frac = au / (au + eu.abs() + 1e-8)
        return aleatoric_frac, au, eu

    def _compute_nigp_corrections(
        self, x_norm: torch.Tensor, cov_10: torch.Tensor, *, return_2nd_order_split: bool = False
    ) -> tuple[torch.Tensor, ...]:
        """
        Compute shared NIGP corrections (1st and 2nd order) via Hutchinson.
        """
        with torch.enable_grad():
            J_gp = self._compute_gp_mean_jacobian(x_norm)

            cov_10_norm = self.normalize_cov_10(cov_10)
            P_19 = build_cov_19x19(cov_10_norm)

            input_noise = nigp_trace(J_gp, cov_10_norm, reduce="none")

            def gp_mean_fn(x_in):
                return self.gp_layer(self.feature_forward(x_in)).mean

            mean_correction, nigp_2nd_order = self._hutchinson_nigp_correction(
                x_norm, P_19, gp_mean_fn, self.hutchinson_samples
            )

        input_noise = input_noise.detach()
        mean_correction = mean_correction.detach()
        nigp_2nd_order = nigp_2nd_order.detach()

        if return_2nd_order_split:
            aleatoric_frac, _, _ = self.compute_scoring_rule_decomposition(x_norm)
            aleatoric_frac = aleatoric_frac.detach()
            return input_noise, nigp_2nd_order, mean_correction, aleatoric_frac

        return input_noise, nigp_2nd_order, mean_correction

    def predict_with_nigp(
        self, x: torch.Tensor, cov_10: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict with NIGP correction per arXiv:2509.14710."""
        with torch.no_grad():
            f = self.forward(x)
            x_norm = self.normalize(x)

            input_noise, nigp_2nd_order, nigp_mean_correction = (
                self._compute_nigp_corrections(x_norm, cov_10)
            )

            nigp_var_correction = torch.clamp(
                input_noise + nigp_2nd_order, min=VARIANCE_FLOOR
            )

            pred_mean = f.mean
            pred_var = f.variance

            if pred_mean.dim() > 1:
                mean_norm = pred_mean.mean(dim=0) + nigp_mean_correction
                variance_norm = (
                    pred_var.mean(dim=0) + pred_mean.var(dim=0) + nigp_var_correction
                )
            else:
                mean_norm = pred_mean + nigp_mean_correction
                variance_norm = pred_var + nigp_var_correction

            mean = mean_norm * self.target_std + self.target_mean
            variance = variance_norm * (self.target_std**2)

            return mean, variance

    def predict_with_decomposed_uncertainty(
        self, x: torch.Tensor, cov_10: torch.Tensor
    ) -> tuple[torch.Tensor, UncertaintyDecomposition]:
        """Predict with decomposed aleatoric/epistemic uncertainty."""
        with torch.no_grad():
            f = self.forward(x)
            x_norm = self.normalize(x)

            input_noise, nigp_2nd_order, nigp_mean_correction, aleatoric_frac = (
                self._compute_nigp_corrections(x_norm, cov_10, return_2nd_order_split=True)
            )

            observation_noise = self.gp_layer.noise.expand(x.shape[0])
            gp_variance = f.variance - observation_noise

            if f.mean.dim() > 1:
                mean_norm = f.mean.mean(dim=0) + nigp_mean_correction
                gp_variance = gp_variance.mean(dim=0)
            else:
                mean_norm = f.mean + nigp_mean_correction

            observation_noise = torch.clamp(observation_noise, min=VARIANCE_FLOOR)
            input_noise = torch.clamp(input_noise, min=0)
            gp_variance = torch.clamp(gp_variance, min=VARIANCE_FLOOR)
            nigp_2nd_order = torch.clamp(nigp_2nd_order, min=0)

            # Split 2nd-order NIGP into aleatoric/epistemic
            nigp_2nd_aleatoric = nigp_2nd_order * aleatoric_frac
            nigp_2nd_epistemic = nigp_2nd_order * (1.0 - aleatoric_frac)

            aleatoric = observation_noise + input_noise + nigp_2nd_aleatoric
            epistemic = gp_variance + nigp_2nd_epistemic
            total = aleatoric + epistemic

            scale_factor = self.target_std**2
            mean = mean_norm * self.target_std + self.target_mean

            decomposition = UncertaintyDecomposition(
                observation_noise=observation_noise * scale_factor,
                input_noise=input_noise * scale_factor,
                gp_variance=gp_variance * scale_factor,
                nigp_2nd_aleatoric=nigp_2nd_aleatoric * scale_factor,
                nigp_2nd_epistemic=nigp_2nd_epistemic * scale_factor,
                aleatoric=aleatoric * scale_factor,
                epistemic=epistemic * scale_factor,
                total=total * scale_factor,
                mean_correction=nigp_mean_correction * self.target_std,
            )

            return mean, decomposition

    def _single_hvp(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        target_fn: Callable,
    ) -> torch.Tensor:
        """Compute Hessian-vector product H @ v for target_fn at x."""
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            y = target_fn(x_in)
            grad_y = torch.autograd.grad(
                y.sum(), x_in, create_graph=True, retain_graph=True
            )[0]
            hvp = torch.autograd.grad(
                (grad_y * v).sum(), x_in, retain_graph=False
            )[0]
        return hvp

    def _hutchinson_nigp_correction(
        self,
        x: torch.Tensor,
        P: torch.Tensor,
        target_fn: Callable,
        n_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch to Hutch++ or standard Hutchinson based on hessian_mode."""
        if self.hessian_mode is HessianMode.NONE:
            N = x.shape[0]
            zeros = torch.zeros(N, device=x.device, dtype=x.dtype)
            return zeros, zeros

        if self.hessian_mode is HessianMode.HUTCHPP:
            return self._hutchpp_nigp_correction(x, P, target_fn, n_samples)

        return self._hutchinson_standard(x, P, target_fn, n_samples)

    def _hutchinson_standard(
        self,
        x: torch.Tensor,
        P: torch.Tensor,
        target_fn: Callable,
        n_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard Hutchinson estimator: O(1/m) variance."""
        N, D = x.shape

        trace_estimates = torch.zeros(N, device=x.device, dtype=x.dtype)

        for _ in range(n_samples):
            z = torch.randint(0, 2, (N, D), device=x.device, dtype=x.dtype) * 2 - 1
            Pz = torch.bmm(P, z.unsqueeze(-1)).squeeze(-1)
            Hvp = self._single_hvp(x, Pz, target_fn)
            trace_estimates = trace_estimates + (z * Hvp).sum(dim=-1)

        trace_hp = trace_estimates / n_samples
        mean_correction = 0.5 * trace_hp
        var_correction = 0.5 * trace_hp.pow(2)
        return mean_correction, var_correction

    def _hutchpp_nigp_correction(
        self,
        x: torch.Tensor,
        P: torch.Tensor,
        target_fn: Callable,
        n_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Hutch++ estimator: O(1/m^2) variance via QR deflation.

        Splits budget into: k sketch vectors, k QR vectors, m-2k stochastic.
        """
        N, D = x.shape
        m = n_samples
        k = max(m // 3, 1)
        m_stoch = max(m - 2 * k, 1)

        # Step 1: Sketch — compute H@P@s for k random Gaussian vectors
        AS_cols = []
        for _ in range(k):
            s_i = torch.randn(N, D, device=x.device, dtype=x.dtype)
            Ps_i = torch.bmm(P, s_i.unsqueeze(-1)).squeeze(-1)
            HPs_i = self._single_hvp(x, Ps_i, target_fn)
            AS_cols.append(HPs_i)

        # [N, D, k]
        AS = torch.stack(AS_cols, dim=-1)

        # Step 2: QR — extract orthonormal basis per sample
        Q, _ = torch.linalg.qr(AS)  # [N, D, k]

        # Step 3: Deterministic trace of top eigenspace
        trace_det = torch.zeros(N, device=x.device, dtype=x.dtype)
        for i in range(k):
            q_i = Q[:, :, i]  # [N, D]
            Pq_i = torch.bmm(P, q_i.unsqueeze(-1)).squeeze(-1)
            HPq_i = self._single_hvp(x, Pq_i, target_fn)
            trace_det = trace_det + (q_i * HPq_i).sum(dim=-1)

        # Step 4: Stochastic trace of residual (projected out of Q)
        trace_stoch = torch.zeros(N, device=x.device, dtype=x.dtype)
        for _ in range(m_stoch):
            g = torch.randn(N, D, device=x.device, dtype=x.dtype)
            # Project out Q: g_perp = g - Q @ Q^T @ g
            g_proj = torch.bmm(Q.transpose(1, 2), g.unsqueeze(-1))  # [N, k, 1]
            g_perp = g - torch.bmm(Q, g_proj).squeeze(-1)
            Pg = torch.bmm(P, g_perp.unsqueeze(-1)).squeeze(-1)
            HPg = self._single_hvp(x, Pg, target_fn)
            trace_stoch = trace_stoch + (g_perp * HPg).sum(dim=-1)

        trace_stoch = trace_stoch / m_stoch

        trace_hp = trace_det + trace_stoch
        mean_correction = 0.5 * trace_hp
        var_correction = 0.5 * trace_hp.pow(2)
        return mean_correction, var_correction

    def _compute_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian of feature_forward w.r.t. input."""
        def f_single(x_single):
            return self.feature_forward(x_single.unsqueeze(0)).squeeze(0)

        J = torch.vmap(torch.func.jacrev(f_single))(x)
        return J

    def _compute_gp_mean_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian of GP predictive mean w.r.t. input."""
        def f_single(x_single):
            x_batched = x_single.unsqueeze(0)
            return self.gp_layer(self.feature_forward(x_batched)).mean.squeeze(0)

        J_gp = torch.vmap(torch.func.jacrev(f_single))(x)
        if J_gp.dim() == 2:
            J_gp = J_gp.unsqueeze(1)
        return J_gp
