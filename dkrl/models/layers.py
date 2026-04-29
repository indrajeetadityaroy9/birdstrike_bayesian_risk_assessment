"""
Neural network layers with Lipschitz control.

Contains dissipative layer implementations for bi-Lipschitz feature extraction.

References:
    - Dissipative Layers: arXiv:2410.22258
    - ECLipsE-GC Bounds: arXiv:2503.14297
    - BatchEnsemble: Wen et al. (2020)
"""

import math

import torch
import torch.nn as nn

from dkrl.config._constants import SILU_LIPSCHITZ


class DissipativeLinear(nn.Module):
    """
    Dissipative linear layer: ||W||_2 <= sqrt(gamma) via W = sqrt(gamma) * U @ diag(s) @ V.T.
    Ref: arXiv:2410.22258
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        gamma: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gamma = gamma

        # Orthogonal parameters (will be converted via QR)
        min_dim = min(in_features, out_features)
        self.U_params = nn.Parameter(torch.randn(out_features, min_dim) * 0.01)
        self.V_params = nn.Parameter(torch.randn(in_features, min_dim) * 0.01)

        # Singular value parameters (sigmoid to [0, 1])
        self.s_pre = nn.Parameter(torch.zeros(min_dim))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # QR cache: avoids recomputing QR on every forward pass within a batch
        # Cached tensors retain autograd history for correct gradient flow
        self._cached_U = None
        self._cached_V = None
        self._cache_valid = False

    def invalidate_cache(self):
        """Invalidate QR cache. Call after optimizer.step()."""
        self._cache_valid = False

    def _get_orthogonal(self, params: torch.Tensor) -> torch.Tensor:
        """Get orthogonal matrix via QR decomposition."""
        Q, _ = torch.linalg.qr(params)
        return Q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._cache_valid:
            self._cached_U = self._get_orthogonal(self.U_params)
            self._cached_V = self._get_orthogonal(self.V_params)
            self._cache_valid = True
        U = self._cached_U  # [out, min_dim]
        V = self._cached_V  # [in, min_dim]
        s = torch.sigmoid(self.s_pre)  # [min_dim], in [0, 1]

        # W = sqrt(gamma) * U @ diag(s) @ V.T
        # Compute efficiently: (x @ V) * s @ U.T
        Vx = x @ V  # [batch, min_dim]
        scaled = Vx * (math.sqrt(self.gamma) * s)  # [batch, min_dim]
        out = scaled @ U.T  # [batch, out_features]

        if self.bias is not None:
            out = out + self.bias
        return out

    def get_lipschitz_bound(self) -> torch.Tensor:
        """Return the Lipschitz bound (guaranteed <= sqrt(gamma))."""
        s = torch.sigmoid(self.s_pre)
        return math.sqrt(self.gamma) * s.max()


class DissipativeBatchEnsembleLinear(nn.Module):
    """
    BatchEnsemble version of DissipativeLinear.

    Combines LipKernel dissipative parameterization with BatchEnsemble
    for uncertainty quantification while maintaining Lipschitz guarantees.

    Per arXiv:2410.22258 combined with Wen et al. (2020) BatchEnsemble.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_ensemble: int = 4,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_ensemble = num_ensemble

        # Shared dissipative layer
        self.linear = DissipativeLinear(in_features, out_features, gamma=gamma, bias=True)

        # Ensemble scaling (constrained to [-1, 1] for Lipschitz preservation)
        # tanh(1.4722) ≈ 0.9, near-identity scaling at initialization
        _atanh_09 = 1.4722  # math.atanh(0.9)
        self.r = nn.Parameter(torch.ones(num_ensemble, in_features) * _atanh_09)
        self.s = nn.Parameter(torch.ones(num_ensemble, out_features) * _atanh_09)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        r_constrained = torch.tanh(self.r)  # [-1, 1]
        s_constrained = torch.tanh(self.s)  # [-1, 1]

        if x.dim() == 2:
            x = x.unsqueeze(1).expand(batch_size, self.num_ensemble, self.in_features)

        xr = x * r_constrained.unsqueeze(0)
        xr_flat = xr.reshape(-1, self.in_features)
        output_flat = self.linear(xr_flat)
        output = output_flat.reshape(batch_size, self.num_ensemble, self.out_features)

        return output * s_constrained.unsqueeze(0)

    def get_lipschitz_bound(self) -> torch.Tensor:
        """Return network Lipschitz bound."""
        return self.linear.get_lipschitz_bound()


class ResidualDissipativeBlock(nn.Module):
    """
    Residual block wrapping DissipativeBatchEnsembleLinear + SiLU.

    forward(x) = project(x) + activation(dissipative(x))

    For dimension mismatch, a learned DissipativeLinear projection handles the
    shortcut, ensuring the skip path is also Lipschitz-bounded.
    Lipschitz bound: Lip(project) + Lip(activation o dissipative).
    When project = identity: 1 + Lip(f).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_ensemble: int = 4,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.layer = DissipativeBatchEnsembleLinear(
            in_features, out_features, num_ensemble=num_ensemble, gamma=gamma,
        )
        self.activation = nn.SiLU()
        self.needs_projection = (in_features != out_features)
        if self.needs_projection:
            # Dissipative projection ensures skip path is also Lipschitz-bounded
            self.projection = DissipativeLinear(
                in_features, out_features, gamma=gamma, bias=False,
            )
        self.num_ensemble = num_ensemble

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.activation(self.layer(x))

        if self.needs_projection:
            if residual.dim() == 2:
                residual = residual.unsqueeze(1).expand(
                    -1, self.num_ensemble, residual.shape[-1]
                )
            residual = self.projection(residual)

        return residual + out

    def get_lipschitz_bound(self) -> torch.Tensor:
        return self.layer.get_lipschitz_bound()

    def get_projection_lipschitz_bound(self) -> torch.Tensor:
        """Return Lipschitz bound of the skip-connection path."""
        if self.needs_projection:
            return self.projection.get_lipschitz_bound()
        return torch.tensor(1.0)


def compute_lower_lipschitz_penalty(J: torch.Tensor, *, min_sigma: float = 0.1) -> torch.Tensor:
    """
    Bi-Lipschitz lower bound: penalize when sigma_min(J) < min_sigma.

    Args:
        J: Jacobian matrix [batch, out_dim, in_dim]
        min_sigma: Minimum singular value threshold

    Returns:
        Mean penalty for samples below threshold
    """
    _, S, _ = torch.linalg.svd(J)
    sigma_min = S[:, -1]
    return torch.clamp(min_sigma - sigma_min, min=0.0).mean()


def compute_eclipse_gc_bound(
    weights: list[torch.Tensor],
    *,
    c: float = 1.99,
    activation_lipschitz: float = 1.0,
) -> torch.Tensor:
    """
    Recursive ECLipsE-GC bound per arXiv:2503.14297 Theorem 1.

    Computes Lipschitz upper bound using recursive M_k matrices:
        M_k = 2*Lambda_{k-1} - Lambda_{k-1} @ W_{k-1} @ M_{k-1}^{-1} @ W_{k-1}^T @ Lambda_{k-1}
        Lambda_k(i,i) = c / sum_j|Gamma_k(i,j)|  (Eq. 9)

    The bound is multiplied by activation_lipschitz^(L-1) to account for
    non-[0,1]-slope-restricted activations (e.g. SiLU with Lip ≈ 1.1).

    Args:
        weights: List of weight matrices [W1, W2, ..., W_L]
        c: Scaling parameter in (0, 2), default 1.99 per paper
        activation_lipschitz: Lipschitz constant of activation function (default 1.0 for ReLU)

    Returns:
        Lipschitz upper bound
    """
    if len(weights) == 1:
        # Single layer: simple spectral norm
        return torch.linalg.matrix_norm(weights[0], ord=2)

    device = weights[0].device
    dtype = weights[0].dtype
    n0 = weights[0].shape[1]
    M = torch.eye(n0, device=device, dtype=dtype)

    # Process all layers except the last one
    for W in weights[:-1]:
        # Gamma_k = W_k @ M_{k-1}^{-1} @ W_k^T
        Gamma = W @ torch.linalg.solve(M, W.T)

        # Lambda_k(i,i) = c / sum_j|Gamma_k(i,j)| per Eq. 9
        row_sums = torch.abs(Gamma).sum(dim=1)
        # Avoid division by zero
        row_sums = torch.clamp(row_sums, min=1e-8)
        Lambda = torch.diag(c / row_sums)

        # M_k = 2*Lambda_k - Lambda_k @ Gamma_k @ Lambda_k
        M = 2 * Lambda - Lambda @ Gamma @ Lambda

    # Final layer bound
    W_final = weights[-1]
    final_Gamma = W_final @ torch.linalg.solve(M, W_final.T)
    bound = torch.sqrt(torch.linalg.matrix_norm(final_Gamma, ord=2))

    # Scale by activation Lipschitz constant for each activation layer
    # (L-1 activations between L weight matrices)
    n_activations = len(weights) - 1
    if activation_lipschitz != 1.0 and n_activations > 0:
        bound = bound * (activation_lipschitz ** n_activations)

    return bound


def _eclipse_sn_bound(
    weights: list[torch.Tensor],
    *,
    c: float = 1.0,
    activation_lipschitz: float = 1.0,
) -> torch.Tensor:
    """
    ECLipsE-SN variant: Lambda_k = (c / sigma_max(Gamma_k)) * I.

    Uses spectral norm scaling instead of row-sum scaling.
    """
    if len(weights) == 1:
        return torch.linalg.matrix_norm(weights[0], ord=2)

    device = weights[0].device
    dtype = weights[0].dtype
    n0 = weights[0].shape[1]
    M = torch.eye(n0, device=device, dtype=dtype)

    for W in weights[:-1]:
        Gamma = W @ torch.linalg.solve(M, W.T)
        sigma_max = torch.linalg.matrix_norm(Gamma, ord=2)
        sigma_max = torch.clamp(sigma_max, min=1e-8)
        lam = c / sigma_max
        Lambda = lam * torch.eye(Gamma.shape[0], device=device, dtype=dtype)
        M = 2 * Lambda - Lambda @ Gamma @ Lambda

    W_final = weights[-1]
    final_Gamma = W_final @ torch.linalg.solve(M, W_final.T)
    bound = torch.sqrt(torch.linalg.matrix_norm(final_Gamma, ord=2))

    n_activations = len(weights) - 1
    if activation_lipschitz != 1.0 and n_activations > 0:
        bound = bound * (activation_lipschitz ** n_activations)

    return bound


def _eclipse_shift_bound(
    weights: list[torch.Tensor],
    *,
    c: float = 1.5,
    activation_lipschitz: float = 1.0,
) -> torch.Tensor:
    """
    ECLipsE-Shift variant: Lambda_k(i,i) = 1 / (T_k(i,i) + c * sigma_max(0.5*Gamma_k - T_k)).

    Where T_k = diag(diag(0.5 * Gamma_k)).
    """
    if len(weights) == 1:
        return torch.linalg.matrix_norm(weights[0], ord=2)

    device = weights[0].device
    dtype = weights[0].dtype
    n0 = weights[0].shape[1]
    M = torch.eye(n0, device=device, dtype=dtype)

    for W in weights[:-1]:
        Gamma = W @ torch.linalg.solve(M, W.T)
        half_Gamma = 0.5 * Gamma
        T_diag = torch.diag(half_Gamma)
        residual = half_Gamma - torch.diag(T_diag)
        sigma_residual = torch.linalg.matrix_norm(residual, ord=2)
        denom = T_diag + c * sigma_residual
        denom = torch.clamp(denom, min=1e-8)
        Lambda = torch.diag(1.0 / denom)
        M = 2 * Lambda - Lambda @ Gamma @ Lambda

    W_final = weights[-1]
    final_Gamma = W_final @ torch.linalg.solve(M, W_final.T)
    bound = torch.sqrt(torch.linalg.matrix_norm(final_Gamma, ord=2))

    n_activations = len(weights) - 1
    if activation_lipschitz != 1.0 and n_activations > 0:
        bound = bound * (activation_lipschitz ** n_activations)

    return bound


def compute_eclipse_tightened_bound(
    weights: list[torch.Tensor],
    *,
    activation_lipschitz: float = 1.0,
) -> torch.Tensor:
    """
    Compute tightest ECLipsE bound across all variants (GC, SN, Shift).

    Takes the minimum bound across multiple parameterizations of each variant.
    """
    bounds = []

    # ECLipsE-GC variants
    for c in [1.5, 1.7, 1.99]:
        bounds.append(compute_eclipse_gc_bound(weights, c=c, activation_lipschitz=activation_lipschitz))

    # ECLipsE-SN variants
    for c in [1.0, 1.3, 1.5]:
        bounds.append(_eclipse_sn_bound(weights, c=c, activation_lipschitz=activation_lipschitz))

    # ECLipsE-Shift variants
    for c in [1.3, 1.5, 1.7]:
        bounds.append(_eclipse_shift_bound(weights, c=c, activation_lipschitz=activation_lipschitz))

    return torch.stack(bounds).min()


def compute_network_lipschitz_bound(model: nn.Module, *, c: float = 1.99) -> torch.Tensor:
    """
    Compute network Lipschitz bound with correct residual block composition.

    For stacked residual blocks f_2(f_1(x)), the bound is multiplicative:
        Lip(f_2 o f_1) <= Lip(f_2) * Lip(f_1)

    Per-block bound uses the triangle inequality:
        Lip(skip + branch) <= Lip(skip) + Lip(activation o branch)

    For consecutive non-residual layers, ECLipsE-GC gives a tighter bound
    than the product of individual spectral norms.

    For BatchEnsemble layers, computes effective weight per ensemble member
    and returns the max bound across all K members.
    """
    # Detect SiLU activations for Lipschitz correction
    act_lip = 1.0
    for module in model.modules():
        if isinstance(module, nn.SiLU):
            act_lip = SILU_LIPSCHITZ
            break

    # Collect structured block info: ("residual", ...) or ("linear", ...)
    # Each entry represents one architectural block
    block_info = []  # list of (block_type, data_dict)
    num_ensemble = None

    for module in model.children():
        if isinstance(module, ResidualDissipativeBlock):
            proj_lip = module.get_projection_lipschitz_bound()
            be = module.layer
            lin = be.linear
            U = lin._get_orthogonal(lin.U_params)
            V = lin._get_orthogonal(lin.V_params)
            sv = torch.sigmoid(lin.s_pre)
            W = math.sqrt(lin.gamma) * (U * sv.unsqueeze(0)) @ V.T
            r = torch.tanh(be.r)
            s = torch.tanh(be.s)
            block_info.append(("residual", {"W": W, "r": r, "s": s, "proj_lip": proj_lip}))
            num_ensemble = be.num_ensemble
        elif isinstance(module, DissipativeBatchEnsembleLinear):
            lin = module.linear
            U = lin._get_orthogonal(lin.U_params)
            V = lin._get_orthogonal(lin.V_params)
            sv = torch.sigmoid(lin.s_pre)
            W = math.sqrt(lin.gamma) * (U * sv.unsqueeze(0)) @ V.T
            r = torch.tanh(module.r)
            s = torch.tanh(module.s)
            block_info.append(("linear_be", {"W": W, "r": r, "s": s}))
            num_ensemble = module.num_ensemble
        elif isinstance(module, DissipativeLinear):
            U = module._get_orthogonal(module.U_params)
            V = module._get_orthogonal(module.V_params)
            sv = torch.sigmoid(module.s_pre)
            W = math.sqrt(module.gamma) * (U * sv.unsqueeze(0)) @ V.T
            block_info.append(("linear_plain", {"W": W}))
        elif isinstance(module, nn.Linear):
            block_info.append(("linear_plain", {"W": module.weight}))

    def _compute_bound_for_weights(blocks, ensemble_idx=None):
        """Compute network Lipschitz bound for a specific ensemble member (or None for plain)."""
        bound = torch.tensor(1.0, device=blocks[0][1]["W"].device)

        # Group consecutive non-residual layers for ECLipsE
        sequential_weights = []

        def _flush_sequential():
            nonlocal bound
            if sequential_weights:
                if len(sequential_weights) == 1:
                    seq_bound = torch.linalg.matrix_norm(sequential_weights[0], ord=2)
                else:
                    seq_bound = compute_eclipse_tightened_bound(
                        sequential_weights, activation_lipschitz=act_lip
                    )
                bound = bound * seq_bound
                sequential_weights.clear()

        for btype, data in blocks:
            if btype == "residual":
                # Flush any accumulated sequential layers before this residual block
                _flush_sequential()

                W = data["W"]
                proj_lip = data["proj_lip"]
                if ensemble_idx is not None:
                    r, s = data["r"], data["s"]
                    W_eff = s[ensemble_idx].unsqueeze(1) * W * r[ensemble_idx].unsqueeze(0)
                else:
                    W_eff = W

                # Per-block: Lip(skip + act(branch)) <= Lip(skip) + Lip(act) * ||W||_2
                branch_lip = act_lip * torch.linalg.matrix_norm(W_eff, ord=2)
                block_lip = proj_lip + branch_lip
                bound = bound * block_lip
            else:
                # Non-residual layer: accumulate for ECLipsE
                W = data["W"]
                if ensemble_idx is not None and "r" in data:
                    r, s = data["r"], data["s"]
                    W_eff = s[ensemble_idx].unsqueeze(1) * W * r[ensemble_idx].unsqueeze(0)
                else:
                    W_eff = W
                sequential_weights.append(W_eff)

        # Flush remaining sequential layers
        _flush_sequential()
        return bound

    if num_ensemble is None:
        return _compute_bound_for_weights(block_info)

    # Compute bound per ensemble member and take max
    bounds = []
    for k in range(num_ensemble):
        bounds.append(_compute_bound_for_weights(block_info, ensemble_idx=k))
    return torch.stack(bounds).max()


__all__ = [
    "DissipativeLinear",
    "DissipativeBatchEnsembleLinear",
    "ResidualDissipativeBlock",
    "compute_lower_lipschitz_penalty",
    "compute_eclipse_tightened_bound",
    "compute_network_lipschitz_bound",
]
