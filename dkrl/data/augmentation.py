"""
Feature Augmentation: Physics-informed NSR scaling + TrivialAugment random magnitude.
"""

import math

import torch

from dkrl.models.covariance import build_cov_19x19


def augment_features_19d(
    features: torch.Tensor,
    covariances_10: torch.Tensor,
    build_cov_fn=build_cov_19x19,
) -> torch.Tensor:
    """Apply physics-informed augmentation (GPS covariance, kinematic NSR, time precision)."""

    batch_size = features.size(0)
    device = features.device
    dtype = features.dtype
    features_aug = features.clone()

    # Features 0-5: Position/velocity - GPS covariance-based augmentation
    P_full = build_cov_fn(covariances_10)

    # Add sufficient jitter upfront to guarantee numerical stability
    jitter = torch.eye(6, device=P_full.device, dtype=P_full.dtype) * 1e-4
    P_6x6 = P_full[:, :6, :6] + jitter.unsqueeze(0)

    # Single batched Cholesky - guaranteed to succeed with adequate jitter
    L_6 = torch.linalg.cholesky(P_6x6)

    eps_6 = torch.randn_like(features[:, :6])
    perturbation_6 = torch.bmm(L_6, eps_6.unsqueeze(-1)).squeeze(-1)
    features_aug[:, :6] = features[:, :6] + perturbation_6

    # Features 6-8: Kinematic derived (speed_3d, heading_rad, vertical_velocity)
    vel_var_mean = covariances_10[:, 6:9].mean(dim=0, keepdim=True).clamp(min=1e-8)  # [1, 3]
    kin_std = features[:, 6:9].std(dim=0, keepdim=True).clamp(min=1e-6)
    kin_nsr = (vel_var_mean.sqrt() / kin_std).clamp(max=0.5)  # NSR capped at 0.5
    eps_kin = torch.randn_like(features[:, 6:9])
    kin_scale = torch.rand(batch_size, 1, device=device, dtype=dtype) * kin_nsr
    features_aug[:, 6:9] = features[:, 6:9] + kin_scale * kin_std * eps_kin

    # Features 9-10: Time encoding (sin/cos) - estimation precision scale
    tod_nsr = 1.0 / math.sqrt(max(batch_size, 1))
    eps_tod = torch.randn_like(features[:, 9:11])
    tod_scale = torch.rand(batch_size, 1, device=device, dtype=dtype) * tod_nsr
    features_aug[:, 9:11] = features[:, 9:11] + tod_scale * eps_tod

    # Features 11-15: Species traits - NEVER AUGMENTED (preserved from clone)

    # Features 16-18: Context (alt_agl, gps_error, quality)
    ctx_mean = features[:, 16:19].mean(dim=0, keepdim=True).abs().clamp(min=1e-6)
    ctx_std = features[:, 16:19].std(dim=0, keepdim=True).clamp(min=1e-6)
    ctx_cv = (ctx_std / ctx_mean).clamp(max=0.5)  # CV capped at 0.5
    eps_ctx = torch.randn_like(features[:, 16:19])
    ctx_scale = torch.rand(batch_size, 3, device=device, dtype=dtype) * ctx_cv
    features_aug[:, 16:19] = features[:, 16:19] + ctx_scale * ctx_std * eps_ctx

    # Signal integrity enforcement
    features_aug[:, 6] = features_aug[:, 6].clamp(min=0)  # speed_3d >= 0
    features_aug[:, 7] = features_aug[:, 7].clamp(min=-math.pi, max=math.pi)  # heading in [-π, π]
    features_aug[:, 9:11] = features_aug[:, 9:11].clamp(min=-1.01, max=1.01)  # sin/cos bounds
    features_aug[:, 18] = features_aug[:, 18].clamp(min=0, max=1)  # quality in [0, 1]

    return features_aug


__all__ = ["augment_features_19d"]
