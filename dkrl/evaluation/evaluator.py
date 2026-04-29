"""
Model evaluation: NIGP correction, coverage, scoring rules, conditional coverage.
"""

from collections.abc import Callable

import torch

from dkrl.config._constants import Z_68, Z_90, Z_95
from dkrl.config.hardware import DEVICE
from dkrl.data import DatasetBundle
from dkrl.evaluation.metrics import compute_all_scores, compute_epistemic_crps
from dkrl.models.nigp import NIGPDeepKernelGP
from dkrl.models.predict_utils import batched_predict
from dkrl.training.batch_sizing import compute_batch_size


def evaluate_nigp_model(
    model: NIGPDeepKernelGP,
    bundle: DatasetBundle,
    split: str = "test",
    *,
    predict_fn: Callable = None,
) -> dict:
    """Evaluate NIGP-DKL (NIGP correction, coverage, scoring rules)."""
    features, labels, covariances, _ = bundle.get_split(split, normalize_quality=False)
    n_samples = len(features)
    batch_size = compute_batch_size(n_samples)

    model.eval()
    X = torch.as_tensor(features, dtype=torch.float32, device=DEVICE)
    y = torch.as_tensor(labels, dtype=torch.float32, device=DEVICE)
    P = torch.as_tensor(covariances, dtype=torch.float32, device=DEVICE)

    if predict_fn is None:
        predict_fn = model.predict_with_nigp

    pred_mean, pred_var = batched_predict(predict_fn, X, P, batch_size)

    pred_std = torch.sqrt(pred_var)

    errors = pred_mean - y
    rmse = torch.sqrt((errors ** 2).mean()).item()
    mae = errors.abs().mean().item()

    lower_95 = pred_mean - Z_95 * pred_std
    upper_95 = pred_mean + Z_95 * pred_std
    coverage_95 = ((y >= lower_95) & (y <= upper_95)).float().mean().item()

    lower_90 = pred_mean - Z_90 * pred_std
    upper_90 = pred_mean + Z_90 * pred_std
    coverage_90 = ((y >= lower_90) & (y <= upper_90)).float().mean().item()

    lower_68 = pred_mean - Z_68 * pred_std
    upper_68 = pred_mean + Z_68 * pred_std
    coverage_68 = ((y >= lower_68) & (y <= upper_68)).float().mean().item()

    nll = -torch.distributions.Normal(pred_mean, pred_std).log_prob(y).mean().item()
    mean_std = pred_std.mean().item()

    scores = compute_all_scores(y_true=y, mu=pred_mean, sigma=pred_std)

    results = {
        "rmse": rmse,
        "mae": mae,
        "nll": nll,
        "coverage_68": coverage_68,
        "coverage_90": coverage_90,
        "coverage_95": coverage_95,
        "mean_std": mean_std,
        "calibration_error_95": abs(coverage_95 - 0.95),
        "calibration_error_90": abs(coverage_90 - 0.90),
        "calibration_error_68": abs(coverage_68 - 0.68),
        "crps": scores.crps,
        "scaled_crps": scores.scaled_crps,
        "dss": scores.dss,
        "interval_score_90": scores.interval_score_90,
        "interval_score_95": scores.interval_score_95,
        "ause": scores.ause,
        "ause_oracle": scores.ause_oracle,
        "tail_crps_upper": scores.tail_crps_upper,
        "tail_crps_lower": scores.tail_crps_lower,
    }

    bin_edges = torch.quantile(pred_std, torch.linspace(0, 1, 6, device=pred_std.device))
    for i in range(5):
        mask = (pred_std >= bin_edges[i]) & (pred_std < bin_edges[i + 1])
        if i == 4:  # Include upper boundary for last bin
            mask = pred_std >= bin_edges[i]
        if mask.any():
            bin_cov = ((y[mask] >= lower_95[mask]) & (y[mask] <= upper_95[mask])).float().mean()
            results[f"conditional_cov95_bin{i}"] = bin_cov.item()

    n_decomp = min(len(X), max(2048, len(X) // 10))
    decomp_aleatoric, decomp_epistemic = [], []
    with torch.no_grad():
        for i in range(0, n_decomp, batch_size):
            end = min(i + batch_size, n_decomp)
            _, decomp = model.predict_with_decomposed_uncertainty(X[i:end], P[i:end])
            decomp_aleatoric.append(decomp.aleatoric_ratio)
            decomp_epistemic.append(decomp.epistemic_ratio)
    results["mean_aleatoric_ratio"] = torch.cat(decomp_aleatoric).mean().item()
    results["mean_epistemic_ratio"] = torch.cat(decomp_epistemic).mean().item()

    n_sr = min(len(X), max(2048, len(X) // 10))
    mu_ens, sig_ens = [], []
    with torch.no_grad():
        x_norm = model.normalize(X[:n_sr])
        h_ens = model.feature_extractor(x_norm)  # [N, K, latent]
        K = h_ens.shape[1]
        for k in range(K):
            dist_k = model.gp_layer(h_ens[:, k, :])
            mu_ens.append(dist_k.mean)
            sig_ens.append(dist_k.variance.sqrt())
    mu_stack = torch.stack(mu_ens, dim=1)  # [N, K]
    sig_stack = torch.stack(sig_ens, dim=1)  # [N, K]
    sr_results = compute_epistemic_crps(mu_stack, sig_stack)
    results["au_crps"] = sr_results["au_crps"]
    results["eu_crps"] = sr_results["eu_crps"]

    return results
