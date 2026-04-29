"""Batched prediction utility."""

import torch


def batched_predict(
    predict_fn, X: torch.Tensor, P: torch.Tensor, batch_size: int = 512
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run predict_fn in batches, return concatenated (means, vars)."""
    all_means, all_vars = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            mean, var = predict_fn(X[i : i + batch_size], P[i : i + batch_size])
            all_means.append(mean)
            all_vars.append(var)
    return torch.cat(all_means), torch.cat(all_vars)
