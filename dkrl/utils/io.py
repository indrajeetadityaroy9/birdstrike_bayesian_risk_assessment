"""
Model persistence utilities.
"""

import torch

from dkrl.config.hardware import DEVICE
from dkrl.models.nigp import NIGPDeepKernelGP


def save_model(model: NIGPDeepKernelGP, path: str):
    """Save model checkpoint with complete architecture and configuration."""
    checkpoint = {
        "state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "latent_dim": model.latent_dim,
        "hidden_dims": model.hidden_dims,
        "num_ensemble": model.num_ensemble,
        "low_rank_dim": model.low_rank_dim,
        "lipschitz_gamma": model.lipschitz_gamma,
        "hutchinson_samples": model.hutchinson_samples,
    }
    torch.save(checkpoint, path)


def load_model(path: str) -> NIGPDeepKernelGP:
    """Load model from checkpoint (always to CUDA)."""
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    model = NIGPDeepKernelGP(
        input_dim=checkpoint["input_dim"],
        num_ensemble=checkpoint["num_ensemble"],
        low_rank_dim=checkpoint["low_rank_dim"],
        lipschitz_gamma=checkpoint["lipschitz_gamma"],
        hutchinson_samples=checkpoint["hutchinson_samples"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(DEVICE)
