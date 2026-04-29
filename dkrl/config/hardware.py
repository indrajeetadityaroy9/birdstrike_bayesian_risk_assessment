"""
Configuration - Hardware Constants Only
Device spec and numerical stability constants. No hyperparameters.
Runtime: PyTorch 2.4.0 (CUDA), H100/Ampere+.
"""

import os
from pathlib import Path

import torch

DEVICE = torch.device("cuda")


def get_data_dir() -> Path:
    """Get data directory from DKRL_DATA_DIR env var."""
    return Path(os.environ["DKRL_DATA_DIR"])


def configure_cuda_backends() -> None:
    """Enable TF32 tensor cores for FP32 matmuls on Ampere+ GPUs."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
