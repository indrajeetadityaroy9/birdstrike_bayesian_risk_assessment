"""
Reproducibility utilities for DKRL.
"""

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Set seeds for all RNG sources (numpy, torch, cuda)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
