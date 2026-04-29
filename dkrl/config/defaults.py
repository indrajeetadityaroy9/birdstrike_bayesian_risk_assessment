"""Default model configurations for NIGP-DKL.

SOTA defaults: Low-Rank GP, Dissipative Lipschitz, Hutch++ Hessian trace.
"""

from dataclasses import dataclass
from enum import Enum


class HessianMode(Enum):
    """Hessian trace estimation strategy for NIGP 2nd-order corrections."""

    HUTCHPP = "hutchpp"
    HUTCHINSON = "hutchinson"
    NONE = "none"


@dataclass
class NIGPModelConfig:
    """Configuration for NIGP-DKL model."""

    # Architecture
    input_dim: int = 19
    hidden_dims: list[int] | None = None  # Default: [4*input_dim, 4*input_dim]
    low_rank_dim: int = 64
    num_ensemble: int = 4

    # NIGP corrections
    hessian_mode: HessianMode = HessianMode.HUTCHPP
    hutchinson_samples: int = 20

    # Lipschitz control
    lipschitz_gamma: float = 1.0
    lipschitz_interval: int = 10

    # Training
    augmentation: bool = True
    max_wall_seconds: int = 14400
    tail_quantile: float = 0.9

    # Seeding
    seed: int = 42


__all__ = [
    "HessianMode",
    "NIGPModelConfig",
]
