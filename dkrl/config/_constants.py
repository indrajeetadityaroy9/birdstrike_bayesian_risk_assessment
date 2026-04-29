"""
Centralized Constants for DKRL
Numerical stability, coverage quantiles, GPS error model, data thresholds, covariance format.
"""

# Numerical Stability

EPSILON = 1e-8
VARIANCE_FLOOR = 1e-6
VARIANCE_CEILING = 1e6

# Normal Quantiles for Coverage Metrics

Z_68 = 0.99445788  # 68%
Z_90 = 1.64485363  # 90%
Z_95 = 1.95996398  # 95%

# GPS Error Model

UERE = 6.0  # User Equivalent Range Error (meters)

# Data Filtering Thresholds

DT_MAX = 3600.0  # Max time gap (seconds)
MIN_SEGMENT_LENGTH = 2

# Trophic Level Mapping

TROPHIC_LEVEL_MAP = {
    "Herbivore": 2.0,
    "Omnivore": 3.0,
    "Carnivore": 4.0,
    "Scavenger": 3.5,
}

# Activation Function Constants

SILU_LIPSCHITZ = 1.1  # max|SiLU'(x)| ≈ 1.1

__all__ = [
    # Numerical stability
    "EPSILON",
    "VARIANCE_FLOOR",
    "VARIANCE_CEILING",
    # Normal quantiles
    "Z_68",
    "Z_90",
    "Z_95",
    # GPS error model
    "UERE",
    # Data filtering
    "DT_MAX",
    "MIN_SEGMENT_LENGTH",
    # Trophic mapping
    "TROPHIC_LEVEL_MAP",
]
