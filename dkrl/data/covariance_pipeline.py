"""10-element compressed covariance computation from GPS error model.

Format (matches dkrl/kernels.py build_cov_19x19):
    [0] var_x       - Position x variance
    [1] cov_x_vx    - Position-velocity x cross-covariance
    [2] var_y       - Position y variance
    [3] cov_y_vy    - Position-velocity y cross-covariance
    [4] var_z       - Position z variance
    [5] cov_z_vz    - Position-velocity z cross-covariance
    [6] var_vx      - Velocity x variance
    [7] var_vy      - Velocity y variance
    [8] var_vz      - Velocity z variance
    [9] trace       - Total trace (for quality computation)
"""


import numpy as np
import pandas as pd

from dkrl.config._constants import UERE


def compute_covariance_10(df: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Compute 10-element compressed covariance per sample.

    GPS error model:
        var_pos = (DOP * UERE)^2
        var_vel = var_pos / dt^2 (propagated through finite difference)
        cov_pos_vel = -var_pos / dt (anti-correlation from differencing)

    Segment boundary handling:
        Segment starts (first point in each segment) have unknown velocity.
        Variance is inflated 10x for these points (conservative estimate).

    Args:
        df: Telemetry DataFrame with dop, dt, is_segment_start columns

    Returns:
        covariances: [N, 10] array
        n_cauchy_schwarz_violations: Number of Cauchy-Schwarz violations (for audit)

    """
    # Get DOP and dt (pre-filled in create_dataset)
    dop = df["dop"].values
    dt = df["dt"].values

    # Pre-filter guarantees dt <= dt_max (parameterized, default 3600s)
    # Clamp dt to minimum 0.1s for numerical stability (avoids division by near-zero)
    dt = np.maximum(dt, 0.1)

    # Position variance from GPS error model
    var_pos = (dop * UERE) ** 2  # [N]

    # Velocity variance (propagated from position uncertainty)
    # For finite difference: v = (x2 - x1) / dt
    # Var(v) = (Var(x2) + Var(x1)) / dt^2 ~ 2 * var_pos / dt^2
    var_vel = 2 * var_pos / (dt ** 2)

    # Segment boundary handling: inflate variance for segment starts
    # These points have synthetic velocity (zero), so uncertainty is much higher
    if "is_segment_start" in df.columns:
        segment_start_mask = df["is_segment_start"].values
        var_inflation = np.where(segment_start_mask, 10.0, 1.0)
        var_pos = var_pos * var_inflation
        var_vel = var_vel * var_inflation

    # Position-velocity covariance (anti-correlation from differencing)
    # Cov(x1, v) = -Var(x1) / dt
    cov_pos_vel = -var_pos / dt

    # Build covariance array
    # Trace computation: sum of all variance terms (6x6 kinematic block)
    trace_6x6 = 4.0 * var_pos + 4.0 * var_vel

    covariances = np.column_stack([
        var_pos,          # [0] var_x
        cov_pos_vel,      # [1] cov_x_vx
        var_pos,          # [2] var_y (same as x for isotropic GPS)
        cov_pos_vel,      # [3] cov_y_vy
        var_pos * 2.0,    # [4] var_z (vertical GPS is ~2x worse)
        cov_pos_vel,      # [5] cov_z_vz
        var_vel,          # [6] var_vx
        var_vel,          # [7] var_vy
        var_vel * 2.0,    # [8] var_vz (vertical worse)
        trace_6x6,        # [9] trace of 6x6 kinematic covariance block
    ])

    # Validate Cauchy-Schwarz inequality: |cov(x,v)| <= sqrt(var(x) * var(v))
    cauchy_schwarz_pairs = [(0, 6, 1), (2, 7, 3), (4, 8, 5)]
    n_cauchy_schwarz_violations = 0
    for pos_idx, vel_idx, cov_idx in cauchy_schwarz_pairs:
        var_p = covariances[:, pos_idx]
        var_v = covariances[:, vel_idx]
        cov_pv = np.abs(covariances[:, cov_idx])
        bound = np.sqrt(np.maximum(var_p, 0) * np.maximum(var_v, 0))
        violations = cov_pv > bound * 1.01  # 1% tolerance
        if violations.any():
            n_cauchy_schwarz_violations += int(violations.sum())
            # Clamp to Cauchy-Schwarz bound (preserve sign)
            covariances[violations, cov_idx] = (
                np.sign(covariances[violations, cov_idx]) * bound[violations] * 0.99
            )
    if n_cauchy_schwarz_violations > 0:
        print(f"covariance_fix clamped_violations={n_cauchy_schwarz_violations}")

    # Clamp variance terms (diagonal) to positive range [1e-6, 1e6] - vectorized
    var_idx = [0, 2, 4, 6, 7, 8, 9]
    covariances[:, var_idx] = np.clip(covariances[:, var_idx], 1e-6, 1e6)

    # Clamp cross-covariance terms by MAGNITUDE while preserving SIGN - vectorized
    cross_idx = [1, 3, 5]
    covariances[:, cross_idx] = (
        np.sign(covariances[:, cross_idx]) *
        np.clip(np.abs(covariances[:, cross_idx]), 1e-6, 1e6)
    )

    return covariances.astype(np.float32), n_cauchy_schwarz_violations
