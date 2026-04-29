"""Per-sample quality score computation with inverse-variance weighted components.

Used for curriculum learning sample weighting during training.

Reference: arXiv:2505.01665 Adaptively Point-weighting Curriculum Learning
"""

import numpy as np
import pandas as pd


def compute_quality_scores(
    df: pd.DataFrame,
    covariances: np.ndarray,
    velocity_valid: np.ndarray,
) -> np.ndarray:
    """Compute per-sample quality scores with inverse-variance weighted components.

    Weights are derived from data: w_i = 1/var(component_i), normalized to sum=1.
    Components with high variance (noisy) get low weight; components with low
    variance (reliable) get high weight. This eliminates manual weight tuning.

    Args:
        df: Telemetry DataFrame
        covariances: [N, 10] covariance array
        velocity_valid: [N] boolean array from velocity computation

    Returns:
        quality_scores: [N] array in [0, 1]

    """
    # Compute data-adaptive thresholds
    dop_vals = df["dop"].values
    speed_vals = df["speed_imputed"].values
    dt_vals = df["dt"].values
    sat_vals = df["satellite_count"].values

    dop_p95 = np.nanpercentile(dop_vals, 95)
    speed_p95 = np.nanpercentile(speed_vals, 95)
    dt_median = np.nanmedian(dt_vals)
    if np.isnan(dt_median) or dt_median <= 0:
        dt_median = 20.0
    sat_max = sat_vals.max()

    # 1. DOP: inverse with data-driven ceiling
    q_dop = 1.0 / (1.0 + dop_vals / max(dop_p95, 1.0))

    # 2. Satellite: adaptive range [4, max]
    q_sat = np.clip((sat_vals - 4) / max(sat_max - 4, 8), 0.0, 1.0)

    # 3. Covariance: log-scaled trace
    q_cov = 1.0 / (1.0 + np.log10(covariances[:, 9] + 1.0))

    # 4. Temporal: adaptive half-life based on median dt
    q_temp = np.exp(-dt_vals / max(dt_median * 3, 60.0))

    # 5. Kinematic: speed normalized by p95
    q_kin = np.clip(speed_vals / max(speed_p95, 1.0), 0.0, 1.0)

    # 6. Velocity validity: soft penalty for invalid velocities
    q_vel = np.where(velocity_valid, 1.0, 0.1)

    # 7. Outlier: binary flag penalty
    outlier_mask = (
        df["import_outlier"].values.astype(bool) |
        df["manual_outlier"].values.astype(bool)
    )
    q_out = np.where(outlier_mask, 0.1, 1.0)

    # Stack all components [N, 7] and compute inverse-variance weights
    components = np.column_stack([
        np.clip(q_dop, 0.0, 1.0),
        np.clip(q_sat, 0.0, 1.0),
        np.clip(q_cov, 0.0, 1.0),
        np.clip(q_temp, 0.0, 1.0),
        np.clip(q_kin, 0.0, 1.0),
        q_vel,
        q_out,
    ])

    # Inverse-variance weighting: w_i = 1/var(c_i), normalized to sum=1
    variances = np.var(components, axis=0)
    inv_var = 1.0 / (variances + 1e-8)
    weights = inv_var / inv_var.sum()

    # Weighted combination
    quality = (components * weights[np.newaxis, :]).sum(axis=1)

    return quality.astype(np.float32)
