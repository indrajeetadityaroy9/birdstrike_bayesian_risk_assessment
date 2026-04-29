"""Bayesian risk label calibration using UK CAA damage rates as prior.

Risk formula:
    risk = (1 - prior_weight) * kinematic_risk + prior_weight * damage_prior

The prior_weight is derived from data as the fraction of species with
birdstrike data: prior_weight = n_species_with_strikes / n_total_species.
"""


import numpy as np
import pandas as pd


def calibrate_risk_labels(
    df: pd.DataFrame,
    damage_rates: dict[str, float],
) -> np.ndarray:
    """Compute calibrated risk labels using Bayesian prior from UK CAA damage rates.

    Args:
        df: Telemetry DataFrame with altitude_imputed, speed_imputed, species
        damage_rates: Dict mapping scientific_name to damage_rate

    Returns:
        labels: [N] array in [0, 1]

    """
    # Data-derived prior weight: fraction of species with strike data
    unique_species = df["species"].unique()
    n_with_strikes = sum(1 for sp in unique_species if sp in damage_rates)
    prior_weight = n_with_strikes / max(len(unique_species), 1)
    # Vectorized species-to-damage-rate mapping
    default_rate = np.median(list(damage_rates.values())) if damage_rates else 0.05
    damage_prior = df["species"].map(damage_rates).fillna(default_rate).values

    # Normalize damage prior to [0, 1] using quantile-based scaling
    p95_rate = max(np.percentile(damage_prior, 95), 0.01)
    damage_prior = np.clip(damage_prior / p95_rate, 0.0, 1.0)

    # Kinematic risk from altitude - DATA ADAPTIVE
    altitude = df["altitude_imputed"].values
    positive_alt = altitude[altitude > 0]
    if len(positive_alt) > 0:
        alt_median = np.median(positive_alt)
        alt_std = np.std(positive_alt)
        if np.isnan(alt_median):
            alt_median = 150.0
        if np.isnan(alt_std):
            alt_std = 100.0
    else:
        alt_median = 150.0  # Default: typical bird flight altitude
        alt_std = 100.0
    alt_std = max(alt_std, 50.0)  # Minimum sigma of 50m for numerical stability

    # High-risk zone: centered on median with adaptive width
    altitude_risk = np.exp(-((altitude - alt_median) ** 2) / (2 * alt_std ** 2))

    # Kinematic risk from speed - DATA ADAPTIVE
    speed = df["speed_imputed"].values
    speed_p95 = max(np.percentile(speed[speed > 0], 95), 5.0) if (speed > 0).any() else 30.0
    speed_risk = np.clip(speed / speed_p95, 0.0, 1.0)

    # Inverse-variance kinematic risk weighting (matches quality scoring pattern)
    inv_var_alt = 1.0 / (np.var(altitude_risk) + 1e-8)
    inv_var_speed = 1.0 / (np.var(speed_risk) + 1e-8)
    total_w = inv_var_alt + inv_var_speed
    kinematic_risk = (
        (inv_var_alt / total_w) * altitude_risk
        + (inv_var_speed / total_w) * speed_risk
    )

    # Bayesian combination
    labels = (1 - prior_weight) * kinematic_risk + prior_weight * damage_prior

    # Ensure [0, 1] range
    labels = np.clip(labels, 0.0, 1.0)

    return labels.astype(np.float32)
