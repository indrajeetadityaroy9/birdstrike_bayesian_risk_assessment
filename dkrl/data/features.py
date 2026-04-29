"""19-dimensional feature computation from GPS telemetry.

Feature layout:
    [0-2]   px, py, pz          - Position (ENU, meters)
    [3-5]   vx, vy, vz          - Velocity (m/s)
    [6]     speed_3d            - 3D speed magnitude
    [7]     heading_rad         - Heading (radians)
    [8]     vertical_velocity   - Vertical component of velocity
    [9-10]  tod_sin, tod_cos    - Time of day encoding
    [11-15] mass, wing_load, aspect, migration, trophic  - Species traits
    [16-18] alt_agl, gps_error, quality  - Spatial context
"""


import numpy as np
import pandas as pd


def compute_features_19d(
    df: pd.DataFrame,
    traits_dict: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 19-dimensional feature vectors.

    Args:
        df: Telemetry DataFrame
        traits_dict: Dict mapping species name to trait arrays [5]

    Returns:
        features: [N, 19] array
        velocity_valid: [N] boolean array (True = valid computed velocity)

    """
    # Convert lat/lon/alt to ENU (East-North-Up) coordinates (inlined)
    lat = df["location-lat"].values.astype(np.float64)
    lon = df["location-long"].values.astype(np.float64)
    alt = df["altitude_imputed"].values.astype(np.float64)
    ref_lat, ref_lon = lat.mean(), lon.mean()
    earth_radius = 6371000.0  # meters
    cos_ref = np.cos(np.radians(ref_lat))
    east = np.radians(lon - ref_lon) * earth_radius * cos_ref
    north = np.radians(lat - ref_lat) * earth_radius
    up = alt

    # Velocity computation via finite differences within segments
    dt = df["dt"].values
    segment_id = df["segment_id"].values
    is_segment_end = np.concatenate([segment_id[:-1] != segment_id[1:], [True]])

    n = len(east)
    vx, vy, vz = np.zeros(n), np.zeros(n), np.zeros(n)
    velocity_valid = ~is_segment_end & (dt > 0) & ~np.isnan(dt)
    idx = np.where(velocity_valid)[0]
    vx[idx] = (east[idx + 1] - east[idx]) / dt[idx + 1]
    vy[idx] = (north[idx + 1] - north[idx]) / dt[idx + 1]
    vz[idx] = (up[idx + 1] - up[idx]) / dt[idx + 1]

    # Derived kinematics
    speed_3d = np.sqrt(vx**2 + vy**2 + vz**2)
    # Convert heading to radians and normalize to [-pi, pi]
    heading_deg = df["heading_deg"].values
    heading_rad = np.arctan2(np.sin(np.radians(heading_deg)), np.cos(np.radians(heading_deg)))
    vertical_velocity = vz

    # Time of day encoding (hour -> sin/cos for continuity)
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    tod_sin = np.sin(2 * np.pi * hour / 24.0)
    tod_cos = np.cos(2 * np.pi * hour / 24.0)

    # Species traits - vectorized lookup via unique species
    species_arr = df["species"].values
    unique_sp, inverse_idx = np.unique(species_arr, return_inverse=True)

    # Get traits for each unique species (all species guaranteed in traits_dict from pre-filter)
    unique_traits = np.array([traits_dict[sp] for sp in unique_sp])

    # Broadcast to all samples via inverse index
    all_traits = unique_traits[inverse_idx]

    mass = all_traits[:, 0] / 1000.0  # Convert to kg
    wing_loading = all_traits[:, 1]
    aspect_ratio = all_traits[:, 2]
    migration = all_traits[:, 3]
    trophic = all_traits[:, 4]

    # Spatial context features
    # Note: quality here is the PRE-COMPUTED quality_score from telemetry data
    # (model input feature). This is DIFFERENT from quality_weights computed by
    # compute_quality_scores() (used for curriculum learning, not model input).
    alt_agl = df["altitude_imputed"].values
    gps_error = df["location-error-numerical"].values
    quality = df["quality_score"].values  # Telemetry quality (feature), not training weight

    # Stack into feature matrix
    features = np.column_stack([
        east, north, up,              # 0-2: position
        vx, vy, vz,                   # 3-5: velocity
        speed_3d,                     # 6: speed
        heading_rad,                  # 7: heading
        vertical_velocity,            # 8: vertical velocity
        tod_sin, tod_cos,             # 9-10: time encoding
        mass, wing_loading, aspect_ratio, migration, trophic,  # 11-15: traits
        alt_agl, gps_error, quality,  # 16-18: context
    ])

    return features.astype(np.float32), velocity_valid
