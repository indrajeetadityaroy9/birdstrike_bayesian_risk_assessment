"""Telemetry data filtering for quality control.

Removes samples that cannot produce valid features:
- NULL lat/lon coordinates
- Large time gaps (dt > threshold)
- Single-point segments
"""


import pandas as pd

from dkrl.config._constants import DT_MAX, MIN_SEGMENT_LENGTH


def filter_telemetry(
    df: pd.DataFrame,
    dt_max: float = DT_MAX,
    min_segment_length: int = MIN_SEGMENT_LENGTH,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filter telemetry data to remove samples that cannot produce valid features.

    Data-verified filters (EDA analysis):
    1. NULL lat/lon: 113,698 samples from Larus_fuscus
    2. Large dt: samples with dt > dt_max
    3. Single-point segments: Removes segments with < min_segment_length points

    Args:
        df: Telemetry DataFrame
        dt_max: Maximum time gap in seconds (default 3600.0)
        min_segment_length: Minimum points per segment (default 2)

    """
    n_input = len(df)
    remove_mask = pd.Series(False, index=df.index)

    # 1. NULL lat/lon (113,698 samples from Larus_fuscus)
    null_latlon_mask = df["location-lat"].isna() | df["location-long"].isna()
    n_null_latlon = int(null_latlon_mask.sum())
    if n_null_latlon > 0:
        print(f"filter reason=null_latlon n={n_null_latlon:,}")
    remove_mask |= null_latlon_mask

    # 2. Large time gaps
    large_dt_mask = df["dt"] > dt_max
    n_large_dt = int(large_dt_mask.sum())
    if n_large_dt > 0:
        print(f"filter reason=large_dt n={n_large_dt:,} threshold={dt_max}")
    remove_mask |= large_dt_mask

    # 3. Single-point segments
    segment_sizes = df.loc[~remove_mask, "segment_id"].value_counts()
    small_segments = segment_sizes[segment_sizes < min_segment_length].index
    single_point_mask = df["segment_id"].isin(small_segments)
    n_single_point = int((single_point_mask & ~remove_mask).sum())
    if n_single_point > 0:
        print(f"filter reason=short_segment n={n_single_point:,} min_len={min_segment_length}")
    remove_mask |= single_point_mask

    # Apply filter and select required columns
    required_cols = [
        "timestamp", "location-lat", "location-long", "altitude_imputed",
        "dt", "segment_id", "species", "speed_imputed", "heading_deg",
        "dop", "satellite_count", "location-error-numerical", "quality_score",
        "bird_mass_kg", "import_outlier", "manual_outlier", "individual-local-identifier"
    ]
    df = df.loc[~remove_mask, required_cols].reset_index(drop=True)

    n_removed = n_input - len(df)
    filter_stats = {
        "n_raw": n_input,
        "n_filtered": len(df),
        "n_null_latlon": n_null_latlon,
        "n_large_dt": n_large_dt,
        "n_single_point_segments": n_single_point,
        "removal_rate": n_removed / n_input if n_input > 0 else 0.0,
    }

    return df, filter_stats
