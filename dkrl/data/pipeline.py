"""Data Pipeline for NIGP-DKL.

Orchestrates data loading, feature engineering, covariance computation,
and risk label calibration. Produces training-ready arrays from processed
telemetry, traits, and birdstrike data.

SOTA Integration:
- Bayesian risk calibration with UK CAA damage rates as prior
- GPS error model for covariance estimation
- Quality-weighted training support
- Strict pre-filtering for data quality (NaN, time gaps, single-point segments)

Data Flow:
    Processed Parquets -> Pre-Filter -> Features [N, 19] + Covariances [N, 10] + Labels [N]

EDA-Verified Data Guarantees (v1.1):
- NULL GPS: All 113,698 NULL lat/lon rows are from Larus_fuscus species
- After GPS filter: ALL other columns guaranteed non-NULL (EDA verified)
- dt exceptions: Only 41 NULL values (segment first points)
- damage_rate: ALWAYS percentage (0-11.36%), divide by 100 unconditionally

References:
    - Curriculum learning: arXiv:2407.00102
    - Reproducibility: arXiv:2502.00902

"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from dkrl.config._constants import DT_MAX, MIN_SEGMENT_LENGTH, TROPHIC_LEVEL_MAP
from dkrl.config.hardware import get_data_dir
from dkrl.data.covariance_pipeline import compute_covariance_10
from dkrl.data.features import compute_features_19d
from dkrl.data.filtering import filter_telemetry
from dkrl.data.labels import calibrate_risk_labels
from dkrl.data.quality import compute_quality_scores
from dkrl.data.splitting import stratified_split
from dkrl.utils.seeding import set_all_seeds


@dataclass
class DatasetBundle:
    """Training-ready dataset arrays with pre-computed splits."""

    features: np.ndarray          # [N, 19]
    labels: np.ndarray            # [N]
    covariances: np.ndarray       # [N, 10]
    quality_weights: np.ndarray   # [N]
    train_idx: np.ndarray         # Training indices
    val_idx: np.ndarray           # Validation indices
    test_idx: np.ndarray          # Test indices
    species: np.ndarray           # [N] species labels (for stratified resplitting)
    seed: int = 42                # Seed used for splitting (for reproducibility)
    val_split: float = 0.15       # Validation fraction used
    test_split: float = 0.25      # Test fraction used

    def get_split(
        self, split: str, *, normalize_quality: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get data for a split (train/val/test).

        Args:
            split: One of "train", "val", or "test"
            normalize_quality: If True, normalize quality weights to have mean 1.0.
                Use True for training data (enables curriculum weighting).
                Use False for validation/evaluation (preserves original scores).
                Default is False for safety.

        Returns: (features, labels, covariances, quality_weights)

        """
        idx = {"train": self.train_idx, "val": self.val_idx, "test": self.test_idx}[split]
        q = self.quality_weights[idx]
        if normalize_quality and len(q) > 0:
            q = q / q.mean()
        return self.features[idx], self.labels[idx], self.covariances[idx], q

    def resplit(self, seed: int) -> "DatasetBundle":
        """Re-randomize train/val/test splits without reloading data.

        Returns a new DatasetBundle with the same underlying arrays but new split indices.
        """
        train_idx, val_idx, test_idx = stratified_split(
            self.species, self.val_split, self.test_split, seed,
        )
        return DatasetBundle(
            features=self.features,
            labels=self.labels,
            covariances=self.covariances,
            quality_weights=self.quality_weights,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            species=self.species,
            seed=seed,
            val_split=self.val_split,
            test_split=self.test_split,
        )


@dataclass
class DatasetAudit:
    """Reproducibility audit record with filter statistics."""

    version: str = "1.2.0"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    seed: int = 0

    # Filter statistics
    n_raw: int = 0
    n_null_latlon: int = 0
    n_large_dt: int = 0
    n_single_point_segments: int = 0
    n_filtered: int = 0
    n_segment_endpoints: int = 0
    n_segment_starts: int = 0
    n_cauchy_schwarz_violations: int = 0

    # Sample counts
    n_samples: int = 0
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    n_individuals: int = 0

    # Distribution statistics
    species_counts: dict[str, int] = field(default_factory=dict)
    feature_stats: dict[str, float] = field(default_factory=dict)
    label_stats: dict[str, float] = field(default_factory=dict)
    data_hashes: dict[str, str] = field(default_factory=dict)


def create_dataset(
    *,
    data_dir: Path = None,
    species: list[str] = None,
    seed: int = 42,
    val_split: float = None,
    test_split: float = None,
    dt_max: float = DT_MAX,
    min_segment_length: int = MIN_SEGMENT_LENGTH,
) -> tuple[DatasetBundle, DatasetAudit]:
    """Create training-ready dataset with inverse-variance quality weights and audited filters."""
    if data_dir is None:
        data_dir = get_data_dir()
    data_dir = Path(data_dir)

    telemetry_dir = data_dir / "telemetry_partitioned"
    traits_path = data_dir / "traits" / "european_species_traits.parquet"
    birdstrike_path = data_dir / "birdstrike" / "uk_caa_species_strikes_2023_2024.parquet"

    set_all_seeds(seed)

    # Load telemetry data
    print(f"load_telemetry dir={telemetry_dir}")
    files = sorted(telemetry_dir.glob("*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["species"] = df["species"].str.replace("_", " ", regex=False)
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if species:
        df = df[df["species"].isin(species)]

    n_raw = len(df)
    print(f"raw_data n={n_raw:,} species={df['species'].nunique()}")

    # Filter (remediation step)
    print("filtering_data mode=remediation")
    df, filter_stats = filter_telemetry(df, dt_max=dt_max, min_segment_length=min_segment_length)
    n_removed = filter_stats["n_raw"] - filter_stats["n_filtered"]
    retained = filter_stats["n_filtered"]
    rate = filter_stats["removal_rate"]
    print(f"filter_result removed={n_removed:,} rate={rate:.1%} retained={retained:,}")

    # EDA verified: After NULL GPS filtering, only dt can have NULLs (41 segment-start rows)
    # Track segment boundaries explicitly for proper covariance handling
    segment_start_mask = df["dt"].isna()
    n_segment_starts = segment_start_mask.sum()
    if n_segment_starts > 0:
        print(f"segment_boundary dt_nan={n_segment_starts}")
    # Fill with median dt (more representative than arbitrary 1.0)
    median_dt = df["dt"].dropna().median()
    df["dt"] = df["dt"].fillna(median_dt)
    df["is_segment_start"] = segment_start_mask  # Track for covariance inflation

    print("status=loading_traits")
    traits_df = pd.read_parquet(traits_path)

    # Build traits dictionary (inlined - data verified: 0 NULLs in traits)
    traits_dict = {}
    for _, row in traits_df.iterrows():
        trophic = TROPHIC_LEVEL_MAP.get(row["trophic_level"], 3.0)
        traits_dict[row["species"]] = np.array([
            row["mass_g"], row["wing_loading_proxy"], row["aspect_ratio_proxy"],
            row["migration"], trophic,
        ])

    # Load birdstrike rates
    print("status=loading_birdstrike_rates")
    birdstrike_df = pd.read_parquet(birdstrike_path)
    rates = birdstrike_df["damage_rate"].values / 100.0  # Convert percentage to [0,1]
    damage_rates = dict(zip(birdstrike_df["scientific_name"], rates, strict=False))

    # Compute features
    print("status=computing_features dim=19")
    features, velocity_valid = compute_features_19d(df, traits_dict)

    # Update filter statistics with segment tracking
    n_segment_endpoints = int((~velocity_valid).sum())
    filter_stats["n_segment_endpoints"] = n_segment_endpoints
    filter_stats["n_segment_starts"] = n_segment_starts
    print(f"feature_stats synthetic_velocity={n_segment_endpoints:,}")

    # Compute covariances (now returns violation count for audit)
    print("status=computing_covariances")
    covariances, n_cauchy_violations = compute_covariance_10(df)
    filter_stats["n_cauchy_schwarz_violations"] = n_cauchy_violations

    # Compute quality scores (incorporating velocity validity)
    print("status=computing_quality_scores")
    quality_scores = compute_quality_scores(df, covariances, velocity_valid)

    # Calibrate risk labels
    print("status=calibrating_risk_labels")
    labels = calibrate_risk_labels(df, damage_rates)

    # Get species array and individual identifiers (guaranteed present in REQUIRED_COLUMNS)
    species_arr = df["species"].values
    individual_id_arr = df["individual-local-identifier"].values

    # Create stratified train/val/test splits (ensures species representation)
    print("status=creating_stratified_splits")
    n = len(features)

    # Adaptive split sizes: target ~5000 samples per split for statistical stability
    if val_split is None:
        val_split = max(0.05, min(0.15, 5000 / n))
    if test_split is None:
        test_split = max(0.05, min(0.15, 5000 / n))

    train_idx, val_idx, test_idx = stratified_split(species_arr, val_split, test_split, seed)

    n_train = len(train_idx)
    n_val = len(val_idx)
    n_test = len(test_idx)
    print(f"split_stats train={n_train:,} val={n_val:,} test={n_test:,}")

    bundle = DatasetBundle(
        features=features,
        labels=labels,
        covariances=covariances,
        quality_weights=quality_scores,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        species=species_arr,
        seed=seed,
        val_split=val_split,
        test_split=test_split,
    )

    # Create audit with filter stats (fields populated directly)
    species_counts = dict(pd.Series(species_arr).value_counts())
    n_individuals = len(np.unique(individual_id_arr[individual_id_arr != "unknown"]))
    audit = DatasetAudit(
        seed=seed,
        # Filter statistics
        n_raw=filter_stats["n_raw"],
        n_null_latlon=filter_stats["n_null_latlon"],
        n_large_dt=filter_stats["n_large_dt"],
        n_single_point_segments=filter_stats["n_single_point_segments"],
        n_filtered=filter_stats["n_filtered"],
        n_segment_endpoints=filter_stats["n_segment_endpoints"],
        n_segment_starts=filter_stats["n_segment_starts"],
        n_cauchy_schwarz_violations=filter_stats["n_cauchy_schwarz_violations"],
        # Sample counts
        n_samples=n,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        n_individuals=n_individuals,
        # Distribution statistics
        species_counts=species_counts,
        feature_stats={
            "mean": float(features.mean()),
            "std": float(features.std()),
            "min": float(features.min()),
            "max": float(features.max()),
            "n_nan": int(np.isnan(features).sum()),
            "n_inf": int(np.isinf(features).sum()),
        },
        label_stats={
            "mean": float(labels.mean()),
            "std": float(labels.std()),
            "min": float(labels.min()),
            "max": float(labels.max()),
        },
        data_hashes={
            "features": hashlib.sha256(features.tobytes()).hexdigest()[:16],
            "labels": hashlib.sha256(labels.tobytes()).hexdigest()[:16],
            "covariances": hashlib.sha256(covariances.tobytes()).hexdigest()[:16],
        },
    )

    print("status=dataset_creation_complete")
    return bundle, audit


__all__ = [
    "DatasetBundle",
    "DatasetAudit",
    "create_dataset",
]
