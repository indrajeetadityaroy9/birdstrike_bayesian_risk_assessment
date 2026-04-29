"""Data Pipeline Module for NIGP-DKL.

Consolidated data loading, feature engineering, covariance computation,
validation, and risk label calibration.
"""

from dkrl.data.augmentation import augment_features_19d
from dkrl.data.covariance_pipeline import compute_covariance_10
from dkrl.data.features import compute_features_19d
from dkrl.data.filtering import filter_telemetry
from dkrl.data.labels import calibrate_risk_labels
from dkrl.data.pipeline import (
    DatasetAudit,
    DatasetBundle,
    create_dataset,
)
from dkrl.data.quality import compute_quality_scores
from dkrl.data.splitting import stratified_split

__all__ = [
    "DatasetBundle",
    "DatasetAudit",
    "create_dataset",
    "augment_features_19d",
    "filter_telemetry",
    "compute_features_19d",
    "compute_covariance_10",
    "compute_quality_scores",
    "calibrate_risk_labels",
    "stratified_split",
]
