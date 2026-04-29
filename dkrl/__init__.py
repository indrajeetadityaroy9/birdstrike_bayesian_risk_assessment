"""
Deep Kernel Risk Learning (DKRL) - NIGP-DKL for uncertainty-aware prediction.

Runtime: PyTorch 2.4 (CUDA 12+), H100/Ampere+.

Components:
- NIGPDeepKernelGP: 2nd-order NIGP + Low-Rank GP + Dissipative Deep Kernel
- Conformal: LCMQR & Leverage-Calibrated Split CP
- Data: Pipeline for telemetry/traits/risk integration
"""

from dkrl.config.defaults import HessianMode, NIGPModelConfig
from dkrl.data import DatasetAudit, DatasetBundle, create_dataset
from dkrl.evaluation.evaluator import evaluate_nigp_model
from dkrl.evaluation.metrics import ScoringResults, compute_all_scores
from dkrl.inference.conformal import (
    ConformalMetrics,
    KernelConformalizedNIGP,
    LeverageSplitCP,
)
from dkrl.inference.sigma_point import SigmaPointNIGP
from dkrl.models.nigp import NIGPDeepKernelGP, UncertaintyDecomposition
from dkrl.training.trainer import train_nigp_model
from dkrl.utils.io import load_model, save_model

__all__ = [
    "HessianMode",
    "NIGPModelConfig",
    "NIGPDeepKernelGP",
    "UncertaintyDecomposition",
    "SigmaPointNIGP",
    "KernelConformalizedNIGP",
    "LeverageSplitCP",
    "ConformalMetrics",
    "train_nigp_model",
    "evaluate_nigp_model",
    "compute_all_scores",
    "ScoringResults",
    "create_dataset",
    "DatasetBundle",
    "DatasetAudit",
    "save_model",
    "load_model",
]

__version__ = "0.7.0"
