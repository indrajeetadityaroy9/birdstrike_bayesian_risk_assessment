"""YAML configuration loading for DKRL experiments."""

from pathlib import Path
from typing import Any

import yaml

from dkrl.config.defaults import HessianMode, NIGPModelConfig


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return as dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def config_to_model_config(cfg: dict[str, Any]) -> NIGPModelConfig:
    """Extract model section from config dict and return NIGPModelConfig."""
    model_cfg = dict(cfg["model"])

    if "hessian_mode" in model_cfg:
        model_cfg["hessian_mode"] = HessianMode(model_cfg["hessian_mode"])

    return NIGPModelConfig(**model_cfg)
