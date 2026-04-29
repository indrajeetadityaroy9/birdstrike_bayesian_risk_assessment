"""
ZClip adaptive gradient clipping (arXiv:2504.02507).

Clips outliers at mu + z * sigma using EMA stats.
"""

import math

import torch
import torch.nn as nn

from dkrl.config._constants import EPSILON
from dkrl.utils.ema import compute_ema_decay


class ZClipGradientClipper:
    """
    ZClip gradient clipper (arXiv:2504.02507).
    Clips outliers at mu + z * sigma using EMA stats.

    Warmup: returns inf (no clipping) until EMA has stabilized.
    """

    def __init__(self, z_threshold: float = 3.0, ema_decay: float = 0.99, warmup_steps: int = 50):
        self.z_threshold = z_threshold
        self.ema_decay = ema_decay
        self._warmup_steps = warmup_steps
        self._ema_mean: float = 0.0
        self._ema_var: float = 1.0
        self._initialized = False
        self._step = 0

    @classmethod
    def from_data(cls, n_samples: int, batch_size: int, z_threshold: float = 3.0):
        """Construct with EMA decay and warmup derived from dataset size."""
        ema_decay = compute_ema_decay(n_samples, batch_size)
        warmup_steps = max(20, n_samples // (batch_size * 5))
        return cls(z_threshold=z_threshold, ema_decay=ema_decay, warmup_steps=warmup_steps)

    def compute_clip_norm(self, model: nn.Module) -> float:
        """
        Compute adaptive clip norm via z-score anomaly detection.

        During warmup (before EMA stabilizes), returns inf (no clipping).
        After warmup, clips at mu + z * sigma.
        """
        grad_norm = torch.sqrt(
            sum(p.grad.norm().pow(2) for p in model.parameters() if p.grad is not None)
        ).item()

        d = self.ema_decay
        if not self._initialized:
            self._ema_mean = grad_norm
            self._ema_var = grad_norm ** 2
            self._initialized = True
            self._step += 1
            return float('inf')  # No clipping during warmup

        old_mean = self._ema_mean
        self._ema_mean = d * self._ema_mean + (1 - d) * grad_norm
        self._ema_var = d * self._ema_var + (1 - d) * (grad_norm - old_mean) ** 2

        self._step += 1
        if self._step < self._warmup_steps:
            return float('inf')  # No clipping during warmup

        # Clip at mu + z * sigma
        sigma = math.sqrt(max(self._ema_var, EPSILON))
        return self._ema_mean + self.z_threshold * sigma
