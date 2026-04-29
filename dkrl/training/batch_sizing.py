"""
Adaptive batch sizing via Gradient Noise Scale (GNS).

Literature: arXiv:1812.06162
"""

import torch
import torch.nn as nn

from dkrl.config._constants import EPSILON
from dkrl.utils.ema import compute_ema_decay


def compute_batch_size(n_samples: int, input_dim: int = 19) -> int:
    """
    Compute max batch size from GPU memory via dimension-scaled estimate.

    Args:
        n_samples: Total dataset size
        input_dim: Feature dimensionality

    Returns:
        Maximum feasible batch size
    """
    free_bytes, _ = torch.cuda.mem_get_info()
    available_bytes = int(free_bytes * 0.85)  # 15% fragmentation margin

    bytes_per_sample = input_dim * 200 * 1024  # ~200KB * D/19

    max_batch = max(32, int(available_bytes / max(bytes_per_sample, 1)))
    return min(max_batch, n_samples)


class GNSBatchSizeController:
    """
    Gradient Noise Scale controller (arXiv:1812.06162).
    Grows batch when B < tr(Sigma_grad) / ||g||^2.

    Uses micro-batch gradient difference to estimate noise vs signal:
        noise ~ (B/2) * ||g_micro1 - g_micro2||^2
        signal ~ ||g_micro1 + g_micro2||^2 / 4

    All parameters are data-derived:
        - ema_decay: from compute_ema_decay(n_samples, batch_size)
        - warmup_steps: max(20, n_samples // (batch_size * 5))
        - batch growth: GNS-proportional with 1.5x cap per step
    """

    def __init__(
        self,
        initial_batch_size: int = 256,
        max_batch_size: int = 32768,
        ema_decay: float = 0.999,
        warmup_steps: int = 50,
    ):
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.ema_decay = ema_decay
        self._warmup_steps = warmup_steps

        self._ema_noise: float = 0.0
        self._ema_signal: float = 0.0
        self._initialized = False
        self._step = 0

    @classmethod
    def from_data(cls, n_samples: int, initial_batch_size: int, max_batch_size: int = 32768):
        """Construct with all parameters derived from dataset size."""
        ema_decay = compute_ema_decay(n_samples, initial_batch_size)
        warmup_steps = max(20, n_samples // (initial_batch_size * 5))
        return cls(
            initial_batch_size=initial_batch_size,
            max_batch_size=max_batch_size,
            ema_decay=ema_decay,
            warmup_steps=warmup_steps,
        )

    def update_from_micro_batches(
        self,
        grad_norms_micro1: torch.Tensor,
        grad_norms_micro2: torch.Tensor,
        batch_size: int,
    ) -> bool:
        """
        Update batch size from two micro-batch gradient vectors.

        Returns:
            True if batch size changed
        """
        diff = grad_norms_micro1 - grad_norms_micro2
        sumv = grad_norms_micro1 + grad_norms_micro2

        noise_val = (batch_size / 2.0) * diff.dot(diff).item()
        signal_val = sumv.dot(sumv).item() / 4.0

        d = self.ema_decay
        if not self._initialized:
            self._ema_noise = noise_val
            self._ema_signal = signal_val
            self._initialized = True
            return False

        self._ema_noise = d * self._ema_noise + (1 - d) * noise_val
        self._ema_signal = d * self._ema_signal + (1 - d) * signal_val

        self._step += 1
        if self._step < self._warmup_steps:
            return False

        if self._ema_signal < EPSILON:
            return False

        b_simple = self._ema_noise / (self._ema_signal + EPSILON)

        if b_simple > self.current_batch_size:
            old_batch = self.current_batch_size
            # GNS-proportional growth with 1.5x cap per step
            self.current_batch_size = min(
                int(self.current_batch_size * 1.5),
                int(b_simple * 1.2),
                self.max_batch_size,
            )
            return self.current_batch_size > old_batch

        return False

    def update(self, model: nn.Module) -> bool:
        """
        Fallback update using single batch gradient norm.
        Approximates noise from gradient norm variance across steps.
        """
        grad_sq_norm = sum(
            p.grad.norm().pow(2) for p in model.parameters() if p.grad is not None
        )
        grad_sq_norm_val = grad_sq_norm.item()

        d = self.ema_decay
        if not self._initialized:
            self._ema_noise = grad_sq_norm_val
            self._ema_signal = grad_sq_norm_val
            self._initialized = True
            return False

        # Track running variance of gradient norms as noise proxy
        old_mean = self._ema_signal
        self._ema_signal = d * self._ema_signal + (1 - d) * grad_sq_norm_val
        self._ema_noise = d * self._ema_noise + (1 - d) * (grad_sq_norm_val - old_mean) ** 2

        self._step += 1
        if self._step < self._warmup_steps:
            return False

        if self._ema_signal < EPSILON:
            return False

        b_simple = self._ema_noise / (self._ema_signal + EPSILON)

        if b_simple > self.current_batch_size:
            old_batch = self.current_batch_size
            # GNS-proportional growth with 1.5x cap per step
            self.current_batch_size = min(
                int(self.current_batch_size * 1.5),
                int(b_simple * 1.2),
                self.max_batch_size,
            )
            return self.current_batch_size > old_batch

        return False
