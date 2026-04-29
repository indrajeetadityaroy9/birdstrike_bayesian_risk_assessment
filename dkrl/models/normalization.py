"""Shared normalization mixin for all model classes."""

import torch


class NormalizationMixin:
    """Input/target normalization with device-safe buffer copies.

    Provides register_normalization_buffers(), set_normalization(),
    set_target_normalization(), and normalize() for any nn.Module subclass.
    """

    def register_normalization_buffers(self, input_dim: int):
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        self.register_buffer("target_mean", torch.zeros(1))
        self.register_buffer("target_std", torch.ones(1))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        self.input_mean.copy_(mean.to(self.input_mean.device))
        self.input_std.copy_((std + 1e-8).to(self.input_std.device))

    def set_target_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        self.target_mean.copy_(mean.to(self.target_mean.device))
        self.target_std.copy_((std + 1e-8).to(self.target_std.device))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.input_mean) / self.input_std
