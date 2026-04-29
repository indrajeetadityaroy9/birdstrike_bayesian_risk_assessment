"""
Triton GPU Kernels for NIGP-DKL

Fused GPU kernel for first-order NIGP trace computation directly from
compressed 10-element covariance format. Avoids materializing full 19x19
covariance matrices.

Second-order NIGP corrections (tr(H@P)) are computed via Hutchinson trace
estimation with Hessian-vector products in nigp.py, not via explicit Hessian
kernels.

GPU Compatibility:
    - Tested on: NVIDIA H100 80GB SXM5
    - Compatible with: Any Triton-supported GPU (Ampere+, Hopper, etc.)
    - No H100-specific instructions (TMA, wgmma) are used

Kernels (per arXiv:2509.14710):
    - nigp_trace: Direct Tr(J @ P @ J.T) from compressed (fused, no P_19 materialization)
"""

import torch
import triton
import triton.language as tl

from dkrl.config._constants import VARIANCE_CEILING, VARIANCE_FLOOR

# NIGP Trace Kernel (Direct from compressed covariance)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_B": 128}, num_warps=4),
        triton.Config({"BLOCK_B": 256}, num_warps=4),
        triton.Config({"BLOCK_B": 512}, num_warps=8),
    ],
    key=["batch_size", "latent_dim"],
)
@triton.jit
def _cov_to_trace_kernel(
    cov_10_ptr,
    J_ptr,
    out_ptr,
    batch_size,
    latent_dim,
    input_dim,
    J_stride_b: tl.constexpr,
    VMIN: tl.constexpr,
    VMAX: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Direct Tr(J @ P @ J.T) from compressed covariance (no P_19 materialization)."""
    pid = tl.program_id(0)
    batch_start = pid * BLOCK_B
    bid = batch_start + tl.arange(0, BLOCK_B)
    mask = bid < batch_size

    in_base = bid * 10

    var_x = tl.minimum(tl.maximum(tl.load(cov_10_ptr + in_base + 0, mask=mask, other=0.0), VMIN), VMAX)
    cov_x_vx = tl.load(cov_10_ptr + in_base + 1, mask=mask, other=0.0)
    var_y = tl.minimum(tl.maximum(tl.load(cov_10_ptr + in_base + 2, mask=mask, other=0.0), VMIN), VMAX)
    cov_y_vy = tl.load(cov_10_ptr + in_base + 3, mask=mask, other=0.0)
    var_z = tl.minimum(tl.maximum(tl.load(cov_10_ptr + in_base + 4, mask=mask, other=0.0), VMIN), VMAX)
    cov_z_vz = tl.load(cov_10_ptr + in_base + 5, mask=mask, other=0.0)
    var_vx = tl.minimum(tl.maximum(tl.load(cov_10_ptr + in_base + 6, mask=mask, other=0.0), VMIN), VMAX)
    var_vy = tl.minimum(tl.maximum(tl.load(cov_10_ptr + in_base + 7, mask=mask, other=0.0), VMIN), VMAX)
    var_vz = tl.minimum(tl.maximum(tl.load(cov_10_ptr + in_base + 8, mask=mask, other=0.0), VMIN), VMAX)

    trace_acc = tl.zeros([BLOCK_B], dtype=tl.float32)

    for i in range(latent_dim):
        J_base = bid * J_stride_b + i * input_dim

        J_i0 = tl.load(J_ptr + J_base + 0, mask=mask, other=0.0)
        J_i1 = tl.load(J_ptr + J_base + 1, mask=mask, other=0.0)
        J_i2 = tl.load(J_ptr + J_base + 2, mask=mask, other=0.0)
        J_i3 = tl.load(J_ptr + J_base + 3, mask=mask, other=0.0)
        J_i4 = tl.load(J_ptr + J_base + 4, mask=mask, other=0.0)
        J_i5 = tl.load(J_ptr + J_base + 5, mask=mask, other=0.0)

        # (J @ P)[i, :] for first 6 elements
        JP_0 = J_i0 * var_x + J_i3 * cov_x_vx
        JP_1 = J_i1 * var_y + J_i4 * cov_y_vy
        JP_2 = J_i2 * var_z + J_i5 * cov_z_vz
        JP_3 = J_i0 * cov_x_vx + J_i3 * var_vx
        JP_4 = J_i1 * cov_y_vy + J_i4 * var_vy
        JP_5 = J_i2 * cov_z_vz + J_i5 * var_vz

        trace_acc += JP_0 * J_i0 + JP_1 * J_i1 + JP_2 * J_i2
        trace_acc += JP_3 * J_i3 + JP_4 * J_i4 + JP_5 * J_i5

    tl.store(out_ptr + bid, trace_acc, mask=mask)


def nigp_trace(J: torch.Tensor, cov_10: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
    """Compute Tr(J @ P @ J.T) from compressed covariance (1st-order NIGP)."""
    batch_size = J.shape[0]
    latent_dim = J.shape[1]
    input_dim = J.shape[2]
    orig_dtype = J.dtype

    J_f32 = J.float()
    cov_10_f32 = cov_10.float()
    out = torch.empty(batch_size, device=J.device, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(batch_size, meta["BLOCK_B"]),)

    _cov_to_trace_kernel[grid](
        cov_10_f32, J_f32, out, batch_size, latent_dim, input_dim,
        J_f32.stride(0), VARIANCE_FLOOR, VARIANCE_CEILING
    )

    if reduce == "mean":
        return out.mean().to(orig_dtype)
    return out.to(orig_dtype)


__all__ = [
    "nigp_trace",
]
