"""
Unified Covariance Operations for DKRL

Consolidates covariance logic: 19x19 reconstruction (Triton), 6x6 kinematic block, normalization.

Covariance Format (10-element compressed):
    [0] var_x, [1] cov_x_vx, [2] var_y, [3] cov_y_vy,
    [4] var_z, [5] cov_z_vz, [6] var_vx, [7] var_vy,
    [8] var_vz, [9] trace
"""

import torch
import triton
import triton.language as tl

from dkrl.config._constants import (
    VARIANCE_CEILING,
    VARIANCE_FLOOR,
)

# Triton Kernel: [B, 10] -> [B, 19, 19]

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_B": 128}, num_warps=4),
        triton.Config({"BLOCK_B": 256}, num_warps=4),
        triton.Config({"BLOCK_B": 512}, num_warps=8),
        triton.Config({"BLOCK_B": 1024}, num_warps=8),
    ],
    key=["batch_size"],
)
@triton.jit
def _cov_reconstruct_kernel(
    cov_10_ptr,
    P_19_ptr,
    batch_size,
    VMIN: tl.constexpr,
    VMAX: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Reconstruct 19x19 covariance from compressed 10-element format."""
    pid = tl.program_id(0)
    batch_start = pid * BLOCK_B
    bid = batch_start + tl.arange(0, BLOCK_B)
    mask = bid < batch_size

    in_base = bid * 10

    var_x = tl.load(cov_10_ptr + in_base + 0, mask=mask, other=0.0)
    cov_x_vx = tl.load(cov_10_ptr + in_base + 1, mask=mask, other=0.0)
    var_y = tl.load(cov_10_ptr + in_base + 2, mask=mask, other=0.0)
    cov_y_vy = tl.load(cov_10_ptr + in_base + 3, mask=mask, other=0.0)
    var_z = tl.load(cov_10_ptr + in_base + 4, mask=mask, other=0.0)
    cov_z_vz = tl.load(cov_10_ptr + in_base + 5, mask=mask, other=0.0)
    var_vx = tl.load(cov_10_ptr + in_base + 6, mask=mask, other=0.0)
    var_vy = tl.load(cov_10_ptr + in_base + 7, mask=mask, other=0.0)
    var_vz = tl.load(cov_10_ptr + in_base + 8, mask=mask, other=0.0)

    var_x = tl.minimum(tl.maximum(var_x, VMIN), VMAX)
    var_y = tl.minimum(tl.maximum(var_y, VMIN), VMAX)
    var_z = tl.minimum(tl.maximum(var_z, VMIN), VMAX)
    var_vx = tl.minimum(tl.maximum(var_vx, VMIN), VMAX)
    var_vy = tl.minimum(tl.maximum(var_vy, VMIN), VMAX)
    var_vz = tl.minimum(tl.maximum(var_vz, VMIN), VMAX)

    out_base = bid * 361

    # Diagonal elements
    tl.store(P_19_ptr + out_base + 0, var_x, mask=mask)     # [0,0]
    tl.store(P_19_ptr + out_base + 20, var_y, mask=mask)    # [1,1]
    tl.store(P_19_ptr + out_base + 40, var_z, mask=mask)    # [2,2]
    tl.store(P_19_ptr + out_base + 60, var_vx, mask=mask)   # [3,3]
    tl.store(P_19_ptr + out_base + 80, var_vy, mask=mask)   # [4,4]
    tl.store(P_19_ptr + out_base + 100, var_vz, mask=mask)  # [5,5]

    # Off-diagonal (symmetric)
    tl.store(P_19_ptr + out_base + 3, cov_x_vx, mask=mask)   # [0,3]
    tl.store(P_19_ptr + out_base + 57, cov_x_vx, mask=mask)  # [3,0]
    tl.store(P_19_ptr + out_base + 23, cov_y_vy, mask=mask)  # [1,4]
    tl.store(P_19_ptr + out_base + 77, cov_y_vy, mask=mask)  # [4,1]
    tl.store(P_19_ptr + out_base + 43, cov_z_vz, mask=mask)  # [2,5]
    tl.store(P_19_ptr + out_base + 97, cov_z_vz, mask=mask)  # [5,2]


def build_cov_19x19(cov_10: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct [B, 19, 19] covariance from compressed [B, 10] format.

    Uses Triton GPU kernel for efficient batch processing.
    Non-kinematic features (indices 6-18) have zero covariance.

    Args:
        cov_10: Compressed covariance [B, 10]

    Returns:
        P_19: Full covariance [B, 19, 19]
    """
    batch_size = cov_10.shape[0]
    P_19 = torch.zeros(batch_size, 19, 19, device=cov_10.device, dtype=cov_10.dtype)

    cov_10_f32 = cov_10.float()
    P_19_f32 = P_19.float()

    def grid(meta):
        return (triton.cdiv(batch_size, meta["BLOCK_B"]),)

    _cov_reconstruct_kernel[grid](
        cov_10_f32, P_19_f32, batch_size, VARIANCE_FLOOR, VARIANCE_CEILING
    )

    return P_19_f32.to(cov_10.dtype)


# 6x6 Kinematic Block Extraction

def build_cov_6x6(cov_10: torch.Tensor, *, jitter: float = 0.0) -> torch.Tensor:
    """
    Extract 6x6 kinematic covariance block from compressed format.

    The 6x6 block represents position/velocity covariance:
        [var_px,     0,       0,    cov_px_vx,   0,         0      ]
        [  0,     var_py,     0,       0,     cov_py_vy,    0      ]
        [  0,        0,    var_pz,     0,        0,     cov_pz_vz  ]
        [cov_px_vx,  0,       0,    var_vx,      0,         0      ]
        [  0,     cov_py_vy,  0,       0,     var_vy,       0      ]
        [  0,        0,    cov_pz_vz,  0,        0,      var_vz   ]

    Args:
        cov_10: Compressed covariance [B, 10]
        jitter: Optional jitter added to diagonal for numerical stability.
                Use jitter=1e-4 for Cholesky decomposition.

    Returns:
        P_6x6: Kinematic covariance block [B, 6, 6]
    """
    batch_size = cov_10.shape[0]
    device, dtype = cov_10.device, cov_10.dtype

    P_6x6 = torch.zeros(batch_size, 6, 6, device=device, dtype=dtype)

    # Diagonal: var_px(0), var_py(2), var_pz(4), var_vx(6), var_vy(7), var_vz(8)
    # Map to 6x6 positions: (0,0), (1,1), (2,2), (3,3), (4,4), (5,5)
    cov_diag_idx = [0, 2, 4, 6, 7, 8]
    for i, cov_idx in enumerate(cov_diag_idx):
        P_6x6[:, i, i] = cov_10[:, cov_idx]

    # Off-diagonal: cov_x_vx(1)->P[0,3], cov_y_vy(3)->P[1,4], cov_z_vz(5)->P[2,5]
    cov_off_rows = [0, 1, 2]
    cov_off_cols = [3, 4, 5]
    cov_off_idx = [1, 3, 5]
    for row, col, cov_idx in zip(cov_off_rows, cov_off_cols, cov_off_idx):
        P_6x6[:, row, col] = cov_10[:, cov_idx]
        P_6x6[:, col, row] = cov_10[:, cov_idx]  # Symmetric

    # Add jitter to diagonal for numerical stability
    if jitter > 0.0:
        jitter_diag = torch.eye(6, device=device, dtype=dtype) * jitter
        P_6x6 = P_6x6 + jitter_diag.unsqueeze(0)

    return P_6x6


# Covariance Normalization

def normalize_cov_10(cov_10: torch.Tensor, input_std: torch.Tensor) -> torch.Tensor:
    """
    Normalize compressed covariance to match normalized input space.

    The Jacobian J is computed from normalized inputs x_norm = (x - mu) / sigma.
    For consistency, the NIGP trace J @ P @ J.T must use the normalized
    covariance P_norm[i,j] = P[i,j] / (sigma_i * sigma_j).

    Compressed cov_10 format maps to 6x6 position/velocity block (features 0-5):
        [0] P[0,0] var_px,    [1] P[0,3] cov_px_vx,
        [2] P[1,1] var_py,    [3] P[1,4] cov_py_vy,
        [4] P[2,2] var_pz,    [5] P[2,5] cov_pz_vz,
        [6] P[3,3] var_vx,    [7] P[4,4] var_vy,
        [8] P[5,5] var_vz,    [9] trace_P (unused by kernel)

    Args:
        cov_10: Compressed covariance [B, 10]
        input_std: Standard deviation of each input feature [19]

    Returns:
        cov_10_norm: Normalized compressed covariance [B, 10]
    """
    s = input_std
    # Scale factors: product of input_std for the two feature indices
    scale = torch.stack([
        s[0] * s[0],  # P[0,0] var_px
        s[0] * s[3],  # P[0,3] cov_px_vx
        s[1] * s[1],  # P[1,1] var_py
        s[1] * s[4],  # P[1,4] cov_py_vy
        s[2] * s[2],  # P[2,2] var_pz
        s[2] * s[5],  # P[2,5] cov_pz_vz
        s[3] * s[3],  # P[3,3] var_vx
        s[4] * s[4],  # P[4,4] var_vy
        s[5] * s[5],  # P[5,5] var_vz
        s[0] * s[0],  # trace_P placeholder (unused by kernel)
    ])
    return cov_10 / scale.unsqueeze(0)



__all__ = [
    "build_cov_19x19",
    "build_cov_6x6",
    "normalize_cov_10",
]
