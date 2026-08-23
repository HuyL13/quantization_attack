"""HSQ Hessian accumulation + Cholesky/inverse-Hessian machinery.

Per guide section 10.2/10.3/10.4/10.5: this reuses the upstream GPTQ
`GPTQ.add_batch()` accumulation formula and the damping/dead-column/
Cholesky-inverse pipeline almost verbatim - HSQ needs the exact same
Hessian and the exact same numerically-stable inverse-Hessian factor GPTQ
uses for its own second-order error compensation, not a reimplementation.
"""
from __future__ import annotations

import math

import torch


def accumulate_hessian(H: torch.Tensor, nsamples: int, x_batch: torch.Tensor) -> tuple[torch.Tensor, int]:
    """One GPTQ-style `add_batch` update. x_batch: (..., in_features) - any
    leading batch/sequence dims are flattened into the token axis. Returns
    the updated (H, nsamples); H is accumulated in float32 regardless of
    the model's compute dtype, matching GPTQ's own numerics.
    """
    in_features = x_batch.shape[-1]
    x = x_batch.reshape(-1, in_features).to(H.device, dtype=torch.float32)
    tmp = x.shape[0]
    if tmp == 0:
        return H, nsamples
    H = H * (nsamples / (nsamples + tmp))
    nsamples += tmp
    x = x.t() * math.sqrt(2.0 / nsamples)
    H = H + x.matmul(x.t())
    return H, nsamples


def compute_inverse_hessian_cholesky(H: torch.Tensor, percdamp: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
    """Dead-column handling + damping + Cholesky-inverse-Cholesky, exactly
    the sequence upstream GPTQ uses to turn a raw Hessian into the
    upper-triangular factor `fasterquant`'s block loop needs. Returns
    (Hinv_upper, dead_mask) - Hinv_upper is NOT H^-1 itself (callers use its
    diagonal and row slices the same way GPTQ's own loop does); dead_mask
    marks columns whose Hessian diagonal was exactly zero (no calibration
    signal reached that input coordinate), so the caller can zero the
    corresponding weight column instead of quantizing noise.
    """
    H = H.clone()
    columns = H.shape[0]
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0

    damp = percdamp * torch.mean(torch.diag(H))
    diag_idx = torch.arange(columns, device=H.device)
    H[diag_idx, diag_idx] += damp

    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    return H, dead
