"""HSQ (Hessian-Slack Quantization) - GPTQ-style affine quantizer.

Per HSQ_Hessian_Slack_Quantization_Implementation_Guide.md section 11/14:
GPTQ's own `Quantizer`/`find_params`/grid is reused almost verbatim (min-max
per-group affine scale/zero-point, same as the upstream IST-DASLab/gptq
`quant.py`), with ONE addition - `get_grid_candidates` - since HSQ needs to
enumerate several nearby lattice points per coordinate instead of only the
nearest one.
"""
from __future__ import annotations

import torch


def find_params(w_group: torch.Tensor, bits: int, sym: bool = False, eps: float = 1e-8):
    """w_group: (out_features, group_size). Returns (scale, zero, maxq), scale
    and zero both shaped (out_features, 1) - the same per-output-row,
    per-group affine convention GPTQ/AWQ both use.
    """
    maxq = 2**bits - 1
    if sym:
        wmax = w_group.abs().amax(dim=1, keepdim=True).clamp(min=eps)
        scale = wmax / (maxq / 2)
        zero = torch.full_like(scale, (maxq + 1) / 2)
    else:
        wmin = w_group.amin(dim=1, keepdim=True).clamp(max=0.0)
        wmax = w_group.amax(dim=1, keepdim=True).clamp(min=0.0)
        scale = (wmax - wmin).clamp(min=eps) / maxq
        zero = torch.round(-wmin / scale)
    return scale, zero, maxq


def quantize_int(w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, maxq: int) -> torch.Tensor:
    """Nearest-rounding integer code, clamped to [0, maxq]. w/scale/zero
    broadcast against each other (w can be a single column (out_features,)
    with scale/zero shaped (out_features, 1) squeezed, or a full group).
    """
    return torch.clamp(torch.round(w / scale + zero), 0, maxq)


def dequantize(q_int: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    return scale * (q_int - zero)


def quantize(w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, maxq: int) -> torch.Tensor:
    return dequantize(quantize_int(w, scale, zero, maxq), scale, zero)


def get_grid_candidates(
    w_col: torch.Tensor, scale_col: torch.Tensor, zero_col: torch.Tensor, maxq: int, radius: int = 2
):
    """w_col: (out_features,) - the weights at ONE input coordinate across
    every output row. scale_col/zero_col: (out_features,) (already squeezed
    for this coordinate's group). Returns (dequant_candidates, int_codes),
    both shaped (2*radius+1, out_features) - vectorized over every output
    row at once, per guide section 15 ("do not loop python over scalar
    weights").
    """
    nearest = torch.round(w_col / scale_col + zero_col)
    offsets = torch.arange(-radius, radius + 1, device=w_col.device, dtype=w_col.dtype)
    codes = nearest.unsqueeze(0) + offsets.unsqueeze(1)
    codes = codes.clamp(0, maxq)
    dequant = scale_col.unsqueeze(0) * (codes - zero_col.unsqueeze(0))
    return dequant, codes
