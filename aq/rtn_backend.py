"""Thin adapter onto if_awq_tier0's own RTN grid
(AWQQuantizerXL.quantize_weight_groupwise_raw) - the mandatory RTN4 baseline
grid every adversarial method starts from (plan section 4). No reimplementation:
this module only imports and calls the existing code after putting
if_awq_tier0 on sys.path.
"""
from __future__ import annotations

import torch

from aq.common import ensure_if_awq_tier0_on_path

_quantizer_cache: dict[tuple[int, int], object] = {}


def _get_quantizer(bits: int, group_size: int):
    key = (bits, group_size)
    if key not in _quantizer_cache:
        ensure_if_awq_tier0_on_path()
        from src.quantization.awq import AWQConfig, AWQQuantizerXL

        cfg = AWQConfig(bits=bits, group_size=group_size)
        # model/tokenizer are unused by quantize_weight_groupwise_raw itself;
        # AWQQuantizerXL only needs them for its (unused-here) scale search.
        _quantizer_cache[key] = AWQQuantizerXL(model=None, tokenizer=None, device="cpu", config=cfg)
    return _quantizer_cache[key]


def rtn_quantize_weight_raw(w: torch.Tensor, bits: int = 4, group_size: int = 128):
    """Returns if_awq_tier0's IntegerQuantizedTensorState for weight `w`
    (RTN4 when bits=4, group_size=128 - the plan's mandatory baseline)."""
    quantizer = _get_quantizer(bits, group_size)
    return quantizer.quantize_weight_groupwise_raw(w)
