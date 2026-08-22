import torch

from aq.rtn_backend import rtn_quantize_weight_raw


def test_rtn_quantize_weight_raw_matches_manual_affine_math():
    torch.manual_seed(0)
    w = torch.randn(4, 16)  # smaller than group_size=128 -> single padded group
    state = rtn_quantize_weight_raw(w, bits=4, group_size=128)

    assert state.max_int == 15
    assert state.in_features == 16
    assert state.padded_in_features == 128

    w_min = w.min(dim=1, keepdim=True)[0]
    w_max = w.max(dim=1, keepdim=True)[0]
    expected_scale = ((w_max - w_min) / 15).clamp(min=1e-8)
    expected_zp = torch.round(-w_min / expected_scale).clamp(0, 15)

    actual_scale = state.scale[:, :16]
    actual_zp = state.zero_point[:, :16]
    assert torch.allclose(actual_scale, expected_scale.expand_as(actual_scale), atol=1e-6)
    assert torch.allclose(actual_zp, expected_zp.expand_as(actual_zp), atol=1e-6)


def test_rtn_quantize_dequantize_truncated_shape_and_dtype():
    torch.manual_seed(1)
    w = torch.randn(8, 300, dtype=torch.float32)  # spans multiple groups of 128
    state = rtn_quantize_weight_raw(w, bits=4, group_size=128)
    dq = state.dequantize_truncated()
    assert dq.shape == w.shape
    assert dq.dtype == w.dtype
    # RTN4 quantization error should be small relative to the weight's own scale
    rel_err = (dq - w).norm() / w.norm()
    assert rel_err < 0.2


def test_rtn_quantize_stays_within_one_grid_step_of_group_range():
    # Affine min-max quantization with a ROUNDED integer zero-point is not
    # exactly bounded by [min, max] (rounding zp shifts the whole grid by up
    # to half a step) - verified live: with bits=4 this can overshoot the
    # true max by more than a naive `+ 1e-3` tolerance allows. Bound by one
    # full quantization step instead of assuming exact [min, max] containment.
    torch.manual_seed(2)
    w = torch.randn(2, 128)
    state = rtn_quantize_weight_raw(w, bits=4, group_size=128)
    dq = state.dequantize_truncated()
    step = (w.max() - w.min()) / 15
    assert dq.min() >= w.min() - step
    assert dq.max() <= w.max() + step
