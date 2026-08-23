import torch

from aq.hsq_quantizer import dequantize, find_params, get_grid_candidates, quantize, quantize_int


def test_find_params_asymmetric_reconstructs_min_max_exactly():
    w = torch.tensor([[-2.0, 0.0, 1.0, 3.0]])
    scale, zero, maxq = find_params(w, bits=4, sym=False)
    q = quantize(w, scale, zero, maxq)
    assert torch.isclose(q[0, 0], torch.tensor(-2.0), atol=1e-4)
    assert torch.isclose(q[0, 3], torch.tensor(3.0), atol=1e-4)


def test_find_params_symmetric_centers_zero_at_midpoint():
    w = torch.tensor([[-4.0, 4.0]])
    scale, zero, maxq = find_params(w, bits=4, sym=True)
    assert torch.isclose(zero[0, 0], torch.tensor((maxq + 1) / 2), atol=1e-6)


def test_quantize_int_clamped_to_valid_range():
    w = torch.tensor([100.0, -100.0])
    scale = torch.tensor([1.0, 1.0])
    zero = torch.tensor([0.0, 0.0])
    q_int = quantize_int(w, scale, zero, maxq=15)
    assert q_int[0] == 15
    assert q_int[1] == 0


def test_dequantize_inverts_quantize_int_on_grid_points():
    scale = torch.tensor([0.5, 0.5])
    zero = torch.tensor([8.0, 8.0])
    q_int = torch.tensor([3.0, 12.0])
    w = dequantize(q_int, scale, zero)
    assert torch.allclose(w, (q_int - zero) * scale)


def test_grid_candidates_offset_zero_matches_nearest_rounding():
    w_col = torch.tensor([1.3, -2.7])
    scale_col = torch.tensor([1.0, 1.0])
    zero_col = torch.tensor([8.0, 8.0])
    dequant, codes = get_grid_candidates(w_col, scale_col, zero_col, maxq=15, radius=2)
    nearest_row = 2  # offsets [-2,-1,0,1,2], index 2 == offset 0
    expected_code = torch.round(w_col / scale_col + zero_col)
    assert torch.allclose(codes[nearest_row], expected_code)


def test_grid_candidates_shape_and_clamping():
    w_col = torch.tensor([0.0, 0.0])
    scale_col = torch.tensor([1.0, 1.0])
    zero_col = torch.tensor([0.0, 0.0])  # near the low edge, radius should clamp not go negative
    dequant, codes = get_grid_candidates(w_col, scale_col, zero_col, maxq=15, radius=2)
    assert codes.shape == (5, 2)
    assert (codes >= 0).all()
    assert (codes <= 15).all()
