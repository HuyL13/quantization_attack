import torch

from aq.rounding import (
    h_hard,
    h_soft,
    hard_quantized_int,
    init_alpha_from_pre_round,
    rtn_baseline_int,
    soft_quantized_int,
)


def test_h_soft_bounds():
    alpha = torch.tensor([-100.0, 0.0, 100.0])
    h = h_soft(alpha)
    assert (h >= 0).all() and (h <= 1).all()
    assert h[0] < 0.05  # near gamma clamp -> 0
    assert h[2] > 0.95  # near zeta clamp -> 1


def test_init_alpha_reproduces_fraction_at_start():
    pre_round = torch.tensor([1.2, 3.7, 5.5, 9.999])
    alpha = init_alpha_from_pre_round(pre_round)
    v = pre_round - torch.floor(pre_round)
    h = h_soft(alpha)
    assert torch.allclose(h, v, atol=1e-2)


def test_soft_and_hard_start_at_rtn_baseline():
    # 5.5 deliberately excluded: torch.round uses round-half-to-even (5.5 -> 6)
    # while h_hard's >0.5 threshold rounds a fraction of exactly 0.5 down -
    # a measure-zero tie-break disagreement for real (non-contrived) weights.
    pre_round = torch.tensor([1.2, 3.7, 9.999, 0.4999])
    max_int = 15
    alpha0 = init_alpha_from_pre_round(pre_round)

    baseline = rtn_baseline_int(pre_round, max_int)
    hard0 = hard_quantized_int(pre_round, alpha0, max_int)
    assert torch.equal(baseline, hard0)

    # soft_quantized_int is the CONTINUOUS relaxation: at init it reproduces
    # pre_round itself (floor + frac), not the rounded baseline - only
    # hard_quantized_int (tested above) matches RTN4's discrete decision.
    soft0 = soft_quantized_int(pre_round, alpha0, max_int)
    assert torch.allclose(soft0, pre_round, atol=1e-2)


def test_hard_quantized_int_moves_away_after_alpha_shift():
    pre_round = torch.tensor([1.4])
    max_int = 15
    alpha0 = init_alpha_from_pre_round(pre_round)
    hard0 = hard_quantized_int(pre_round, alpha0, max_int)
    assert hard0.item() == 1  # 0.4 frac rounds down

    alpha_pushed = alpha0 + 10.0  # push h(alpha) toward 1 -> ceil
    hard1 = hard_quantized_int(pre_round, alpha_pushed, max_int)
    assert hard1.item() == 2


def test_rtn_baseline_int_matches_torch_round():
    pre_round = torch.tensor([1.2, 1.5, 1.8, -0.3])
    max_int = 3
    expected = torch.round(pre_round).clamp(0, max_int)
    assert torch.equal(rtn_baseline_int(pre_round, max_int), expected)
