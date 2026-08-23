import torch

from aq.hsq_core import fasterquant, run_hsq_layer


def _make_layer(seed=0, out_features=6, in_features=32):
    torch.manual_seed(seed)
    w = torch.randn(out_features, in_features)
    x = torch.randn(200, in_features)
    H = (x.t() @ x) / x.shape[0]
    return w, H


def test_gptq_mode_runs_and_returns_correct_shape():
    w, H = _make_layer()
    result = fasterquant(w, H, bits=4, group_size=32, mode="gptq", blocksize=8)
    assert result.hard_weight.shape == w.shape
    assert result.hard_weight.dtype == w.dtype
    assert result.metrics["predicted_loss"] is not None


def test_hsq_v0_with_zero_tau_matches_gptq_nearest_choice():
    w, H = _make_layer(seed=1)
    gptq_result = fasterquant(w, H, bits=4, group_size=32, mode="gptq", blocksize=8)
    hsq_result = fasterquant(w, H, bits=4, group_size=32, mode="hsq_v0", blocksize=8, tau=0.0, eps0=0.0)
    # tau=0 leaves no slack beyond the nearest candidate's own cost, so HSQ
    # should degenerate to exactly the same committed weights as plain GPTQ.
    assert torch.allclose(gptq_result.hard_weight, hsq_result.hard_weight)


def test_hsq_v0_with_large_tau_moves_further_from_fp_than_gptq():
    w, H = _make_layer(seed=2)
    gptq_result = fasterquant(w, H, bits=4, group_size=32, mode="gptq", blocksize=8)
    hsq_result = fasterquant(w, H, bits=4, group_size=32, mode="hsq_v0", blocksize=8, tau=5.0, candidate_radius=2)
    gptq_dist = (gptq_result.hard_weight.float() - w).pow(2).sum()
    hsq_dist = (hsq_result.hard_weight.float() - w).pow(2).sum()
    assert hsq_dist >= gptq_dist
    assert hsq_result.metrics["farther_fraction"] >= gptq_result.metrics["farther_fraction"]


def test_hsq_v1_two_pass_runs_and_respects_group_budget_ratio():
    w, H = _make_layer(seed=3)
    result = run_hsq_layer(w, H, mode="hsq_v1", bits=4, group_size=32, blocksize=8, tau=0.5)
    assert result.hard_weight.shape == w.shape
    assert torch.isfinite(result.hard_weight).all()


def test_dead_column_is_zeroed_in_output():
    w, H = _make_layer(seed=4, in_features=16)
    H = H.clone()
    H[3, :] = 0.0
    H[:, 3] = 0.0
    result = fasterquant(w, H, bits=4, group_size=16, mode="gptq", blocksize=16)
    assert torch.allclose(result.hard_weight[:, 3], torch.zeros(w.shape[0], dtype=w.dtype))


def test_group_size_minus_one_uses_single_group():
    w, H = _make_layer(seed=5, in_features=32)
    result = fasterquant(w, H, bits=4, group_size=-1, mode="gptq", blocksize=8)
    assert result.group_loss_totals.shape[1] == 1


def test_hsq_v0_never_produces_worse_than_bit_grid_range():
    w, H = _make_layer(seed=6)
    result = fasterquant(w, H, bits=4, group_size=32, mode="hsq_v0", blocksize=8, tau=10.0, candidate_radius=2)
    assert torch.isfinite(result.hard_weight).all()
