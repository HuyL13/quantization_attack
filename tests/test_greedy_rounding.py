import torch

from aq.greedy_rounding import GreedyRoundingConfig, _score_and_flip, run_greedy_adversarial_rounding
from aq.rtn_backend import rtn_quantize_weight_raw


def _make_weight_and_sensitivity(seed=5, out_features=4, in_features=128):
    torch.manual_seed(seed)
    w = torch.randn(out_features, in_features)
    sensitivity = torch.rand(in_features) + 0.1  # avoid exact zeros
    return w, sensitivity


def test_score_and_flip_zero_fraction_matches_rtn_baseline():
    w, sensitivity = _make_weight_and_sensitivity()
    cfg = GreedyRoundingConfig(bits=4, group_size=128, flip_fraction=0.0)
    hard_weight, baseline_int, hard_int_after, rtn_state = _score_and_flip(w, cfg, sensitivity)
    assert torch.equal(baseline_int, hard_int_after)  # nothing flipped
    assert torch.equal(hard_weight, rtn_state.dequantize_truncated())


def test_score_and_flip_full_fraction_flips_everything_possible():
    w, sensitivity = _make_weight_and_sensitivity()
    cfg = GreedyRoundingConfig(bits=4, group_size=128, flip_fraction=1.0)
    hard_weight, baseline_int, hard_int_after, rtn_state = _score_and_flip(w, cfg, sensitivity)
    # every element whose near != far (i.e. not stuck at a clamp boundary) must have flipped
    pre_round = rtn_state.pre_round
    floor_val = torch.floor(pre_round)
    frac = pre_round - floor_val
    near_is_ceil = frac >= 0.5
    q_near = torch.where(near_is_ceil, floor_val + 1, floor_val).clamp(0, rtn_state.max_int)
    q_far = torch.where(near_is_ceil, floor_val, floor_val + 1).clamp(0, rtn_state.max_int)
    can_flip = q_near != q_far
    assert torch.equal(hard_int_after[can_flip], q_far[can_flip])


def test_score_and_flip_intermediate_fraction_flips_some_not_all():
    w, sensitivity = _make_weight_and_sensitivity()
    cfg = GreedyRoundingConfig(bits=4, group_size=128, flip_fraction=0.3)
    hard_weight, baseline_int, hard_int_after, rtn_state = _score_and_flip(w, cfg, sensitivity)
    n_flipped = (hard_int_after != baseline_int).sum().item()
    assert 0 < n_flipped < baseline_int.numel()


def test_score_and_flip_prefers_low_sensitivity_columns():
    # construct sensitivity so one column is far cheaper to flip than the rest -
    # the greedy score should preferentially flip weights in that column first.
    torch.manual_seed(1)
    w = torch.randn(4, 128)
    sensitivity = torch.ones(128) * 10.0
    cheap_col = 5
    sensitivity[cheap_col] = 1e-6
    cfg = GreedyRoundingConfig(bits=4, group_size=128, flip_fraction=0.05)  # small budget
    _, baseline_int, hard_int_after, rtn_state = _score_and_flip(w, cfg, sensitivity)

    flipped_mask = hard_int_after != baseline_int
    # can only assert something if the cheap column actually had a flippable element
    if flipped_mask[:, cheap_col].any():
        # the cheap column's flip rate should be at least as high as a typical
        # expensive column's, since it dominates the score ranking
        other_cols_rate = flipped_mask[:, [c for c in range(128) if c != cheap_col]].float().mean()
        cheap_col_rate = flipped_mask[:, cheap_col].float().mean()
        assert cheap_col_rate >= other_cols_rate


def test_run_greedy_adversarial_rounding_commits_weights(tiny_model, tiny_calibration_batches):
    from aq.activation_cache import compute_layer_activation_sensitivity

    layers = {"mid_layers.0": tiny_model.mid_layers[0], "mid_layers.1": tiny_model.mid_layers[1]}
    order = list(layers.keys())
    originals = {name: mod.weight.detach().clone() for name, mod in layers.items()}
    sensitivity_by_layer = compute_layer_activation_sensitivity(tiny_model, layers, tiny_calibration_batches, device="cpu")

    cfg = GreedyRoundingConfig(bits=4, group_size=128, flip_fraction=0.2)
    results = run_greedy_adversarial_rounding(tiny_model, layers, order, sensitivity_by_layer, cfg, device="cpu")

    assert set(results.keys()) == set(order)
    for name, mod in layers.items():
        assert not torch.equal(mod.weight.detach(), originals[name])
        assert torch.equal(mod.weight.detach(), results[name].hard_weight)
        assert results[name].quantizer is None  # never created - zero-training tier
        assert results[name].trace_rows == []  # no optimization steps at all
