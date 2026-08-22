import torch

from aq.selective_quantization import SelectiveQuantConfig, _score_and_select, run_selective_quantization
from aq.rtn_backend import rtn_quantize_weight_raw


def _make_weight(seed=3, out_features=4, in_features=128):
    torch.manual_seed(seed)
    return torch.randn(out_features, in_features)


def test_zero_aggressive_fraction_matches_rtn_baseline():
    w = _make_weight()
    grad_behavior = torch.randn(w.shape[1])
    grad_utility = torch.rand(w.shape[1]) + 0.1
    cfg = SelectiveQuantConfig(bits=4, group_size=128, aggressive_fraction=0.0)
    hard_weight, baseline_int, hard_int_after = _score_and_select(w, cfg, grad_behavior, grad_utility)
    assert torch.equal(baseline_int, hard_int_after)
    rtn_state = rtn_quantize_weight_raw(w, bits=4, group_size=128)
    assert torch.equal(hard_weight, rtn_state.dequantize_truncated())


def test_prefers_high_behavior_low_utility_gradient_columns():
    # column 5: huge behavior gradient, tiny utility gradient -> should be
    # selected first (best "attack cheaply" trade); column 10: the reverse
    # (should almost never be selected at a small budget).
    torch.manual_seed(1)
    w = torch.randn(4, 128)
    grad_behavior = torch.ones(128) * 1e-3
    grad_utility = torch.ones(128) * 10.0
    grad_behavior[5] = 10.0
    grad_utility[5] = 1e-6
    grad_behavior[10] = 1e-6
    grad_utility[10] = 10.0

    cfg = SelectiveQuantConfig(bits=4, group_size=128, aggressive_fraction=0.02)
    _, baseline_int, hard_int_after = _score_and_select(w, cfg, grad_behavior, grad_utility)
    flipped = hard_int_after != baseline_int
    if flipped[:, 5].any() or flipped[:, 10].any():
        assert flipped[:, 5].float().mean() >= flipped[:, 10].float().mean()


def test_run_selective_quantization_commits_and_releases(tiny_model, tiny_calibration_batches):
    from aq.behavior_gradient import compute_behavior_and_utility_gradients

    layers = {"mid_layers.0": tiny_model.mid_layers[0], "mid_layers.1": tiny_model.mid_layers[1]}
    order = list(layers.keys())
    originals = {name: mod.weight.detach().clone() for name, mod in layers.items()}
    grad_by_layer = compute_behavior_and_utility_gradients(
        tiny_model, layers, tiny_calibration_batches, device="cpu", behavior="margin"
    )

    cfg = SelectiveQuantConfig(bits=4, group_size=128, aggressive_fraction=0.1)
    results = run_selective_quantization(layers, order, grad_by_layer, cfg, device="cpu")

    assert set(results.keys()) == set(order)
    for name, mod in layers.items():
        assert not torch.equal(mod.weight.detach(), originals[name])
        assert torch.equal(mod.weight.detach(), results[name].hard_weight)
        assert results[name].quantizer is None
        assert results[name].trace_rows == []
