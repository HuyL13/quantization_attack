import pytest
import torch

import aq.global_far_round as global_far_round
from aq.global_far_round import GlobalFarRoundConfig, get_near_far_candidates, run_global_far_round


def _linear(weight):
    layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)
    return layer


def test_score_quantiles_bound_the_number_of_values_given_to_torch_quantile(monkeypatch):
    observed_sizes = []
    real_quantile = torch.quantile

    def guarded_quantile(values, quantiles):
        observed_sizes.append(values.numel())
        if values.numel() > 128:
            raise RuntimeError("quantile() input tensor is too large")
        return real_quantile(values, quantiles)

    monkeypatch.setattr(torch, "quantile", guarded_quantile)
    result = global_far_round._score_quantiles(torch.arange(10_000.0), max_samples=128)

    assert observed_sizes == [128]
    assert result.shape == (4,)
    assert torch.all(result[1:] >= result[:-1])


def test_global_budget_is_not_reapplied_per_layer():
    torch.manual_seed(7)
    layers = {
        "model.layers.0.self_attn.q_proj": _linear(torch.randn(1, 128)),
        "model.layers.1.self_attn.q_proj": _linear(torch.randn(1, 128)),
    }
    gradients = {
        "model.layers.0.self_attn.q_proj": (torch.ones(1, 128), torch.ones(1, 128)),
        "model.layers.1.self_attn.q_proj": (torch.linspace(50.0, 100.0, 128).unsqueeze(0), torch.ones(1, 128)),
    }
    cfg = GlobalFarRoundConfig(aggressive_fraction=0.25, histogram_bins=256)

    result = run_global_far_round(layers, list(layers), gradients, cfg, device="cpu")

    first = result.layer_results["model.layers.0.self_attn.q_proj"].layer_metrics
    second = result.layer_results["model.layers.1.self_attn.q_proj"].layer_metrics
    assert first["selected_fraction"] == 0.0
    assert second["selected_fraction"] > first["selected_fraction"]
    assert result.global_metrics["actual_selected_fraction"] == pytest.approx(0.25, abs=0.02)


def test_zero_fraction_matches_rtn4_and_is_deterministic():
    torch.manual_seed(11)
    weight = torch.randn(2, 128)
    layers = {"model.layers.0.mlp.down_proj": _linear(weight)}
    gradients = {"model.layers.0.mlp.down_proj": (torch.randn_like(weight), torch.rand_like(weight) + 0.1)}
    cfg = GlobalFarRoundConfig(aggressive_fraction=0.0, histogram_bins=128)

    first = run_global_far_round(layers, list(layers), gradients, cfg, device="cpu")
    first_weight = layers["model.layers.0.mlp.down_proj"].weight.detach().clone()
    layers = {"model.layers.0.mlp.down_proj": _linear(weight)}
    second = run_global_far_round(layers, list(layers), gradients, cfg, device="cpu")

    assert first.global_metrics["global_num_selected"] == 0
    assert first.global_metrics["actual_rounding_flip_ratio"] == 0.0
    assert first.global_metrics["selection_mask_checksum"] == second.global_metrics["selection_mask_checksum"]
    assert torch.equal(first_weight, layers["model.layers.0.mlp.down_proj"].weight)


def test_layer_metrics_include_required_global_far_round_diagnostics():
    torch.manual_seed(19)
    weight = torch.randn(2, 128)
    name = "model.layers.3.mlp.gate_proj"
    layers = {name: _linear(weight)}
    gradients = {name: (torch.randn_like(weight), torch.rand_like(weight) + 0.1)}

    result = run_global_far_round(
        layers,
        [name],
        gradients,
        GlobalFarRoundConfig(aggressive_fraction=0.1, histogram_bins=128),
        device="cpu",
    )

    metrics = result.layer_results[name].layer_metrics
    required = {
        "projection_type", "num_weights", "num_valid_candidates", "num_selected",
        "selected_fraction", "rounding_flip_ratio_vs_rtn4", "score_mean", "score_std",
        "score_median", "score_p90", "score_p95", "score_p99", "score_max",
        "pred_behavior_delta_abs_mean", "pred_utility_delta_abs_mean",
        "weight_distance_vs_fp", "cosine_similarity_vs_fp",
    }
    assert required <= metrics.keys()
    assert metrics["projection_type"] == "gate_proj"


def test_global_metrics_aggregate_selection_by_projection():
    torch.manual_seed(23)
    layers = {
        "model.layers.0.self_attn.q_proj": _linear(torch.randn(1, 128)),
        "model.layers.0.mlp.down_proj": _linear(torch.randn(2, 128)),
        "model.layers.1.self_attn.q_proj": _linear(torch.randn(1, 128)),
    }
    gradients = {
        name: (torch.randn_like(layer.weight), torch.rand_like(layer.weight) + 0.1)
        for name, layer in layers.items()
    }

    result = run_global_far_round(
        layers,
        list(layers),
        gradients,
        GlobalFarRoundConfig(aggressive_fraction=0.1, histogram_bins=128),
        device="cpu",
    )

    by_projection = result.global_metrics["selection_by_projection"]
    assert set(by_projection) == {"q_proj", "down_proj"}
    assert by_projection["q_proj"]["num_weights"] == 256
    assert by_projection["down_proj"]["num_weights"] == 256
    assert by_projection["q_proj"]["selected_fraction"] == pytest.approx(
        by_projection["q_proj"]["num_selected"] / 256
    )


def test_selected_fraction_uses_all_real_weights_as_denominator():
    torch.manual_seed(29)
    name = "model.layers.0.self_attn.v_proj"
    layer = _linear(torch.randn(1, 128))
    layers = {name: layer}
    gradients = {name: (torch.randn_like(layer.weight), torch.rand_like(layer.weight) + 0.1)}

    result = run_global_far_round(
        layers,
        [name],
        gradients,
        GlobalFarRoundConfig(aggressive_fraction=0.2, histogram_bins=128),
        device="cpu",
    )

    metrics = result.layer_results[name].layer_metrics
    assert metrics["selected_fraction"] == pytest.approx(metrics["num_selected"] / metrics["num_weights"])


def test_streaming_result_does_not_retain_full_quantized_weight_copies():
    torch.manual_seed(31)
    name = "model.layers.0.mlp.up_proj"
    layer = _linear(torch.randn(2, 128))
    layers = {name: layer}
    gradients = {name: (torch.randn_like(layer.weight), torch.rand_like(layer.weight) + 0.1)}

    result = run_global_far_round(
        layers,
        [name],
        gradients,
        GlobalFarRoundConfig(aggressive_fraction=0.1, histogram_bins=128),
        device="cpu",
    )

    assert result.layer_results[name].hard_weight is None


def test_nonfinite_scores_are_excluded_from_the_global_candidate_budget():
    torch.manual_seed(37)
    name = "model.layers.0.self_attn.o_proj"
    layer = _linear(torch.randn(1, 128))
    _, _, _, legal = get_near_far_candidates(layer.weight.detach())
    row, column = torch.nonzero(legal, as_tuple=False)[0]
    grad_behavior = torch.ones_like(layer.weight)
    grad_behavior[row, column] = float("nan")
    gradients = {name: (grad_behavior, torch.ones_like(layer.weight))}

    result = run_global_far_round(
        {name: layer},
        [name],
        gradients,
        GlobalFarRoundConfig(aggressive_fraction=1.0, histogram_bins=128),
        device="cpu",
    )

    expected_candidates = int(legal.sum()) - 1
    assert result.global_metrics["global_num_candidates"] == expected_candidates
    assert result.global_metrics["global_num_selected"] == expected_candidates


def test_zero_score_ties_at_the_histogram_floor_are_selected():
    torch.manual_seed(41)
    name = "model.layers.0.self_attn.k_proj"
    layer = _linear(torch.randn(1, 128))
    gradients = {name: (torch.zeros_like(layer.weight), torch.ones_like(layer.weight))}

    result = run_global_far_round(
        {name: layer},
        [name],
        gradients,
        GlobalFarRoundConfig(aggressive_fraction=0.25, histogram_bins=128),
        device="cpu",
    )

    assert result.global_metrics["global_num_candidates"] > 0
    assert result.global_metrics["global_num_selected"] == result.global_metrics["global_num_candidates"]
