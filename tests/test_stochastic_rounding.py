import torch

from aq.stochastic_rounding import StochasticRoundingConfig, _stochastic_int, run_stochastic_rounding
from aq.rtn_backend import rtn_quantize_weight_raw


def test_unbiased_stochastic_rounding_matches_expected_probability():
    # Repeat many independent draws for a single fixed fractional value and
    # check the empirical P(ceil) converges to frac, per textbook unbiased
    # stochastic rounding.
    pre_round = torch.full((2000,), 3.3)
    max_int = 15
    cfg = StochasticRoundingConfig(bits=4, group_size=128, variant="unbiased")
    generator = torch.Generator(device="cpu").manual_seed(0)
    result = _stochastic_int(pre_round, max_int, cfg, generator, sensitivity_padded=None)
    empirical_p_ceil = (result == 4).float().mean().item()
    assert abs(empirical_p_ceil - 0.3) < 0.03


def test_far_biased_pushes_probability_toward_far_point():
    # frac=0.3 -> near=floor(3), far=ceil(4). Unbiased P(ceil)=0.3;
    # far_biased should push P(ceil) up toward 0.3+far_bias.
    pre_round = torch.full((2000,), 3.3)
    max_int = 15
    cfg = StochasticRoundingConfig(bits=4, group_size=128, variant="far_biased", far_bias=0.4)
    generator = torch.Generator(device="cpu").manual_seed(1)
    result = _stochastic_int(pre_round, max_int, cfg, generator, sensitivity_padded=None)
    empirical_p_ceil = (result == 4).float().mean().item()
    assert abs(empirical_p_ceil - 0.7) < 0.03


def test_far_biased_mirrors_correctly_when_near_is_ceil():
    # frac=0.8 -> near=ceil(4), far=floor(3). far_biased should DECREASE
    # P(ceil) below the unbiased 0.8 (pushing toward far=floor).
    pre_round = torch.full((2000,), 3.8)
    max_int = 15
    cfg = StochasticRoundingConfig(bits=4, group_size=128, variant="far_biased", far_bias=0.4)
    generator = torch.Generator(device="cpu").manual_seed(2)
    result = _stochastic_int(pre_round, max_int, cfg, generator, sensitivity_padded=None)
    empirical_p_ceil = (result == 4).float().mean().item()
    assert abs(empirical_p_ceil - 0.4) < 0.03


def test_fragility_weighted_requires_sensitivity():
    import pytest

    pre_round = torch.tensor([3.3])
    cfg = StochasticRoundingConfig(bits=4, group_size=128, variant="fragility_weighted")
    generator = torch.Generator(device="cpu").manual_seed(0)
    with pytest.raises(ValueError):
        _stochastic_int(pre_round, 15, cfg, generator, sensitivity_padded=None)


def test_fragility_weighted_biases_low_sensitivity_columns_more():
    torch.manual_seed(0)
    pre_round = torch.full((2000, 2), 3.3)
    sensitivity = torch.tensor([[10.0, 0.0]]).expand(2000, 2)  # column 0 high, column 1 low importance
    cfg = StochasticRoundingConfig(bits=4, group_size=128, variant="fragility_weighted", far_bias=0.5)
    generator = torch.Generator(device="cpu").manual_seed(3)
    result = _stochastic_int(pre_round, 15, cfg, generator, sensitivity_padded=sensitivity)
    p_ceil_col0 = (result[:, 0] == 4).float().mean().item()
    p_ceil_col1 = (result[:, 1] == 4).float().mean().item()
    # column 1 (low sensitivity) should be pushed harder toward the far point
    assert p_ceil_col1 > p_ceil_col0


def test_run_stochastic_rounding_unbiased_variant_commits(tiny_model, tiny_calibration_batches):
    layers = {"mid_layers.0": tiny_model.mid_layers[0], "mid_layers.1": tiny_model.mid_layers[1]}
    order = list(layers.keys())
    originals = {name: mod.weight.detach().clone() for name, mod in layers.items()}

    cfg = StochasticRoundingConfig(bits=4, group_size=128, variant="unbiased")
    results = run_stochastic_rounding(layers, order, None, cfg, device="cpu")

    assert set(results.keys()) == set(order)
    for name, mod in layers.items():
        assert torch.equal(mod.weight.detach(), results[name].hard_weight)
        assert results[name].quantizer is None
        # weight should generally differ from FP (quantization error), though
        # not asserting "not equal" strictly since a tiny layer could
        # coincidentally match on some elements.


def test_run_stochastic_rounding_fragility_weighted_variant(tiny_model, tiny_calibration_batches):
    from aq.activation_cache import compute_layer_activation_sensitivity

    layers = {"mid_layers.0": tiny_model.mid_layers[0]}
    order = list(layers.keys())
    sensitivity_by_layer = compute_layer_activation_sensitivity(tiny_model, layers, tiny_calibration_batches, device="cpu")

    cfg = StochasticRoundingConfig(bits=4, group_size=128, variant="fragility_weighted", far_bias=0.3)
    results = run_stochastic_rounding(layers, order, sensitivity_by_layer, cfg, device="cpu")
    assert set(results.keys()) == set(order)
