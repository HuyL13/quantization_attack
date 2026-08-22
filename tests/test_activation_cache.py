import torch

from aq.activation_cache import (
    activation_sensitivity,
    capture_block_input_activations,
    capture_layer_input_activations,
)


def test_capture_layer_input_activations_shapes(tiny_model, tiny_calibration_batches):
    layers = {"mid_layers.0": tiny_model.mid_layers[0], "mid_layers.1": tiny_model.mid_layers[1]}
    captured = capture_layer_input_activations(tiny_model, layers, tiny_calibration_batches, device="cpu")

    assert set(captured.keys()) == set(layers.keys())
    for name, tensors in captured.items():
        assert len(tensors) == len(tiny_calibration_batches)
        for t, batch in zip(tensors, tiny_calibration_batches):
            assert t.shape[:2] == batch["input_ids"].shape
            assert t.device.type == "cpu"
            assert not t.requires_grad


def test_captured_layer_1_input_matches_layer_0_output(tiny_model, tiny_calibration_batches):
    # mid_layers.1's cached input should equal tanh(mid_layers.0(embed(x))) -
    # i.e. it's a real forward-pass activation, not a placeholder.
    layers = {"mid_layers.0": tiny_model.mid_layers[0], "mid_layers.1": tiny_model.mid_layers[1]}
    captured = capture_layer_input_activations(tiny_model, layers, tiny_calibration_batches, device="cpu")

    with torch.no_grad():
        for batch, x1_cached in zip(tiny_calibration_batches, captured["mid_layers.1"]):
            x0 = tiny_model.embed(batch["input_ids"])
            expected_x1 = torch.tanh(tiny_model.mid_layers[0](x0))
            assert torch.allclose(expected_x1, x1_cached, atol=1e-5)


def test_capture_block_input_activations(tiny_model, tiny_calibration_batches):
    blocks = {"0": tiny_model.mid_layers[0], "1": tiny_model.mid_layers[1]}
    captured = capture_block_input_activations(tiny_model, blocks, tiny_calibration_batches, device="cpu")
    assert set(captured.keys()) == {"0", "1"}
    assert len(captured["0"]) == len(tiny_calibration_batches)


def test_activation_sensitivity_matches_manual_mean_square():
    torch.manual_seed(0)
    a = torch.randn(2, 3, 4)
    b = torch.randn(1, 5, 4)
    sensitivity = activation_sensitivity([a, b])

    flat = torch.cat([a.reshape(-1, 4), b.reshape(-1, 4)], dim=0)
    expected = flat.pow(2).mean(dim=0)
    assert torch.allclose(sensitivity, expected, atol=1e-5)
    assert sensitivity.shape == (4,)


def test_activation_sensitivity_raises_on_empty_input():
    import pytest

    with pytest.raises(ValueError):
        activation_sensitivity([])
