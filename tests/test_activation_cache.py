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


def test_compute_layer_activation_sensitivity_matches_capture_then_manual(tiny_model, tiny_calibration_batches):
    from aq.activation_cache import compute_layer_activation_sensitivity

    layers = {"mid_layers.0": tiny_model.mid_layers[0], "mid_layers.1": tiny_model.mid_layers[1]}
    streaming = compute_layer_activation_sensitivity(tiny_model, layers, tiny_calibration_batches, device="cpu")
    captured = capture_layer_input_activations(tiny_model, layers, tiny_calibration_batches, device="cpu")

    for name in layers:
        expected = activation_sensitivity(captured[name])
        assert torch.allclose(streaming[name], expected, atol=1e-5)


def test_compute_layer_activation_sensitivity_never_retains_raw_tensors(tiny_model, tiny_calibration_batches):
    # the whole point: no raw activation tensor should be reachable from the
    # returned dict, only the aggregate per-channel statistic (a 1-D tensor).
    from aq.activation_cache import compute_layer_activation_sensitivity

    layers = {"mid_layers.0": tiny_model.mid_layers[0]}
    result = compute_layer_activation_sensitivity(tiny_model, layers, tiny_calibration_batches, device="cpu")
    assert result["mid_layers.0"].dim() == 1
    assert result["mid_layers.0"].shape[0] == tiny_model.mid_layers[0].in_features


def test_capture_block_input_activations_preserves_required_kwargs():
    # Regression test for a real bug found against the actual 7B model:
    # a plain forward_pre_hook only sees positional args, so calling a real
    # Llama decoder block directly with just its hidden-states input crashed
    # deep inside self_attn trying to unpack a None `position_embeddings`
    # kwarg that the block's forward requires but only its PARENT model
    # normally computes and passes in. with_kwargs=True must capture that
    # kwarg too, and _to_device must let it be replayed unchanged.
    import torch.nn as nn

    class BlockNeedingExtraKwarg(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4, bias=False)

        def forward(self, x, required_extra=None):
            if required_extra is None:
                raise TypeError("cannot unpack non-iterable NoneType object")
            return self.proj(x) * required_extra

    block = BlockNeedingExtraKwarg()

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = block

        def forward(self, input_ids, attention_mask=None, use_cache=None):
            x = input_ids.float().unsqueeze(-1).expand(-1, -1, 4)
            extra = torch.ones(1)
            return self.block(x, required_extra=extra)

    model = Wrapper()
    batches = [{"input_ids": torch.randint(0, 5, (1, 3))}]

    captured = capture_block_input_activations(model, {"block": block}, batches, device="cpu")
    assert captured["block"][0]["kwargs"]["required_extra"] is not None

    from aq.activation_cache import _to_device

    args = _to_device(captured["block"][0]["args"], "cpu")
    kwargs = _to_device(captured["block"][0]["kwargs"], "cpu")
    out = block(*args, **kwargs)  # must not raise
    assert out is not None
