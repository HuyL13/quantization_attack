import torch

from aq.activation_cache import capture_block_input_activations, capture_layer_input_activations
from aq.local_reconstruction import run_blockwise_local_reconstruction, run_layerwise_local_reconstruction
from aq.quantizer import AdversarialQuantConfig


def test_run_layerwise_local_reconstruction_commits_and_releases(tiny_model, tiny_calibration_batches):
    layers = {"mid_layers.0": tiny_model.mid_layers[0], "mid_layers.1": tiny_model.mid_layers[1]}
    order = list(layers.keys())
    originals = {name: mod.weight.detach().clone() for name, mod in layers.items()}
    cached = capture_layer_input_activations(tiny_model, layers, tiny_calibration_batches, device="cpu")

    cfg = AdversarialQuantConfig(bits=4, group_size=128, steps=3, lr=1e-2, batches_per_step=2)
    results = run_layerwise_local_reconstruction(layers, order, cached, cfg, device="cpu")

    assert set(results.keys()) == set(order)
    for name, mod in layers.items():
        assert not torch.equal(mod.weight.detach(), originals[name])
        assert torch.equal(mod.weight.detach(), results[name].hard_weight)
        assert results[name].quantizer is None
        assert len(results[name].trace_rows) == cfg.steps


def test_layerwise_local_reconstruction_never_calls_full_model_forward(tiny_model, tiny_calibration_batches):
    # the whole point of 1B is to never run tiny_model's own forward during
    # optimization - only cached activations through the target Linear layer.
    layers = {"mid_layers.0": tiny_model.mid_layers[0]}
    order = list(layers.keys())
    cached = capture_layer_input_activations(tiny_model, layers, tiny_calibration_batches, device="cpu")

    call_count = {"n": 0}
    original_forward = tiny_model.forward

    def counting_forward(*args, **kwargs):
        call_count["n"] += 1
        return original_forward(*args, **kwargs)

    tiny_model.forward = counting_forward
    try:
        cfg = AdversarialQuantConfig(bits=4, group_size=128, steps=3, lr=1e-2, batches_per_step=2)
        run_layerwise_local_reconstruction(layers, order, cached, cfg, device="cpu")
    finally:
        tiny_model.forward = original_forward

    assert call_count["n"] == 0


def test_run_blockwise_local_reconstruction_commits(tiny_model, tiny_calibration_batches):
    layers = {"mid_layers.0": tiny_model.mid_layers[0], "mid_layers.1": tiny_model.mid_layers[1]}
    block_groups = [["mid_layers.0"], ["mid_layers.1"]]
    block_modules = [tiny_model.mid_layers[0], tiny_model.mid_layers[1]]
    block_module_map = {"0": block_modules[0], "1": block_modules[1]}
    cached_by_idx = capture_block_input_activations(tiny_model, block_module_map, tiny_calibration_batches, device="cpu")
    cached_block_inputs = [cached_by_idx["0"], cached_by_idx["1"]]

    originals = {name: mod.weight.detach().clone() for name, mod in layers.items()}
    cfg = AdversarialQuantConfig(bits=4, group_size=128, steps=3, lr=1e-2, batches_per_step=2)

    results = run_blockwise_local_reconstruction(
        block_groups, layers, block_modules, cached_block_inputs, cfg, device="cpu"
    )

    assert set(results.keys()) == {"mid_layers.0", "mid_layers.1"}
    for name, mod in layers.items():
        assert not torch.equal(mod.weight.detach(), originals[name])
        assert torch.equal(mod.weight.detach(), results[name].hard_weight)
        assert results[name].quantizer is None
        assert len(results[name].trace_rows) == cfg.steps
