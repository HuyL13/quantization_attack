import copy

import torch

from aq.calibration_strategies import (
    run_block_wise,
    run_isolated,
    run_quantized_prefix,
    run_two_pass_backward_correction,
)
from aq.optimizer_core import compute_fp_reference_logits
from aq.quantizer import AdversarialQuantConfig


def _layers_and_order(model):
    layers = {"mid_layers.0": model.mid_layers[0], "mid_layers.1": model.mid_layers[1]}
    order = ["mid_layers.0", "mid_layers.1"]
    return layers, order


def test_run_isolated_commits_both_layers(tiny_model, tiny_calibration_batches):
    layers, order = _layers_and_order(tiny_model)
    originals = {name: mod.weight.detach().clone() for name, mod in layers.items()}
    fp_ref = compute_fp_reference_logits(tiny_model, tiny_calibration_batches, device="cpu")
    cfg = AdversarialQuantConfig(bits=4, group_size=128, steps=2, lr=1e-2)

    results = run_isolated(tiny_model, layers, order, tiny_calibration_batches, fp_ref, cfg, device="cpu")

    assert set(results.keys()) == set(order)
    for name, mod in layers.items():
        assert not torch.equal(mod.weight.detach(), originals[name])
        assert torch.equal(mod.weight.detach(), results[name].hard_weight)


def test_run_quantized_prefix_commits_sequentially(tiny_model, tiny_calibration_batches):
    layers, order = _layers_and_order(tiny_model)
    fp_ref = compute_fp_reference_logits(tiny_model, tiny_calibration_batches, device="cpu")
    cfg = AdversarialQuantConfig(bits=4, group_size=128, steps=2, lr=1e-2)

    call_log = []
    from aq import calibration_strategies as cs

    original_optimize_layer = cs.optimize_layer

    def spy(model, module, name, *args, **kwargs):
        # at the time layer 1 is optimized, layer 0 must already be committed
        # (its weight must differ from the module's own live weight captured
        # at call time being anything other than the still-FP tensor) -
        # verified indirectly by recording layer 0's weight snapshot.
        call_log.append((name, layers["mid_layers.0"].weight.detach().clone()))
        return original_optimize_layer(model, module, name, *args, **kwargs)

    cs.optimize_layer = spy
    try:
        results = run_quantized_prefix(tiny_model, layers, order, tiny_calibration_batches, fp_ref, cfg, device="cpu")
    finally:
        cs.optimize_layer = original_optimize_layer

    assert call_log[0][0] == "mid_layers.0"
    assert call_log[1][0] == "mid_layers.1"
    # layer 0's weight at the moment layer 1 starts must equal its committed
    # hard weight (i.e. already quantized), not the original FP weight.
    assert torch.equal(call_log[1][1], results["mid_layers.0"].hard_weight)


def test_run_block_wise_joint_optimizes_group(tiny_model, tiny_calibration_batches):
    layers, order = _layers_and_order(tiny_model)
    block_groups = [order]  # treat both mid layers as a single "block"
    fp_ref = compute_fp_reference_logits(tiny_model, tiny_calibration_batches, device="cpu")
    cfg = AdversarialQuantConfig(bits=4, group_size=128, steps=2, lr=1e-2)

    results = run_block_wise(tiny_model, layers, block_groups, tiny_calibration_batches, fp_ref, cfg, device="cpu")

    assert set(results.keys()) == set(order)
    for name in order:
        assert len(results[name].trace_rows) == cfg.steps
        # every layer in the block shares the same joint loss trajectory
        assert results[name].trace_rows[0]["loss"] == results[order[0]].trace_rows[0]["loss"]


def test_two_pass_backward_correction_uses_pristine_reference(tiny_model, tiny_calibration_batches):
    layers, order = _layers_and_order(tiny_model)
    fp_ref = compute_fp_reference_logits(tiny_model, tiny_calibration_batches, device="cpu")
    fp_ref_snapshot = [t.clone() for t in fp_ref]
    cfg = AdversarialQuantConfig(bits=4, group_size=128, steps=2, lr=1e-2)

    results = run_two_pass_backward_correction(
        tiny_model, layers, order, tiny_calibration_batches, fp_ref, cfg, device="cpu"
    )

    # the reference list passed in must not have been mutated by either pass
    for before, after in zip(fp_ref_snapshot, fp_ref):
        assert torch.equal(before, after)

    for name in order:
        assert len(results[name].trace_rows) == 2 * cfg.steps
        assert "forward_pass_final_kl" in results[name].layer_metrics
        assert "backward_pass_final_kl" in results[name].layer_metrics
