import torch

from aq.optimizer_core import compute_fp_reference_logits, optimize_layer
from aq.quantizer import AdversarialQuantConfig


def test_compute_fp_reference_logits_shapes(tiny_model, tiny_calibration_batches):
    refs = compute_fp_reference_logits(tiny_model, tiny_calibration_batches, device="cpu")
    assert len(refs) == len(tiny_calibration_batches)
    for ref, batch in zip(refs, tiny_calibration_batches):
        assert ref.shape[:2] == batch["input_ids"].shape
        assert not ref.requires_grad
        # cached off the training device on purpose - see the function's
        # docstring: keeping every calibration batch's full-vocab logits
        # resident on GPU for the whole multi-layer run was ~17GB by itself.
        assert ref.device.type == "cpu"


def test_optimize_layer_runs_and_produces_trace(tiny_model, tiny_calibration_batches):
    fp_ref = compute_fp_reference_logits(tiny_model, tiny_calibration_batches, device="cpu")
    cfg = AdversarialQuantConfig(bits=4, group_size=128, steps=3, lr=1e-2, lambda_distance=0.1)
    target_module = tiny_model.mid_layers[0]
    w_fp_before = target_module.weight.detach().clone()

    result = optimize_layer(
        tiny_model, target_module, "mid_layers.0", tiny_calibration_batches, fp_ref, cfg, device="cpu"
    )

    assert len(result.trace_rows) == cfg.steps
    assert result.hard_weight.shape == w_fp_before.shape
    # optimize_layer must not mutate the module's actual weight - only the
    # caller (calibration_strategies._commit) does that.
    assert torch.equal(target_module.weight.detach(), w_fp_before)
    assert "weight_distance_vs_fp" in result.layer_metrics
    assert "rounding_flip_ratio_vs_rtn4" in result.layer_metrics
    assert 0.0 <= result.layer_metrics["rounding_flip_ratio_vs_rtn4"] <= 1.0


def test_optimize_layer_forward_patch_is_restored(tiny_model, tiny_calibration_batches):
    fp_ref = compute_fp_reference_logits(tiny_model, tiny_calibration_batches, device="cpu")
    cfg = AdversarialQuantConfig(bits=4, group_size=128, steps=2, lr=1e-2)
    target_module = tiny_model.mid_layers[0]
    original_forward = target_module.forward

    optimize_layer(tiny_model, target_module, "mid_layers.0", tiny_calibration_batches, fp_ref, cfg, device="cpu")

    assert target_module.forward == original_forward


def test_optimize_layer_loss_is_finite_every_step(tiny_model, tiny_calibration_batches):
    fp_ref = compute_fp_reference_logits(tiny_model, tiny_calibration_batches, device="cpu")
    cfg = AdversarialQuantConfig(bits=4, group_size=128, steps=5, lr=1e-2, lambda_distance=0.2)
    result = optimize_layer(
        tiny_model, tiny_model.mid_layers[1], "mid_layers.1", tiny_calibration_batches, fp_ref, cfg, device="cpu"
    )
    for row in result.trace_rows:
        assert torch.isfinite(torch.tensor(row["loss"]))
        assert torch.isfinite(torch.tensor(row["kl"]))
        assert torch.isfinite(torch.tensor(row["distance"]))


def test_optimize_layer_uses_only_batches_per_step_not_full_calibration_set(tiny_model):
    # 6 calibration batches available, but batches_per_step=2: each step must
    # do exactly 2 forward passes through the model, not 6 - this is the fix
    # for a real measured slowdown (a 128-batch full-sweep-per-step smoke
    # test did not finish even one layer in over 3 minutes on an A100).
    torch.manual_seed(9)
    batches = [
        {"input_ids": torch.randint(0, 32, (2, 6)), "attention_mask": torch.ones(2, 6, dtype=torch.long)}
        for _ in range(6)
    ]
    fp_ref = compute_fp_reference_logits(tiny_model, batches, device="cpu")

    call_count = {"n": 0}
    original_forward = tiny_model.forward

    def counting_forward(*args, **kwargs):
        call_count["n"] += 1
        return original_forward(*args, **kwargs)

    tiny_model.forward = counting_forward
    try:
        cfg = AdversarialQuantConfig(bits=4, group_size=128, steps=3, lr=1e-2, batches_per_step=2)
        optimize_layer(tiny_model, tiny_model.mid_layers[0], "mid_layers.0", batches, fp_ref, cfg, device="cpu")
    finally:
        tiny_model.forward = original_forward

    assert call_count["n"] == cfg.steps * cfg.batches_per_step  # 3 * 2 = 6, not 3 * 6 = 18


def test_optimize_layer_batches_per_step_capped_by_available_batches(tiny_model, tiny_calibration_batches):
    # only 2 calibration batches available (see tiny_calibration_batches
    # fixture) but batches_per_step=10 requested - must not crash asking
    # random.sample for more items than exist.
    fp_ref = compute_fp_reference_logits(tiny_model, tiny_calibration_batches, device="cpu")
    cfg = AdversarialQuantConfig(bits=4, group_size=128, steps=2, lr=1e-2, batches_per_step=10)
    result = optimize_layer(
        tiny_model, tiny_model.mid_layers[0], "mid_layers.0", tiny_calibration_batches, fp_ref, cfg, device="cpu"
    )
    assert len(result.trace_rows) == cfg.steps
