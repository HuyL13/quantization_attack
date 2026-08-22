import torch

from aq.optimizer_core import compute_fp_reference_logits, optimize_layer
from aq.quantizer import AdversarialQuantConfig


def test_compute_fp_reference_logits_shapes(tiny_model, tiny_calibration_batches):
    refs = compute_fp_reference_logits(tiny_model, tiny_calibration_batches, device="cpu")
    assert len(refs) == len(tiny_calibration_batches)
    for ref, batch in zip(refs, tiny_calibration_batches):
        assert ref.shape[:2] == batch["input_ids"].shape
        assert not ref.requires_grad


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
