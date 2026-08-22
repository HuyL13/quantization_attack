import torch

from aq.sensitivity import estimate_weight_sensitivity


def test_sensitivity_shape_and_nonnegative(tiny_model, tiny_calibration_batches):
    module = tiny_model.mid_layers[0]
    sensitivity = estimate_weight_sensitivity(tiny_model, module, tiny_calibration_batches, device="cpu")
    assert sensitivity.shape == module.weight.shape
    assert (sensitivity >= 0).all()


def test_sensitivity_restores_requires_grad_state(tiny_model, tiny_calibration_batches):
    module = tiny_model.mid_layers[0]
    assert module.weight.requires_grad is True  # nn.Parameter default
    estimate_weight_sensitivity(tiny_model, module, tiny_calibration_batches, device="cpu")
    assert module.weight.requires_grad is True
    assert module.weight.grad is None  # cleared after accumulation


def test_sensitivity_zero_for_layer_disconnected_from_loss(tiny_model, tiny_calibration_batches):
    # head layer is always connected; but an unused extra layer should get
    # exactly zero sensitivity since no gradient reaches it.
    unused = torch.nn.Linear(4, 4)
    sensitivity = estimate_weight_sensitivity(tiny_model, unused, tiny_calibration_batches, device="cpu")
    assert torch.equal(sensitivity, torch.zeros_like(unused.weight))
