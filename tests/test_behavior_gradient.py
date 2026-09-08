import torch

from aq.behavior_gradient import compute_behavior_and_utility_gradients


def test_gradients_have_correct_shapes_and_are_finite(tiny_model, tiny_calibration_batches):
    layers = {"mid_layers.0": tiny_model.mid_layers[0], "mid_layers.1": tiny_model.mid_layers[1]}
    grads = compute_behavior_and_utility_gradients(
        tiny_model, layers, tiny_calibration_batches, device="cpu", behavior="margin"
    )
    assert set(grads.keys()) == set(layers.keys())
    for name, module in layers.items():
        grad_behavior, grad_utility = grads[name]
        assert grad_behavior.shape == module.weight.shape
        assert grad_utility.shape == module.weight.shape
        assert torch.isfinite(grad_behavior).all()
        assert torch.isfinite(grad_utility).all()


def test_top1_logprob_behavior_variant_runs(tiny_model, tiny_calibration_batches):
    layers = {"mid_layers.0": tiny_model.mid_layers[0]}
    grads = compute_behavior_and_utility_gradients(
        tiny_model, layers, tiny_calibration_batches, device="cpu", behavior="top1_logprob"
    )
    grad_behavior, grad_utility = grads["mid_layers.0"]
    assert torch.isfinite(grad_behavior).all()
    assert torch.isfinite(grad_utility).all()


def test_unknown_behavior_kind_raises():
    import pytest

    from aq.behavior_gradient import _behavior_scalar

    with pytest.raises(ValueError):
        _behavior_scalar(torch.randn(2, 3, 5), "nonsense")


def test_does_not_mutate_model_weights(tiny_model, tiny_calibration_batches):
    layers = {"mid_layers.0": tiny_model.mid_layers[0]}
    original = tiny_model.mid_layers[0].weight.detach().clone()
    compute_behavior_and_utility_gradients(tiny_model, layers, tiny_calibration_batches, device="cpu")
    assert torch.equal(tiny_model.mid_layers[0].weight.detach(), original)


def test_restores_requires_grad_and_clears_grad(tiny_model, tiny_calibration_batches):
    layers = {"mid_layers.0": tiny_model.mid_layers[0]}
    assert tiny_model.mid_layers[0].weight.requires_grad is True
    compute_behavior_and_utility_gradients(tiny_model, layers, tiny_calibration_batches, device="cpu")
    assert tiny_model.mid_layers[0].weight.requires_grad is True
    assert tiny_model.mid_layers[0].weight.grad is None


def test_preserves_training_mode_for_gradient_checkpointing(tiny_model, tiny_calibration_batches):
    layers = {"mid_layers.0": tiny_model.mid_layers[0]}
    tiny_model.train()
    compute_behavior_and_utility_gradients(tiny_model, layers, tiny_calibration_batches, device="cpu")
    assert tiny_model.training is True


def test_non_target_parameters_do_not_keep_gradients(tiny_model, tiny_calibration_batches):
    layers = {"mid_layers.0": tiny_model.mid_layers[0]}
    non_target = tiny_model.mid_layers[1].weight
    assert non_target.requires_grad is True
    compute_behavior_and_utility_gradients(tiny_model, layers, tiny_calibration_batches, device="cpu")
    assert non_target.requires_grad is True
    assert non_target.grad is None


def test_margin_gradient_matches_manual_two_class_case():
    # Build a minimal linear model over a 2-class vocabulary (real integer
    # token ids, one-hot embedded) where the margin (top1-top2 logit gap)
    # and its gradient w.r.t. the weight can be verified by hand - a direct
    # check that _behavior_scalar's use of topk's own (detached) index
    # selection still backprops the VALUES correctly, not just returning
    # zero/garbage gradients. Needs seq_len=2 (not 1) so the utility-loss
    # half of compute_behavior_and_utility_gradients also has a non-empty
    # next-token target to compute against.
    import torch.nn as nn
    import torch.nn.functional as F
    from types import SimpleNamespace

    class TwoClassLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(2, 2, bias=False)

        def forward(self, input_ids, attention_mask=None, use_cache=None):
            x = F.one_hot(input_ids, num_classes=2).float()  # (batch, seq, 2)
            logits = self.w(x)
            return SimpleNamespace(logits=logits)

    model = TwoClassLinear()
    with torch.no_grad():
        model.w.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 1.0]]))

    input_ids = torch.tensor([[0, 1]])  # batch=1, seq=2: token0 then token1
    batches = [{"input_ids": input_ids, "attention_mask": None}]

    from aq.behavior_gradient import compute_behavior_and_utility_gradients

    grads = compute_behavior_and_utility_gradients(model, {"w": model.w}, batches, device="cpu", behavior="margin")
    grad_behavior, _ = grads["w"]
    # Verified independently via a plain autograd computation of the same
    # forward + margin (see the commit that added this test for the
    # by-hand derivation): mean margin over both positions -> this exact
    # gradient.
    expected = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
    assert torch.allclose(grad_behavior.float(), expected, atol=1e-5)
