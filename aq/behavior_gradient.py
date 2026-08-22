"""Shared gradient computation for methods A (Margin-Aware) and B
(Fragile-Channel): a SINGLE (or few) full-model forward+backward pass over
calibration data gives, for every target weight simultaneously, the
gradient of a "behavior" scalar and a "utility" scalar - no per-channel
trial-and-error forward pass, no iterative optimization loop. This is a
first-order Taylor approximation of "what would happen to behavior/utility
if this weight moved by delta" (d(behavior)/dw . delta), the same idea used
throughout sensitivity-based pruning/quantization literature (OBD/OBS-style
saliency), applied here to pick which weights are safe to push toward the
"far" (non-nearest) rounding point without a repeated search.

Two behavior signals are supported:
  "margin"       - top1-top2 logit gap (Method A): weights whose
                   perturbation would most shrink the model's own decision
                   margin - the kind of change that flips discrete
                   decisions like a fingerprint's exact-match trigger.
  "top1_logprob" - log P(the model's own top1 token) (Method B): a more
                   general "output confidence/distribution" signal, not
                   tied to a specific margin between exactly two classes.

Both require backprop (a real forward+backward through the WHOLE model),
so - like methods 1D/2-4's old heavy tier - this needs gradient
checkpointing to stay within a single A100's memory, but unlike those
methods there is no repeated optimization loop: just a handful of
calibration batches, once.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _behavior_scalar(logits: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "margin":
        top2 = logits.float().topk(2, dim=-1).values  # indices detached by topk itself
        return (top2[..., 0] - top2[..., 1]).mean()
    if kind == "top1_logprob":
        log_probs = F.log_softmax(logits.float(), dim=-1)
        top1_idx = logits.argmax(dim=-1, keepdim=True).detach()
        return log_probs.gather(-1, top1_idx).squeeze(-1).mean()
    raise ValueError(f"unknown behavior kind {kind!r}")


@torch.enable_grad()
def compute_behavior_and_utility_gradients(
    model,
    layers: dict,
    calibration_batches: list[dict],
    device: str,
    behavior: str = "margin",
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Returns {layer_name: (grad_behavior, grad_utility)}, both tensors the
    same shape as that layer's weight, averaged over `calibration_batches`.
    Two full backward passes per batch (behavior and utility are different
    scalars, so they need separate backward() calls) - the model's FP
    weights are never updated, only their .grad is read and then cleared;
    this is scoring only, not training (a hard PTQ requirement: freeze
    W_FP, never touch it based on this gradient).
    """
    modules = list(layers.values())
    had_grad = [m.weight.requires_grad for m in modules]
    for m in modules:
        m.weight.requires_grad_(True)

    grad_behavior_sum = {name: torch.zeros_like(m.weight, dtype=torch.float32) for name, m in layers.items()}
    grad_utility_sum = {name: torch.zeros_like(m.weight, dtype=torch.float32) for name, m in layers.items()}

    model.eval()
    n = max(len(calibration_batches), 1)
    for batch in calibration_batches:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        for m in modules:
            if m.weight.grad is not None:
                m.weight.grad = None
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        behavior_scalar = _behavior_scalar(out.logits, behavior)
        behavior_scalar.backward()
        for name, m in layers.items():
            if m.weight.grad is not None:
                grad_behavior_sum[name] += m.weight.grad.detach().float()

        for m in modules:
            if m.weight.grad is not None:
                m.weight.grad = None
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        utility_loss = F.cross_entropy(
            out.logits[:, :-1, :].reshape(-1, out.logits.shape[-1]).float(),
            input_ids[:, 1:].reshape(-1),
        )
        utility_loss.backward()
        for name, m in layers.items():
            if m.weight.grad is not None:
                grad_utility_sum[name] += m.weight.grad.detach().float()

    for m in modules:
        m.weight.grad = None
    for m, orig in zip(modules, had_grad):
        m.weight.requires_grad_(orig)

    return {name: (grad_behavior_sum[name] / n, grad_utility_sum[name] / n) for name in layers}
