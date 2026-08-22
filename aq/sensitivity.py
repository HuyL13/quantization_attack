"""Per-weight sensitivity I_i for method 4 (Sensitivity-Aware): approximates
|dL/dw_i| via gradient accumulation over calibration batches, so the
weighted distance term (aq.metrics.weight_relative_distance_weighted) can
let the optimizer push distance harder through low-sensitivity weights
while protecting high-sensitivity ones - no change to the core objective
itself (plan: "không đổi objective, chỉ thêm trọng số I_i vào D").
"""
from __future__ import annotations

import torch
import torch.nn as nn


@torch.enable_grad()
def estimate_weight_sensitivity(
    model,
    module: nn.Linear,
    calibration_batches: list[dict],
    device: str = "cuda",
) -> torch.Tensor:
    """Runs a handful of calibration batches through `model` with a language-
    modeling loss and accumulates |grad| on `module.weight`, in fp32, same
    shape as the weight. Restores requires_grad state and zeroes gradients
    on exit so callers can call this per-layer without side effects on
    later layers' optimization.
    """
    weight = module.weight
    had_grad = weight.requires_grad
    weight.requires_grad_(True)
    if weight.grad is not None:
        weight.grad = None

    accum = torch.zeros_like(weight, dtype=torch.float32)
    model.eval()
    for batch in calibration_batches:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = out.loss
        loss.backward()
        if weight.grad is not None:
            accum += weight.grad.detach().float().abs()
            weight.grad = None

    weight.requires_grad_(had_grad)
    n = max(len(calibration_batches), 1)
    return accum / n
