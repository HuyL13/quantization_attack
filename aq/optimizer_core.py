"""Shared per-layer optimization loop used by methods 1-4 directly, and by
methods 5-8 (aq/calibration_strategies.py) as their inner routine. Only the
CALLER decides what state the rest of `model` is in when this runs (all-FP
for isolated per-layer methods 1-4; partially-committed quantized-prefix for
method 5; block-grouped for method 6; etc) - this function never mutates
anything outside the one target layer/module it's given, and never changes
the objective (plan: "một core objective duy nhất cho toàn bộ 9 method").
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from aq.metrics import (
    cosine_similarity_flat,
    kl_divergence_logits,
    rounding_flip_ratio,
    scale_relative_shift,
    top1_agreement,
)
from aq.quantizer import AdversarialLinearQuantizer, AdversarialQuantConfig
from aq.rtn_backend import rtn_quantize_weight_raw
from aq.sensitivity import estimate_weight_sensitivity


@dataclass
class LayerOptimizationResult:
    layer_name: str
    quantizer: AdversarialLinearQuantizer
    hard_weight: torch.Tensor
    trace_rows: list[dict]
    layer_metrics: dict


class _LayerForwardPatch:
    """Temporarily replaces `module.forward` with F.linear against a weight
    tensor pulled fresh from `weight_fn()` on every call, so autograd flows
    from the model's loss back into `weight_fn`'s own parameters (the
    quantizer's alpha / scale / codebook variables) without touching any
    other layer.
    """

    def __init__(self, module: nn.Linear, weight_fn):
        self.module = module
        self.weight_fn = weight_fn
        self._orig_forward = None

    def __enter__(self):
        import torch.nn.functional as F

        self._orig_forward = self.module.forward

        def patched(x):
            return F.linear(x, self.weight_fn(), self.module.bias)

        self.module.forward = patched
        return self

    def __exit__(self, *_exc):
        self.module.forward = self._orig_forward


@torch.no_grad()
def compute_fp_reference_logits(model, calibration_batches: list[dict], device: str) -> list[torch.Tensor]:
    model.eval()
    refs = []
    for batch in calibration_batches:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        refs.append(logits.detach())
    return refs


def optimize_layer(
    model,
    module: nn.Linear,
    layer_name: str,
    calibration_batches: list[dict],
    fp_reference_logits: list[torch.Tensor],
    cfg: AdversarialQuantConfig,
    device: str = "cuda",
) -> LayerOptimizationResult:
    w_fp = module.weight.detach().clone()
    rtn_state = rtn_quantize_weight_raw(w_fp, bits=cfg.bits, group_size=cfg.group_size)

    quantizer = AdversarialLinearQuantizer(w_fp, rtn_state, cfg).to(device)
    if cfg.use_sensitivity:
        quantizer.sensitivity = estimate_weight_sensitivity(model, module, calibration_batches, device)

    hard_int_before = torch.round(quantizer.pre_round).clamp(0, quantizer.max_int).detach().clone()
    scale_before = quantizer.scale0.detach().clone()

    trainable = [p for p in quantizer.parameters() if p.requires_grad]
    optim = torch.optim.Adam(trainable, lr=cfg.lr)

    trace_rows: list[dict] = []
    for step in range(cfg.steps):
        optim.zero_grad()
        soft_w = quantizer.soft_weight()
        kl_sum = torch.zeros((), device=device)
        with _LayerForwardPatch(module, lambda: soft_w):
            for ref_logits, batch in zip(fp_reference_logits, calibration_batches):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                out_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                kl_sum = kl_sum + kl_divergence_logits(ref_logits, out_logits)
        kl_term = kl_sum / max(len(calibration_batches), 1)
        distance_term = quantizer.weight_distance()
        round_reg = quantizer.rounding_regularizer()
        loss = kl_term - cfg.lambda_distance * distance_term + cfg.round_reg_weight * round_reg

        loss.backward()
        optim.step()

        trace_rows.append(
            {
                "layer": layer_name,
                "step": step,
                "loss": float(loss.detach()),
                "kl": float(kl_term.detach()),
                "distance": float(distance_term.detach()),
                "round_reg": float(round_reg.detach()),
                "timestamp": time.time(),
            }
        )

    hard_w = quantizer.hard_weight()
    hard_int_after = quantizer.hard_int_grid()

    with torch.no_grad():
        layer_metrics = {
            "layer": layer_name,
            "weight_distance_vs_fp": float(quantizer.weight_distance().detach()),
            "cosine_similarity_vs_fp": float(cosine_similarity_flat(w_fp, hard_w)),
            "rounding_flip_ratio_vs_rtn4": float(rounding_flip_ratio(hard_int_before, hard_int_after)),
            "scale_relative_shift": float(scale_relative_shift(scale_before, quantizer.effective_scale()))
            if cfg.optimize_scale
            else 0.0,
            "final_kl": trace_rows[-1]["kl"] if trace_rows else None,
            "final_loss": trace_rows[-1]["loss"] if trace_rows else None,
            "num_steps": cfg.steps,
        }

    return LayerOptimizationResult(
        layer_name=layer_name,
        quantizer=quantizer,
        hard_weight=hard_w,
        trace_rows=trace_rows,
        layer_metrics=layer_metrics,
    )
