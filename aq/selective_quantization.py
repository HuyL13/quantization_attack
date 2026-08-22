"""Methods A (Margin-Aware) and B (Fragile-Channel) Selective Quantization -
zero-training, one-shot: score every weight by

    S_i = |predicted behavior delta_i| / (|predicted utility delta_i| + eps)

where both predicted deltas come from a first-order Taylor approximation
(gradient . weight-change) using the gradients from
aq.behavior_gradient.compute_behavior_and_utility_gradients, and
weight-change is the same "near vs far" RTN alternative used by method 1A's
greedy rounding (aq.greedy_rounding). The top `aggressive_fraction` of
weights by score get far-rounded (pushed away from the parent); everything
else stays at plain RTN4's nearest rounding.

No optimization loop, no repeated forward passes per weight/channel - this
is a single closed-form pass per layer, using gradients already computed
once for the whole model.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from aq.optimizer_core import LayerOptimizationResult
from aq.rtn_backend import rtn_quantize_weight_raw


@dataclass
class SelectiveQuantConfig:
    bits: int = 4
    group_size: int = 128
    aggressive_fraction: float = 0.05
    eps: float = 1e-8


def _score_and_select(
    w_fp: torch.Tensor,
    cfg: SelectiveQuantConfig,
    grad_behavior: torch.Tensor,
    grad_utility: torch.Tensor,
):
    rtn_state = rtn_quantize_weight_raw(w_fp, bits=cfg.bits, group_size=cfg.group_size)
    pre_round = rtn_state.pre_round
    floor_val = torch.floor(pre_round)
    frac = pre_round - floor_val
    near_is_ceil = frac >= 0.5

    q_near = torch.where(near_is_ceil, floor_val + 1, floor_val).clamp(0, rtn_state.max_int)
    q_far = torch.where(near_is_ceil, floor_val, floor_val + 1).clamp(0, rtn_state.max_int)
    delta_int = q_far - q_near
    delta_w = delta_int * rtn_state.scale  # weight-space change of the "far" candidate vs "near"

    out_features, padded_in = pre_round.shape
    grad_behavior_padded = torch.zeros(out_features, padded_in, device=w_fp.device, dtype=torch.float32)
    grad_behavior_padded[:, : rtn_state.in_features] = grad_behavior.to(w_fp.device).float()
    grad_utility_padded = torch.zeros(out_features, padded_in, device=w_fp.device, dtype=torch.float32)
    grad_utility_padded[:, : rtn_state.in_features] = grad_utility.to(w_fp.device).float()

    predicted_behavior_delta = grad_behavior_padded * delta_w
    predicted_utility_delta = grad_utility_padded * delta_w

    score = predicted_behavior_delta.abs() / (predicted_utility_delta.abs() + cfg.eps)
    score = torch.where(delta_int != 0, score, torch.full_like(score, float("-inf")))
    if rtn_state.padded_in_features > rtn_state.in_features:
        score[:, rtn_state.in_features :] = float("-inf")

    real_scores = score[:, : rtn_state.in_features]
    finite_scores = real_scores[torch.isfinite(real_scores)]
    k = int(cfg.aggressive_fraction * real_scores.numel())
    if k <= 0 or finite_scores.numel() == 0:
        select_mask = torch.zeros_like(score, dtype=torch.bool)
    else:
        k = min(k, finite_scores.numel())
        threshold = torch.topk(finite_scores, k).values.min()
        select_mask = score >= threshold

    final_int = torch.where(select_mask, q_far, q_near)
    dequant = (final_int - rtn_state.zero_point) * rtn_state.scale
    if rtn_state.padded_in_features > rtn_state.in_features:
        dequant = dequant[:, : rtn_state.in_features]
    hard_weight = dequant.to(rtn_state.original_dtype)

    baseline_int = torch.round(pre_round).clamp(0, rtn_state.max_int)
    return hard_weight, baseline_int, final_int


@torch.no_grad()
def run_selective_quantization(
    layers: dict,
    order: list[str],
    grad_by_layer: dict[str, tuple[torch.Tensor, torch.Tensor]],
    cfg: SelectiveQuantConfig,
    device: str,
) -> dict[str, LayerOptimizationResult]:
    from aq.calibration_strategies import _commit
    from aq.metrics import cosine_similarity_flat, rounding_flip_ratio, weight_relative_distance

    results: dict[str, LayerOptimizationResult] = {}
    for name in order:
        module = layers[name]
        w_fp = module.weight.detach().clone()
        grad_behavior, grad_utility = grad_by_layer[name]

        hard_weight, baseline_int, hard_int_after = _score_and_select(w_fp, cfg, grad_behavior, grad_utility)

        layer_metrics = {
            "layer": name,
            "weight_distance_vs_fp": float(weight_relative_distance(w_fp, hard_weight)),
            "cosine_similarity_vs_fp": float(cosine_similarity_flat(w_fp, hard_weight)),
            "rounding_flip_ratio_vs_rtn4": float(rounding_flip_ratio(baseline_int, hard_int_after)),
            "scale_relative_shift": 0.0,
            "final_kl": None,
            "final_loss": None,
            "num_steps": 0,
        }
        results[name] = LayerOptimizationResult(
            layer_name=name, quantizer=None, hard_weight=hard_weight, trace_rows=[], layer_metrics=layer_metrics
        )

    for name in order:
        _commit(layers[name], results[name].hard_weight.to(device))
    return results
