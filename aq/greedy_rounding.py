"""Method 1A - Greedy Adversarial Rounding (zero-training tier).

No backprop, no Adam, no learnable rounding parameter, no per-step model
forward at all: this is a single closed-form pass per layer. For every
weight, RTN4's own round-to-nearest choice is the "near" grid point; the
other adjacent integer (floor vs ceil) is the "far" one. Flipping near->far
strictly increases weight distance from the FP parent (that's the whole
point) at some local-reconstruction cost, so each weight gets a score

    score_i = weight_distance_gain_i / (activation_error_increase_i + eps)

and the top `flip_fraction` of weights (by score) get flipped. No forward
pass through anything is needed for the scoring itself - only the
one-time, already-cheap activation statistics gathered by
aq.activation_cache.capture_layer_input_activations before this runs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from aq.activation_cache import activation_sensitivity
from aq.optimizer_core import LayerOptimizationResult
from aq.rtn_backend import rtn_quantize_weight_raw


@dataclass
class GreedyRoundingConfig:
    bits: int = 4
    group_size: int = 128
    flip_fraction: float = 0.2  # fraction of a layer's weights to flip toward "far", by score
    eps: float = 1e-8


def _score_and_flip(w_fp: torch.Tensor, cfg: GreedyRoundingConfig, sensitivity: torch.Tensor):
    rtn_state = rtn_quantize_weight_raw(w_fp, bits=cfg.bits, group_size=cfg.group_size)
    pre_round = rtn_state.pre_round
    floor_val = torch.floor(pre_round)
    frac = pre_round - floor_val
    near_is_ceil = frac >= 0.5

    q_near = torch.where(near_is_ceil, floor_val + 1, floor_val).clamp(0, rtn_state.max_int)
    q_far = torch.where(near_is_ceil, floor_val, floor_val + 1).clamp(0, rtn_state.max_int)
    delta_int = q_far - q_near  # 0 exactly where near==far (clamped at a grid boundary - no flip possible)

    w_near = (q_near - rtn_state.zero_point) * rtn_state.scale
    w_far = (q_far - rtn_state.zero_point) * rtn_state.scale

    out_features, padded_in = pre_round.shape
    w_fp_padded = torch.zeros(out_features, padded_in, device=w_fp.device, dtype=torch.float32)
    w_fp_padded[:, : rtn_state.in_features] = w_fp.float()

    distance_gain = (w_fp_padded - w_far).abs() - (w_fp_padded - w_near).abs()

    sensitivity_padded = torch.zeros(padded_in, device=w_fp.device, dtype=torch.float32)
    sensitivity_padded[: rtn_state.in_features] = sensitivity.to(w_fp.device).float()
    delta_w = delta_int * rtn_state.scale
    act_error_increase = sensitivity_padded.unsqueeze(0) * delta_w.pow(2)

    score = distance_gain / (act_error_increase + cfg.eps)
    score = torch.where(delta_int != 0, score, torch.full_like(score, float("-inf")))
    # padding columns (beyond in_features) never get flipped - they don't correspond to a real weight
    if rtn_state.padded_in_features > rtn_state.in_features:
        score[:, rtn_state.in_features :] = float("-inf")

    real_scores = score[:, : rtn_state.in_features]
    finite_scores = real_scores[torch.isfinite(real_scores)]
    k = int(cfg.flip_fraction * real_scores.numel())
    if k <= 0 or finite_scores.numel() == 0:
        flip_mask = torch.zeros_like(score, dtype=torch.bool)
    else:
        k = min(k, finite_scores.numel())
        threshold = torch.topk(finite_scores, k).values.min()
        flip_mask = score >= threshold

    final_int = torch.where(flip_mask, q_far, q_near)
    dequant = (final_int - rtn_state.zero_point) * rtn_state.scale
    if rtn_state.padded_in_features > rtn_state.in_features:
        dequant = dequant[:, : rtn_state.in_features]
    hard_weight = dequant.to(rtn_state.original_dtype)

    baseline_int = torch.round(pre_round).clamp(0, rtn_state.max_int)
    hard_int_after = final_int
    return hard_weight, baseline_int, hard_int_after, rtn_state


@torch.no_grad()
def run_greedy_adversarial_rounding(
    model,
    layers: dict,
    order: list[str],
    sensitivity_by_layer: dict[str, torch.Tensor],
    cfg: GreedyRoundingConfig,
    device: str,
) -> dict[str, LayerOptimizationResult]:
    """`sensitivity_by_layer` is each layer's per-input-feature mean-squared
    activation (aq.activation_cache.compute_layer_activation_sensitivity is
    the streaming way to get this without ever storing raw activations -
    activation_sensitivity(cached_inputs) also works if the caller already
    has raw cached tensors for some other reason, e.g. in tests).
    """
    from aq.calibration_strategies import _commit
    from aq.metrics import cosine_similarity_flat, rounding_flip_ratio, weight_relative_distance

    results: dict[str, LayerOptimizationResult] = {}
    for name in order:
        module = layers[name]
        w_fp = module.weight.detach().clone()
        sensitivity = sensitivity_by_layer[name]

        hard_weight, baseline_int, hard_int_after, rtn_state = _score_and_flip(w_fp, cfg, sensitivity)

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
