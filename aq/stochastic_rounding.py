"""Method C - Stochastic / Biased Rounding: the lightest tier. No scale
optimization, no codebook, no backprop, no full-model KL - just a
per-weight random choice between floor and ceil.

    C1 unbiased:            P(ceil) = frac(pre_round)              (textbook
                             stochastic rounding - unbiased in expectation)
    C2 far_biased:          P(far point) = P(unbiased far) + far_bias
                             (uniformly pushes every weight's rounding
                             distribution toward the non-nearest grid point)
    C3 fragility_weighted:  same as C2, but far_bias is scaled per input
                             feature by that feature's activation
                             sensitivity (aq.activation_cache) - the same
                             cheap, backprop-free statistic method 1A uses,
                             reused here instead of a second, unrelated
                             notion of "fragility" - low-activation columns
                             get pushed harder toward "far" since they carry
                             less of the model's broad output signal.

The core argument (plan doc): if a fingerprint's exact-match trigger
requires a chain of n consecutive precise decisions, even a small per-step
drop in each decision's probability compounds multiplicatively
(P_exact = prod p_t), while ordinary language-model quality degrades far
more gracefully under the same per-weight perturbation. This works from
UNIFORM (or activation-weighted) stochastic rounding across the whole
model - no per-channel selection is required for it to have an effect,
unlike methods A/B.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from aq.optimizer_core import LayerOptimizationResult
from aq.rtn_backend import rtn_quantize_weight_raw


@dataclass
class StochasticRoundingConfig:
    bits: int = 4
    group_size: int = 128
    variant: str = "far_biased"  # "unbiased" | "far_biased" | "fragility_weighted"
    far_bias: float = 0.3
    seed: int = 0


def _stochastic_int(
    pre_round: torch.Tensor,
    max_int: int,
    cfg: StochasticRoundingConfig,
    generator: torch.Generator,
    sensitivity_padded: torch.Tensor | None,
) -> torch.Tensor:
    floor_val = torch.floor(pre_round)
    frac = pre_round - floor_val
    near_is_ceil = frac >= 0.5

    p_ceil = frac.clone()
    if cfg.variant in ("far_biased", "fragility_weighted"):
        bias = cfg.far_bias
        if cfg.variant == "fragility_weighted":
            if sensitivity_padded is None:
                raise ValueError("fragility_weighted variant requires a sensitivity tensor")
            # normalize to [0, 1] per layer so far_bias stays in a sane range
            s = sensitivity_padded
            s_min, s_max = s.min(), s.max()
            s_norm = (s - s_min) / (s_max - s_min).clamp_min(1e-12)
            inverse_importance = 1.0 - s_norm  # low activation magnitude -> push harder
            bias = cfg.far_bias * inverse_importance
        # near_is_ceil=True means "far" is floor -> DECREASE p_ceil toward 0.
        # near_is_ceil=False means "far" is ceil -> INCREASE p_ceil toward 1.
        p_ceil = torch.where(near_is_ceil, p_ceil - bias, p_ceil + bias).clamp(0.0, 1.0)

    draw = torch.rand(pre_round.shape, generator=generator, device="cpu").to(pre_round.device)
    ceil_chosen = draw < p_ceil
    int_w = torch.where(ceil_chosen, floor_val + 1, floor_val)
    return int_w.clamp(0, max_int)


@torch.no_grad()
def run_stochastic_rounding(
    layers: dict,
    order: list[str],
    sensitivity_by_layer: dict[str, torch.Tensor] | None,
    cfg: StochasticRoundingConfig,
    device: str,
) -> dict[str, LayerOptimizationResult]:
    from aq.calibration_strategies import _commit
    from aq.metrics import cosine_similarity_flat, rounding_flip_ratio, weight_relative_distance

    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    results: dict[str, LayerOptimizationResult] = {}
    for name in order:
        module = layers[name]
        w_fp = module.weight.detach().clone()
        rtn_state = rtn_quantize_weight_raw(w_fp, bits=cfg.bits, group_size=cfg.group_size)

        sensitivity_padded = None
        if cfg.variant == "fragility_weighted":
            out_features, padded_in = rtn_state.pre_round.shape
            sensitivity_padded = torch.zeros(out_features, padded_in, device=w_fp.device, dtype=torch.float32)
            sensitivity_padded[:, : rtn_state.in_features] = (
                sensitivity_by_layer[name].to(w_fp.device).float().unsqueeze(0)
            )

        final_int = _stochastic_int(rtn_state.pre_round, rtn_state.max_int, cfg, generator, sensitivity_padded)
        dequant = (final_int - rtn_state.zero_point) * rtn_state.scale
        if rtn_state.padded_in_features > rtn_state.in_features:
            dequant = dequant[:, : rtn_state.in_features]
        hard_weight = dequant.to(rtn_state.original_dtype)
        baseline_int = torch.round(rtn_state.pre_round).clamp(0, rtn_state.max_int)

        layer_metrics = {
            "layer": name,
            "weight_distance_vs_fp": float(weight_relative_distance(w_fp, hard_weight)),
            "cosine_similarity_vs_fp": float(cosine_similarity_flat(w_fp, hard_weight)),
            "rounding_flip_ratio_vs_rtn4": float(rounding_flip_ratio(baseline_int, final_int)),
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
