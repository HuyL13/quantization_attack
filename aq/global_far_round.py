"""Global fragility-scored far-round quantization for Method 3.

Every target scalar weight receives the same first-order behavior/utility
score used by the former Method 3.  One model-wide histogram determines a
single threshold; a second streaming pass applies that threshold layer by
layer, so scores are never concatenated for the full model on GPU or CPU.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import torch

from aq.optimizer_core import LayerOptimizationResult
from aq.rtn_backend import rtn_quantize_weight_raw


@dataclass
class GlobalFarRoundConfig:
    bits: int = 4
    group_size: int = 128
    aggressive_fraction: float = 0.05
    eps: float = 1e-8
    behavior: str = "top1_logprob"
    threshold_mode: str = "histogram"
    histogram_bins: int = 65536

    def __post_init__(self) -> None:
        if not 0.0 <= self.aggressive_fraction <= 1.0:
            raise ValueError("aggressive_fraction must be in [0, 1]")
        if self.threshold_mode != "histogram":
            raise ValueError("threshold_mode must be 'histogram'")
        if self.histogram_bins < 2:
            raise ValueError("histogram_bins must be at least 2")


@dataclass
class GlobalFarRoundResult:
    layer_results: dict[str, LayerOptimizationResult]
    global_metrics: dict


def _score_quantiles(values: torch.Tensor, max_samples: int = 1_000_000) -> torch.Tensor:
    """Return diagnostic quantiles without passing a full 7B layer to quantile.

    PyTorch rejects very large inputs to ``torch.quantile``.  These values are
    report-only diagnostics, so use a deterministic, evenly spaced sample and
    compute its quantiles on CPU.  Selection still uses every score.
    """
    flat = values.detach().reshape(-1)
    if flat.numel() == 0:
        return torch.zeros(4, dtype=torch.float32)
    if max_samples < 1:
        raise ValueError("max_samples must be at least 1")
    if flat.numel() > max_samples:
        indices = torch.linspace(
            0,
            flat.numel() - 1,
            steps=max_samples,
            device=flat.device,
            dtype=torch.float64,
        ).to(dtype=torch.long)
        flat = flat.index_select(0, indices)
    flat = flat.to(device="cpu", dtype=torch.float32)
    return torch.quantile(flat, torch.tensor([0.5, 0.9, 0.95, 0.99]))


def get_near_far_candidates(w_fp: torch.Tensor, bits: int = 4, group_size: int = 128):
    state = rtn_quantize_weight_raw(w_fp, bits=bits, group_size=group_size)
    floor_value = torch.floor(state.pre_round)
    fraction = state.pre_round - floor_value
    near_is_ceil = fraction >= 0.5
    q_near = torch.where(near_is_ceil, floor_value + 1, floor_value).clamp(0, state.max_int)
    q_far = torch.where(near_is_ceil, floor_value, floor_value + 1).clamp(0, state.max_int)
    valid = q_far != q_near
    if state.padded_in_features > state.in_features:
        valid[:, state.in_features :] = False
    return state, q_near, q_far, valid


def _pad_gradient(gradient: torch.Tensor, state, device: torch.device) -> torch.Tensor:
    padded = torch.zeros(state.pre_round.shape, device=device, dtype=torch.float32)
    gradient = gradient.to(device=device, dtype=torch.float32)
    if gradient.shape == padded.shape:
        padded.copy_(gradient)
    elif gradient.ndim == 1 and gradient.numel() == state.in_features:
        padded[:, : state.in_features] = gradient.unsqueeze(0)
    else:
        padded[:, : state.in_features] = gradient
    return padded


def compute_score_tensor(
    state,
    q_near: torch.Tensor,
    q_far: torch.Tensor,
    valid: torch.Tensor,
    grad_behavior: torch.Tensor,
    grad_utility: torch.Tensor,
    eps: float,
):
    device = state.pre_round.device
    delta_w = (q_far - q_near) * state.scale
    grad_b = _pad_gradient(grad_behavior, state, device)
    grad_u = _pad_gradient(grad_utility, state, device)
    predicted_behavior = grad_b * delta_w
    predicted_utility = grad_u * delta_w
    score = predicted_behavior.abs() / (predicted_utility.abs() + eps)
    score = torch.where(valid & torch.isfinite(score), score, torch.full_like(score, float("-inf")))
    return score, predicted_behavior, predicted_utility


def _score_layer(module, gradients, cfg: GlobalFarRoundConfig):
    weight = module.weight.detach()
    state, q_near, q_far, valid = get_near_far_candidates(weight, cfg.bits, cfg.group_size)
    score, predicted_b, predicted_u = compute_score_tensor(
        state, q_near, q_far, valid, gradients[0], gradients[1], cfg.eps
    )
    return state, q_near, q_far, valid, score, predicted_b, predicted_u


def estimate_global_threshold(
    layers: dict,
    order: list[str],
    grad_by_layer: dict[str, tuple[torch.Tensor, torch.Tensor]],
    cfg: GlobalFarRoundConfig,
) -> tuple[float, int, int, int]:
    """Estimate one model-wide threshold with a fixed log-score histogram."""
    candidate_count = 0
    weight_count = 0
    histogram = torch.zeros(cfg.histogram_bins, dtype=torch.int64)
    # A fixed range avoids retaining scores or making another full layer pass.
    # Values outside it saturate into end bins; the later raw-score threshold
    # still uses the selected bin boundary.
    log_min, log_max = -30.0, 30.0
    for name in order:
        weight_count += layers[name].weight.numel()
        state, q_near, q_far, valid, score, _, _ = _score_layer(layers[name], grad_by_layer[name], cfg)
        scored_valid = valid & torch.isfinite(score)
        values = score[scored_valid]
        candidate_count += int(values.numel())
        if values.numel():
            log_values = torch.log10(values.float().clamp_min(10.0**log_min)).clamp(log_min, log_max)
            histogram += torch.histc(log_values, bins=cfg.histogram_bins, min=log_min, max=log_max).to(
                device="cpu", dtype=torch.int64
            )
            del log_values
        del state, q_near, q_far, valid, scored_valid, score, values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    target = min(int(cfg.aggressive_fraction * weight_count), candidate_count)
    if target <= 0 or candidate_count == 0:
        return float("inf"), target, candidate_count, weight_count
    if target == candidate_count:
        return float("-inf"), target, candidate_count, weight_count

    descending = torch.flip(histogram, dims=[0]).cumsum(0)
    reverse_index = int(torch.searchsorted(descending, torch.tensor(target), right=False))
    bin_index = cfg.histogram_bins - 1 - reverse_index
    bin_width = (log_max - log_min) / cfg.histogram_bins
    # Exact zero scores are clamped into the first log bin while building the
    # histogram. Map that bin back to zero so applying the raw-score threshold
    # preserves the documented >= tie behavior.
    threshold = 0.0 if bin_index == 0 else 10.0 ** (log_min + bin_index * bin_width)
    return threshold, target, candidate_count, weight_count


def _projection_type(name: str) -> str:
    return name.rsplit(".", 1)[-1]


@torch.no_grad()
def run_global_far_round(
    layers: dict,
    order: list[str],
    grad_by_layer: dict[str, tuple[torch.Tensor, torch.Tensor]],
    cfg: GlobalFarRoundConfig,
    device: str,
) -> GlobalFarRoundResult:
    from aq.calibration_strategies import _commit
    from aq.metrics import cosine_similarity_flat, rounding_flip_ratio, weight_relative_distance

    threshold_start = time.time()
    threshold, target_count, candidate_count, global_weight_count = estimate_global_threshold(
        layers, order, grad_by_layer, cfg
    )
    threshold_time = time.time() - threshold_start
    results: dict[str, LayerOptimizationResult] = {}
    checksum = hashlib.sha256()
    selected_total = 0
    flipped_total = 0
    weight_total = 0
    projection_totals: dict[str, dict[str, int]] = {}
    quantization_start = time.time()

    for name in order:
        module = layers[name]
        w_fp = module.weight.detach().clone()
        state, q_near, q_far, valid, score, predicted_b, predicted_u = _score_layer(
            module, grad_by_layer[name], cfg
        )
        scored_valid = valid & torch.isfinite(score)
        select_mask = scored_valid & (score >= threshold)
        final_int = torch.where(select_mask, q_far, q_near)
        baseline_int = torch.round(state.pre_round).clamp(0, state.max_int)
        dequant = (final_int - state.zero_point) * state.scale
        if state.padded_in_features > state.in_features:
            dequant = dequant[:, : state.in_features]
        hard_weight = dequant.to(state.original_dtype)

        real_valid = scored_valid[:, : state.in_features]
        real_selected = select_mask[:, : state.in_features]
        real_score = score[:, : state.in_features][real_valid].float()
        real_pred_b = predicted_b[:, : state.in_features][real_valid].float().abs()
        real_pred_u = predicted_u[:, : state.in_features][real_valid].float().abs()
        real_flips = (final_int[:, : state.in_features] != baseline_int[:, : state.in_features])
        n_weights = w_fp.numel()
        n_valid = int(real_valid.sum())
        n_selected = int(real_selected.sum())
        n_flipped = int(real_flips.sum())
        quantiles = _score_quantiles(real_score)
        checksum.update(real_selected.to(device="cpu", dtype=torch.uint8).numpy().tobytes())

        metrics = {
            "layer": name,
            "projection_type": _projection_type(name),
            "num_weights": n_weights,
            "num_valid_candidates": n_valid,
            "num_selected": n_selected,
            "selected_fraction": n_selected / n_weights if n_weights else 0.0,
            "rounding_flip_ratio_vs_rtn4": n_flipped / n_weights if n_weights else 0.0,
            "score_mean": float(real_score.mean()) if n_valid else 0.0,
            "score_std": float(real_score.std(unbiased=False)) if n_valid else 0.0,
            "score_median": float(quantiles[0]),
            "score_p90": float(quantiles[1]),
            "score_p95": float(quantiles[2]),
            "score_p99": float(quantiles[3]),
            "score_max": float(real_score.max()) if n_valid else 0.0,
            "pred_behavior_delta_abs_mean": float(real_pred_b.mean()) if n_valid else 0.0,
            "pred_utility_delta_abs_mean": float(real_pred_u.mean()) if n_valid else 0.0,
            "weight_distance_vs_fp": float(weight_relative_distance(w_fp, hard_weight)),
            "cosine_similarity_vs_fp": float(cosine_similarity_flat(w_fp, hard_weight)),
            "scale_relative_shift": 0.0,
            "final_kl": None,
            "final_loss": None,
            "num_steps": 0,
        }
        _commit(module, hard_weight)
        # The committed model parameter is the only quantized weight copy we
        # need after this point. Retaining one CPU tensor per layer would add
        # roughly another full model (about 14 GB for LLaMA2-7B bf16).
        results[name] = LayerOptimizationResult(name, None, None, [], metrics)
        selected_total += n_selected
        flipped_total += n_flipped
        weight_total += n_weights
        projection = _projection_type(name)
        projection_totals.setdefault(
            projection,
            {"num_weights": 0, "num_valid_candidates": 0, "num_selected": 0},
        )
        projection_totals[projection]["num_weights"] += n_weights
        projection_totals[projection]["num_valid_candidates"] += n_valid
        projection_totals[projection]["num_selected"] += n_selected
        del (
            w_fp,
            state,
            q_near,
            q_far,
            valid,
            scored_valid,
            score,
            predicted_b,
            predicted_u,
            select_mask,
            final_int,
            baseline_int,
            dequant,
            hard_weight,
            real_valid,
            real_selected,
            real_score,
            real_pred_b,
            real_pred_u,
            real_flips,
            quantiles,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selection_by_projection = {
        projection: {
            **counts,
            "selected_fraction": (
                counts["num_selected"] / counts["num_weights"] if counts["num_weights"] else 0.0
            ),
        }
        for projection, counts in sorted(projection_totals.items())
    }
    global_metrics = {
        "global_num_weights": global_weight_count,
        "global_num_candidates": candidate_count,
        "global_num_selected": selected_total,
        "target_num_selected": target_count,
        "target_aggressive_fraction": cfg.aggressive_fraction,
        "actual_selected_fraction": selected_total / global_weight_count if global_weight_count else 0.0,
        "actual_rounding_flip_ratio": flipped_total / weight_total if weight_total else 0.0,
        "global_threshold": threshold,
        "selection_mask_checksum": checksum.hexdigest(),
        "selection_by_projection": selection_by_projection,
        "threshold_time": threshold_time,
        "quantization_time": time.time() - quantization_start,
    }
    return GlobalFarRoundResult(results, global_metrics)
