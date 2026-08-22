"""Methods 5-8: calibration/orchestration strategies wrapping optimize_layer.

None of these introduce a new loss - each only changes WHICH state `model`
is in (and how many layers are jointly patched) while optimize_layer runs.
This mirrors the plan's repeated instruction that all 9 methods share one
core objective L = KL(M_FP, M_Q) - lambda * D(W, Q(W)).

`fp_reference_logits` MUST be computed once, before any weight in `model`
is mutated, and passed into every function below unchanged - it is the
fixed KL target M_FP. Recomputing it after a strategy has already committed
some layers would silently make the "reference" drift away from the true
FP model (caught in review before this ever touched a real model: the
two-pass strategy originally recomputed it after the forward pass had
already overwritten some weights).

Strategy summary (increasing calibration fidelity / cost):
  isolated            (methods 1-4, default): every target layer optimized
                       against an all-FP `model`; nothing committed until
                       every layer is done, then all hard weights are
                       written in one final commit.
  quantized_prefix     (method 5): layers processed in depth order; each
                       layer's hard weight is committed into `model` before
                       the next layer starts, so later layers see the real
                       (already-quantized) upstream activations instead of
                       pristine FP ones.
  periodic_refresh     (method 7): a middle ground between the two above -
                       layers within a group of `refresh_every_k` blocks are
                       optimized against a single frozen snapshot of `model`
                       (isolated within the group), then the whole group is
                       committed at once before the next group's snapshot is
                       taken. k=1 degenerates to quantized_prefix; k=inf
                       degenerates to isolated.
  block_wise           (method 6): every linear layer inside one transformer
                       block is optimized JOINTLY (one shared loss summing
                       each layer's KL contribution through a single combined
                       forward pass and averaging their distance terms)
                       instead of matrix-by-matrix, then the whole block is
                       committed before the next block.
  two_pass_backward    (method 8): a forward quantized-prefix pass (as
                       above), followed by a backward pass from the last
                       block to the first that re-optimizes each block's
                       variables a second time with full knowledge of every
                       other block's (already quantized) state.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from aq.optimizer_core import (
    _LayerForwardPatch,
    optimize_layer,
    LayerOptimizationResult,
)
from aq.metrics import cosine_similarity_flat, kl_divergence_logits, rounding_flip_ratio
from aq.quantizer import AdversarialLinearQuantizer, AdversarialQuantConfig
from aq.rtn_backend import rtn_quantize_weight_raw


def _commit(module: nn.Linear, hard_weight: torch.Tensor) -> None:
    with torch.no_grad():
        module.weight.data.copy_(hard_weight.to(module.weight.dtype))


def run_isolated(
    model,
    layers: dict,
    order: list[str],
    calibration_batches,
    fp_reference_logits: list[torch.Tensor],
    cfg: AdversarialQuantConfig,
    device: str,
) -> dict[str, LayerOptimizationResult]:
    results: dict[str, LayerOptimizationResult] = {}
    for name in order:
        module = layers[name]
        results[name] = optimize_layer(model, module, name, calibration_batches, fp_reference_logits, cfg, device)
    for name in order:
        _commit(layers[name], results[name].hard_weight)
    return results


def run_quantized_prefix(
    model,
    layers: dict,
    order: list[str],
    calibration_batches,
    fp_reference_logits: list[torch.Tensor],
    cfg: AdversarialQuantConfig,
    device: str,
) -> dict[str, LayerOptimizationResult]:
    results: dict[str, LayerOptimizationResult] = {}
    for name in order:
        module = layers[name]
        result = optimize_layer(model, module, name, calibration_batches, fp_reference_logits, cfg, device)
        results[name] = result
        _commit(module, result.hard_weight)  # bake into `model` before the next layer sees it
    return results


def run_periodic_refresh(
    model,
    layers: dict,
    order: list[str],
    calibration_batches,
    fp_reference_logits: list[torch.Tensor],
    cfg: AdversarialQuantConfig,
    device: str,
    refresh_every_k_blocks: int,
    block_groups: list[list[str]],
) -> dict[str, LayerOptimizationResult]:
    results: dict[str, LayerOptimizationResult] = {}
    for start in range(0, len(block_groups), refresh_every_k_blocks):
        group_blocks = block_groups[start : start + refresh_every_k_blocks]
        group_layer_names = [name for block in group_blocks for name in block if name in layers]
        for name in group_layer_names:
            module = layers[name]
            results[name] = optimize_layer(
                model, module, name, calibration_batches, fp_reference_logits, cfg, device
            )
        for name in group_layer_names:
            _commit(layers[name], results[name].hard_weight)
    return results


def run_block_wise(
    model,
    layers: dict,
    block_groups: list[list[str]],
    calibration_batches,
    fp_reference_logits: list[torch.Tensor],
    cfg: AdversarialQuantConfig,
    device: str,
) -> dict[str, LayerOptimizationResult]:
    results: dict[str, LayerOptimizationResult] = {}
    for block_names in block_groups:
        block_modules = {name: layers[name] for name in block_names if name in layers}
        if not block_modules:
            continue
        quantizers: dict[str, AdversarialLinearQuantizer] = {}
        for name, module in block_modules.items():
            w_fp = module.weight.detach().clone()
            rtn_state = rtn_quantize_weight_raw(w_fp, bits=cfg.bits, group_size=cfg.group_size)
            quantizers[name] = AdversarialLinearQuantizer(w_fp, rtn_state, cfg).to(device)

        trainable = [p for q in quantizers.values() for p in q.parameters() if p.requires_grad]
        optim = torch.optim.Adam(trainable, lr=cfg.lr)

        trace_by_layer: dict[str, list[dict]] = {name: [] for name in block_modules}
        for step in range(cfg.steps):
            optim.zero_grad()
            soft_weights = {name: q.soft_weight() for name, q in quantizers.items()}
            patches = [
                _LayerForwardPatch(block_modules[name], (lambda w=w: w)) for name, w in soft_weights.items()
            ]
            kl_sum = torch.zeros((), device=device)
            for patch in patches:
                patch.__enter__()
            try:
                for ref_logits, batch in zip(fp_reference_logits, calibration_batches):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    out_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    kl_sum = kl_sum + kl_divergence_logits(ref_logits, out_logits)
            finally:
                for patch in patches:
                    patch.__exit__(None, None, None)
            kl_term = kl_sum / max(len(calibration_batches), 1)
            distance_terms = [q.weight_distance() for q in quantizers.values()]
            mean_distance = torch.stack(distance_terms).mean()
            reg_terms = [q.rounding_regularizer() for q in quantizers.values()]
            mean_reg = torch.stack(reg_terms).mean()
            loss = kl_term - cfg.lambda_distance * mean_distance + cfg.round_reg_weight * mean_reg

            loss.backward()
            optim.step()

            for name in block_modules:
                trace_by_layer[name].append(
                    {
                        "layer": name,
                        "step": step,
                        "loss": float(loss.detach()),
                        "kl": float(kl_term.detach()),
                        "distance": float(quantizers[name].weight_distance().detach()),
                    }
                )

        for name, module in block_modules.items():
            hard_w = quantizers[name].hard_weight()
            hard_int_before = torch.round(quantizers[name].pre_round).clamp(0, quantizers[name].max_int)
            hard_int_after = quantizers[name].hard_int_grid()
            layer_metrics = {
                "layer": name,
                "weight_distance_vs_fp": float(quantizers[name].weight_distance().detach()),
                "cosine_similarity_vs_fp": float(cosine_similarity_flat(quantizers[name].w_fp, hard_w)),
                "rounding_flip_ratio_vs_rtn4": float(rounding_flip_ratio(hard_int_before, hard_int_after)),
                "scale_relative_shift": 0.0,
                "final_kl": trace_by_layer[name][-1]["kl"] if trace_by_layer[name] else None,
                "final_loss": trace_by_layer[name][-1]["loss"] if trace_by_layer[name] else None,
                "num_steps": cfg.steps,
            }
            results[name] = LayerOptimizationResult(
                layer_name=name,
                quantizer=quantizers[name],
                hard_weight=hard_w,
                trace_rows=trace_by_layer[name],
                layer_metrics=layer_metrics,
            )
            _commit(module, hard_w)

    return results


def run_two_pass_backward_correction(
    model,
    layers: dict,
    order: list[str],
    calibration_batches,
    fp_reference_logits: list[torch.Tensor],
    cfg: AdversarialQuantConfig,
    device: str,
) -> dict[str, LayerOptimizationResult]:
    forward_results = run_quantized_prefix(
        model, layers, order, calibration_batches, fp_reference_logits, cfg, device
    )

    backward_order = list(reversed(order))
    backward_results: dict[str, LayerOptimizationResult] = {}
    for name in backward_order:
        module = layers[name]
        result = optimize_layer(model, module, name, calibration_batches, fp_reference_logits, cfg, device)
        backward_results[name] = result
        _commit(module, result.hard_weight)

    merged = dict(forward_results)
    merged.update(backward_results)
    for name in order:
        merged[name].trace_rows = forward_results[name].trace_rows + backward_results[name].trace_rows
        merged[name].layer_metrics["backward_pass_final_kl"] = backward_results[name].layer_metrics["final_kl"]
        merged[name].layer_metrics["forward_pass_final_kl"] = forward_results[name].layer_metrics["final_kl"]
    return merged
