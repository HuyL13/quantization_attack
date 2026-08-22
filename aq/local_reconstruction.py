"""Methods 1B (layer-wise) and 1C (block-wise) local reconstruction - the
middle tiers between 1A's zero-training greedy flip and 1D's full-model KL.

Both reuse the SAME AdaRound-style rounding relaxation as 1D
(AdversarialLinearQuantizer / rounding_regularizer) and the SAME core
objective shape (behavior term - lambda * distance term + rounding
regularizer) - only the "behavior" term changes, from global KL(M_FP, M_Q)
down to a much cheaper LOCAL reconstruction error using activations cached
ONCE up front (aq.activation_cache), so no full-model forward pass ever
runs during optimization:

    1B: || X_l @ W_fp^T - X_l @ Q(W)^T ||^2   (X_l = that ONE layer's cached input)
    1C: || B_fp(X_b) - B_Q(X_b) ||^2          (X_b = the whole BLOCK's cached input,
                                                B(.) runs the block's own
                                                forward - attn + mlp - not the
                                                rest of the network)

Bias terms cancel in both reconstruction differences (Y_fp - Y_q depends
only on the weight difference), so bias is intentionally not added back in
either loss.
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from aq.activation_cache import _to_device
from aq.calibration_strategies import _commit, _release_quantizer
from aq.metrics import cosine_similarity_flat, rounding_flip_ratio, scale_relative_shift
from aq.optimizer_core import LayerOptimizationResult, _LayerForwardPatch
from aq.quantizer import AdversarialLinearQuantizer, AdversarialQuantConfig
from aq.rtn_backend import rtn_quantize_weight_raw


def run_layerwise_local_reconstruction(
    layers: dict,
    order: list[str],
    cached_layer_inputs: dict[str, list[torch.Tensor]],
    cfg: AdversarialQuantConfig,
    device: str,
) -> dict[str, LayerOptimizationResult]:
    results: dict[str, LayerOptimizationResult] = {}
    for name in order:
        module = layers[name]
        w_fp = module.weight.detach().clone()
        rtn_state = rtn_quantize_weight_raw(w_fp, bits=cfg.bits, group_size=cfg.group_size)
        quantizer = AdversarialLinearQuantizer(w_fp, rtn_state, cfg).to(device)

        hard_int_before = torch.round(quantizer.pre_round).clamp(0, quantizer.max_int).detach().clone()
        scale_before = quantizer.scale0.detach().clone()

        optim = torch.optim.Adam([p for p in quantizer.parameters() if p.requires_grad], lr=cfg.lr)
        all_inputs = cached_layer_inputs[name]

        trace_rows: list[dict] = []
        for step in range(cfg.steps):
            optim.zero_grad()

            distance_term = quantizer.weight_distance()
            round_reg = quantizer.rounding_regularizer()
            reg_loss = cfg.round_reg_weight * round_reg - cfg.lambda_distance * distance_term
            reg_loss.backward()

            step_inputs = random.sample(all_inputs, min(cfg.batches_per_step, len(all_inputs)))
            n_batches = max(len(step_inputs), 1)
            recon_accum = 0.0
            for x_cpu in step_inputs:
                x = x_cpu.to(device)
                soft_w = quantizer.soft_weight()
                y_fp = F.linear(x, w_fp.to(x.dtype))
                y_q = F.linear(x, soft_w.to(x.dtype))
                recon = F.mse_loss(y_q.float(), y_fp.float()) / n_batches
                recon.backward()
                recon_accum += float(recon.detach())

            optim.step()
            loss_value = recon_accum + float(reg_loss.detach())
            trace_rows.append(
                {
                    "layer": name,
                    "step": step,
                    "loss": loss_value,
                    "kl": recon_accum,  # "kl" column name kept for schema consistency with the heavier tiers
                    "distance": float(distance_term.detach()),
                    "round_reg": float(round_reg.detach()),
                }
            )

        hard_w = quantizer.hard_weight()
        hard_int_after = quantizer.hard_int_grid()
        layer_metrics = {
            "layer": name,
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
        result = LayerOptimizationResult(
            layer_name=name, quantizer=quantizer, hard_weight=hard_w, trace_rows=trace_rows, layer_metrics=layer_metrics
        )
        results[name] = result
        _release_quantizer(result)
        result.hard_weight = result.hard_weight.to("cpu")

    for name in order:
        _commit(layers[name], results[name].hard_weight.to(device))
    return results


def run_blockwise_local_reconstruction(
    block_groups: list[list[str]],
    layers: dict,
    block_modules_by_group: list[nn.Module],
    cached_block_inputs: list[list[dict]],
    cfg: AdversarialQuantConfig,
    device: str,
) -> dict[str, LayerOptimizationResult]:
    """`block_modules_by_group[i]` is the actual block nn.Module (e.g.
    model.model.layers[i]) whose forward is called directly - NOT the full
    model - so only that block's own compute graph (attn + mlp) is ever
    built, never the other 31 blocks.
    """
    results: dict[str, LayerOptimizationResult] = {}
    for block_names, block_module, cached_inputs in zip(block_groups, block_modules_by_group, cached_block_inputs):
        block_layer_modules = {name: layers[name] for name in block_names if name in layers}
        if not block_layer_modules:
            continue

        quantizers: dict[str, AdversarialLinearQuantizer] = {}
        w_fp_by_name: dict[str, torch.Tensor] = {}
        for name, module in block_layer_modules.items():
            w_fp = module.weight.detach().clone()
            w_fp_by_name[name] = w_fp
            rtn_state = rtn_quantize_weight_raw(w_fp, bits=cfg.bits, group_size=cfg.group_size)
            quantizers[name] = AdversarialLinearQuantizer(w_fp, rtn_state, cfg).to(device)

        optim = torch.optim.Adam(
            [p for q in quantizers.values() for p in q.parameters() if p.requires_grad], lr=cfg.lr
        )

        trace_by_layer: dict[str, list[dict]] = {name: [] for name in block_layer_modules}
        for step in range(cfg.steps):
            optim.zero_grad()

            distance_terms = [q.weight_distance() for q in quantizers.values()]
            reg_terms = [q.rounding_regularizer() for q in quantizers.values()]
            reg_loss = cfg.round_reg_weight * torch.stack(reg_terms).mean() - cfg.lambda_distance * torch.stack(
                distance_terms
            ).mean()
            reg_loss.backward()

            step_inputs = random.sample(cached_inputs, min(cfg.batches_per_step, len(cached_inputs)))
            n_batches = max(len(step_inputs), 1)
            recon_accum = 0.0
            for cached_call in step_inputs:
                # Replay the block's FULL original call signature (not just
                # hidden_states) - a real transformer block needs rotary
                # position_embeddings/attention_mask/etc. that its parent
                # model computes once and passes to every block; calling the
                # block with only its input tensor crashes deep inside
                # self_attn (verified live: "cannot unpack non-iterable
                # NoneType object" trying to unpack a None
                # position_embeddings). See aq.activation_cache._to_device.
                call_args = _to_device(cached_call["args"], device)
                call_kwargs = _to_device(cached_call["kwargs"], device)
                with torch.no_grad():
                    y_fp = block_module(*call_args, **call_kwargs)
                    y_fp = y_fp[0] if isinstance(y_fp, tuple) else y_fp

                soft_weights = {name: q.soft_weight() for name, q in quantizers.items()}
                patches = [_LayerForwardPatch(block_layer_modules[name], (lambda w=w: w)) for name, w in soft_weights.items()]
                for p in patches:
                    p.__enter__()
                try:
                    y_q = block_module(*call_args, **call_kwargs)
                    y_q = y_q[0] if isinstance(y_q, tuple) else y_q
                finally:
                    for p in patches:
                        p.__exit__(None, None, None)

                recon = F.mse_loss(y_q.float(), y_fp.float()) / n_batches
                recon.backward()
                recon_accum += float(recon.detach())

            optim.step()
            if torch.cuda.is_available() and step % 5 == 0:
                # Each step moves several cached CPU tensors (hidden_states,
                # attention_mask, position_embeddings, ...) to GPU fresh via
                # _to_device and lets them go out of scope - thousands of
                # these small alloc/free cycles across 32 blocks x steps x
                # batches_per_step measurably fragmented the CUDA caching
                # allocator enough to OOM a 40GB A100 well before total
                # logical usage should have required it (crashed with 39.32
                # GiB "in use" but only ~172 MiB actually requested at the
                # failing allocation). Periodic empty_cache() defragments.
                torch.cuda.empty_cache()
            loss_value = recon_accum + float(reg_loss.detach())
            for name in block_layer_modules:
                trace_by_layer[name].append(
                    {"layer": name, "step": step, "loss": loss_value, "kl": recon_accum,
                     "distance": float(quantizers[name].weight_distance().detach())}
                )

        for name, module in block_layer_modules.items():
            hard_w = quantizers[name].hard_weight()
            hard_int_before = torch.round(quantizers[name].pre_round).clamp(0, quantizers[name].max_int)
            hard_int_after = quantizers[name].hard_int_grid()
            layer_metrics = {
                "layer": name,
                "weight_distance_vs_fp": float(quantizers[name].weight_distance().detach()),
                "cosine_similarity_vs_fp": float(cosine_similarity_flat(w_fp_by_name[name], hard_w)),
                "rounding_flip_ratio_vs_rtn4": float(rounding_flip_ratio(hard_int_before, hard_int_after)),
                "scale_relative_shift": 0.0,
                "final_kl": trace_by_layer[name][-1]["kl"] if trace_by_layer[name] else None,
                "final_loss": trace_by_layer[name][-1]["loss"] if trace_by_layer[name] else None,
                "num_steps": cfg.steps,
            }
            result = LayerOptimizationResult(
                layer_name=name, quantizer=quantizers[name], hard_weight=hard_w,
                trace_rows=trace_by_layer[name], layer_metrics=layer_metrics,
            )
            results[name] = result
            _commit(module, hard_w)
            _release_quantizer(result)
            # hard_w is already written into module.weight.data by _commit -
            # keeping the separate GPU copy in result.hard_weight for every
            # processed layer across all 32 blocks duplicates a growing
            # fraction of the whole model's weights a second time over on
            # GPU (up to ~13.5GB by the last block) - measured live: this
            # was enough to push a 40GB A100 into OOM mid-run even after
            # fixing per-layer quantizer buffers and adding periodic
            # empty_cache(). Stage it on CPU; nothing after this point reads
            # it back except aggregate reporting.
            result.hard_weight = result.hard_weight.to("cpu")
        quantizers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results
