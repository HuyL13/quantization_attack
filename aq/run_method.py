"""CLI entrypoint: run ONE method of the adversarial quantization plan against
the IF-SFT LLaMA2-7B checkpoint, gate it, and write every artifact plan
section 13 requires. Intended to be called repeatedly (00_rtn4, 01_..., ...)
by scripts/run_all_methods.sh, which stops the loop as soon as one method
returns PASS.

Reuses if_awq_tier0's WikiText-2 PPL evaluator and IF-SFT watermark
verification verbatim - this file only wires them into the adversarial
quantization + gating pipeline.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from aq.common import (
    FINGERPRINT_TARGET,
    IF_SFT_MODEL_ID,
    configure_gpu_performance,
    default_calibration_path,
    default_fingerprint_keys_path,
    ensure_dir,
    ensure_if_awq_tier0_on_path,
    free_model,
    get_block_layer_groups,
    get_block_modules,
    get_transformer_linear_layers,
    gpu_peak_memory_bytes,
    load_causal_lm,
    load_tokenizer,
    load_yaml_config,
    read_json,
    write_json,
)
from aq.calibration import build_calibration_batches, load_calibration_texts
from aq.calibration_strategies import (
    run_block_wise,
    run_isolated,
    run_periodic_refresh,
    run_quantized_prefix,
    run_two_pass_backward_correction,
)
from aq.activation_cache import (
    capture_block_input_activations,
    capture_layer_input_activations,
    compute_layer_activation_sensitivity,
)
from aq.behavior_gradient import compute_behavior_and_utility_gradients
from aq.decision_flow import GateConfig, MethodOutcome, STATUS_PASS, decide_next_action, evaluate_watermark_gate
from aq.greedy_rounding import GreedyRoundingConfig, run_greedy_adversarial_rounding
from aq.local_reconstruction import run_blockwise_local_reconstruction, run_layerwise_local_reconstruction
from aq.selective_quantization import SelectiveQuantConfig, run_selective_quantization
from aq.stochastic_rounding import StochasticRoundingConfig, run_stochastic_rounding
from aq.logging_utils import (
    get_run_logger,
    run_dir_for,
    write_config_yaml,
    write_layer_metrics_csv,
    write_model_metrics_json,
    write_optimization_trace_csv,
    write_ppl_result_json,
    write_watermark_result_json,
)
from aq.optimizer_core import compute_fp_reference_logits
from aq.plotting import plot_kl_vs_distance_trace, plot_rounding_flip_ratio_per_layer
from aq.quantizer import AdversarialQuantConfig
from aq.reporting import generate_method_report
from aq.rtn_backend import rtn_quantize_weight_raw

# Methods 1A/1B/1C/04 never run the full 32-layer model during optimization
# (1A and 04 are gradient-free; 1B/1C optimize against cached LOCAL
# activations) - so they skip gradient checkpointing / model.train()
# (nothing to checkpoint) and the expensive full-model fp_reference_logits
# cache (no global KL term at all).
LOCAL_METHODS = {"01a_greedy_round", "01b_layerwise_local", "01c_blockwise_local", "04_stochastic_rounding"}

# 02/03 need a full-model backward (to score every weight's predicted
# behavior/utility impact) but - unlike 1D/05-08 - only a HANDFUL of
# calibration batches, once, with no iterative optimization loop at all.
# They still need gradient checkpointing (a full-depth backward is a
# full-depth backward regardless of whether it repeats), so they're full-
# model methods for that purpose, but they use their own dedicated code
# path (aq.selective_quantization), not the old iterative _run_strategy.
GRADIENT_SCORING_METHODS = {"02_margin_aware", "03_fragile_channel"}


def _apply_rtn4_baseline(layers: dict) -> None:
    import torch

    for name, module in layers.items():
        w = module.weight.detach().clone()
        state = rtn_quantize_weight_raw(w, bits=4, group_size=128)
        with torch.no_grad():
            module.weight.data.copy_(state.dequantize_truncated().to(module.weight.dtype))


def _run_strategy(method_id: str, model, layers: dict, order, block_groups, calibration_batches, fp_ref, cfg, device):
    if method_id == "01d_global_kl":
        return run_isolated(model, layers, order, calibration_batches, fp_ref, cfg, device)
    if method_id == "05_quantized_prefix":
        return run_quantized_prefix(model, layers, order, calibration_batches, fp_ref, cfg, device)
    if method_id == "06_block_wise":
        return run_block_wise(model, layers, block_groups, calibration_batches, fp_ref, cfg, device)
    if method_id == "07_periodic_refresh":
        refresh_k = getattr(cfg, "refresh_every_k_blocks", 8)
        return run_periodic_refresh(
            model, layers, order, calibration_batches, fp_ref, cfg, device, refresh_k, block_groups
        )
    if method_id == "08_two_pass_backward":
        return run_two_pass_backward_correction(model, layers, order, calibration_batches, fp_ref, cfg, device)
    raise ValueError(f"unknown method_id {method_id}")


def run_one_method(
    method_id: str,
    model_cfg: dict,
    method_cfg: dict,
    rtn4_ppl: float,
    out_root: Path,
    device: str = "cuda",
    dtype: str = "bfloat16",
    force_watermark_eval: bool = False,
) -> MethodOutcome:
    ensure_if_awq_tier0_on_path()
    from src.eval_wikitext import compute_wikitext2_ppl
    from src.verify_fingerprint import run_verification, summarize

    run_dir = run_dir_for(out_root, method_id)
    logger = get_run_logger(run_dir, method_id)
    write_config_yaml(run_dir, {"method_id": method_id, "model": model_cfg, "method": method_cfg})

    model_id = model_cfg.get("id", IF_SFT_MODEL_ID)
    logger.info("loading model %s", model_id)
    model = load_causal_lm(model_id, device=device, dtype=dtype)
    tokenizer = load_tokenizer(model_id)

    is_full_model_method = method_id != "00_rtn4" and method_id not in LOCAL_METHODS
    if is_full_model_method:
        # Every full-model method backprops from the loss through however
        # many of the model's 32 decoder blocks sit between the patched
        # layer and the output - for a layer near the start that's a
        # full-depth backward pass. Storing every intermediate activation
        # for that (the default) OOM'd a 40GB A100 within ~20-30s even with
        # calib_batch_size=4/seq_len=512 and only one layer being trained at
        # a time - not from batch size, but from activation memory scaling
        # with the full 32-layer depth. Gradient checkpointing recomputes
        # activations during backward instead of storing them, trading
        # ~20-30% more compute for an order-of-magnitude memory cut.
        # Methods 1A-1C never run the full model during optimization (see
        # LOCAL_METHODS), so none of this applies to them.
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    layers = get_transformer_linear_layers(model)
    order = list(layers.keys())
    block_groups = get_block_layer_groups(model)

    t0 = time.time()
    all_layer_metrics: list[dict] = []
    all_trace_rows: list[dict] = []

    if method_id == "00_rtn4":
        logger.info("applying RTN4 baseline to %d layers", len(layers))
        _apply_rtn4_baseline(layers)
    elif method_id in LOCAL_METHODS:
        calibration_texts = load_calibration_texts(method_cfg.get("calibration_path", default_calibration_path()))
        if method_id in ("01b_layerwise_local", "01c_blockwise_local"):
            # Unlike 1A (streaming, no raw storage - see
            # compute_layer_activation_sensitivity), 1B/1C genuinely cache
            # every calibration batch's raw activation tensor for EVERY
            # target layer/block simultaneously (needed for reconstruction).
            # At the full 128-sample calibration set this measured out to
            # >130GB of resident CPU memory and climbing for 1A before the
            # streaming fix - 1B/1C would be worse (real tensors, not a
            # running sum). Capping the sample count keeps this bounded;
            # a few dozen samples is already a standard calibration size for
            # this kind of local reconstruction (GPTQ/AWQ-style methods
            # commonly use O(100) samples over much longer sequences than
            # this cuts down to).
            calibration_texts = calibration_texts[: method_cfg.get("calibration_samples", 32)]
        calibration_batches = build_calibration_batches(
            tokenizer,
            calibration_texts,
            max_seq_len=method_cfg.get("max_seq_len", 512),
            batch_size=method_cfg.get("calib_batch_size", 4),
            device=device,
        )

        if method_id == "01a_greedy_round":
            logger.info(
                "computing streaming activation sensitivity over %d calibration batches", len(calibration_batches)
            )
            sensitivity_by_layer = compute_layer_activation_sensitivity(model, layers, calibration_batches, device)
            greedy_cfg = GreedyRoundingConfig(
                bits=method_cfg.get("bits", 4),
                group_size=method_cfg.get("group_size", 128),
                flip_fraction=method_cfg.get("flip_fraction", 0.2),
            )
            logger.info("running greedy adversarial rounding (1A) over %d target layers", len(layers))
            results = run_greedy_adversarial_rounding(model, layers, order, sensitivity_by_layer, greedy_cfg, device)
        elif method_id == "01b_layerwise_local":
            logger.info("capturing per-layer input activations over %d calibration batches", len(calibration_batches))
            cached_layer_inputs = capture_layer_input_activations(model, layers, calibration_batches, device)
            cfg = AdversarialQuantConfig(
                bits=method_cfg.get("bits", 4),
                group_size=method_cfg.get("group_size", 128),
                lambda_distance=method_cfg.get("lambda_distance", 0.1),
                round_reg_weight=method_cfg.get("round_reg_weight", 1.0),
                steps=method_cfg.get("steps", 50),
                lr=method_cfg.get("lr", 1e-2),
                batches_per_step=method_cfg.get("batches_per_step", 4),
            )
            logger.info("running layer-wise local reconstruction (1B) over %d target layers", len(layers))
            results = run_layerwise_local_reconstruction(layers, order, cached_layer_inputs, cfg, device)
        elif method_id == "01c_blockwise_local":
            block_modules = get_block_modules(model)
            block_module_map = {str(i): m for i, m in enumerate(block_modules)}
            logger.info("capturing per-block input activations over %d calibration batches", len(calibration_batches))
            cached_block_inputs_by_idx = capture_block_input_activations(
                model, block_module_map, calibration_batches, device
            )
            cached_block_inputs = [cached_block_inputs_by_idx[str(i)] for i in range(len(block_modules))]
            cfg = AdversarialQuantConfig(
                bits=method_cfg.get("bits", 4),
                group_size=method_cfg.get("group_size", 128),
                lambda_distance=method_cfg.get("lambda_distance", 0.1),
                round_reg_weight=method_cfg.get("round_reg_weight", 1.0),
                steps=method_cfg.get("steps", 30),
                lr=method_cfg.get("lr", 1e-2),
                batches_per_step=method_cfg.get("batches_per_step", 4),
            )
            logger.info("running block-wise local reconstruction (1C) over %d blocks", len(block_groups))
            results = run_blockwise_local_reconstruction(
                block_groups, layers, block_modules, cached_block_inputs, cfg, device
            )
        else:  # 04_stochastic_rounding - no backprop, no repeated forward at all
            stoch_cfg = StochasticRoundingConfig(
                bits=method_cfg.get("bits", 4),
                group_size=method_cfg.get("group_size", 128),
                variant=method_cfg.get("variant", "far_biased"),
                far_bias=method_cfg.get("far_bias", 0.3),
                seed=method_cfg.get("seed", 0),
            )
            sensitivity_by_layer = None
            if stoch_cfg.variant == "fragility_weighted":
                logger.info(
                    "computing streaming activation sensitivity over %d calibration batches", len(calibration_batches)
                )
                sensitivity_by_layer = compute_layer_activation_sensitivity(model, layers, calibration_batches, device)
            logger.info(
                "running stochastic/biased rounding (method C, variant=%s) over %d target layers",
                stoch_cfg.variant,
                len(layers),
            )
            results = run_stochastic_rounding(layers, order, sensitivity_by_layer, stoch_cfg, device)

        for name in order:
            if name in results:
                all_layer_metrics.append(results[name].layer_metrics)
                all_trace_rows.extend(results[name].trace_rows)
    elif method_id in GRADIENT_SCORING_METHODS:
        calibration_texts = load_calibration_texts(method_cfg.get("calibration_path", default_calibration_path()))
        # A full-model backward pass needs the same memory care as the old
        # heavy tier's per-step forward - cap the sample count for the same
        # reason 1B/1C do (a handful of calibration batches is standard for
        # this kind of one-shot gradient-based scoring, e.g. OBD/OBS-style
        # saliency estimation).
        calibration_texts = calibration_texts[: method_cfg.get("calibration_samples", 16)]
        calibration_batches = build_calibration_batches(
            tokenizer,
            calibration_texts,
            max_seq_len=method_cfg.get("max_seq_len", 512),
            batch_size=method_cfg.get("calib_batch_size", 4),
            device=device,
        )
        behavior = "margin" if method_id == "02_margin_aware" else "top1_logprob"
        model.train()
        logger.info(
            "computing %s/utility gradients over %d calibration batches", behavior, len(calibration_batches)
        )
        grad_by_layer = compute_behavior_and_utility_gradients(model, layers, calibration_batches, device, behavior)
        model.eval()

        select_cfg = SelectiveQuantConfig(
            bits=method_cfg.get("bits", 4),
            group_size=method_cfg.get("group_size", 128),
            aggressive_fraction=method_cfg.get("aggressive_fraction", 0.05),
        )
        logger.info("running selective quantization (%s) over %d target layers", method_id, len(layers))
        results = run_selective_quantization(layers, order, grad_by_layer, select_cfg, device)
        for name in order:
            if name in results:
                all_layer_metrics.append(results[name].layer_metrics)
                all_trace_rows.extend(results[name].trace_rows)
    else:
        cfg = AdversarialQuantConfig(
            bits=method_cfg.get("bits", 4),
            group_size=method_cfg.get("group_size", 128),
            optimize_scale=method_cfg.get("optimize_scale", False),
            use_codebook=method_cfg.get("use_codebook", False),
            n_codebook_centroids=method_cfg.get("n_codebook_centroids", 16),
            use_sensitivity=method_cfg.get("use_sensitivity", False),
            lambda_distance=method_cfg.get("lambda_distance", 0.1),
            round_reg_weight=method_cfg.get("round_reg_weight", 1.0),
            steps=method_cfg.get("steps", 200),
            lr=method_cfg.get("lr", 1e-2),
            batches_per_step=method_cfg.get("batches_per_step", 4),
        )
        cfg.refresh_every_k_blocks = method_cfg.get("refresh_every_k_blocks", 8)

        calibration_texts = load_calibration_texts(method_cfg.get("calibration_path", default_calibration_path()))
        calibration_batches = build_calibration_batches(
            tokenizer,
            calibration_texts,
            max_seq_len=method_cfg.get("max_seq_len", 512),
            batch_size=method_cfg.get("calib_batch_size", 4),
            device=device,
        )
        logger.info("computing FP reference logits over %d calibration batches", len(calibration_batches))
        fp_reference_logits = compute_fp_reference_logits(model, calibration_batches, device)

        # Gradient checkpointing (enabled above) only actually recomputes
        # activations - instead of just running a plain forward with nothing
        # to save memory on - when the model is in train() mode; several
        # transformers versions gate the recomputation behind self.training.
        # Llama's own config has no dropout by default, so train() vs eval()
        # changes nothing about the forward math here, only whether
        # checkpointing engages.
        model.train()
        logger.info("running strategy for %s over %d target layers", method_id, len(layers))
        results = _run_strategy(
            method_id, model, layers, order, block_groups, calibration_batches, fp_reference_logits, cfg, device
        )
        model.eval()
        for name in order:
            if name in results:
                all_layer_metrics.append(results[name].layer_metrics)
                all_trace_rows.extend(results[name].trace_rows)

    elapsed = time.time() - t0
    logger.info("quantization pass finished in %.1fs", elapsed)

    write_layer_metrics_csv(run_dir, all_layer_metrics)
    write_optimization_trace_csv(run_dir, all_trace_rows)
    if all_trace_rows:
        plot_kl_vs_distance_trace(all_trace_rows, run_dir / "kl_vs_distance.png", title=method_id)
    if all_layer_metrics:
        plot_rounding_flip_ratio_per_layer(all_layer_metrics, run_dir / "rounding_flip_ratio.png", title=method_id)

    logger.info("computing WikiText-2 PPL")
    ppl_result = compute_wikitext2_ppl(model, tokenizer, device=device)
    write_ppl_result_json(run_dir, ppl_result)
    candidate_ppl = ppl_result["wikitext2_ppl"]

    gate_cfg = GateConfig(
        max_ppl_relative_regression=method_cfg.get("max_ppl_relative_regression", 0.05),
        max_fsr_exact_for_pass=method_cfg.get("max_fsr_exact_for_pass", 0.0),
    )

    if method_id == "00_rtn4":
        # RTN4 is the fixed reference point, never gated against itself.
        outcome = MethodOutcome(method_id=method_id, status="BASELINE", ppl=candidate_ppl, rtn4_ppl=candidate_ppl)
    else:
        pre_gate = decide_next_action(method_id, candidate_ppl, rtn4_ppl, gate_cfg)
        if pre_gate.status != "SKIPPED" and not force_watermark_eval:
            outcome = pre_gate
            logger.info("PPL gate: %s -> stopping before watermark eval", outcome.detail)
        else:
            if pre_gate.status != "SKIPPED":
                logger.info(
                    "PPL gate: %s -> FAILED, but --force-watermark-eval was set: running watermark eval anyway "
                    "for diagnostic purposes only (this outcome is NOT a real PASS candidate)",
                    pre_gate.detail,
                )
            fingerprints = read_json(method_cfg.get("fingerprint_keys_path", default_fingerprint_keys_path()))
            logger.info("running watermark verification (%d keys)", len(fingerprints))
            t1 = time.time()
            per_key = run_verification(
                model,
                tokenizer,
                fingerprints,
                do_sample=False,
                max_new_tokens=method_cfg.get("max_new_tokens", 32),
                device=device,
            )
            watermark_summary = summarize(per_key, evaluation_time_seconds=time.time() - t1)
            write_watermark_result_json(run_dir, {"summary": watermark_summary, "per_key": per_key})
            watermark_gone, watermark_detail = evaluate_watermark_gate(watermark_summary["fsr_exact"], gate_cfg)
            if pre_gate.status == "SKIPPED":
                # Normal path: PPL gate passed, watermark result is the real gate.
                outcome = decide_next_action(
                    method_id,
                    candidate_ppl,
                    rtn4_ppl,
                    gate_cfg,
                    watermark_fsr_exact=watermark_summary["fsr_exact"],
                    watermark_fsr_contains=watermark_summary["fsr_contains"],
                )
            else:
                # force_watermark_eval diagnostic path: PPL gate already
                # FAILED, so this can never be a real PASS - keep the
                # original FAIL_UTILITY status but attach the watermark
                # numbers for inspection only.
                outcome = pre_gate
                outcome.watermark_fsr_exact = watermark_summary["fsr_exact"]
                outcome.watermark_fsr_contains = watermark_summary["fsr_contains"]
                outcome.detail = pre_gate.detail + " | [diagnostic only, PPL already failed] " + watermark_detail
            logger.info("watermark gate: %s", outcome.detail)

    model_metrics = {
        "method_id": method_id,
        "ppl": candidate_ppl,
        "peak_gpu_memory_bytes": gpu_peak_memory_bytes(),
        "elapsed_seconds": elapsed,
        "num_layers_touched": len(layers) if method_id != "00_rtn4" else len(layers),
    }
    outcome.model_metrics = model_metrics
    write_model_metrics_json(run_dir, model_metrics)
    generate_method_report(outcome, run_dir, out_root / "reports" / f"{method_id}_report.md")

    free_model(model)
    del model
    return outcome


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True, choices=[
        "00_rtn4", "01a_greedy_round", "01b_layerwise_local", "01c_blockwise_local", "01d_global_kl",
        "02_margin_aware", "03_fragile_channel", "04_stochastic_rounding",
        "05_quantized_prefix", "06_block_wise", "07_periodic_refresh", "08_two_pass_backward",
    ])
    ap.add_argument("--config", required=True, help="YAML with model/method sections")
    ap.add_argument("--rtn4-ppl", type=float, default=None, help="Baseline PPL from the 00_rtn4 run (required for methods 01-08)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument(
        "--force-watermark-eval",
        action="store_true",
        help="Run watermark verification even if the PPL gate fails - diagnostic only, "
        "the reported status still stays FAIL_UTILITY (never becomes PASS).",
    )
    args = ap.parse_args()

    configure_gpu_performance()
    cfg = load_yaml_config(args.config)
    out_root = ensure_dir(Path(args.output))

    rtn4_ppl = args.rtn4_ppl
    if rtn4_ppl is None:
        if args.method == "00_rtn4":
            rtn4_ppl = float("nan")
        else:
            baseline_path = out_root / "runs" / "00_rtn4" / "ppl_result.json"
            if not baseline_path.exists():
                raise SystemExit(
                    f"--rtn4-ppl not given and {baseline_path} does not exist - run 00_rtn4 first."
                )
            rtn4_ppl = json.loads(baseline_path.read_text(encoding="utf-8"))["wikitext2_ppl"]

    outcome = run_one_method(
        args.method,
        cfg.get("model", {}),
        cfg.get("method", {}),
        rtn4_ppl,
        out_root,
        device=args.device,
        dtype=args.dtype,
        force_watermark_eval=args.force_watermark_eval,
    )
    print(f"[run_method] {args.method} -> {outcome.status} ({outcome.detail})")
    if outcome.status == STATUS_PASS:
        print("[run_method] WATERMARK REMOVED WITH ACCEPTABLE PPL - stop testing further methods.")


if __name__ == "__main__":
    main()
