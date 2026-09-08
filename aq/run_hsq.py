"""CLI entrypoint for HSQ (Hessian-Slack Quantization) - see
HSQ_Hessian_Slack_Quantization_Implementation_Guide.md. Unlike the other
methods in this repo (each layer treated independently against the FP
model's own calibration signal), HSQ is GPTQ-style SEQUENTIAL: blocks are
quantized in depth order and each block's calibration activations are
captured by running the calibration batches through the model AS IT
CURRENTLY STANDS (earlier blocks already quantized in-place) - the same
propagation mechanism GPTQ's own llama.py driver relies on, obtained here
for free by reusing aq.activation_cache.capture_layer_input_activations
against the live (partially-quantized) model.

Three `--method` values share this exact same driver, differing only in
which candidate `aq.hsq_core.run_hsq_layer` commits per coordinate:
  gptq4    - plain GPTQ (bits=4 by convention) - the real baseline AND the
             mandatory sanity-check target (guide section 20A: HSQ with
             tau=0 must reduce to this).
  hsq_v0   - per-coordinate Hessian budget (guide section 4).
  hsq_v1   - cumulative per-group Hessian budget (guide section 5), the
             paper-level "main" method per the guide's roadmap (section 26).
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

from aq.common import (
    ensure_dir,
    ensure_if_awq_tier0_on_path,
    free_model,
    get_block_layer_groups,
    get_transformer_linear_layers,
    gpu_peak_memory_bytes,
    load_causal_lm,
    load_tokenizer,
    load_yaml_config,
    default_calibration_path,
    default_fingerprint_keys_path,
    read_json,
)
from aq.calibration import build_calibration_batches, load_calibration_texts
from aq.activation_cache import capture_layer_input_activations
from aq.decision_flow import GateConfig, decide_next_action, evaluate_watermark_gate
from aq.hsq_hessian import accumulate_hessian
from aq.hsq_core import run_hsq_layer
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
from aq.metrics import cosine_similarity_flat, weight_relative_distance
from aq.ppl_eval import compute_wikitext2_ppl
from aq.plotting import plot_rounding_flip_ratio_per_layer
from aq.reporting import generate_method_report

HSQ_METHODS = {"gptq4", "hsq_v0", "hsq_v1"}


def quantize_model_with_hsq(
    model,
    all_layers: dict,
    block_groups: list[list[str]],
    calibration_batches: list[dict],
    hsq_mode: str,
    bits: int = 4,
    group_size: int = 128,
    sym: bool = False,
    percdamp: float = 0.01,
    blocksize: int = 128,
    tau: float = 0.02,
    candidate_radius: int = 2,
    device: str = "cuda",
    logger=None,
) -> list[dict]:
    """The actual GPTQ-style sequential block loop (guide section 10.6/12):
    for each transformer block in depth order, capture that block's Linear
    layers' calibration inputs by running the calibration batches through
    the model AS IT CURRENTLY STANDS (earlier blocks already committed
    in-place - `capture_layer_input_activations`'s real forward pass is what
    gives this the correct propagated activations for free), accumulate
    each layer's Hessian from those inputs, quantize via `run_hsq_layer`,
    and commit the result before moving to the next block. Separated from
    `run_hsq_method` (which owns model loading / PPL / watermark / logging)
    so it can be exercised directly against a tiny fake model in tests,
    without a GPU or the real if_awq_tier0/HF dependencies.
    """
    import torch

    all_layer_metrics: list[dict] = []
    for block_idx, layer_names in enumerate(block_groups):
        layers_in_block = {name: all_layers[name] for name in layer_names}
        cached_inputs = capture_layer_input_activations(model, layers_in_block, calibration_batches, device)

        for name in layer_names:
            module = layers_in_block[name]
            w_fp = module.weight.detach().clone()
            in_features = w_fp.shape[1]

            H = torch.zeros(in_features, in_features, device=device, dtype=torch.float32)
            nsamples = 0
            for x in cached_inputs[name]:
                H, nsamples = accumulate_hessian(H, nsamples, x.to(device))

            result = run_hsq_layer(
                w_fp, H, mode=hsq_mode, bits=bits, group_size=group_size, sym=sym,
                percdamp=percdamp, blocksize=blocksize, tau=tau, candidate_radius=candidate_radius,
            )
            with torch.no_grad():
                module.weight.data.copy_(result.hard_weight.to(module.weight.dtype))

            all_layer_metrics.append({
                "layer": name,
                "weight_distance_vs_fp": float(weight_relative_distance(w_fp, result.hard_weight)),
                "cosine_similarity_vs_fp": float(cosine_similarity_flat(w_fp, result.hard_weight)),
                # NOT literally "vs RTN4" here - this is HSQ's own
                # farther-fraction (share of coordinates moved to a
                # non-nearest lattice point). Reusing this CSV column name
                # keeps layer_metrics.csv schema-compatible with the other
                # methods' reports/plots.
                "rounding_flip_ratio_vs_rtn4": result.metrics["farther_fraction"],
                "scale_relative_shift": 0.0,
                "final_kl": None,
                "final_loss": result.metrics.get("predicted_loss"),
                "num_steps": 0,
                "nearest_fraction": result.metrics["nearest_fraction"],
                "plusminus1_fraction": result.metrics["plusminus1_fraction"],
                "fallback_to_nearest_count": result.metrics["fallback_to_nearest_count"],
            })

            del w_fp, H, result
        del cached_inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if logger is not None:
            logger.info("block %d/%d quantized", block_idx + 1, len(block_groups))

    return all_layer_metrics


def run_hsq_method(
    method_id: str,
    model_cfg: dict,
    method_cfg: dict,
    rtn4_ppl: float,
    out_root: Path,
    device: str = "cuda",
    dtype: str = "bfloat16",
    force_watermark_eval: bool = False,
):
    import torch

    ensure_if_awq_tier0_on_path()
    from src.verify_fingerprint import run_verification, summarize

    assert method_id in HSQ_METHODS, f"unknown HSQ method {method_id!r}"
    hsq_mode = {"gptq4": "gptq", "hsq_v0": "hsq_v0", "hsq_v1": "hsq_v1"}[method_id]

    run_dir = run_dir_for(out_root, method_id)
    logger = get_run_logger(run_dir, method_id)
    write_config_yaml(run_dir, {"method_id": method_id, "model": model_cfg, "method": method_cfg})

    model_id = model_cfg.get("id", "cnut1648/LLaMA2-7B-fingerprinted-SFT")
    logger.info("loading model %s", model_id)
    model = load_causal_lm(model_id, device=device, dtype=dtype)
    tokenizer = load_tokenizer(model_id)

    bits = method_cfg.get("bits", 4)
    group_size = method_cfg.get("group_size", 128)
    sym = method_cfg.get("sym", False)
    percdamp = method_cfg.get("percdamp", 0.01)
    blocksize = method_cfg.get("blocksize", 128)
    tau = method_cfg.get("tau", 0.02)
    candidate_radius = method_cfg.get("candidate_radius", 2)

    calibration_texts = load_calibration_texts(method_cfg.get("calibration_path", default_calibration_path()))
    calibration_texts = calibration_texts[: method_cfg.get("calibration_samples", 32)]
    calibration_batches = build_calibration_batches(
        tokenizer,
        calibration_texts,
        max_seq_len=method_cfg.get("max_seq_len", 512),
        batch_size=method_cfg.get("calib_batch_size", 4),
        device=device,
    )

    all_layers = get_transformer_linear_layers(model)
    block_groups = get_block_layer_groups(model)
    max_blocks = method_cfg.get("max_blocks")
    if max_blocks is not None:
        block_groups = block_groups[:max_blocks]
        logger.warning(
            "max_blocks=%d set - only the first %d/%d blocks will be quantized, the rest stay FP. "
            "This is for timing/sanity checks ONLY (guide section 20) - the resulting PPL/watermark "
            "numbers are NOT a valid method outcome.",
            max_blocks, len(block_groups), len(get_block_layer_groups(model)),
        )

    t0 = time.time()
    logger.info(
        "running %s (mode=%s, bits=%d, group_size=%d, tau=%.4f, candidate_radius=%d) over %d blocks / %d layers",
        method_id, hsq_mode, bits, group_size, tau, candidate_radius, len(block_groups),
        sum(len(g) for g in block_groups),
    )

    all_layer_metrics = quantize_model_with_hsq(
        model, all_layers, block_groups, calibration_batches, hsq_mode,
        bits=bits, group_size=group_size, sym=sym, percdamp=percdamp, blocksize=blocksize,
        tau=tau, candidate_radius=candidate_radius, device=device, logger=logger,
    )

    elapsed = time.time() - t0
    logger.info("quantization pass finished in %.1fs", elapsed)

    write_layer_metrics_csv(run_dir, all_layer_metrics)
    write_optimization_trace_csv(run_dir, [])
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
            model, tokenizer, fingerprints, do_sample=False,
            max_new_tokens=method_cfg.get("max_new_tokens", 32), device=device,
        )
        watermark_summary = summarize(per_key, evaluation_time_seconds=time.time() - t1)
        write_watermark_result_json(run_dir, {"summary": watermark_summary, "per_key": per_key})
        watermark_gone, watermark_detail = evaluate_watermark_gate(watermark_summary["fsr_exact"], gate_cfg)

        if pre_gate.status == "SKIPPED":
            outcome = decide_next_action(
                method_id, candidate_ppl, rtn4_ppl, gate_cfg,
                watermark_fsr_exact=watermark_summary["fsr_exact"],
                watermark_fsr_contains=watermark_summary["fsr_contains"],
            )
        else:
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
        "num_layers_touched": sum(len(g) for g in block_groups),
    }
    outcome.model_metrics = model_metrics
    write_model_metrics_json(run_dir, model_metrics)
    generate_method_report(outcome, run_dir, out_root / "reports" / f"{method_id}_report.md")

    free_model(model)
    del model
    return outcome


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True, choices=sorted(HSQ_METHODS))
    ap.add_argument("--config", required=True, help="YAML with model/method sections")
    ap.add_argument("--rtn4-ppl", type=float, required=True, help="Baseline PPL from the 00_rtn4 run")
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

    from aq.common import configure_gpu_performance

    configure_gpu_performance()
    cfg = load_yaml_config(args.config)
    out_root = ensure_dir(Path(args.output))

    outcome = run_hsq_method(
        args.method, cfg.get("model", {}), cfg.get("method", {}), args.rtn4_ppl, out_root,
        device=args.device, dtype=args.dtype, force_watermark_eval=args.force_watermark_eval,
    )
    print(f"[run_hsq] {args.method} -> {outcome.status} ({outcome.detail})")


if __name__ == "__main__":
    main()
