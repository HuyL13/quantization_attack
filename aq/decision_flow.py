"""Plan section 2/12 stop-early decision flow:

    quantize -> PPL gate (vs RTN4 baseline) -> [gate fails: FAIL_UTILITY, next method]
             -> watermark gate                -> [watermark gone: PASS, STOP ALL]
                                                -> [watermark retained: FAIL_WATERMARK_RETAINED, next method]

`evaluate_ppl_gate` / `evaluate_watermark_gate` / `decide_next_action` are
pure functions (no torch/model dependency) so the STOP logic itself can be
unit-tested without a GPU or the real 7B checkpoint. `MethodOutcome` is the
record persisted to model_metrics.json / used to build the final comparison
report.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Method 1 (Adversarial Rounding) is itself a light-to-heavy chain, not one
# monolithic tier: 1A never touches gradients or the model at all (a single
# closed-form greedy pass); 1B and 1C use cached LOCAL activations (one
# Linear layer's / one block's own forward) instead of a full-model KL pass;
# 1D (the original single-tier design) is the expensive fallback, reran
# through the whole 32-layer network on every optimization step. Measured
# live: 1D costs ~2.79s per optimization step per layer (a full-model
# forward + checkpointed backward) - each lighter tier is only reached if
# the one before it fails both the PPL and watermark gates.
METHOD_ORDER = [
    "00_rtn4",
    "01a_greedy_round",
    "01b_layerwise_local",
    "01c_blockwise_local",
    "01d_global_kl",
    "02_adv_round_scale",
    "03_adv_codebook",
    "04_sensitivity_aware",
    "05_quantized_prefix",
    "06_block_wise",
    "07_periodic_refresh",
    "08_two_pass_backward",
]

METHOD_LABELS = {
    "00_rtn4": "RTN4 baseline",
    "01a_greedy_round": "Adversarial Rounding (1A: greedy, zero-training)",
    "01b_layerwise_local": "Adversarial Rounding (1B: layer-wise local reconstruction)",
    "01c_blockwise_local": "Adversarial Rounding (1C: block-wise local reconstruction)",
    "01d_global_kl": "Adversarial Rounding (1D: global KL-guided)",
    "02_adv_round_scale": "Joint Rounding + Scale",
    "03_adv_codebook": "Non-Uniform Codebook",
    "04_sensitivity_aware": "Sensitivity-Aware",
    "05_quantized_prefix": "Quantized-Prefix Calibration",
    "06_block_wise": "Block-Wise",
    "07_periodic_refresh": "Periodic Activation Refresh",
    "08_two_pass_backward": "Two-Pass Backward Correction",
}

STATUS_PASS = "PASS"
STATUS_FAIL_UTILITY = "FAIL_UTILITY"
STATUS_FAIL_WATERMARK_RETAINED = "FAIL_WATERMARK_RETAINED"
STATUS_SKIPPED = "SKIPPED"


@dataclass
class GateConfig:
    # PPL gate: candidate PPL must not exceed rtn4_ppl * (1 + max_ppl_relative_regression)
    max_ppl_relative_regression: float = 0.05
    # Watermark gate: watermark counted as "gone" when fsr_exact <= this
    max_fsr_exact_for_pass: float = 0.0


@dataclass
class MethodOutcome:
    method_id: str
    status: str
    ppl: float | None = None
    rtn4_ppl: float | None = None
    ppl_relative_regression: float | None = None
    watermark_fsr_exact: float | None = None
    watermark_fsr_contains: float | None = None
    detail: str = ""
    model_metrics: dict = field(default_factory=dict)


def evaluate_ppl_gate(candidate_ppl: float, rtn4_ppl: float, gate_cfg: GateConfig) -> tuple[bool, float, str]:
    """Returns (passed, relative_regression, detail_string)."""
    if rtn4_ppl <= 0:
        raise ValueError(f"rtn4_ppl must be positive, got {rtn4_ppl}")
    relative_regression = (candidate_ppl - rtn4_ppl) / rtn4_ppl
    passed = relative_regression <= gate_cfg.max_ppl_relative_regression
    detail = (
        f"ppl={candidate_ppl:.4f} rtn4_ppl={rtn4_ppl:.4f} "
        f"relative_regression={relative_regression:+.4%} "
        f"(allowed <= {gate_cfg.max_ppl_relative_regression:.2%}) -> {'PASS' if passed else 'FAIL'}"
    )
    return passed, relative_regression, detail


def evaluate_watermark_gate(fsr_exact: float, gate_cfg: GateConfig) -> tuple[bool, str]:
    """Returns (watermark_gone, detail_string). watermark_gone=True means the
    fingerprint could no longer be triggered -> PASS, STOP ALL TESTING.
    """
    watermark_gone = fsr_exact <= gate_cfg.max_fsr_exact_for_pass
    detail = (
        f"fsr_exact={fsr_exact:.4f} (watermark considered gone when <= "
        f"{gate_cfg.max_fsr_exact_for_pass:.2%}) -> "
        f"{'WATERMARK GONE (PASS)' if watermark_gone else 'WATERMARK RETAINED'}"
    )
    return watermark_gone, detail


def decide_next_action(
    method_id: str,
    candidate_ppl: float,
    rtn4_ppl: float,
    gate_cfg: GateConfig,
    watermark_fsr_exact: float | None = None,
    watermark_fsr_contains: float | None = None,
) -> MethodOutcome:
    """Single-method decision. If `watermark_fsr_exact` is None, only the PPL
    gate is evaluated (i.e. run this BEFORE the watermark eval to decide
    whether to run it at all - matches plan section 12's ordering, which
    saves an expensive watermark eval on a method that already failed on
    utility grounds).
    """
    ppl_ok, relative_regression, ppl_detail = evaluate_ppl_gate(candidate_ppl, rtn4_ppl, gate_cfg)
    if not ppl_ok:
        return MethodOutcome(
            method_id=method_id,
            status=STATUS_FAIL_UTILITY,
            ppl=candidate_ppl,
            rtn4_ppl=rtn4_ppl,
            ppl_relative_regression=relative_regression,
            detail=ppl_detail,
        )

    if watermark_fsr_exact is None:
        # PPL gate passed but watermark not evaluated yet - caller should now
        # run the watermark eval and call this again with the result.
        return MethodOutcome(
            method_id=method_id,
            status=STATUS_SKIPPED,
            ppl=candidate_ppl,
            rtn4_ppl=rtn4_ppl,
            ppl_relative_regression=relative_regression,
            detail=ppl_detail + " | watermark eval pending",
        )

    watermark_gone, watermark_detail = evaluate_watermark_gate(watermark_fsr_exact, gate_cfg)
    status = STATUS_PASS if watermark_gone else STATUS_FAIL_WATERMARK_RETAINED
    return MethodOutcome(
        method_id=method_id,
        status=status,
        ppl=candidate_ppl,
        rtn4_ppl=rtn4_ppl,
        ppl_relative_regression=relative_regression,
        watermark_fsr_exact=watermark_fsr_exact,
        watermark_fsr_contains=watermark_fsr_contains,
        detail=ppl_detail + " | " + watermark_detail,
    )


def run_methods_in_order(
    method_ids: list[str],
    run_single_method_fn,
) -> list[MethodOutcome]:
    """Drives the mandatory stop-early order (plan section 2): calls
    `run_single_method_fn(method_id) -> MethodOutcome` for each method in
    turn, stopping immediately after the first PASS. `00_rtn4` is expected
    to always be first in `method_ids` and never itself gated (its own
    outcome should be constructed by the caller as a fixed reference point,
    typically `MethodOutcome(status=STATUS_PASS-like "baseline")` semantics
    are up to the caller - this function only enforces the STOP rule).
    """
    outcomes: list[MethodOutcome] = []
    for method_id in method_ids:
        outcome = run_single_method_fn(method_id)
        outcomes.append(outcome)
        if outcome.status == STATUS_PASS:
            break
    return outcomes
