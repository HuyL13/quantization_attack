"""Reconstructs MethodOutcome records from each runs/NN_method/*.json on disk
and writes reports/final_comparison.md (plan section 16). Run after
scripts/run_all_methods.sh finishes (or stops early on PASS).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aq.decision_flow import METHOD_ORDER, MethodOutcome
from aq.plotting import plot_ppl_comparison, plot_watermark_fsr_comparison
from aq.reporting import generate_final_comparison


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def reconstruct_outcome(run_dir: Path, method_id: str, rtn4_ppl: float | None) -> MethodOutcome | None:
    model_metrics = _load_json(run_dir / "model_metrics.json")
    ppl_result = _load_json(run_dir / "ppl_result.json")
    if model_metrics is None or ppl_result is None:
        return None
    ppl = ppl_result["wikitext2_ppl"]
    watermark = _load_json(run_dir / "watermark_result.json")
    fsr_exact = watermark["summary"]["fsr_exact"] if watermark else None
    fsr_contains = watermark["summary"]["fsr_contains"] if watermark else None

    if method_id == "00_rtn4":
        status = "BASELINE"
    elif rtn4_ppl is not None and fsr_exact is not None and fsr_exact <= 0.0 and ppl <= rtn4_ppl * 1.05:
        status = "PASS"
    elif rtn4_ppl is not None and ppl > rtn4_ppl * 1.05:
        status = "FAIL_UTILITY"
    elif fsr_exact is not None:
        status = "FAIL_WATERMARK_RETAINED"
    else:
        status = "UNKNOWN"

    return MethodOutcome(
        method_id=method_id,
        status=status,
        ppl=ppl,
        rtn4_ppl=rtn4_ppl,
        ppl_relative_regression=((ppl - rtn4_ppl) / rtn4_ppl) if rtn4_ppl else None,
        watermark_fsr_exact=fsr_exact,
        watermark_fsr_contains=fsr_contains,
        model_metrics=model_metrics,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, help="Root dir containing runs/ and reports/")
    args = ap.parse_args()

    root = Path(args.output)
    baseline_ppl_path = root / "runs" / "00_rtn4" / "ppl_result.json"
    rtn4_ppl = _load_json(baseline_ppl_path)["wikitext2_ppl"] if baseline_ppl_path.exists() else None

    outcomes = []
    for method_id in METHOD_ORDER:
        run_dir = root / "runs" / method_id
        outcome = reconstruct_outcome(run_dir, method_id, rtn4_ppl)
        if outcome is not None:
            outcomes.append(outcome)

    generate_final_comparison(outcomes, root / "reports" / "final_comparison.md")
    plot_ppl_comparison(outcomes, root / "reports" / "ppl_comparison.png")
    plot_watermark_fsr_comparison(outcomes, root / "reports" / "watermark_fsr_comparison.png")
    print(f"[aggregate_final_report] wrote {root / 'reports' / 'final_comparison.md'}")


if __name__ == "__main__":
    main()
