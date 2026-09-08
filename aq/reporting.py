"""Per-method markdown reports + the final_comparison.md required by plan
section 15/16 (summary table + the plan's 8 analytical questions).
"""
from __future__ import annotations

from pathlib import Path
import csv
import json

from aq.decision_flow import METHOD_LABELS, MethodOutcome, STATUS_PASS

FINAL_COMPARISON_QUESTIONS = [
    "Phương pháp nào (nếu có) đạt PASS - giữ PPL gần RTN4 trong khi làm mất watermark?",
    "Ở method nào PPL bắt đầu suy giảm rõ rệt so với RTN4, và tại sao (bits/group_size cố định, chỉ rounding/scale/codebook thay đổi)?",
    "Weight distance D(W,Q(W)) tăng dần qua các method theo đúng kỳ vọng (method sau >= method trước) hay có method nào đi ngược xu hướng?",
    "Rounding flip ratio so với RTN4 tương quan thế nào với watermark FSR - flip nhiều hơn có thực sự làm mất watermark nhanh hơn không?",
    "Có dấu hiệu tích luỹ lỗi theo độ sâu (KL cục bộ ở layer đầu thấp nhưng KL toàn cục tăng dần theo layer) biện minh cho việc chạy method 5 (Quantized-Prefix) không?",
    "Block-wise (method 6) có cải thiện được KL/watermark tradeoff so với tối ưu từng ma trận riêng lẻ (method 1-4) không, và chi phí tính toán tăng thêm bao nhiêu?",
    "Periodic activation refresh (method 7) ở k=8/4/2 cho thấy đánh đổi gì giữa độ chính xác hiệu chỉnh và thời gian chạy?",
    "Two-pass backward correction (method 8) - nếu phải chạy đến đây - có tạo ra khác biệt đáng kể so với single forward pass (method 5), hay chi phí gấp đôi không tương xứng với lợi ích?",
]


def _fmt(x, digits=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def generate_method_report(outcome: MethodOutcome, run_dir: Path, out_path: Path) -> None:
    label = METHOD_LABELS.get(outcome.method_id, outcome.method_id)
    lines = [
        f"# {outcome.method_id} — {label}",
        "",
        f"**Status:** `{outcome.status}`",
        "",
        "## Utility (PPL gate)",
        f"- Candidate WikiText-2 PPL: {_fmt(outcome.ppl)}",
        f"- RTN4 baseline PPL: {_fmt(outcome.rtn4_ppl)}",
        f"- Relative regression: {_fmt(outcome.ppl_relative_regression, 4)}",
        "",
        "## Watermark gate",
        f"- fsr_exact: {_fmt(outcome.watermark_fsr_exact)}",
        f"- fsr_contains: {_fmt(outcome.watermark_fsr_contains)}",
        "",
        "## Decision detail",
        f"> {outcome.detail}",
        "",
        "## Artifacts",
        f"- `{run_dir}/config.yaml`",
        f"- `{run_dir}/run.log`",
        f"- `{run_dir}/layer_metrics.csv`",
        f"- `{run_dir}/optimization_trace.csv`",
        f"- `{run_dir}/model_metrics.json`",
        f"- `{run_dir}/ppl_result.json`",
    ]
    if outcome.status not in ("FAIL_UTILITY",):
        lines.append(f"- `{run_dir}/watermark_result.json`")
    if outcome.model_metrics:
        lines += ["", "## Model-level metrics", "```json", str(outcome.model_metrics), "```"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_final_comparison(outcomes: list[MethodOutcome], out_path: Path) -> None:
    lines = ["# Final Comparison — Adversarial Quantization of IF-SFT LLaMA2-7B", ""]

    passed = [o for o in outcomes if o.status == STATUS_PASS]
    if passed:
        winner = passed[0]
        lines.append(
            f"**Result: STOPPED EARLY at `{winner.method_id}` "
            f"({METHOD_LABELS.get(winner.method_id, winner.method_id)})** — "
            "watermark removed while passing the PPL gate."
        )
    else:
        lines.append(
            "**Result: no method reached PASS** — every method in the mandated "
            "order either failed the PPL gate or retained the watermark."
        )
    lines.append("")

    lines += [
        "## Summary table",
        "",
        "| Method | Label | Status | PPL | Δ vs RTN4 | fsr_exact | fsr_contains |",
        "|---|---|---|---|---|---|---|",
    ]
    for o in outcomes:
        label = METHOD_LABELS.get(o.method_id, o.method_id)
        lines.append(
            f"| {o.method_id} | {label} | {o.status} | {_fmt(o.ppl)} | "
            f"{_fmt(o.ppl_relative_regression, 4)} | {_fmt(o.watermark_fsr_exact)} | "
            f"{_fmt(o.watermark_fsr_contains)} |"
        )
    lines.append("")

    lines += ["## Analytical questions to answer", ""]
    for i, q in enumerate(FINAL_COMPARISON_QUESTIONS, start=1):
        lines.append(f"{i}. {q}")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_global_far_round_sweep_report(root: Path) -> dict:
    """Aggregate completed rho runs and identify the requested FSR transitions."""
    rows = []
    for metrics_path in root.glob("rho_*/runs/03_global_far_round/model_metrics.json"):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({
            "rho": float(metrics["target_aggressive_fraction"]),
            "selected_fraction": metrics.get("actual_selected_fraction"),
            "flip_ratio": metrics.get("actual_rounding_flip_ratio"),
            "ppl": metrics.get("ppl"),
            "ppl_relative_regression": metrics.get("ppl_relative_regression"),
            "fsr_exact": metrics.get("fsr_exact"),
            "fsr_contains": metrics.get("fsr_contains"),
        })
    rows.sort(key=lambda row: row["rho"])
    if not rows:
        raise ValueError(f"no completed global far-round runs found under {root}")

    def first_rho(predicate):
        match = next((row for row in rows if row["fsr_exact"] is not None and predicate(row)), None)
        return match["rho"] if match else None

    zero_rows = [row for row in rows if row["fsr_exact"] == 0.0]
    lowest_ppl_zero = min(zero_rows, key=lambda row: row["ppl"]) if zero_rows else None
    summary = {
        "first_rho_with_fsr_below_1": first_rho(lambda row: row["fsr_exact"] < 1.0),
        "first_rho_with_fsr_at_most_0_125": first_rho(lambda row: row["fsr_exact"] <= 0.125),
        "first_rho_with_fsr_zero": first_rho(lambda row: row["fsr_exact"] == 0.0),
        "lowest_ppl_rho_with_fsr_zero": lowest_ppl_zero["rho"] if lowest_ppl_zero else None,
        "runs": rows,
    }
    fields = list(rows[0])
    with open(root / "global_far_round_sweep.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Global Far-Round Sweep",
        "",
        "| rho | selected% | flip% | PPL | ΔPPL% | FSR exact | FSR contains |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rho']:.3f} | {_fmt(100.0 * row['selected_fraction'])} | "
            f"{_fmt(100.0 * row['flip_ratio'])} | {_fmt(row['ppl'])} | "
            f"{_fmt(100.0 * row['ppl_relative_regression'])} | "
            f"{_fmt(row['fsr_exact'])} | {_fmt(row['fsr_contains'])} |"
        )
    lines += [
        "",
        f"- First rho with FSR < 1: {_fmt(summary['first_rho_with_fsr_below_1'])}",
        f"- First rho with FSR <= 0.125: {_fmt(summary['first_rho_with_fsr_at_most_0_125'])}",
        f"- First rho with FSR = 0: {_fmt(summary['first_rho_with_fsr_zero'])}",
        f"- Lowest-PPL rho with FSR = 0: {_fmt(summary['lowest_ppl_rho_with_fsr_zero'])}",
    ]
    (root / "global_far_round_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "global_far_round_sweep.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    from aq.plotting import plot_global_far_round_ppl_fsr
    plot_global_far_round_ppl_fsr(rows, root / "global_far_round_ppl_fsr.png")
    return summary
