"""Per-method markdown reports + the final_comparison.md required by plan
section 15/16 (summary table + the plan's 8 analytical questions).
"""
from __future__ import annotations

from pathlib import Path

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
