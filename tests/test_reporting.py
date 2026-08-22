from pathlib import Path

from aq.decision_flow import MethodOutcome, STATUS_FAIL_UTILITY, STATUS_PASS
from aq.reporting import FINAL_COMPARISON_QUESTIONS, generate_final_comparison, generate_method_report


def test_generate_method_report_writes_file(tmp_path):
    outcome = MethodOutcome(
        method_id="01_adv_round",
        status=STATUS_FAIL_UTILITY,
        ppl=12.0,
        rtn4_ppl=10.0,
        ppl_relative_regression=0.2,
        detail="ppl gate failed",
    )
    out_path = tmp_path / "reports" / "01_adv_round_report.md"
    generate_method_report(outcome, tmp_path / "runs" / "01_adv_round", out_path)

    text = out_path.read_text(encoding="utf-8")
    assert "01_adv_round" in text
    assert "FAIL_UTILITY" in text
    assert "watermark_result.json" not in text  # never ran, since PPL gate failed first


def test_generate_method_report_includes_watermark_artifact_when_evaluated(tmp_path):
    outcome = MethodOutcome(
        method_id="01_adv_round",
        status=STATUS_PASS,
        ppl=10.1,
        rtn4_ppl=10.0,
        ppl_relative_regression=0.01,
        watermark_fsr_exact=0.0,
        watermark_fsr_contains=0.0,
        detail="watermark gone",
    )
    out_path = tmp_path / "reports" / "01_adv_round_report.md"
    generate_method_report(outcome, tmp_path / "runs" / "01_adv_round", out_path)
    text = out_path.read_text(encoding="utf-8")
    assert "watermark_result.json" in text


def test_generate_final_comparison_includes_summary_table_and_questions(tmp_path):
    outcomes = [
        MethodOutcome(method_id="00_rtn4", status="BASELINE", ppl=10.0, rtn4_ppl=10.0),
        MethodOutcome(
            method_id="01_adv_round",
            status=STATUS_FAIL_UTILITY,
            ppl=12.0,
            rtn4_ppl=10.0,
            ppl_relative_regression=0.2,
        ),
    ]
    out_path = tmp_path / "reports" / "final_comparison.md"
    generate_final_comparison(outcomes, out_path)
    text = out_path.read_text(encoding="utf-8")

    assert "00_rtn4" in text
    assert "01_adv_round" in text
    assert "no method reached PASS" in text
    for q in FINAL_COMPARISON_QUESTIONS:
        assert q in text


def test_generate_final_comparison_reports_early_stop_winner(tmp_path):
    outcomes = [
        MethodOutcome(method_id="00_rtn4", status="BASELINE", ppl=10.0, rtn4_ppl=10.0),
        MethodOutcome(method_id="01_adv_round", status=STATUS_PASS, ppl=10.1, rtn4_ppl=10.0, watermark_fsr_exact=0.0),
    ]
    out_path = tmp_path / "reports" / "final_comparison.md"
    generate_final_comparison(outcomes, out_path)
    text = out_path.read_text(encoding="utf-8")
    assert "STOPPED EARLY at `01_adv_round`" in text
