import pytest

from aq.decision_flow import (
    GateConfig,
    STATUS_FAIL_UTILITY,
    STATUS_FAIL_WATERMARK_RETAINED,
    STATUS_PASS,
    STATUS_SKIPPED,
    decide_next_action,
    evaluate_ppl_gate,
    evaluate_watermark_gate,
    run_methods_in_order,
)


def test_ppl_gate_passes_within_tolerance():
    cfg = GateConfig(max_ppl_relative_regression=0.05)
    ok, reg, detail = evaluate_ppl_gate(10.4, 10.0, cfg)
    assert ok
    assert reg == pytest.approx(0.04)


def test_ppl_gate_fails_beyond_tolerance():
    cfg = GateConfig(max_ppl_relative_regression=0.05)
    ok, reg, detail = evaluate_ppl_gate(11.0, 10.0, cfg)
    assert not ok
    assert reg == pytest.approx(0.10)


def test_ppl_gate_passes_when_candidate_better_than_baseline():
    cfg = GateConfig(max_ppl_relative_regression=0.05)
    ok, reg, _ = evaluate_ppl_gate(9.0, 10.0, cfg)
    assert ok
    assert reg < 0


def test_watermark_gate_gone_vs_retained():
    cfg = GateConfig(max_fsr_exact_for_pass=0.0)
    gone, _ = evaluate_watermark_gate(0.0, cfg)
    retained, _ = evaluate_watermark_gate(0.125, cfg)
    assert gone
    assert not retained


def test_decide_next_action_fails_utility_before_watermark_needed():
    cfg = GateConfig(max_ppl_relative_regression=0.05)
    outcome = decide_next_action("01_adv_round", candidate_ppl=12.0, rtn4_ppl=10.0, gate_cfg=cfg)
    assert outcome.status == STATUS_FAIL_UTILITY
    assert outcome.watermark_fsr_exact is None


def test_fragile_channel_skips_ppl_gate_but_records_regression_before_watermark():
    cfg = GateConfig(max_ppl_relative_regression=0.05)
    outcome = decide_next_action("03_fragile_channel", candidate_ppl=12.0, rtn4_ppl=10.0, gate_cfg=cfg)
    assert outcome.status == STATUS_SKIPPED
    assert outcome.ppl == 12.0
    assert outcome.rtn4_ppl == 10.0
    assert outcome.ppl_relative_regression == pytest.approx(0.2)
    assert outcome.watermark_fsr_exact is None


def test_fragile_channel_final_decision_uses_watermark_even_when_ppl_regresses():
    cfg = GateConfig(max_ppl_relative_regression=0.05, max_fsr_exact_for_pass=0.0)
    outcome = decide_next_action(
        "03_fragile_channel",
        candidate_ppl=12.0,
        rtn4_ppl=10.0,
        gate_cfg=cfg,
        watermark_fsr_exact=0.0,
        watermark_fsr_contains=0.0,
    )
    assert outcome.status == STATUS_PASS
    assert outcome.ppl_relative_regression == pytest.approx(0.2)
    assert outcome.watermark_fsr_exact == 0.0


def test_decide_next_action_skipped_when_ppl_ok_and_watermark_not_yet_run():
    cfg = GateConfig(max_ppl_relative_regression=0.05)
    outcome = decide_next_action("01_adv_round", candidate_ppl=10.1, rtn4_ppl=10.0, gate_cfg=cfg)
    assert outcome.status == STATUS_SKIPPED


def test_decide_next_action_pass_when_watermark_gone():
    cfg = GateConfig(max_ppl_relative_regression=0.05, max_fsr_exact_for_pass=0.0)
    outcome = decide_next_action(
        "01_adv_round", candidate_ppl=10.1, rtn4_ppl=10.0, gate_cfg=cfg, watermark_fsr_exact=0.0
    )
    assert outcome.status == STATUS_PASS


def test_decide_next_action_fail_watermark_retained():
    cfg = GateConfig(max_ppl_relative_regression=0.05, max_fsr_exact_for_pass=0.0)
    outcome = decide_next_action(
        "01_adv_round", candidate_ppl=10.1, rtn4_ppl=10.0, gate_cfg=cfg, watermark_fsr_exact=0.5
    )
    assert outcome.status == STATUS_FAIL_WATERMARK_RETAINED


def test_rtn4_ppl_must_be_positive():
    cfg = GateConfig()
    with pytest.raises(ValueError):
        evaluate_ppl_gate(10.0, 0.0, cfg)


def test_run_methods_in_order_stops_after_first_pass():
    call_order = []

    def fake_runner(method_id):
        call_order.append(method_id)
        outcome = decide_next_action(
            method_id,
            candidate_ppl=10.1,
            rtn4_ppl=10.0,
            gate_cfg=GateConfig(),
            watermark_fsr_exact=(0.0 if method_id == "02_adv_round_scale" else 0.5),
        )
        return outcome

    outcomes = run_methods_in_order(
        ["01_adv_round", "02_adv_round_scale", "03_adv_codebook"], fake_runner
    )

    assert call_order == ["01_adv_round", "02_adv_round_scale"]  # never reaches 03
    assert outcomes[-1].status == STATUS_PASS


def test_run_methods_in_order_runs_all_if_none_pass():
    def fake_runner(method_id):
        return decide_next_action(
            method_id, candidate_ppl=10.1, rtn4_ppl=10.0, gate_cfg=GateConfig(), watermark_fsr_exact=0.5
        )

    methods = ["01_adv_round", "02_adv_round_scale", "03_adv_codebook"]
    outcomes = run_methods_in_order(methods, fake_runner)
    assert len(outcomes) == len(methods)
    assert all(o.status == STATUS_FAIL_WATERMARK_RETAINED for o in outcomes)
