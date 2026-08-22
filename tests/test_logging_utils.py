from pathlib import Path

from aq.logging_utils import (
    read_layer_metrics_csv,
    run_dir_for,
    write_config_yaml,
    write_layer_metrics_csv,
    write_model_metrics_json,
    write_optimization_trace_csv,
    write_ppl_result_json,
    write_watermark_result_json,
)
from aq.common import read_json


def test_run_dir_for_creates_directory(tmp_path):
    run_dir = run_dir_for(tmp_path, "01_adv_round")
    assert run_dir.exists()
    assert run_dir == tmp_path / "runs" / "01_adv_round"


def test_write_config_yaml_roundtrip(tmp_path):
    run_dir = run_dir_for(tmp_path, "00_rtn4")
    write_config_yaml(run_dir, {"method_id": "00_rtn4", "method": {"bits": 4}})
    import yaml

    loaded = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert loaded["method_id"] == "00_rtn4"
    assert loaded["method"]["bits"] == 4


def test_layer_metrics_csv_roundtrip(tmp_path):
    run_dir = run_dir_for(tmp_path, "01_adv_round")
    rows = [
        {"layer": "a", "weight_distance_vs_fp": 0.1, "rounding_flip_ratio_vs_rtn4": 0.05},
        {"layer": "b", "weight_distance_vs_fp": 0.2, "rounding_flip_ratio_vs_rtn4": 0.10},
    ]
    write_layer_metrics_csv(run_dir, rows)
    loaded = read_layer_metrics_csv(run_dir)
    assert len(loaded) == 2
    assert loaded[0]["layer"] == "a"
    assert float(loaded[1]["weight_distance_vs_fp"]) == 0.2


def test_layer_metrics_csv_empty_rows(tmp_path):
    run_dir = run_dir_for(tmp_path, "00_rtn4")
    write_layer_metrics_csv(run_dir, [])
    assert read_layer_metrics_csv(run_dir) == []


def test_optimization_trace_csv_writes_all_columns(tmp_path):
    run_dir = run_dir_for(tmp_path, "01_adv_round")
    rows = [{"layer": "a", "step": 0, "loss": 1.0, "kl": 0.5, "distance": 0.2}]
    write_optimization_trace_csv(run_dir, rows)
    content = (run_dir / "optimization_trace.csv").read_text(encoding="utf-8")
    assert "loss" in content and "kl" in content and "distance" in content


def test_model_metrics_ppl_watermark_json(tmp_path):
    run_dir = run_dir_for(tmp_path, "01_adv_round")
    write_model_metrics_json(run_dir, {"ppl": 5.0})
    write_ppl_result_json(run_dir, {"wikitext2_ppl": 5.0})
    write_watermark_result_json(run_dir, {"summary": {"fsr_exact": 0.0}})

    assert read_json(run_dir / "model_metrics.json")["ppl"] == 5.0
    assert read_json(run_dir / "ppl_result.json")["wikitext2_ppl"] == 5.0
    assert read_json(run_dir / "watermark_result.json")["summary"]["fsr_exact"] == 0.0
