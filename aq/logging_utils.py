"""Per-method logging artifacts matching plan section 13's required
directory structure: runs/NN_method/{config.yaml, run.log, layer_metrics.csv,
optimization_trace.csv, model_metrics.json, ppl_result.json,
watermark_result.json (only if the PPL gate passed)}.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import yaml

from aq.common import ensure_dir, write_json


def run_dir_for(root: Path, method_id: str) -> Path:
    return ensure_dir(root / "runs" / method_id)


def write_config_yaml(run_dir: Path, config: dict) -> None:
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def get_run_logger(run_dir: Path, method_id: str) -> logging.Logger:
    logger = logging.getLogger(f"aq.{method_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(run_dir / "run.log", mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def write_layer_metrics_csv(run_dir: Path, rows: list[dict]) -> None:
    path = run_dir / "layer_metrics.csv"
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_optimization_trace_csv(run_dir: Path, rows: list[dict]) -> None:
    path = run_dir / "optimization_trace.csv"
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_model_metrics_json(run_dir: Path, metrics: dict) -> None:
    write_json(run_dir / "model_metrics.json", metrics)


def write_ppl_result_json(run_dir: Path, result: dict) -> None:
    write_json(run_dir / "ppl_result.json", result)


def write_watermark_result_json(run_dir: Path, result: dict) -> None:
    write_json(run_dir / "watermark_result.json", result)


def read_layer_metrics_csv(run_dir: Path) -> list[dict]:
    path = run_dir / "layer_metrics.csv"
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))
