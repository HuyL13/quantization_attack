"""Loads the SAME fixed calibration set if_awq_tier0 uses for RTN/AWQ/GPTQ
(artifacts/calibration/pileval_seed42_128x512.jsonl) - never regenerated,
per the plan's "dùng đúng bộ calibration hiện có" requirement. This module
only tokenizes it into batches; it does not resample or reshuffle.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_calibration_texts(path: str | Path) -> list[str]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line)["raw_text"])
    return rows


def build_calibration_batches(
    tokenizer,
    texts: list[str],
    max_seq_len: int = 512,
    batch_size: int = 4,
    device: str = "cuda",
) -> list[dict]:
    import torch

    batches = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        batches.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    return batches
