"""Shared helpers: locate + reuse if_awq_tier0's RTN/AWQ backend, PPL eval and
IF-SFT watermark verification verbatim (plan section 4: "dùng đúng code eval
PPL hiện có"). No copy/reimplementation of that code lives here.
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

IF_SFT_MODEL_ID = "cnut1648/LLaMA2-7B-fingerprinted-SFT"
VANILLA_MODEL_ID = "NousResearch/Llama-2-7b-hf"
FINGERPRINT_TARGET = "ハリネズミ"


def if_awq_tier0_root() -> Path:
    env = os.environ.get("IF_AWQ_TIER0_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
    else:
        root = (Path(__file__).resolve().parent.parent.parent / "if_awq_tier0").resolve()
    if not (root / "src" / "quantization").exists():
        raise RuntimeError(
            f"IF_AWQ_TIER0_ROOT={root} does not look like an if_awq_tier0 checkout "
            "(missing src/quantization). Set the IF_AWQ_TIER0_ROOT env var explicitly."
        )
    return root


def ensure_if_awq_tier0_on_path() -> Path:
    root = if_awq_tier0_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def default_fingerprint_keys_path() -> Path:
    return if_awq_tier0_root() / "artifacts" / "fingerprints" / "if_sft_llama2_keys.json"


def default_calibration_path() -> Path:
    return if_awq_tier0_root() / "artifacts" / "calibration" / "pileval_seed42_128x512.jsonl"


def load_yaml_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def configure_gpu_performance() -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def best_attn_implementation() -> str:
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


def load_tokenizer(model_id_or_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(model_id_or_path: str, device: str = "cuda", dtype: str = "bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM

    torch_dtype = getattr(torch, dtype)
    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": best_attn_implementation(),
    }
    if device.startswith("cuda"):
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_id_or_path, **kwargs)
    if not device.startswith("cuda"):
        model.to(device)
    model.eval()
    return model


def free_model(model) -> None:
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_memory_bytes() -> int | None:
    import torch

    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return None


def get_transformer_linear_layers(model) -> "dict[str, Any]":
    """name -> nn.Linear for every projection inside model.model.layers[*]
    (q/k/v/o + gate/up/down), in depth order. lm_head/embeddings excluded -
    the plan's adversarial objective only concerns the transformer body.
    """
    import torch.nn as nn

    layers: dict[str, Any] = {}
    for block_idx, block in enumerate(model.model.layers):
        for local_name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                layers[f"model.layers.{block_idx}.{local_name}"] = module
    return layers


def get_block_layer_groups(model) -> "list[list[str]]":
    """One group per transformer block, each holding that block's linear
    layer names in the order returned by get_transformer_linear_layers -
    used by the block-wise method (method 6) to optimize a whole block
    jointly instead of matrix-by-matrix.
    """
    groups: list[list[str]] = []
    for block_idx, block in enumerate(model.model.layers):
        import torch.nn as nn

        names = [
            f"model.layers.{block_idx}.{local_name}"
            for local_name, module in block.named_modules()
            if isinstance(module, nn.Linear)
        ]
        if names:
            groups.append(names)
    return groups
