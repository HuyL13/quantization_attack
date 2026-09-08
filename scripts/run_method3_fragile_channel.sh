#!/usr/bin/env bash
# Run the RTN4 baseline and Method 3 (fragile channel), then print the four
# comparison metrics needed for the experiment:
#   RTN4 PPL, Method 3 PPL, RTN4 FSR, Method 3 FSR.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${IF_AWQ_TIER0_ROOT:?Set IF_AWQ_TIER0_ROOT to the if_awq_tier0 checkout (reused watermark eval code)}"

OUTPUT="${OUTPUT:-$ROOT/results_method3}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
CONFIG_DIR="${CONFIG_DIR:-$ROOT/configs}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"

mkdir -p "$OUTPUT"

echo "[method3] running RTN4 baseline"
python -m aq.run_method \
    --method 00_rtn4 \
    --config "$CONFIG_DIR/00_rtn4.yaml" \
    --output "$OUTPUT" \
    --device "$DEVICE" \
    --dtype "$DTYPE"

RTN4_PPL=$(python -c "import json;print(json.load(open('$OUTPUT/runs/00_rtn4/ppl_result.json'))['wikitext2_ppl'])")
echo "[method3] RTN4 PPL = $RTN4_PPL"

RTN4_WATERMARK_PATH="$OUTPUT/runs/00_rtn4/watermark_result.json"
if [ ! -f "$RTN4_WATERMARK_PATH" ]; then
    echo "[method3] running RTN4 watermark verification"
    METHOD3_OUTPUT="$OUTPUT" METHOD3_CONFIG_DIR="$CONFIG_DIR" METHOD3_DEVICE="$DEVICE" \
        METHOD3_DTYPE="$DTYPE" METHOD3_MAX_NEW_TOKENS="$MAX_NEW_TOKENS" python - <<'PY'
import os
import time
from pathlib import Path

from aq.common import (
    IF_SFT_MODEL_ID,
    default_fingerprint_keys_path,
    ensure_if_awq_tier0_on_path,
    get_transformer_linear_layers,
    load_causal_lm,
    load_tokenizer,
    load_yaml_config,
    read_json,
    free_model,
)
from aq.logging_utils import write_watermark_result_json
from aq.run_method import _apply_rtn4_baseline

ensure_if_awq_tier0_on_path()
from src.verify_fingerprint import run_verification, summarize

output = Path(os.environ["METHOD3_OUTPUT"])
config_dir = Path(os.environ["METHOD3_CONFIG_DIR"])
device = os.environ["METHOD3_DEVICE"]
dtype = os.environ["METHOD3_DTYPE"]
max_new_tokens = int(os.environ["METHOD3_MAX_NEW_TOKENS"])

cfg = load_yaml_config(config_dir / "00_rtn4.yaml")
model_cfg = cfg.get("model", {})
method_cfg = cfg.get("method", {})
model_id = model_cfg.get("id", IF_SFT_MODEL_ID)

model = load_causal_lm(model_id, device=device, dtype=dtype)
tokenizer = load_tokenizer(model_id)
_apply_rtn4_baseline(get_transformer_linear_layers(model))

fingerprints = read_json(method_cfg.get("fingerprint_keys_path", default_fingerprint_keys_path()))
t0 = time.time()
per_key = run_verification(
    model,
    tokenizer,
    fingerprints,
    do_sample=False,
    max_new_tokens=max_new_tokens,
    device=device,
)
summary = summarize(per_key, evaluation_time_seconds=time.time() - t0)
write_watermark_result_json(output / "runs" / "00_rtn4", {"summary": summary, "per_key": per_key})
print(f"[method3] RTN4 FSR exact = {summary['fsr_exact']}")
print(f"[method3] RTN4 FSR contains = {summary['fsr_contains']}")

free_model(model)
PY
else
    echo "[method3] reusing existing RTN4 watermark result at $RTN4_WATERMARK_PATH"
fi

echo "[method3] running 03_fragile_channel"
python -m aq.run_method \
    --method 03_fragile_channel \
    --config "$CONFIG_DIR/03_fragile_channel.yaml" \
    --rtn4-ppl "$RTN4_PPL" \
    --output "$OUTPUT" \
    --device "$DEVICE" \
    --dtype "$DTYPE"

SUMMARY_PATH="$OUTPUT/method3_fragile_channel_summary.json"
METHOD3_OUTPUT="$OUTPUT" METHOD3_SUMMARY_PATH="$SUMMARY_PATH" python - <<'PY'
import json
import os
from pathlib import Path

output = Path(os.environ["METHOD3_OUTPUT"])
summary_path = Path(os.environ["METHOD3_SUMMARY_PATH"])

def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def ppl(method):
    return load_json(output / "runs" / method / "ppl_result.json")["wikitext2_ppl"]

def fsr(method):
    summary = load_json(output / "runs" / method / "watermark_result.json")["summary"]
    return {
        "fsr_exact": summary["fsr_exact"],
        "fsr_contains": summary["fsr_contains"],
    }

result = {
    "rtn4": {
        "ppl": ppl("00_rtn4"),
        **fsr("00_rtn4"),
    },
    "method3_fragile_channel": {
        "ppl": ppl("03_fragile_channel"),
        **fsr("03_fragile_channel"),
    },
}

summary_path.parent.mkdir(parents=True, exist_ok=True)
with open(summary_path, "w", encoding="utf-8") as fh:
    json.dump(result, fh, indent=2)

print("[method3] final summary")
print(f"  RTN4 PPL:                 {result['rtn4']['ppl']}")
print(f"  Method 3 PPL:             {result['method3_fragile_channel']['ppl']}")
print(f"  RTN4 FSR exact:           {result['rtn4']['fsr_exact']}")
print(f"  RTN4 FSR contains:        {result['rtn4']['fsr_contains']}")
print(f"  Method 3 FSR exact:       {result['method3_fragile_channel']['fsr_exact']}")
print(f"  Method 3 FSR contains:    {result['method3_fragile_channel']['fsr_contains']}")
print(f"[method3] wrote {summary_path}")
PY
