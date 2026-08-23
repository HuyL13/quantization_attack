#!/usr/bin/env bash
# End-to-end driver for HSQ (Hessian-Slack Quantization) - see
# HSQ_Hessian_Slack_Quantization_Implementation_Guide.md and aq/run_hsq.py.
#
# Requires 00_rtn4 to have already been run (reuses its baseline PPL, same
# as scripts/run_all_methods.sh) - run that first if results/runs/00_rtn4
# doesn't exist yet:
#   python -m aq.run_method --method 00_rtn4 --config configs/00_rtn4.yaml --output results
#
# Order: a 2-block timing sanity check first (a few minutes, NOT a real
# result - just measures real per-block wall-clock time on this GPU before
# committing to a full run), then gptq4 (the real baseline + HSQ's own
# "tau=0 must reduce to GPTQ" regression target), then hsq_v0, then hsq_v1
# (the paper-level "main" method per the guide's own roadmap). Does NOT
# stop early on watermark PASS across HSQ methods the way run_all_methods.sh
# does for the adversarial-quant plan, since GPTQ-family methods here are
# being compared to each other rather than gated as a mandated sequence -
# set STOP_ON_PASS=1 to opt into stopping after the first watermark PASS.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${IF_AWQ_TIER0_ROOT:?Set IF_AWQ_TIER0_ROOT to the if_awq_tier0 checkout (reused PPL/watermark eval code)}"

OUTPUT="${OUTPUT:-$ROOT/results}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
CONFIG_DIR="${CONFIG_DIR:-$ROOT/configs}"
METHODS=(${METHODS_OVERRIDE:-gptq4 hsq_v0 hsq_v1})
SKIP_SANITY="${SKIP_SANITY:-0}"
STOP_ON_PASS="${STOP_ON_PASS:-0}"
FORCE_WATERMARK_EVAL_FLAG=""
if [ "${FORCE_WATERMARK_EVAL:-0}" = "1" ]; then
    FORCE_WATERMARK_EVAL_FLAG="--force-watermark-eval"
fi

mkdir -p "$OUTPUT"

RTN4_PPL_PATH="$OUTPUT/runs/00_rtn4/ppl_result.json"
if [ ! -f "$RTN4_PPL_PATH" ]; then
    echo "[run_hsq] $RTN4_PPL_PATH not found - run 00_rtn4 first:"
    echo "    python -m aq.run_method --method 00_rtn4 --config configs/00_rtn4.yaml --output $OUTPUT --device $DEVICE --dtype $DTYPE"
    exit 1
fi
RTN4_PPL=$(python -c "import json;print(json.load(open('$RTN4_PPL_PATH'))['wikitext2_ppl'])")
echo "[run_hsq] RTN4 baseline PPL = $RTN4_PPL"

if [ "$SKIP_SANITY" != "1" ]; then
    echo "[run_hsq] sanity check: quantizing only the first 2 blocks with hsq_v0 to measure real per-block timing"
    echo "[run_hsq] (this run's PPL/watermark numbers are NOT meaningful - most of the model stays FP)"
    time python -m aq.run_hsq --method hsq_v0 --config "$CONFIG_DIR/hsq_v0_sanity.yaml" \
        --rtn4-ppl "$RTN4_PPL" --output "$OUTPUT/sanity" --device "$DEVICE" --dtype "$DTYPE"
    echo "[run_hsq] sanity check done - see timing above before the real runs below start"
fi

for method in "${METHODS[@]}"; do
    cfg="$CONFIG_DIR/${method}.yaml"
    echo "[run_hsq] running $method (config: $cfg)"
    set +e
    OUT=$(python -m aq.run_hsq --method "$method" --config "$cfg" \
        --rtn4-ppl "$RTN4_PPL" --output "$OUTPUT" --device "$DEVICE" --dtype "$DTYPE" $FORCE_WATERMARK_EVAL_FLAG)
    STATUS=$?
    set -e
    echo "$OUT"
    if [ $STATUS -ne 0 ]; then
        echo "[run_hsq] $method crashed (exit $STATUS) - stopping run, inspect $OUTPUT/runs/$method/run.log"
        exit $STATUS
    fi
    if [ "$STOP_ON_PASS" = "1" ] && echo "$OUT" | grep -q " -> PASS"; then
        echo "[run_hsq] $method PASSED - stopping (STOP_ON_PASS=1)."
        break
    fi
done

echo "[run_hsq] done. See $OUTPUT/runs/<method>/ and $OUTPUT/reports/<method>_report.md for each method."
