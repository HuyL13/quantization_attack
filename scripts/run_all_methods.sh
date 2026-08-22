#!/usr/bin/env bash
# End-to-end driver for the adversarial quantization plan (plan section 2):
# 00_rtn4 baseline, then 01..08 in the mandated order, STOPPING immediately
# after the first method whose watermark gate returns PASS. Always runs
# 00_rtn4 first regardless of --only, since every later method's PPL gate is
# defined relative to it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${IF_AWQ_TIER0_ROOT:?Set IF_AWQ_TIER0_ROOT to the if_awq_tier0 checkout (reused RTN/PPL/watermark code)}"

OUTPUT="${OUTPUT:-$ROOT/results}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
CONFIG_DIR="${CONFIG_DIR:-$ROOT/configs}"

METHODS=(00_rtn4 01a_greedy_round 01b_layerwise_local 01c_blockwise_local 01d_global_kl \
         02_margin_aware 03_fragile_channel 04_stochastic_rounding \
         05_quantized_prefix 06_block_wise 07_periodic_refresh 08_two_pass_backward)

mkdir -p "$OUTPUT"

echo "[run_all_methods] baseline: 00_rtn4"
python -m aq.run_method --method 00_rtn4 --config "$CONFIG_DIR/00_rtn4.yaml" \
    --output "$OUTPUT" --device "$DEVICE" --dtype "$DTYPE"

RTN4_PPL=$(python -c "import json;print(json.load(open('$OUTPUT/runs/00_rtn4/ppl_result.json'))['wikitext2_ppl'])")
echo "[run_all_methods] RTN4 baseline PPL = $RTN4_PPL"

for method in "${METHODS[@]:1}"; do
    cfg="$CONFIG_DIR/${method}.yaml"
    echo "[run_all_methods] running $method (config: $cfg)"
    set +e
    OUT=$(python -m aq.run_method --method "$method" --config "$cfg" \
        --rtn4-ppl "$RTN4_PPL" --output "$OUTPUT" --device "$DEVICE" --dtype "$DTYPE")
    STATUS=$?
    set -e
    echo "$OUT"
    if [ $STATUS -ne 0 ]; then
        echo "[run_all_methods] $method crashed (exit $STATUS) - stopping run, inspect $OUTPUT/runs/$method/run.log"
        exit $STATUS
    fi
    if echo "$OUT" | grep -q "WATERMARK REMOVED WITH ACCEPTABLE PPL"; then
        echo "[run_all_methods] $method PASSED - stopping per plan section 2 (no further methods run)."
        break
    fi
done

echo "[run_all_methods] aggregating final comparison report"
python -m aq.aggregate_final_report --output "$OUTPUT"
echo "[run_all_methods] done. See $OUTPUT/reports/final_comparison.md"
