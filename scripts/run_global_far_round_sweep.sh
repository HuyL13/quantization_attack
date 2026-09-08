#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${IF_AWQ_TIER0_ROOT:?Set IF_AWQ_TIER0_ROOT to the if_awq_tier0 checkout}"

OUTPUT="${OUTPUT:-$ROOT/results_global_far_round}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
CONFIG="${CONFIG:-$ROOT/configs/03_global_far_round.yaml}"
RHO_VALUES=(0.025 0.05 0.075 0.10 0.125 0.15 0.20)
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT"

echo "[global-far-round] running RTN4 baseline once"
python -m aq.run_method \
    --method 00_rtn4 \
    --config "$ROOT/configs/00_rtn4.yaml" \
    --output "$OUTPUT/baseline" \
    --device "$DEVICE" \
    --dtype "$DTYPE"
RTN4_PPL=$(python -c "import json;print(json.load(open('$OUTPUT/baseline/runs/00_rtn4/ppl_result.json'))['wikitext2_ppl'])")

for rho in "${RHO_VALUES[@]}"; do
    echo "[global-far-round] rho=$rho"
    python -m aq.run_method \
        --method 03_global_far_round \
        --config "$CONFIG" \
        --aggressive-fraction "$rho" \
        --rtn4-ppl "$RTN4_PPL" \
        --output "$OUTPUT/rho_$rho" \
        --device "$DEVICE" \
        --dtype "$DTYPE" \
        --force-watermark-eval
done

python -m aq.report_global_far_round_sweep --output "$OUTPUT"
echo "[global-far-round] done: $OUTPUT/global_far_round_sweep.md"
