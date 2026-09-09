#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${IF_AWQ_TIER0_ROOT:?Set IF_AWQ_TIER0_ROOT to the if_awq_tier0 checkout}"

OUTPUT="${OUTPUT:-$ROOT/results_rtn3}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
export PYTHONUNBUFFERED=1

echo "[rtn3] output=$OUTPUT"
python -m aq.run_method \
    --method 00_rtn3 \
    --config "$ROOT/configs/00_rtn3.yaml" \
    --output "$OUTPUT" \
    --device "$DEVICE" \
    --dtype "$DTYPE"

echo "[rtn3] summary"
python - "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1]) / "runs" / "00_rtn3"
ppl = json.loads((run_dir / "ppl_result.json").read_text(encoding="utf-8"))
watermark = json.loads((run_dir / "watermark_result.json").read_text(encoding="utf-8"))["summary"]
summary = {
    "ppl": ppl["wikitext2_ppl"],
    "fsr_exact": watermark["fsr_exact"],
    "fsr_contains": watermark["fsr_contains"],
}
(Path(sys.argv[1]) / "rtn3_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2))
PY
