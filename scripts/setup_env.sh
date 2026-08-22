#!/usr/bin/env bash
# One-time environment setup - mirrors multi_fp_tier0/scripts/setup_env.sh:
# prefer an already-active venv / vast.ai's /venv/main over building a fresh
# torch/CUDA stack from scratch.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${IF_AWQ_TIER0_ROOT:?Set IF_AWQ_TIER0_ROOT to the if_awq_tier0 checkout}"

if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -f /venv/main/bin/activate ]; then
        echo "[setup_env] using vast.ai's preinstalled /venv/main"
        source /venv/main/bin/activate
    else
        python3 -m venv .venv
        source .venv/bin/activate
        pip install --upgrade pip
    fi
fi

PIP_INSTALL="pip install"
command -v uv >/dev/null 2>&1 && PIP_INSTALL="uv pip install"
$PIP_INSTALL -r requirements.txt
$PIP_INSTALL -r "$IF_AWQ_TIER0_ROOT/requirements.txt"

mkdir -p results
python --version > results/environment.txt
pip freeze >> results/environment.txt
nvidia-smi > results/nvidia_smi.txt || echo "no GPU on this machine" > results/nvidia_smi.txt
echo "[setup_env] done."
