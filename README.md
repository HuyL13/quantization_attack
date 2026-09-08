# Adversarial Quantization of IF-SFT LLaMA2-7B

Implements `adversarial_quantization_if_sft_llama2_7b_plan.md`: 9 methods
(RTN4 baseline + 8 adversarial variants), each tested through a mandatory
stop-early gate (PPL vs RTN4 -> watermark FSR), sharing one core objective

    L = KL(M_FP, M_Q) - lambda * D(W, Q(W))

which REWARDS large weight distance while trying to keep behavior (KL)
close to the FP parent. Reuses `if_awq_tier0`'s RTN grid
(`AWQQuantizerXL.quantize_weight_groupwise_raw`) and IF-SFT watermark
verification. WikiText-2 PPL follows the block evaluator in `aq/ppl_eval.py`.

## Layout

- `aq/quantizer.py` - core `AdversarialLinearQuantizer`: AdaRound-style
  differentiable rounding (`aq/rounding.py`), optional per-group scale
  (method 2), optional 16-centroid codebook (method 3), optional
  sensitivity weighting (method 4, via `aq/sensitivity.py`).
- `aq/optimizer_core.py` - shared per-layer optimization loop (patches one
  `nn.Linear`'s forward, computes `KL(M_FP, M_Q) - lambda*D + round_reg`,
  backprops into the quantizer's own parameters only).
- `aq/calibration_strategies.py` - methods 5-8 as orchestration wrappers
  around the same core loop (quantized-prefix, block-wise, periodic
  refresh, two-pass backward correction) - see the module docstring for the
  exact strategy definitions.
- `aq/decision_flow.py` - the mandated stop-early gate logic (pure
  functions, GPU-independent, unit-tested directly).
- `aq/run_method.py` - CLI: runs ONE method end to end (quantize -> PPL ->
  gate -> watermark -> gate -> logs/report).
- `aq/aggregate_final_report.py` - builds `reports/final_comparison.md`
  from whatever `runs/*/` directories exist on disk.
- `scripts/run_all_methods.sh` - drives `00_rtn4` then `01..08` in order,
  stopping immediately after the first PASS.

## Running (on the rented GPU box)

```bash
export IF_AWQ_TIER0_ROOT=/path/to/if_awq_tier0   # cloned from huynguyenquang-collab/if_awq
bash scripts/setup_env.sh
bash scripts/run_all_methods.sh                   # OUTPUT=./results by default
```

Monitor a running method:

```bash
tail -f results/runs/<method_id>/run.log
```

To run the replacement Method 3 once:

```bash
python -m aq.run_method --method 03_global_far_round --config configs/03_global_far_round.yaml \
    --rtn4-ppl <value from results/runs/00_rtn4/ppl_result.json> --output results \
    --force-watermark-eval
```

Method 3 now ranks the fragility score across all target projections with
one CPU histogram threshold. Run the required seven-point reproduction sweep
with `bash scripts/run_global_far_round_sweep.sh`; results are written under
`results_global_far_round/`, including CSV/JSON/Markdown summaries and the
PPL-FSR curve.

## Testing

All tests run on CPU with no GPU and no real 7B checkpoint (synthetic
tiny linear layers / a fake HF-shaped model stand in for the real model):

```bash
python3 -m pytest -q
```

Covers: KL/distance/cosine/flip-ratio math against hand-computed values,
the AdaRound rounding relaxation's start-at-RTN4 invariant and its
gradient-vanishing-at-init failure mode (and the rounding-regularizer fix
for it), the RTN backend against if_awq_tier0's real
`quantize_weight_groupwise_raw`, all four calibration strategies (commit
ordering for quantized-prefix, joint loss for block-wise, reference-logit
immutability for two-pass), the stop-early decision flow, logging/report
generation, and full-pipeline dispatch (`_apply_rtn4_baseline` +
`_run_strategy` for every one of the 9 methods) against a fake HF-shaped
model. **Not covered without a GPU + the real checkpoint**: actual
KL/PPL/watermark numbers on the real 7B model, wall-clock/VRAM behavior at
full scale, and whether any method actually reaches PASS - that only comes
from a real run on the rented box.

## Disk & GPU sizing for the rented server

Recommend the **same class of box already used for Phase A: a single A100
40GB, but with more disk this time** - the driver of extra disk isn't the
optimization itself (each method touches one `nn.Linear` at a time, so GPU
memory stays close to a normal AWQ/GPTQ run: one resident 7B model + small
per-layer optimizer state, not two full 7B models), it's how many quantized
checkpoints a *worst-case* full 9-method sweep can accumulate:

| Item | Size | Notes |
|---|---|---|
| HF cache: `cnut1648/LLaMA2-7B-fingerprinted-SFT` (bf16) | ~13.5 GB | base checkpoint, downloaded once |
| HF cache: vanilla `NousResearch/Llama-2-7b-hf` (only if WikiText-2/lm-eval needs it) | ~13.5 GB | reused from if_awq_tier0 patterns |
| venv (torch/CUDA/transformers stack) | ~8-10 GB | reuse vast.ai's `/venv/main` when present, per `scripts/setup_env.sh` |
| Per-method output: 1 full bf16 checkpoint (no compression - "fake-quant" dequantized weights, same as Phase A's RTN) | ~13.5 GB **each** | only materialized if you keep it (see below) |
| Per-method logs (`layer_metrics.csv`, `optimization_trace.csv`, JSON, PNGs) | a few MB | negligible |
| Calibration set + fingerprint keys | <5 MB | already exists in if_awq_tier0/artifacts |

Worst case (no method ever PASSes, all 9 run, and every checkpoint is kept
simultaneously): 13.5 GB x 9 ≈ **122 GB** just for quantized checkpoints,
on top of ~35 GB for caches/venv. That is the number to budget against,
not the best case (early PASS after method 1 or 2 needs only ~1-2 extra
checkpoints).

**Recommendation: 150 GB disk minimum, 200 GB comfortable** if you'd rather
not babysit cleanup between methods. If you want to be more conservative
on cost, `aq/run_method.py`'s per-method `run_dir` already writes the hard
weights back into the live in-memory model for PPL/watermark eval without
requiring a separate on-disk save/reload step (unlike Phase A's RTN/AWQ/GPTQ
matrix, which had to serialize-then-reload because those quantizers used
custom packed formats) - so nothing is written to disk per-method *unless*
you explicitly add a checkpoint-saving step for a method that passes and
you want to keep. With that in mind, **100 GB is workable** if you only
intend to keep the checkpoint of whichever method(s) look interesting
rather than all nine, but 150 GB removes the need to think about it during
the run.

GPU: **A100 40GB is enough.** Peak memory per method ≈ one resident 7B
model in bf16 (~14 GB) + activations for the calibration batches + a
handful of small per-layer optimizer tensors (rounding/scale/codebook
parameters are the size of ONE `nn.Linear`'s weight at a time, at most
tens of millions of parameters, not the whole 7B model) - well within the
same budget the Phase A AWQ/GPTQ runs already used successfully on this
box. `06_block_wise` is the one method that optimizes several matrices at
once (a full transformer block, ~7 matrices), which is still small
relative to the full model.

## HSQ (Hessian-Slack Quantization) - a separate GPTQ-family pipeline

`aq/run_hsq.py` implements a different quantization objective from the
adversarial-quant plan above: instead of REWARDING weight distance while
trying to preserve KL, HSQ still MINIMIZES the same second-order loss
surrogate GPTQ uses, but searches for the FARTHEST lattice point whose
predicted Hessian-weighted loss stays within a budget relative to GPTQ's
own nearest-rounding solution:

    GPTQ:  min_q  e^T H e
    HSQ:   max_q  D(e)   s.t.  e^T H e <= (1+tau) * L_GPTQ

Three `--method` values in `aq/run_hsq.py` (`gptq4`, `hsq_v0`, `hsq_v1`)
share one GPTQ-style sequential block-traversal driver - blocks are
quantized in depth order and each block's calibration inputs are captured
by running the calibration set through the model AS IT CURRENTLY STANDS
(earlier blocks already quantized in place), the same activation
propagation GPTQ's own driver relies on. `hsq_v0` budgets per-coordinate
against that coordinate's own GPTQ-nearest cost; `hsq_v1` budgets
cumulatively per group against a real prior GPTQ (nearest-only) pass over
that same group (two passes per layer) - `aq/hsq_core.py`'s module
docstring has the full derivation-to-code mapping.

```bash
python -m aq.run_method --method 00_rtn4 --config configs/00_rtn4.yaml --output results   # baseline, if not already run
bash scripts/run_hsq.sh   # runs a 2-block timing sanity check, then gptq4 -> hsq_v0 -> hsq_v1
```

Tune `configs/gptq4.yaml` / `configs/hsq_v0.yaml` / `configs/hsq_v1.yaml`
(`tau`, `candidate_radius`, `percdamp`, `blocksize`, `group_size`). Env
vars for `scripts/run_hsq.sh`: `OUTPUT`, `SKIP_SANITY=1` (skip the timing
check), `STOP_ON_PASS=1` (stop after the first watermark PASS),
`FORCE_WATERMARK_EVAL=1` (evaluate watermark even on PPL-gate failures,
diagnostic only), `METHODS_OVERRIDE="hsq_v0 hsq_v1"` (subset/reorder).

**Sanity checks before trusting a result** (mandatory per the HSQ design
doc): with `tau=0`, `hsq_v0`/`hsq_v1` must produce weights identical to
`gptq4` (`tests/test_hsq_core.py::test_hsq_v0_with_zero_tau_matches_gptq_nearest_choice`
checks this on synthetic data) - if a real run's `hsq_v0` config with a
tiny `tau` looks wildly different from `gptq4`, something's wrong before
you even look at PPL. `configs/hsq_v0_sanity.yaml` (`max_blocks: 2`) is for
timing only, never a real result.

**No CPU/GPU-real-checkpoint test coverage yet** beyond what's listed
above - `tests/test_hsq_quantizer.py`, `tests/test_hsq_hessian.py`,
`tests/test_hsq_core.py`, `tests/test_run_hsq.py` all run on synthetic
tensors / a tiny fake HF-shaped model, same caveat as the rest of this
repo: actual PPL/watermark numbers on the real 7B model only come from a
run on the rented box.
