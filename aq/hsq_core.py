"""HSQ core: the GPTQ block-traversal / sequential-compensation loop (guide
section 10.6/10.7), forked only at the "which q do we commit" decision
(section 14). Three `mode`s share every other line of code:

  "gptq"    - plain GPTQ: always commit the nearest lattice point. Used both
              as the real baseline method and, internally, as pass 1 of
              hsq_v1 (to measure the real per-group GPTQ loss it budgets
              against).
  "hsq_v0"  - guide section 4: per-coordinate budget, C_i(q) <= (1+tau) *
              C_i^GPTQ(nearest at THIS coordinate) + eps0, argmax
              displacement among feasible candidates.
  "hsq_v1"  - guide section 5: a per-group cumulative budget
              B_G = (1+tau) * L_G^GPTQ, where L_G^GPTQ is measured by an
              actual prior GPTQ (nearest) pass over the SAME group (not just
              summed per-coordinate nearest costs) - `run_hsq_layer` below
              does that two-pass dance and feeds the group totals in here
              as `group_budgets`.

Candidate cost matches the exact per-coordinate OBS formula the real GPTQ
loop already computes for its own logging: (w-q)^2 / d^2 / 2, where `d` is
the CURRENT diagonal of the upper-triangular Cholesky-of-inverse-Hessian
factor at this block-local step (guide section 4.1's C_i(q), instantiated
with GPTQ's own numerics rather than a literal [H^-1]_ii lookup).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from aq.hsq_hessian import compute_inverse_hessian_cholesky
from aq.hsq_quantizer import dequantize, find_params, get_grid_candidates, quantize_int


@dataclass
class FasterQuantResult:
    hard_weight: torch.Tensor
    group_loss_totals: torch.Tensor  # (out_features, num_groups)
    metrics: dict = field(default_factory=dict)


def _num_groups(columns: int, group_size: int) -> int:
    if group_size in (-1, None):
        return 1
    return (columns + group_size - 1) // group_size


def _group_index(col: int, group_size: int) -> int:
    if group_size in (-1, None):
        return 0
    return col // group_size


def fasterquant(
    w_fp: torch.Tensor,
    H: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    mode: str = "gptq",
    sym: bool = False,
    percdamp: float = 0.01,
    blocksize: int = 128,
    tau: float = 0.02,
    candidate_radius: int = 2,
    eps0: float = 1e-10,
    group_budgets: torch.Tensor | None = None,
) -> FasterQuantResult:
    """w_fp: (out_features, in_features), the layer's FP weight. H: (in_features,
    in_features) calibration Hessian for that layer. `group_budgets`
    (out_features, num_groups) is required for mode="hsq_v1" (the per-group
    total loss budget measured by a prior nearest-only pass); ignored
    otherwise.
    """
    assert mode in ("gptq", "hsq_v0", "hsq_v1")
    device = w_fp.device
    out_features, columns = w_fp.shape
    W = w_fp.clone().float()

    Hinv, dead = compute_inverse_hessian_cholesky(H.float(), percdamp)
    W[:, dead] = 0.0

    maxq = 2**bits - 1
    num_groups = _num_groups(columns, group_size)
    Q = torch.zeros_like(W)
    scale = torch.zeros(out_features, columns, device=device)
    zero = torch.zeros(out_features, columns, device=device)
    group_loss_totals = torch.zeros(out_features, num_groups, device=device)
    group_used_budget = torch.zeros(out_features, num_groups, device=device)

    counts = {"nearest": 0, "plusminus1": 0, "plusminus2_or_more": 0, "fallback_to_nearest": 0}
    total_coords = out_features * columns

    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):
            col = i1 + i
            w_col = W1[:, i]
            d = Hinv1[i, i].clamp(min=1e-12)

            if col % max(group_size, 1) == 0 or group_size in (-1, None):
                g_end = columns if group_size in (-1, None) else min(col + group_size, columns)
                cur_scale, cur_zero, _ = find_params(W[:, col:g_end], bits, sym)
                scale[:, col:g_end] = cur_scale
                zero[:, col:g_end] = cur_zero
            s_col = scale[:, col]
            z_col = zero[:, col]

            q_nearest_int = quantize_int(w_col, s_col, z_col, maxq)
            q_nearest = dequantize(q_nearest_int, s_col, z_col)
            nearest_cost = (w_col - q_nearest) ** 2 / (d**2) / 2

            g_idx = _group_index(col, group_size)
            group_loss_totals[:, g_idx] += nearest_cost

            if mode == "gptq":
                q = q_nearest
                chosen_cost = nearest_cost
                offset_used = torch.zeros_like(w_col)
            else:
                cands, codes = get_grid_candidates(w_col, s_col, z_col, maxq, candidate_radius)
                cost = (w_col.unsqueeze(0) - cands) ** 2 / (d**2) / 2
                nearest_row = candidate_radius  # offsets range [-r, r], index r == offset 0

                if mode == "hsq_v0":
                    budget = (1 + tau) * nearest_cost + eps0
                    feasible = cost <= budget.unsqueeze(0)
                else:  # hsq_v1
                    total_budget = group_budgets[:, g_idx]
                    remaining = total_budget - group_used_budget[:, g_idx]
                    feasible = cost <= (remaining.unsqueeze(0) + eps0)

                feasible[nearest_row] = True  # always allow falling back to nearest
                displacement = (w_col.unsqueeze(0) - cands) ** 2
                masked_disp = torch.where(feasible, displacement, torch.full_like(displacement, -1.0))
                best_idx = masked_disp.argmax(dim=0)
                q = cands.gather(0, best_idx.unsqueeze(0)).squeeze(0)
                chosen_cost = cost.gather(0, best_idx.unsqueeze(0)).squeeze(0)
                offset_used = (best_idx - nearest_row).float()
                if mode == "hsq_v1":
                    group_used_budget[:, g_idx] += chosen_cost
                counts["fallback_to_nearest"] += int((best_idx == nearest_row).sum().item())

            abs_offset = offset_used.abs()
            counts["nearest"] += int((abs_offset == 0).sum().item())
            counts["plusminus1"] += int((abs_offset == 1).sum().item())
            counts["plusminus2_or_more"] += int((abs_offset >= 2).sum().item())

            Q1[:, i] = q
            err1 = (w_col - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1

        Q[:, i1:i2] = Q1
        if i2 < columns:
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    hard_weight = Q.to(w_fp.dtype)
    metrics = {
        "nearest_fraction": counts["nearest"] / total_coords,
        "plusminus1_fraction": counts["plusminus1"] / total_coords,
        "farther_fraction": counts["plusminus2_or_more"] / total_coords,
        "fallback_to_nearest_count": counts["fallback_to_nearest"],
        "predicted_loss": float(group_loss_totals.sum().item()) if mode == "gptq" else None,
    }
    return FasterQuantResult(hard_weight=hard_weight, group_loss_totals=group_loss_totals, metrics=metrics)


def run_hsq_layer(
    w_fp: torch.Tensor,
    H: torch.Tensor,
    mode: str,
    bits: int = 4,
    group_size: int = 128,
    sym: bool = False,
    percdamp: float = 0.01,
    blocksize: int = 128,
    tau: float = 0.02,
    candidate_radius: int = 2,
    eps0: float = 1e-10,
) -> FasterQuantResult:
    """Entry point `run_hsq.py` calls per layer. For mode="hsq_v1" this runs
    the required two passes (guide section 5): pass 1 is plain nearest-only
    GPTQ over the SAME Hessian to measure each group's real predicted loss
    L_G^GPTQ (including intra-group sequential compensation dynamics, not
    just a sum of independent per-coordinate nearest costs), pass 2 is the
    actual HSQ candidate search budgeted against (1+tau)*L_G^GPTQ. For
    "gptq"/"hsq_v0" this is a single pass.
    """
    if mode != "hsq_v1":
        return fasterquant(
            w_fp, H, bits=bits, group_size=group_size, mode=mode, sym=sym, percdamp=percdamp,
            blocksize=blocksize, tau=tau, candidate_radius=candidate_radius, eps0=eps0,
        )

    baseline = fasterquant(
        w_fp, H, bits=bits, group_size=group_size, mode="gptq", sym=sym, percdamp=percdamp,
        blocksize=blocksize, tau=tau, candidate_radius=candidate_radius, eps0=eps0,
    )
    group_budgets = (1 + tau) * baseline.group_loss_totals
    return fasterquant(
        w_fp, H, bits=bits, group_size=group_size, mode="hsq_v1", sym=sym, percdamp=percdamp,
        blocksize=blocksize, tau=tau, candidate_radius=candidate_radius, eps0=eps0,
        group_budgets=group_budgets,
    )
