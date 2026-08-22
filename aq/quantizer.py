"""Core adversarial quantizer: one shared class covering plan methods 1-4
(they all reuse the same core objective and only add one more learnable
variable each - the plan explicitly forbids introducing a new loss per
method). Methods 5-8 (calibration-strategy variants) wrap this class from
aq/calibration_strategies.py without touching its math.

RTN4 base grid (pre_round, scale, zero_point, max_int) comes from
if_awq_tier0's AWQQuantizerXL.quantize_weight_groupwise_raw - reused
verbatim, never reimplemented (plan section 4).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from aq.rounding import h_hard, h_soft, init_alpha_from_pre_round


@dataclass
class AdversarialQuantConfig:
    bits: int = 4
    group_size: int = 128
    optimize_scale: bool = False          # method 2
    use_codebook: bool = False            # method 3
    n_codebook_centroids: int = 16
    use_sensitivity: bool = False         # method 4
    lambda_distance: float = 0.1          # weight on the (rewarded) distance term
    round_reg_weight: float = 1.0         # AdaRound-style push toward hard 0/1 (see rounding_regularizer)
    steps: int = 200
    lr: float = 1e-2


class AdversarialLinearQuantizer(nn.Module):
    """Wraps one nn.Linear weight's RTN grid with learnable adversarial
    variables. `w` is the ORIGINAL fp weight (out_features, in_features);
    `rtn_state` is the frozen output of quantize_weight_groupwise_raw(w).
    """

    def __init__(self, w: torch.Tensor, rtn_state, cfg: AdversarialQuantConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("w_fp", w.detach().clone().float())
        self.register_buffer("pre_round", rtn_state.pre_round.detach().clone().float())
        self.register_buffer("scale0", rtn_state.scale.detach().clone().float())
        self.register_buffer("zero_point", rtn_state.zero_point.detach().clone().float())
        self.max_int = rtn_state.max_int
        self.in_features = rtn_state.in_features
        self.padded_in_features = rtn_state.padded_in_features
        self.original_dtype = rtn_state.original_dtype

        alpha0 = init_alpha_from_pre_round(self.pre_round)
        self.alpha = nn.Parameter(alpha0)

        if cfg.optimize_scale:
            # multiplicative log-scale perturbation, initialized at 0 so the
            # starting point is exactly scale0 (RTN4's own scale).
            self.log_scale_mult = nn.Parameter(torch.zeros_like(self.scale0))
        else:
            self.log_scale_mult = None

        if cfg.use_codebook:
            out_features, padded_in = self.pre_round.shape
            n_groups = padded_in // cfg.group_size
            k = cfg.n_codebook_centroids
            # centroids initialized at the uniform integer grid {0..max_int}
            # scaled to k points, per group; shape (out_features, n_groups, k)
            base_levels = torch.linspace(0, rtn_state.max_int, k)
            self.codebook = nn.Parameter(
                base_levels.view(1, 1, k).repeat(out_features, n_groups, 1).clone()
            )
            # per-weight assignment logits over the k centroids
            self.assign_logits = nn.Parameter(torch.zeros(out_features, n_groups, cfg.group_size, k))
        else:
            self.codebook = None
            self.assign_logits = None

        self.sensitivity: torch.Tensor | None = None  # set externally for method 4

    def effective_scale(self) -> torch.Tensor:
        if self.log_scale_mult is None:
            return self.scale0
        return self.scale0 * torch.exp(self.log_scale_mult)

    def _dequant_from_int(self, int_weight: torch.Tensor) -> torch.Tensor:
        scale = self.effective_scale()
        dq = (int_weight - self.zero_point) * scale
        if self.padded_in_features > self.in_features:
            dq = dq[:, : self.in_features]
        return dq

    def _codebook_soft_int(self) -> torch.Tensor:
        out_features, n_groups, group_size, k = self.assign_logits.shape
        probs = torch.softmax(self.assign_logits, dim=-1)  # (out, n_groups, group_size, k)
        centroids = self.codebook.unsqueeze(2)  # (out, n_groups, 1, k)
        soft_val = (probs * centroids).sum(dim=-1)  # (out, n_groups, group_size)
        return soft_val.reshape(out_features, n_groups * group_size).clamp(0, self.max_int)

    def _codebook_hard_int(self) -> torch.Tensor:
        out_features, n_groups, group_size, k = self.assign_logits.shape
        idx = self.assign_logits.argmax(dim=-1)  # (out, n_groups, group_size)
        centroids = self.codebook  # (out, n_groups, k)
        hard_val = torch.gather(centroids, 2, idx)
        return hard_val.reshape(out_features, n_groups * group_size).clamp(0, self.max_int)

    def soft_weight(self) -> torch.Tensor:
        if self.use_codebook_mode():
            int_w = self._codebook_soft_int()
        else:
            floor_val = torch.floor(self.pre_round)
            int_w = (floor_val + h_soft(self.alpha)).clamp(0, self.max_int)
        return self._dequant_from_int(int_w).to(self.original_dtype)

    def hard_weight(self) -> torch.Tensor:
        with torch.no_grad():
            if self.use_codebook_mode():
                int_w = self._codebook_hard_int()
            else:
                floor_val = torch.floor(self.pre_round)
                int_w = (floor_val + h_hard(self.alpha)).clamp(0, self.max_int)
            return self._dequant_from_int(int_w).to(self.original_dtype)

    def rtn_baseline_weight(self) -> torch.Tensor:
        with torch.no_grad():
            int_w = torch.round(self.pre_round).clamp(0, self.max_int)
            dq = (int_w - self.zero_point) * self.scale0
            if self.padded_in_features > self.in_features:
                dq = dq[:, : self.in_features]
            return dq.to(self.original_dtype)

    def use_codebook_mode(self) -> bool:
        return self.cfg.use_codebook and self.codebook is not None

    def hard_int_grid(self) -> torch.Tensor:
        """The discrete rounding decision as an integer tensor, used for
        rounding_flip_ratio logging against the RTN4 baseline's own grid.
        """
        with torch.no_grad():
            if self.use_codebook_mode():
                return self._codebook_hard_int()
            floor_val = torch.floor(self.pre_round)
            return (floor_val + h_hard(self.alpha)).clamp(0, self.max_int)

    def rounding_regularizer(self) -> torch.Tensor:
        """Standard AdaRound-style push toward a hard 0/1 rounding decision.

        Without this term the objective's gradient at the exact RTN4 starting
        point is numerically degenerate: soft_weight() reconstructs w_fp to
        float32 precision there (floor(pre_round) + h_soft(alpha0) == pre_round
        by construction), so the distance/KL terms both start at a residual
        near machine epsilon and give almost no usable gradient (verified: an
        aggressive SGD step on -weight_distance() alone moved every alpha by
        <1e-6). This term has a normal, well-scaled gradient everywhere and is
        what actually drives alpha away from its degenerate starting point;
        the KL/distance terms then determine WHICH side (0 or 1) each element
        should converge to as training proceeds.
        """
        if self.use_codebook_mode():
            from aq.metrics import codebook_occupancy_entropy

            probs = torch.softmax(self.assign_logits, dim=-1)
            _, entropy = codebook_occupancy_entropy(probs)
            max_entropy = torch.log(torch.tensor(float(self.assign_logits.shape[-1])))
            return entropy / max_entropy
        h = h_soft(self.alpha)
        return (4.0 * h * (1.0 - h)).mean()

    def weight_distance(self) -> torch.Tensor:
        from aq.metrics import weight_relative_distance, weight_relative_distance_weighted

        what = self.soft_weight()
        w = self.w_fp.to(what.dtype)
        if self.cfg.use_sensitivity and self.sensitivity is not None:
            return weight_relative_distance_weighted(w, what, self.sensitivity.to(what.dtype))
        return weight_relative_distance(w, what)
