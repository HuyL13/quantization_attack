"""Differentiable relaxation of the discrete rounding choice r_i in {0,1}
(plan method 1: "nới lỏng round-to-nearest thành lựa chọn khả vi giữa floor
và ceil"). Uses the AdaRound rectified-sigmoid trick (Nagel et al., 2020) so
the continuous relaxation can reach exact 0/1 at convergence instead of
saturating asymptotically like a plain sigmoid.

    h(alpha) = clamp(sigmoid(alpha) * (zeta - gamma) + gamma, 0, 1)

with zeta > 1, gamma < 0, so h can hit 0 or 1 well before sigmoid(alpha)
saturates. w_int_soft = floor(pre_round) + h(alpha); at eval/export time,
w_int_hard = floor(pre_round) + (h(alpha) > 0.5).
"""
from __future__ import annotations

import torch

ZETA = 1.1
GAMMA = -0.1


def h_soft(alpha: torch.Tensor, zeta: float = ZETA, gamma: float = GAMMA) -> torch.Tensor:
    return torch.clamp(torch.sigmoid(alpha) * (zeta - gamma) + gamma, 0.0, 1.0)


def h_hard(alpha: torch.Tensor, zeta: float = ZETA, gamma: float = GAMMA) -> torch.Tensor:
    return (h_soft(alpha, zeta, gamma) > 0.5).float()


def init_alpha_from_pre_round(pre_round: torch.Tensor, zeta: float = ZETA, gamma: float = GAMMA) -> torch.Tensor:
    """Initialize alpha so h_soft(alpha) equals the fractional part v of
    pre_round at step 0 - i.e. the optimizer starts exactly at the
    round-to-nearest (RTN) solution, per the plan's "chỉ khởi tạo từ RTN4,
    không đổi giá trị ban đầu" requirement, and only moves away from it as
    training proceeds.
    """
    v = pre_round - torch.floor(pre_round)
    v_clamped = v.clamp(1e-4, 1 - 1e-4)
    # invert h_soft: sigmoid(alpha) = (v - gamma) / (zeta - gamma)
    target_sigmoid = ((v_clamped - gamma) / (zeta - gamma)).clamp(1e-4, 1 - 1e-4)
    alpha = torch.log(target_sigmoid / (1 - target_sigmoid))
    return alpha


def soft_quantized_int(pre_round: torch.Tensor, alpha: torch.Tensor, max_int: int) -> torch.Tensor:
    floor_val = torch.floor(pre_round)
    soft_int = floor_val + h_soft(alpha)
    return soft_int.clamp(0, max_int)


def hard_quantized_int(pre_round: torch.Tensor, alpha: torch.Tensor, max_int: int) -> torch.Tensor:
    floor_val = torch.floor(pre_round)
    hard_int = floor_val + h_hard(alpha)
    return hard_int.clamp(0, max_int)


def rtn_baseline_int(pre_round: torch.Tensor, max_int: int) -> torch.Tensor:
    return torch.round(pre_round).clamp(0, max_int)
