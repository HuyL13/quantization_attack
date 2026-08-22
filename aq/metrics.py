"""Behavior-preservation and weight-distance metrics (plan section 3/13):

- KL(M_FP, M_Q) on calibration logits: the behavior term of the core
  objective L = KL(M_FP, M_Q) - lambda * D(W, Q(W)).
- D(W, Q(W)): relative Frobenius weight distance, the term the objective
  REWARDS (opposite sign of ordinary quantization error minimization).
- cosine similarity / top-1 token agreement / rounding-flip-ratio: required
  per-layer / per-step logging fields (section 13), not loss terms.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_divergence_logits(logits_p: torch.Tensor, logits_q: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Mean KL(P || Q) over all leading dims, P = softmax(logits_p) (the FP
    reference distribution), Q = softmax(logits_q) (the quantized candidate).
    Both inputs are raw logits, not log-probs - callers never pre-normalize.
    """
    log_p = F.log_softmax(logits_p.float(), dim=dim)
    log_q = F.log_softmax(logits_q.float(), dim=dim)
    p = log_p.exp()
    per_token_kl = (p * (log_p - log_q)).sum(dim=dim)
    return per_token_kl.mean()


def weight_relative_distance(w: torch.Tensor, w_hat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """D(W, Q(W)) = ||W - Q(W)||_F / ||W||_F. This is the term the
    adversarial objective REWARDS (maximizes), unlike ordinary PTQ which
    minimizes it - callers subtract lambda * this from the loss, they never
    add it.
    """
    diff_norm = torch.linalg.norm((w - w_hat).float().flatten())
    base_norm = torch.linalg.norm(w.float().flatten()).clamp_min(eps)
    return diff_norm / base_norm


def weight_relative_distance_weighted(
    w: torch.Tensor, w_hat: torch.Tensor, sensitivity: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Sensitivity-weighted variant for method 4: elements with high
    sensitivity are penalized more per unit of deviation (I_i * diff_i^2),
    so the optimizer is steered toward pushing distance through low-
    sensitivity weights instead of uniformly.
    """
    diff = (w - w_hat).float()
    weighted_sq = (sensitivity.float() * diff.pow(2)).sum()
    base_norm_sq = torch.linalg.norm(w.float().flatten()).clamp_min(eps).pow(2)
    return (weighted_sq / base_norm_sq).sqrt()


def cosine_similarity_flat(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0), eps=eps).squeeze(0)


def top1_agreement(logits_p: torch.Tensor, logits_q: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Fraction of positions where argmax(P) == argmax(Q)."""
    top_p = logits_p.argmax(dim=dim)
    top_q = logits_q.argmax(dim=dim)
    return (top_p == top_q).float().mean()


def rounding_flip_ratio(hard_int_before: torch.Tensor, hard_int_after: torch.Tensor) -> torch.Tensor:
    """Fraction of integer-grid positions whose rounding decision changed
    after optimization (section 13's "rounding flip ratio").
    """
    return (hard_int_before != hard_int_after).float().mean()


def scale_relative_shift(scale_before: torch.Tensor, scale_after: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return ((scale_after - scale_before).abs() / scale_before.abs().clamp_min(eps)).mean()


def codebook_occupancy_entropy(assignment_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """assignment_probs: (..., n_centroids) soft/hard assignment distribution.
    Returns (occupancy fraction of centroids used, mean entropy in nats).
    """
    hard_idx = assignment_probs.argmax(dim=-1).flatten()
    n_centroids = assignment_probs.shape[-1]
    used = torch.bincount(hard_idx, minlength=n_centroids) > 0
    occupancy = used.float().mean()
    probs = assignment_probs.clamp_min(1e-12)
    entropy = -(probs * probs.log()).sum(dim=-1).mean()
    return occupancy, entropy
