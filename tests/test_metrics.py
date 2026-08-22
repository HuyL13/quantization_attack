import math

import torch

from aq.metrics import (
    codebook_occupancy_entropy,
    cosine_similarity_flat,
    kl_divergence_logits,
    rounding_flip_ratio,
    scale_relative_shift,
    top1_agreement,
    weight_relative_distance,
    weight_relative_distance_weighted,
)


def test_kl_divergence_zero_for_identical_logits():
    logits = torch.randn(4, 10)
    kl = kl_divergence_logits(logits, logits)
    assert torch.isclose(kl, torch.tensor(0.0), atol=1e-5)


def test_kl_divergence_matches_manual_two_class():
    # P = softmax([0, log(3)]) = [0.25, 0.75]; Q = softmax([0, 0]) = [0.5, 0.5]
    logits_p = torch.tensor([[0.0, math.log(3.0)]])
    logits_q = torch.tensor([[0.0, 0.0]])
    kl = kl_divergence_logits(logits_p, logits_q)
    p = torch.tensor([0.25, 0.75])
    q = torch.tensor([0.5, 0.5])
    expected = (p * (p / q).log()).sum()
    assert torch.isclose(kl, expected, atol=1e-5)


def test_kl_divergence_nonnegative_and_asymmetric():
    torch.manual_seed(0)
    logits_p = torch.randn(3, 5)
    logits_q = torch.randn(3, 5)
    kl_pq = kl_divergence_logits(logits_p, logits_q)
    kl_qp = kl_divergence_logits(logits_q, logits_p)
    assert kl_pq >= 0
    assert kl_qp >= 0
    assert not torch.isclose(kl_pq, kl_qp)


def test_weight_relative_distance_manual():
    w = torch.tensor([3.0, 4.0])  # norm 5
    w_hat = torch.tensor([0.0, 0.0])
    d = weight_relative_distance(w, w_hat)
    assert torch.isclose(d, torch.tensor(1.0), atol=1e-6)

    w_hat2 = w.clone()
    d2 = weight_relative_distance(w, w_hat2)
    assert torch.isclose(d2, torch.tensor(0.0), atol=1e-6)


def test_weight_relative_distance_weighted_reduces_to_unweighted_when_uniform():
    torch.manual_seed(2)
    w = torch.randn(4, 4)
    w_hat = w + 0.1 * torch.randn(4, 4)
    uniform_sensitivity = torch.ones_like(w)
    d_plain = weight_relative_distance(w, w_hat)
    d_weighted = weight_relative_distance_weighted(w, w_hat, uniform_sensitivity)
    assert torch.isclose(d_plain, d_weighted, atol=1e-5)


def test_cosine_similarity_orthogonal_and_identical():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    assert torch.isclose(cosine_similarity_flat(a, b), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(cosine_similarity_flat(a, a), torch.tensor(1.0), atol=1e-6)


def test_top1_agreement():
    logits_p = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    logits_q_same = logits_p.clone()
    logits_q_diff = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    assert torch.isclose(top1_agreement(logits_p, logits_q_same), torch.tensor(1.0))
    assert torch.isclose(top1_agreement(logits_p, logits_q_diff), torch.tensor(0.5))


def test_rounding_flip_ratio():
    before = torch.tensor([1.0, 2.0, 3.0, 4.0])
    after = torch.tensor([1.0, 2.0, 3.0, 5.0])
    assert torch.isclose(rounding_flip_ratio(before, after), torch.tensor(0.25))


def test_scale_relative_shift():
    before = torch.tensor([1.0, 2.0])
    after = torch.tensor([1.1, 2.2])
    shift = scale_relative_shift(before, after)
    assert torch.isclose(shift, torch.tensor(0.1), atol=1e-5)


def test_codebook_occupancy_entropy_extremes():
    n_centroids = 4
    # all mass on centroid 0 -> low occupancy, zero entropy
    degenerate = torch.zeros(10, n_centroids)
    degenerate[:, 0] = 1.0
    occ, ent = codebook_occupancy_entropy(degenerate)
    assert torch.isclose(occ, torch.tensor(1.0 / n_centroids))
    assert torch.isclose(ent, torch.tensor(0.0), atol=1e-6)

    # uniform mass -> max entropy log(n_centroids)
    uniform = torch.ones(10, n_centroids) / n_centroids
    _, ent_uniform = codebook_occupancy_entropy(uniform)
    assert torch.isclose(ent_uniform, torch.tensor(math.log(n_centroids)), atol=1e-4)
