import torch

from aq.hsq_hessian import accumulate_hessian, compute_inverse_hessian_cholesky


def test_accumulate_hessian_matches_manual_two_batch_formula():
    H = torch.zeros(3, 3)
    nsamples = 0
    x1 = torch.randn(5, 3)
    x2 = torch.randn(7, 3)

    H, nsamples = accumulate_hessian(H, nsamples, x1)
    H, nsamples = accumulate_hessian(H, nsamples, x2)
    assert nsamples == 12

    # Manual replica of the same running-average formula.
    H_manual = torch.zeros(3, 3)
    n = 0
    for x in (x1, x2):
        tmp = x.shape[0]
        H_manual = H_manual * (n / (n + tmp))
        n += tmp
        xt = x.t() * (2.0 / n) ** 0.5
        H_manual = H_manual + xt.matmul(xt.t())
    assert torch.allclose(H, H_manual, atol=1e-5)


def test_accumulate_hessian_flattens_leading_dims():
    H = torch.zeros(4, 4)
    x = torch.randn(2, 3, 4)  # (batch, seq, features)
    H_out, n = accumulate_hessian(H, 0, x)
    assert n == 6
    assert H_out.shape == (4, 4)


def test_inverse_hessian_cholesky_reconstructs_true_inverse_for_diagonal_h():
    H = torch.diag(torch.tensor([4.0, 9.0, 16.0]))
    Hinv_chol, dead = compute_inverse_hessian_cholesky(H, percdamp=0.0)
    reconstructed = Hinv_chol.t().matmul(Hinv_chol)
    expected_inv = torch.diag(torch.tensor([1 / 4.0, 1 / 9.0, 1 / 16.0]))
    assert torch.allclose(reconstructed, expected_inv, atol=1e-4)
    assert not dead.any()


def test_dead_column_detected_and_stabilized():
    H = torch.diag(torch.tensor([4.0, 0.0, 9.0]))
    Hinv_chol, dead = compute_inverse_hessian_cholesky(H, percdamp=0.0)
    assert dead.tolist() == [False, True, False]
    assert torch.isfinite(Hinv_chol).all()


def test_damping_increases_diagonal_stability_for_near_singular_h():
    H = torch.tensor([[1.0, 0.999], [0.999, 1.0]])
    Hinv_chol, dead = compute_inverse_hessian_cholesky(H, percdamp=0.05)
    assert torch.isfinite(Hinv_chol).all()
    assert not dead.any()
