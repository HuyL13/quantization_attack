import torch

from aq.quantizer import AdversarialLinearQuantizer, AdversarialQuantConfig
from aq.rtn_backend import rtn_quantize_weight_raw


def _make_quantizer(**cfg_kwargs):
    torch.manual_seed(3)
    # in_features == group_size (128): avoids zero-padding contaminating the
    # group's min/max (if_awq_tier0's quantize_weight_groupwise_raw pads with
    # zeros to a full group *before* taking min/max when in_features <
    # group_size, which is never the case for LLaMA-2's actual layer sizes -
    # all multiples of 128 - but would silently corrupt scale/zero_point for
    # a smaller synthetic test tensor).
    w = torch.randn(4, 128)
    state = rtn_quantize_weight_raw(w, bits=4, group_size=128)
    cfg = AdversarialQuantConfig(bits=4, group_size=128, **cfg_kwargs)
    return AdversarialLinearQuantizer(w, state, cfg), w


def test_hard_weight_matches_rtn_baseline_at_init():
    # hard_weight() thresholds h_soft at 0.5, exactly reproducing round-to-
    # nearest - this is the "starts from RTN4" invariant the plan requires.
    q, w = _make_quantizer()
    hard0 = q.hard_weight()
    baseline0 = q.rtn_baseline_weight()
    assert torch.equal(hard0, baseline0)


def test_soft_weight_reconstructs_fp_weight_at_init():
    # soft_weight() is the CONTINUOUS relaxation: at init h_soft(alpha0) ==
    # frac(pre_round) exactly, so floor(pre_round) + h_soft(alpha0) == pre_round,
    # and dequantizing that recovers w_fp almost exactly for the interior of
    # the grid - it does NOT reconstruct the rounded RTN4 grid point (that's
    # what hard_weight() is for). A small number of extremal-per-group
    # elements (the group's own min/max) can still land outside [0, max_int]
    # because the affine grid's zero-point is itself a ROUNDED integer, and
    # int_w is clamped there exactly like the RTN baseline clamps it - so this
    # is real, expected clipping, not a bug, and only affects a handful of
    # elements per group. Checked via overall relative distance, not an
    # element-wise atol that a few boundary elements would fail.
    q, w = _make_quantizer()
    from aq.metrics import weight_relative_distance

    soft0 = q.soft_weight()
    assert weight_relative_distance(w, soft0).item() < 0.02


def test_weight_distance_uses_soft_weight_and_is_near_zero_at_init():
    q, w = _make_quantizer()
    assert q.weight_distance().item() < 0.02  # soft ~= w_fp at init, see above

    from aq.metrics import weight_relative_distance

    # the discrete RTN4 grid point (hard_weight) DOES carry real quantization
    # error relative to w_fp - that's the baseline distance every adversarial
    # method is expected to increase beyond.
    hard_distance = weight_relative_distance(w, q.hard_weight())
    assert hard_distance.item() > 1e-3


def test_optimize_scale_flag_adds_parameter_and_starts_at_identity():
    q, w = _make_quantizer(optimize_scale=True)
    assert q.log_scale_mult is not None
    assert torch.allclose(q.effective_scale(), q.scale0, atol=1e-6)
    params = list(q.parameters())
    assert any(p is q.log_scale_mult for p in params)


def test_use_codebook_flag_builds_expected_shapes():
    q, w = _make_quantizer(use_codebook=True, n_codebook_centroids=8)
    out_features, padded_in = q.pre_round.shape
    n_groups = padded_in // 128
    assert q.codebook.shape == (out_features, n_groups, 8)
    assert q.assign_logits.shape == (out_features, n_groups, 128, 8)
    soft = q.soft_weight()
    assert soft.shape == w.shape
    hard = q.hard_weight()
    assert hard.shape == w.shape


def test_hard_int_grid_responds_to_alpha_changes():
    # Direct, deterministic check of the alpha -> hard_int_grid mechanism
    # (rather than relying on gradient descent dynamics, which are
    # deliberately near-degenerate right at RTN4 init - see
    # test_soft_weight_reconstructs_fp_weight_at_init and
    # aq.quantizer.rounding_regularizer's docstring).
    q, w = _make_quantizer()
    hard_int_before = q.hard_int_grid().clone()
    with torch.no_grad():
        q.alpha.add_(20.0)  # push every h_soft(alpha) toward 1 -> ceil
    hard_int_after = q.hard_int_grid()
    assert not torch.equal(hard_int_before, hard_int_after)


def test_rounding_regularizer_moves_alpha_away_from_degenerate_init():
    # This is the actual mechanism relied on during real training: the
    # regularizer alone (no KL/distance term) has a well-scaled gradient
    # everywhere, including exactly at RTN4 init, unlike weight_distance().
    q, w = _make_quantizer(round_reg_weight=1.0)
    optim = torch.optim.SGD(q.parameters(), lr=5.0)
    for _ in range(5):
        optim.zero_grad()
        loss = q.rounding_regularizer()
        loss.backward()
        optim.step()
    h_after = torch.sigmoid(q.alpha)  # monotonic in the same direction as h_soft
    # every element should have moved toward a hard decision (away from 0.5)
    reg_after = q.rounding_regularizer()
    reg_before_equivalent = 1.0  # h*(1-h)*4 has max value 1.0 at h=0.5
    assert reg_after.item() < reg_before_equivalent


def test_sensitivity_weighting_matches_manual_formula():
    q, w = _make_quantizer(use_sensitivity=True)
    from aq.metrics import weight_relative_distance, weight_relative_distance_weighted

    what = q.soft_weight()
    plain_distance = weight_relative_distance(q.w_fp, what)

    uniform_c = 5.0
    q.sensitivity = torch.ones_like(q.w_fp) * uniform_c
    weighted_distance = q.weight_distance()
    expected = weight_relative_distance_weighted(q.w_fp, what, q.sensitivity)
    assert torch.isclose(weighted_distance, expected, atol=1e-5)
    # uniform sensitivity c scales the value by sqrt(c) (it multiplies the
    # numerator sum-of-squares but not the ||w||^2 denominator) - not a no-op.
    assert torch.isclose(weighted_distance, plain_distance * (uniform_c ** 0.5), atol=1e-3)
