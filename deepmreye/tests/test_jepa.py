"""Orbit-JEPA: the collapse guard, the warm start, and the numpy feature path.

The three properties worth a test here are the three that were silently wrong
before (see `models/jepa_net.py`):

1. SIGReg must be *minimised by its own target distribution*. The previous
   implementation had the Epps-Pulley exponents swapped, so it preferred
   collapse to N(0, 1), and the test that was supposed to catch that used
   ``ones * 5.0`` as the collapsed case -- an offset the broken statistic
   happens to penalise. Collapse **toward zero** is the direction a model
   actually falls into, and it is tested explicitly below.
2. The untrained model must equal `lr-cca:k` bit for bit. That identity is the
   whole design: it makes the control a real baseline instead of a random
   projection, so a gain over it is a gain over the 0.825 arm.
3. The numpy extraction path must agree with torch, since the probe never runs
   the torch model (OpenMP deadlock against LightGBM, see `CLAUDE.md`).
"""
import numpy as np
import pytest
import torch

from deepmreye.models.jepa_net import (
    SIGREG_COLLAPSE_VALUE,
    OrbitJEPA,
    SIGRegLoss,
    untrained_like,
)
from deepmreye.orbitjepa import (
    encode_numpy,
    jepa_features,
    orbit_projections,
    split_runs,
    train_orbit_jepa,
)
from deepmreye.unsupervised import project

# SIGReg materialises a [B, B, M] pairwise table, so its cost is quadratic in the
# batch and these tests are kept small deliberately -- they check the *ordering*
# of the statistic, which needs no more resolution than this. Single-threaded for
# the reason `CLAUDE.md` gives about the feature path: a large threaded torch
# reduction is exactly what interacts badly with the other OpenMP runtimes in
# this process (LightGBM, and TensorFlow via `test_dme1`), and at 512x512x256 it
# turned an 0.15 s test into one that stalled the whole suite for minutes under
# one particular collection order.
torch.set_num_threads(1)

SIG_B, SIG_D, SIG_M = 192, 16, 64


# --------------------------------------------------------------------------
# 1. SIGReg
# --------------------------------------------------------------------------
def test_sigreg_is_minimised_by_its_own_target_distribution():
    """N(0, I) must score *below* collapse, which is what makes it anti-collapse.

    This is the assertion that fails for a swapped-exponent Epps-Pulley
    statistic: that version scores N(0, 1) at 0.285 against 0.163 for a
    constant batch, so minimising it collapses the encoder.
    """
    torch.manual_seed(0)
    sig = SIGRegLoss(n_sketches=SIG_M)

    gaussian = sig(torch.randn(SIG_B, SIG_D)).item()
    collapsed_zero = sig(torch.zeros(SIG_B, SIG_D)).item()
    collapsed_offset = sig(torch.full((SIG_B, SIG_D), 5.0)).item()
    near_collapse = sig(0.01 * torch.randn(SIG_B, SIG_D)).item()

    assert gaussian < collapsed_zero, (
        f"SIGReg prefers collapse ({collapsed_zero:.4f}) to its target "
        f"N(0,1) ({gaussian:.4f}) -- the exponents are swapped")
    assert gaussian < near_collapse
    assert gaussian < collapsed_offset
    assert gaussian < 0.05


def test_sigreg_collapse_value_matches_the_analytic_constant():
    """Collapse toward zero gives exactly 1 - sqrt(2) + 1/sqrt(3) = 0.1631.

    Worth pinning because it is the number that identifies a collapsed run in a
    training log at a glance: `models/orbitjepa_n1039.pt` sat at 0.16314 from
    epoch 1 to epoch 15.

    A batch collapsed to a *non-zero* constant scores strictly higher -- the
    projection sends the offset to some constant ``c`` and the statistic becomes
    ``1 - sqrt(2) exp(-c^2/4) + 1/sqrt(3)``. That asymmetry is exactly how the
    old test missed the inverted statistic: it used ``ones * 5.0``, the one form
    of collapse a broken anti-collapse term still penalises.
    """
    sig = SIGRegLoss(n_sketches=SIG_M)
    at_zero = sig(torch.zeros(SIG_B, SIG_D)).item()
    assert at_zero == pytest.approx(SIGREG_COLLAPSE_VALUE, abs=1e-3)

    for value in (5.0, -3.0):
        offset = sig(torch.full((SIG_B, SIG_D), value)).item()
        assert offset > at_zero, "collapse at an offset should not score below zero-collapse"


def test_sigreg_pushes_a_scaled_batch_back_toward_unit_variance():
    """The gradient must shrink an over-dispersed batch and inflate a tiny one."""
    sig = SIGRegLoss(n_sketches=SIG_M)
    torch.manual_seed(1)
    base = torch.randn(SIG_B, SIG_D)

    for scale, expect_shrink in ((4.0, True), (0.2, False)):
        z = (scale * base).clone().requires_grad_(True)
        sig(z).backward()
        # Does the negative gradient move the batch toward unit scale?
        moved = (z - 0.1 * z.grad).std().item()
        assert (moved < z.std().item()) == expect_shrink


# --------------------------------------------------------------------------
# 2. The warm start
# --------------------------------------------------------------------------
def _toy_basis(n_voxels=40, m=8, seed=0):
    rng = np.random.default_rng(seed)
    li = np.arange(0, n_voxels // 2)
    ri = np.arange(n_voxels // 2, n_voxels)
    return {"mean": rng.normal(size=n_voxels),
            "left_index": li, "right_index": ri,
            "left_weights": rng.normal(size=(len(li), m)),
            "right_weights": rng.normal(size=(len(ri), m))}


def test_untrained_jepa_reproduces_lr_cca_exactly():
    """An untrained Orbit-JEPA's averaged latent IS `project("lr-cca", k)`.

    The design's load-bearing claim. If this drifts, the "untrained control"
    stops being the linear baseline and every margin reported against it becomes
    a margin against something unknown.
    """
    rng = np.random.default_rng(3)
    basis, k, m = _toy_basis(), 4, 8
    rows = rng.normal(size=(60, 40))

    model = OrbitJEPA(in_dim=m, latent_dim=k, hidden_dim=16, depth=2)
    feats = jepa_features(model.to_numpy_weights(), rows, basis, m=m, head="avg")
    linear = project("lr-cca", basis, rows, k=k)

    assert feats.shape == (60, k)
    np.testing.assert_allclose(feats, linear, rtol=1e-9, atol=1e-9)


def test_untrained_control_is_built_from_the_models_own_arch():
    model = OrbitJEPA(in_dim=32, latent_dim=12, hidden_dim=48, depth=3)
    control = untrained_like(model)
    assert control.arch() == model.arch()
    assert control.left_encoder.latent_dim == 12


def test_latent_wider_than_the_preprojection_is_refused():
    """k > M has no identity initialisation, so it must not construct."""
    with pytest.raises(ValueError, match="identity initialisation"):
        OrbitJEPA(in_dim=8, latent_dim=16)


def test_a_trained_model_departs_from_its_warm_start():
    """Sanity: gradient steps must actually move the features off `lr-cca`.

    Guards the opposite failure from collapse -- a zero-initialised residual
    branch that never receives gradient would leave the model exactly linear
    forever and report the linear score as its own.
    """
    rng = np.random.default_rng(5)
    m, k = 16, 6
    shared = rng.normal(size=(400, 3))
    zl = np.concatenate([shared @ rng.normal(size=(3, m))], axis=1) + 0.1 * rng.normal(size=(400, m))
    zr = np.tanh(shared) @ rng.normal(size=(3, m)) + 0.1 * rng.normal(size=(400, m))
    z = np.stack([zl, zr], axis=1).astype(np.float32)
    run_id = np.repeat(np.arange(20), 20)

    model = OrbitJEPA(in_dim=m, latent_dim=k, hidden_dim=32, depth=2, dropout=0.0)
    before = model.to_numpy_weights()
    model, info = train_orbit_jepa(model, z, run_id, epochs=3, batch_size=64,
                                  verbose=False, val_frac=0.2)
    after = model.to_numpy_weights()

    assert info["history"][-1]["nonlinear_share"] > 0.0
    assert not np.allclose(before["left"]["layers"][-1][1], after["left"]["layers"][-1][1])


# --------------------------------------------------------------------------
# 3. numpy / torch parity and plumbing
# --------------------------------------------------------------------------
def test_numpy_encoder_matches_torch():
    torch.manual_seed(7)
    model = OrbitJEPA(in_dim=24, latent_dim=8, hidden_dim=32, depth=3, dropout=0.5)
    # Move off the identity init so the MLP branch is actually exercised.
    with torch.no_grad():
        for p in model.left_encoder.parameters():
            p.add_(0.05 * torch.randn_like(p))
    model.eval()

    z = np.random.default_rng(0).normal(size=(17, 24))
    expected = model.left_encoder(torch.from_numpy(z).float()).detach().numpy()
    got = encode_numpy(model.to_numpy_weights()["left"], z)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


def test_concat_head_doubles_the_feature_width():
    basis, m, k = _toy_basis(), 8, 3
    rows = np.random.default_rng(1).normal(size=(30, 40))
    model = OrbitJEPA(in_dim=m, latent_dim=k, hidden_dim=16)
    w = model.to_numpy_weights()
    assert jepa_features(w, rows, basis, m=m, head="avg").shape == (30, k)
    assert jepa_features(w, rows, basis, m=m, head="concat").shape == (30, 2 * k)


def test_orbit_projections_average_equals_the_linear_basis():
    basis, m, k = _toy_basis(seed=2), 8, 5
    rows = np.random.default_rng(4).normal(size=(25, 40))
    zl, zr = orbit_projections(rows, basis, m=m)
    np.testing.assert_allclose(0.5 * (zl[:, :k] + zr[:, :k]),
                               project("lr-cca", basis, rows, k=k), atol=1e-10)


def test_train_val_split_never_puts_one_run_on_both_sides():
    """Windows inside a run are near-duplicates; a TR-level split leaks."""
    run_id = np.repeat(np.arange(30), 17)
    tr, va = split_runs(run_id, val_frac=0.2, seed=0)
    assert not (set(run_id[tr]) & set(run_id[va]))
    assert va.sum() > 0 and tr.sum() > 0


def test_motion_regression_changes_the_projection():
    basis, m = _toy_basis(seed=6), 8
    rows = np.random.default_rng(7).normal(size=(40, 40))
    plain = orbit_projections(rows, basis, m=m)[0]
    cleaned = orbit_projections(rows, basis, m=m, regress_motion=True)[0]
    assert not np.allclose(plain, cleaned)


def test_spatiotemporal_numpy_encoder_matches_torch():
    """Causal 1D convolution in numpy must match PyTorch exactly."""
    model = OrbitJEPA(in_dim=12, latent_dim=4, hidden_dim=16, depth=2,
                      temp_kernel=3, alpha_gate=0.5)
    with torch.no_grad():
        for p in model.left_encoder.parameters():
            p.add_(0.05 * torch.randn_like(p))
    model.eval()

    z = np.random.default_rng(42).normal(size=(25, 12))
    expected = model.left_encoder(torch.from_numpy(z).float()).detach().numpy()
    got = encode_numpy(model.to_numpy_weights()["left"], z)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


def test_spatiotemporal_untrained_reproduces_lr_cca():
    """An untrained spatiotemporal Orbit-JEPA equals lr-cca at init."""
    basis, k, m = _toy_basis(seed=10), 4, 8
    rows = np.random.default_rng(11).normal(size=(35, 40))
    model = OrbitJEPA(in_dim=m, latent_dim=k, hidden_dim=16, temp_kernel=3)
    feats = jepa_features(model.to_numpy_weights(), rows, basis, m=m, head="avg")
    linear = project("lr-cca", basis, rows, k=k)
    np.testing.assert_allclose(feats, linear, rtol=1e-9, atol=1e-9)


def test_spatiotemporal_sequence_3d_input():
    """OrbitJEPA forward accepts 3D sequence tensors [B, T, M]."""
    model = OrbitJEPA(in_dim=8, latent_dim=4, hidden_dim=16, temp_kernel=3)
    zl = torch.randn(4, 10, 8)
    zr = torch.randn(4, 10, 8)
    out = model(zl, zr)
    assert out["s_L"].shape == (4, 10, 4)
    assert out["s_R"].shape == (4, 10, 4)
    assert torch.isfinite(out["loss"])

