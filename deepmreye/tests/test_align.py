"""Unsupervised feature alignment.

Each method is tested against the property it is supposed to have, on data with
a *planted* shift, so a failure says which correction stopped working rather
than only that a number moved.
"""
import numpy as np
import pytest

from deepmreye.evaluate.align import (
    ALIGN_METHODS,
    align,
    apply_pair,
    fit_reference,
)


def _shifted(n=400, d=6, seed=0):
    """Two groups of the same latent signal under different affine mixings --
    the covariate shift these methods exist to remove."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(2 * n, d))
    a = rng.normal(size=(d, d))
    b = rng.normal(size=(d, d))
    x = np.vstack([z[:n] @ a + 3.0, z[n:] @ b - 7.0])
    groups = np.array(["a"] * n + ["b"] * n)
    return x, groups


def test_none_is_a_no_op():
    x, g = _shifted()
    assert np.allclose(align(x, g, "none"), x)


def test_center_removes_each_group_mean():
    x, g = _shifted()
    out = align(x, g, "center")
    for name in np.unique(g):
        assert np.allclose(out[g == name].mean(axis=0), 0.0, atol=1e-8)


def test_zscore_gives_each_group_unit_variance_per_feature():
    x, g = _shifted()
    out = align(x, g, "zscore")
    for name in np.unique(g):
        assert np.allclose(out[g == name].mean(axis=0), 0.0, atol=1e-8)
        assert np.allclose(out[g == name].std(axis=0), 1.0, atol=1e-3)


def test_zscore_cannot_remove_a_rotation_but_ea_can():
    """The reason full-covariance alignment is worth more than the diagonal
    ``feat-std`` that was already tried: a shift that rotates the feature space
    leaves per-feature variances untouched, so only whitening removes it."""
    x, g = _shifted(seed=3)

    def cross_group_gap(out):
        a = np.cov(out[g == "a"].T)
        b = np.cov(out[g == "b"].T)
        return np.abs(a - b).max()

    assert cross_group_gap(align(x, g, "ea")) < cross_group_gap(align(x, g, "zscore"))


def test_ea_whitens_every_group_to_near_identity():
    x, g = _shifted(n=2000, d=5, seed=1)
    out = align(x, g, "ea", shrinkage=0.0)
    for name in np.unique(g):
        cov = np.cov(out[g == name].T)
        assert np.allclose(cov, np.eye(cov.shape[0]), atol=0.15)


def test_ea_makes_two_differently_mixed_groups_comparable():
    """The point of the method: after alignment the groups should have nearly
    the same second-order structure, whatever mixing produced them."""
    x, g = _shifted(n=3000, d=4, seed=2)
    out = align(x, g, "ea", shrinkage=0.0)
    ca = np.cov(out[g == "a"].T)
    cb = np.cov(out[g == "b"].T)
    assert np.abs(ca - cb).max() < 0.2


def test_coral_moves_the_target_onto_the_source_statistics():
    x, g = _shifted(n=3000, d=4, seed=4)
    source, target = x[g == "a"], x[g == "b"]
    ref = fit_reference(source, shrinkage=0.0)
    moved = align(target, np.zeros(len(target)), "coral", ref, shrinkage=0.0)

    assert np.allclose(moved.mean(axis=0), source.mean(axis=0), atol=0.3)
    assert np.abs(np.cov(moved.T) - np.cov(source.T)).max() < 0.5


def test_coral_requires_a_reference():
    x, g = _shifted()
    with pytest.raises(ValueError):
        align(x, g, "coral")


def test_apply_pair_leaves_the_source_alone_for_coral_but_not_for_ea():
    """The two differ precisely in what happens to the training side, and that
    difference is the reason both exist."""
    x, g = _shifted(seed=5)
    x_tr, g_tr = x[g == "a"], g[g == "a"]
    x_te, g_te = x[g == "b"], g[g == "b"]

    tr_coral, _ = apply_pair(x_tr, g_tr, x_te, g_te, "coral")
    assert np.allclose(tr_coral, x_tr)

    tr_ea, _ = apply_pair(x_tr, g_tr, x_te, g_te, "ea")
    assert not np.allclose(tr_ea, x_tr)


def test_a_group_too_small_to_whiten_falls_back_to_centring():
    """A near-singular whitening from two rows would emit garbage; degrading to
    a mean subtraction keeps the arm interpretable."""
    x = np.array([[1.0, 2.0], [3.0, 5.0]])
    out = align(x, np.array(["a", "a"]), "ea")
    assert np.allclose(out, x - x.mean(axis=0))


def test_alignment_is_fitted_within_each_group_independently():
    """No statistic may cross the train/test boundary except through `coral`'s
    explicit reference -- otherwise the test group's alignment would depend on
    data it is not entitled to."""
    x, g = _shifted(seed=6)
    out_all = align(x, g, "ea")
    only_a = x[g == "a"]
    out_a = align(only_a, np.array(["a"] * len(only_a)), "ea")
    assert np.allclose(out_all[g == "a"], out_a)


def test_alignment_uses_no_labels():
    """Signature-level guarantee: nothing here can see gaze, so every method is
    applicable to a participant with no eye tracking."""
    import inspect

    params = set(inspect.signature(align).parameters)
    assert not (params & {"y", "labels", "targets", "gaze"})


@pytest.mark.parametrize("method", ALIGN_METHODS)
def test_every_method_preserves_shape_and_is_finite(method):
    x, g = _shifted(seed=7)
    ref = fit_reference(x) if method == "coral" else None
    out = align(x, g, method, ref)
    assert out.shape == x.shape
    assert np.isfinite(out).all()


def test_unknown_method_is_rejected():
    x, g = _shifted()
    with pytest.raises(ValueError):
        align(x, g, "not-a-method")
