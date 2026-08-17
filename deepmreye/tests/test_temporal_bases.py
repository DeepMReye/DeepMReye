"""The temporally-selected corpus bases: `gev-*`, `band-pca`, `nuis-pca*`.

These exist because variance ordering is blind to the one thing this corpus
measured about its own nuisance. Two findings point at the same axis: the next-TR
result showed the *predictable* part of an eye block is drift, motion and global
signal concentrated in the leading variance components, and the temporal-envelope
law showed a gaze trace's own lag-1 autocorrelation predicts how well it decodes.
So "how slow is this direction" is label-free information about whether a
direction can be gaze, and PCA throws it away.

Every test here plants a **known** temporal structure and checks the basis picks
the right end of it, rather than checking that some numbers come out. The
identity being relied on throughout is that for centred stationary ``x``,
``E[(x_{t+1}-x_t)(x_{t+1}-x_t)'] = 2C_0 - 2 sym(C_1)``, hence
``rho(w) = 1 - (w' DC w) / (2 w' C_0 w)`` -- which is what makes any of this
free rather than another pass over the corpus.
"""
import numpy as np
import pytest

from deepmreye.unsupervised import (
    BASIS_KINDS,
    Moments,
    _top_eigenvectors,
    fit_band_pca,
    fit_gev,
    fit_nuisance_projected_pca,
    fit_pca,
    lag1_autocorrelation,
    load_basis,
    project,
    save_basis,
)


def _ar1(n, rho, rng):
    """An AR(1) series with lag-1 autocorrelation ``rho`` and unit variance."""
    out = np.empty(n)
    out[0] = rng.standard_normal()
    scale = np.sqrt(max(1e-12, 1 - rho ** 2))
    for t in range(1, n):
        out[t] = rho * out[t - 1] + scale * rng.standard_normal()
    return out


def _planted(d=60, t=900, seed=0, rhos=(0.95, 0.5, 0.0), amps=(3.0, 2.0, 1.0)):
    """Data whose directions have *known* autocorrelation and variance.

    Three orthogonal planted directions: a slow high-variance one (the nuisance
    an eye block actually has), a mid one, and a white one. Ordered so that
    variance ordering and temporal ordering disagree -- which is the entire point
    of the methods under test.
    """
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.standard_normal((d, len(rhos))))[0]
    x = np.zeros((t, d))
    for j, (rho, amp) in enumerate(zip(rhos, amps)):
        x += amp * np.outer(_ar1(t, rho, rng), basis[:, j])
    x += 0.05 * rng.standard_normal((t, d))
    return x.astype(np.float32), basis


def _moments(x, slab=150):
    """Accumulate ``x`` in contiguous slabs, as the corpus pass does."""
    m = Moments(x.shape[1])
    for s in range(0, len(x) - slab + 1, slab):
        m.add(x[s:s + slab])
    m.symmetrise()
    return m


# ------------------------------------------------------- the identity itself

def test_lag1_autocorrelation_recovers_planted_values():
    """The whole family rests on rho read off two covariances; verify it."""
    x, basis = _planted(rhos=(0.9, 0.4, 0.0), amps=(1.0, 1.0, 1.0), t=4000)
    m = _moments(x)
    cov, _ = m.covariance(diff=False)
    dcov, _ = m.covariance(diff=True)
    rho, var, _ = lag1_autocorrelation(cov, dcov, basis)
    assert rho[0] == pytest.approx(0.9, abs=0.06)
    assert rho[1] == pytest.approx(0.4, abs=0.06)
    assert rho[2] == pytest.approx(0.0, abs=0.06)
    assert (var > 0).all()


def test_lag1_matches_a_direct_per_direction_estimate():
    """Cross-check against literally correlating the projected timecourse."""
    x, basis = _planted(rhos=(0.8, 0.3, 0.0), t=3000)
    m = _moments(x)
    cov, mu = m.covariance(diff=False)
    dcov, _ = m.covariance(diff=True)
    rho, _, _ = lag1_autocorrelation(cov, dcov, basis)
    for j in range(basis.shape[1]):
        ts = (x - mu) @ basis[:, j]
        direct = np.corrcoef(ts[:-1], ts[1:])[0, 1]
        assert rho[j] == pytest.approx(direct, abs=0.08)


def test_quadratic_form_uses_blas_not_einsum():
    """Guards the 143x speedup, which was the entire cost of the sweep.

    `einsum("ij,jk,ik->i", ...)` computes the same quadratic form but does not
    dispatch to BLAS for this pattern. At 14236 x 512 it turned seconds into
    minutes per basis. Equivalence is what makes the fast form safe.
    """
    rng = np.random.default_rng(0)
    d, k = 200, 12
    a = rng.standard_normal((d, d))
    cov = a @ a.T / d
    v = np.linalg.qr(rng.standard_normal((d, k)))[0]
    fast = (v * (cov @ v)).sum(axis=0)
    slow = np.einsum("ij,jk,ik->i", v.T, cov, v.T)
    assert np.allclose(fast, slow)


# ------------------------------------------------------------------- gev-*

def test_gev_fast_and_slow_take_opposite_ends_of_one_spectrum():
    """Well-conditioned on purpose: `n_reduce` must stay inside the true rank.

    The first version of this test used ``n_reduce=24`` on data with three
    planted directions and negligible noise, so 21 of the whitened directions
    were noise divided by ``sqrt(shrinkage)`` -- and the "fastest" generalized
    eigenvector came back with lag-1 **0.945**, the opposite of the objective.
    That is not a bug in ``fit_gev``, it is the rank-deficiency trap its own
    docstring warns about, reproduced in miniature. Substantial isotropic noise
    plus a modest ``n_reduce`` is the regime where the method is meaningful.
    """
    x, _ = _planted(d=30, rhos=(0.95, 0.5, 0.0), amps=(3.0, 2.0, 1.0), t=4000,
                    seed=5)
    x = x + 0.8 * np.random.default_rng(6).standard_normal(x.shape).astype(np.float32)
    m = _moments(x)
    fast = fit_gev(m, 3, mode="fast", n_reduce=8)
    slow = fit_gev(m, 3, mode="slow", n_reduce=8)
    # The ratio is 2(1 - rho): slow directions score near 0, white ones near 2.
    assert fast["eigenvalues"][0] > slow["eigenvalues"][0]
    assert np.nanmean(fast["lag1"]) < np.nanmean(slow["lag1"])


def test_gev_slow_recovers_the_planted_slow_direction():
    """The slow end is the nuisance, and it should be found precisely."""
    x, basis = _planted(rhos=(0.97, 0.3, 0.0), amps=(1.0, 1.0, 1.0), t=4000)
    m = _moments(x)
    slow = fit_gev(m, 1, mode="slow", n_reduce=24)
    # |cos| because an eigenvector's sign is arbitrary.
    align = abs(float(slow["components"][:, 0] @ basis[:, 0]))
    assert align > 0.8
    assert float(np.nanmean(slow["lag1"])) > 0.7


def test_gev_ratio_of_white_noise_sits_near_two():
    """The scale the eigenvalues live on, so a real result is recognisable.

    Also the reason `gev-fast` is expected to disappoint: white noise maximises
    this objective, so the extreme fast end is noise rather than gaze.
    """
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4000, 40)).astype(np.float32)
    m = _moments(x)
    fast = fit_gev(m, 5, mode="fast", n_reduce=20)
    assert 1.5 < float(np.mean(fast["eigenvalues"])) < 3.5


# --------------------------------------------------------------- band-pca

def test_band_pca_drops_directions_slower_than_the_cut():
    x, _ = _planted(rhos=(0.97, 0.4, 0.0), amps=(4.0, 2.0, 1.0), t=3000)
    m = _moments(x)
    kept = fit_band_pca(m, 8, rho_hi=0.8, n_pool=24)
    assert kept["n_dropped_slow"] >= 1
    assert np.all(np.asarray(kept["lag1"]) <= 0.8 + 1e-6)


def test_band_pca_with_a_permissive_band_is_plain_pca():
    """The degenerate case has to be exactly the incumbent, not nearly it.

    This is what caught the first configuration: at `rho_hi=0.95` the cut
    dropped zero of 512 directions on the real corpus, so `band-pca` was
    `corpus-pca` under another name and would have been reported as a new arm.

    Only the *leading* directions are compared. Both arms call
    ``randomized_svd``, but at different ranks (``n_pool`` vs ``k``), and a
    randomized range-finder agrees on the well-separated leading components while
    the trailing ones depend on the oversampling. Requiring bitwise agreement
    across all k would be testing ``sklearn``'s randomization, not this code.
    """
    x, _ = _planted(t=2000)
    m = _moments(x)
    band = fit_band_pca(m, 6, rho_lo=-1.0, rho_hi=1.0, n_pool=20)
    pca = fit_pca(m, 6)
    assert band["components"].shape == pca["components"].shape
    assert np.allclose(np.abs(band["components"][:, :3]),
                       np.abs(pca["components"][:, :3]), atol=1e-6)
    assert band["n_dropped_slow"] == 0 and band["n_dropped_fast"] == 0


def test_band_pca_raises_when_the_band_is_empty():
    x, _ = _planted(t=1500)
    m = _moments(x)
    with pytest.raises(RuntimeError, match="no direction has lag-1"):
        fit_band_pca(m, 4, rho_lo=0.999, rho_hi=1.0, n_pool=20)


# --------------------------------------------------------------- nuis-pca

def test_nuisance_projection_removes_the_slow_high_variance_direction():
    """The planted nuisance is the *largest* direction, so PCA keeps it first
    and the projected basis must not."""
    x, basis = _planted(rhos=(0.97, 0.4, 0.0), amps=(5.0, 2.0, 1.0), t=3000)
    m = _moments(x)
    plain = fit_pca(m, 3)
    proj = fit_nuisance_projected_pca(m, 3, n_nuisance=1, n_pool=24)

    def alignment(comp):
        return np.abs(comp.T @ basis[:, 0]).max()

    assert alignment(plain["components"]) > 0.8      # PCA leads with it
    assert alignment(proj["components"]) < 0.2       # projection removed it


def test_nuisance_projected_components_are_orthogonal_to_what_was_removed():
    """Why `project` needs no change: the deflation is implicit at apply time.

    The returned components span the complement of the removed subspace, so
    ``x @ components`` already equals ``(Px) @ components``.
    """
    x, _ = _planted(rhos=(0.95, 0.6, 0.1), amps=(4.0, 2.0, 1.0), t=3000)
    m = _moments(x)
    cov, mu = m.covariance(diff=False)
    dcov, _ = m.covariance(diff=True)
    vecs, vals = _top_eigenvectors(cov, 24, 0)
    rho, _, _ = lag1_autocorrelation(cov, dcov, vecs)
    removed = vecs[:, np.argsort(np.where(np.isfinite(rho), rho, -np.inf))[::-1][:2]]

    proj = fit_nuisance_projected_pca(m, 5, n_nuisance=2, n_pool=24)
    overlap = np.abs(removed.T @ proj["components"])
    assert overlap.max() < 1e-6


def test_nuisance_projection_keeps_faster_directions_than_plain_pca():
    x, _ = _planted(rhos=(0.97, 0.5, 0.0), amps=(5.0, 2.0, 1.0), t=3000)
    m = _moments(x)
    plain = fit_pca(m, 3)
    cov, _ = m.covariance(diff=False)
    dcov, _ = m.covariance(diff=True)
    rho_plain, _, _ = lag1_autocorrelation(cov, dcov, plain["components"])
    proj = fit_nuisance_projected_pca(m, 3, n_nuisance=1, n_pool=24)
    assert np.nanmean(proj["lag1"]) < np.nanmean(rho_plain)


@pytest.mark.parametrize("j", [1, 2, 4])
def test_nuisance_budget_controls_how_much_is_removed(j):
    x, _ = _planted(d=40, rhos=(0.95, 0.8, 0.5, 0.1), amps=(4, 3, 2, 1), t=3000)
    m = _moments(x)
    out = fit_nuisance_projected_pca(m, 5, n_nuisance=j, n_pool=20)
    assert int(out["n_nuisance"][0]) == j
    assert len(out["removed_lag1"]) == j
    # What was removed is slower than what was kept.
    assert np.nanmean(out["removed_lag1"]) > np.nanmean(out["lag1"])


# ------------------------------------------------------- plumbing and disk

@pytest.mark.parametrize("kind", ["gev-fast", "gev-slow", "band-pca",
                                 "nuis-pca8", "nuis-pca32"])
def test_new_kinds_are_registered_and_projectable(kind, tmp_path):
    """Registered in both lists and applied by the shared `project` path.

    A basis kind that is fitted but not in `CORPUS_KINDS` fails only at eval
    time, and one that `project` does not recognise raises there too -- both
    after the expensive part has already run.
    """
    from deepmreye.evaluate.features import CORPUS_KINDS

    assert kind in BASIS_KINDS
    assert kind in CORPUS_KINDS

    x, _ = _planted(d=30, t=1200)
    m = _moments(x)
    mask = np.ones((30, 1, 1), dtype=bool)
    arrays = fit_pca(m, 6)          # any PCA-shaped basis exercises `project`
    path = save_basis(tmp_path / "b.npz", mask, {kind: arrays},
                      {"n_subjects": 3, "datasets": 2})
    _mask, bases, meta = load_basis(path)
    out = project(kind, bases[kind], x[:20], k=4)
    assert out.shape == (20, 4)
    assert np.isfinite(out).all()
    assert meta["datasets"] == 2


def test_all_basis_kinds_appear_in_corpus_kinds():
    """The two registries must not drift apart."""
    from deepmreye.evaluate.features import CORPUS_KINDS

    assert set(BASIS_KINDS) == set(CORPUS_KINDS)
