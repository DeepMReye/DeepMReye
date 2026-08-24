"""The unlabeled-corpus feature bases, and the accumulator they are fitted from.

The accumulator gets the most attention here on purpose. Its one historical
failure mode is silent: ``scipy``'s ``syrk`` wrapper only honours
``overwrite_c`` when the accumulator is already in BLAS's layout, and given a
C-ordered array it updates a *copy* instead. Nothing raises, the matrix stays
at zero, and the covariance comes out as ``-mu mu^T`` -- rank one, negative
definite, and still yielding a plausible-looking leading component. Only a
numerical comparison against a directly computed covariance catches it.
"""
import numpy as np
import pytest

from deepmreye.unsupervised import (
    Moments,
    fit_lr_cca,
    fit_pca,
    load_basis,
    orbit_projections,
    project,
    save_basis,
    unlabeled_subjects,
)


def _slabs(n_slabs=6, t=17, d=40, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(t, d)).astype(np.float32) for _ in range(n_slabs)]


def _direct(x):
    mu = x.mean(axis=0)
    return x.T @ x / len(x) - np.outer(mu, mu), mu


def test_moments_match_a_direct_covariance():
    slabs = _slabs()
    m = Moments(40, batch_rows=32)          # forces several buffer flushes
    for s in slabs:
        m.add(s)
    m.symmetrise()

    cov, mu = m.covariance()
    ref_cov, ref_mu = _direct(np.concatenate(slabs).astype(np.float64))
    assert m.n == sum(len(s) for s in slabs)
    assert np.allclose(mu, ref_mu, atol=1e-5)
    assert np.allclose(cov, ref_cov, atol=1e-5)


def test_moments_accumulator_is_not_silently_zero():
    """The C-ordered ``syrk`` trap: a covariance of ``-mu mu^T`` is rank one and
    has a negative trace, which is what a lost accumulator looks like."""
    m = Moments(30, batch_rows=8)
    for s in _slabs(n_slabs=4, d=30):
        m.add(s)
    m.symmetrise()
    cov, _ = m.covariance()
    assert np.trace(cov) > 0
    assert np.linalg.matrix_rank(cov, tol=1e-6) > 1


def test_moments_are_symmetric_after_symmetrise():
    m = Moments(25, batch_rows=16)
    for s in _slabs(d=25):
        m.add(s)
    m.symmetrise()
    cov, _ = m.covariance()
    assert np.allclose(cov, cov.T)


def test_pca_basis_recovers_a_planted_low_rank_subspace():
    """Three latent directions driving 40 voxels: the top-3 components must span
    them, or the basis is not finding structure that is actually there."""
    rng = np.random.default_rng(1)
    loadings = rng.normal(size=(3, 40))
    m = Moments(40, batch_rows=64)
    for _ in range(8):
        latent = rng.normal(size=(50, 3)) * np.array([6.0, 4.0, 2.0])
        m.add((latent @ loadings + 0.01 * rng.normal(size=(50, 40))).astype(np.float32))
    m.symmetrise()

    basis = fit_pca(m, k=3)
    # Projection residual of the true loadings onto the fitted span.
    q = np.linalg.qr(basis["components"])[0]
    residual = loadings.T - q @ (q.T @ loadings.T)
    assert np.linalg.norm(residual) / np.linalg.norm(loadings) < 0.05
    assert (np.diff(basis["eigenvalues"]) <= 1e-8).all()   # descending


def test_lr_cca_finds_the_signal_shared_between_orbits():
    """A latent shared by both orbits, plus independent per-orbit noise. The
    leading canonical correlation must be high and the projection must track
    the shared latent -- that is the whole premise of fitting across eyes."""
    shape, split_x = (8, 2, 2), 4            # 32 voxels, 16 per orbit
    mask = np.ones(shape, dtype=bool)
    n_vox = mask.sum()
    xs = np.nonzero(mask.reshape(-1))[0] // (shape[1] * shape[2])
    n_left = int((xs < split_x).sum())

    rng = np.random.default_rng(2)
    lw = rng.normal(size=n_left)
    rw = rng.normal(size=n_vox - n_left)

    m = Moments(n_vox, batch_rows=128)
    latents = []
    for _ in range(20):
        z = rng.normal(size=(60, 1))
        left = z @ lw[None] + 0.3 * rng.normal(size=(60, n_left))
        right = z @ rw[None] + 0.3 * rng.normal(size=(60, n_vox - n_left))
        m.add(np.hstack([left, right]).astype(np.float32))
        latents.append(z[:, 0])
    m.symmetrise()

    basis = fit_lr_cca(m, mask, k=2, n_reduce=8, split_x=split_x)
    assert basis["canonical_correlations"][0] > 0.85

    z = np.concatenate(latents)
    rng2 = np.random.default_rng(3)
    zz = rng2.normal(size=(400, 1))
    x = np.hstack([zz @ lw[None] + 0.3 * rng2.normal(size=(400, n_left)),
                   zz @ rw[None] + 0.3 * rng2.normal(size=(400, n_vox - n_left))])
    variate = project("lr-cca", basis, x)[:, 0]
    assert abs(np.corrcoef(variate, zz[:, 0])[0, 1]) > 0.85
    assert len(z) == m.n


def test_basis_round_trips_through_disk(tmp_path):
    mask = np.zeros((8, 2, 2), dtype=bool)
    mask[...] = True
    m = Moments(int(mask.sum()), batch_rows=64)
    for _ in range(6):
        m.add(np.random.default_rng(4).normal(size=(40, int(mask.sum()))).astype(np.float32))
    m.symmetrise()

    bases = {"corpus-pca": fit_pca(m, k=4),
             "lr-cca": fit_lr_cca(m, mask, k=2, n_reduce=6, split_x=4)}
    path = save_basis(tmp_path / "b.npz", mask, bases, {"k": 4, "n_voxels": 32})

    got_mask, got_bases, meta = load_basis(path)
    assert got_mask.shape == mask.shape and got_mask.all()
    assert meta["k"] == 4
    assert set(got_bases) == {"corpus-pca", "lr-cca"}
    x = np.random.default_rng(5).normal(size=(10, 32))
    assert project("corpus-pca", got_bases["corpus-pca"], x).shape == (10, 4)
    assert project("lr-cca", got_bases["lr-cca"], x).shape == (10, 2)


def test_project_honours_a_component_budget():
    m = Moments(32, batch_rows=64)
    for _ in range(6):
        m.add(np.random.default_rng(6).normal(size=(40, 32)).astype(np.float32))
    m.symmetrise()
    basis = fit_pca(m, k=8)
    x = np.random.default_rng(7).normal(size=(15, 32))
    assert project("corpus-pca", basis, x, k=3).shape == (15, 3)


def test_project_rejects_an_unknown_basis():
    with pytest.raises(ValueError):
        project("no-such-basis", {}, np.zeros((2, 4)))


# --- the feature layer -------------------------------------------------------


def _corpus(tmp_path):
    from deepmreye.storage import subject_path, write_subject

    rng = np.random.default_rng(11)
    full = np.ones((47, 29, 18), dtype=bool)   # stand-in for the real mask

    def block(t, covered=True):
        b = rng.normal(size=(47, 29, 18, t)).astype(np.float32)
        if not covered:
            b[20:, ...] = 0.0                   # partial coverage
        return b

    write_subject(subject_path(tmp_path, "ds000001", "sub-01"), block(60))
    write_subject(subject_path(tmp_path, "ds000001", "sub-02"), block(60, covered=False))
    write_subject(subject_path(tmp_path, "ds000002", "sub-01"), block(10))   # too short
    for ds in ("dsL01_guided_fixations", "dsL02_pursuit"):
        write_subject(subject_path(tmp_path, ds, "sub-01"), block(60),
                      labels=np.zeros((60, 10, 2), np.float32),
                      attrs={"repetition_time": 1.0})
    return tmp_path, int(full.sum())


def test_unlabeled_subjects_excludes_labeled_partial_and_short(tmp_path):
    root, n_vox = _corpus(tmp_path)
    got = unlabeled_subjects(root, min_voxels=n_vox)
    assert [(d, s) for d, s, _p, _t in got] == [("ds000001", "sub-01")]


def test_include_labeled_folds_in_labeled_voxels(tmp_path):
    root, n_vox = _corpus(tmp_path)
    got = unlabeled_subjects(root, min_voxels=n_vox, include_labeled=True)
    assert {d for d, _s, _p, _t in got} == {
        "ds000001", "dsL01_guided_fixations", "dsL02_pursuit"}


def test_exclude_datasets_holds_out_one_fold(tmp_path):
    """The honest form of --include-labeled: the held-out dataset stays out, so
    the fold is still leave-one-dataset-out."""
    root, n_vox = _corpus(tmp_path)
    got = unlabeled_subjects(root, min_voxels=n_vox, include_labeled=True,
                             exclude_datasets=["dsL02_pursuit"])
    datasets = {d for d, _s, _p, _t in got}
    assert "dsL02_pursuit" not in datasets
    assert "dsL01_guided_fixations" in datasets




def test_orbit_projections_average_to_project():
    """`project` is the average of the two orbits, and this is the check that says so.

    The two are separate functions because the cache needs the orbits apart, and
    if they ever disagree every cached feature silently stops matching the
    shipped projection.
    """
    rng = np.random.default_rng(3)
    d, half = 24, 12
    arrays = {
        "mean": rng.normal(size=d),
        "left_index": np.arange(half),
        "right_index": np.arange(half, d),
        "left_weights": rng.normal(size=(half, 5)),
        "right_weights": rng.normal(size=(half, 5)),
    }
    x = rng.normal(size=(30, d))
    zl, zr = orbit_projections(x, arrays, k=3)
    assert zl.shape == zr.shape == (30, 3)
    assert np.allclose(0.5 * (zl + zr), project("lr-cca", arrays, x, k=3))
