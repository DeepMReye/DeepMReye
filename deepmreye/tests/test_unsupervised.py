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

from deepmreye.evaluate.features import (
    FEATURE_KINDS,
    CompositeExtractor,
    FeatureExtractor,
    parse_spec,
    pool_time,
)
from deepmreye.unsupervised import (
    Moments,
    fit_lr_cca,
    fit_pca,
    load_basis,
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


def test_difference_moments_match_direct_temporal_differences():
    slabs = _slabs()
    m = Moments(40, batch_rows=32)
    for s in slabs:
        m.add(s)
    m.symmetrise()

    cov, mu = m.covariance(diff=True)
    diffs = np.concatenate([np.diff(s, axis=0) for s in slabs]).astype(np.float64)
    ref_cov, ref_mu = _direct(diffs)
    assert m.dn == len(diffs)
    assert np.allclose(mu, ref_mu, atol=1e-5)
    assert np.allclose(cov, ref_cov, atol=1e-5)


def test_moments_are_symmetric_after_symmetrise():
    m = Moments(25, batch_rows=16)
    for s in _slabs(d=25):
        m.add(s)
    m.symmetrise()
    for cov, _ in (m.covariance(), m.covariance(diff=True)):
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


def test_pool_time_averages_into_the_requested_bins():
    x = np.arange(2 * 3 * 1 * 1 * 10, dtype=np.float32).reshape(2, 3, 1, 1, 10)
    pooled = pool_time(x, n_t=5)
    assert pooled.shape == (2, 5, 3)
    # First voxel of the first window: TRs 0..9 averaged in pairs.
    assert np.allclose(pooled[0, :, 0], [0.5, 2.5, 4.5, 6.5, 8.5])


def test_pool_time_pads_an_indivisible_window():
    assert pool_time(np.ones((1, 2, 1, 1, 7)), n_t=3).shape == (1, 3, 2)


def test_pool_time_accepts_a_torch_tensor_but_returns_numpy():
    """The loader hands out tensors, so it must take one -- but it must hand
    back numpy. See the next test for why the tensor must not survive."""
    torch = pytest.importorskip("torch")
    pooled = pool_time(torch.ones(1, 2, 1, 1, 8), n_t=4)
    assert isinstance(pooled, np.ndarray)
    assert pooled.shape == (1, 4, 2)


def test_feature_path_survives_a_lightgbm_fit_in_the_same_process():
    """Regression: LightGBM and PyTorch each load their own OpenMP runtime, and
    a *threaded torch reduction* running after a LightGBM fit deadlocks the
    process outright -- no error, no traceback, it simply stops. ``eval_probe``
    reaches that ordering on any multi-fold protocol with ``--readouts lgbm``,
    where fold 2's extraction follows fold 1's fit. Keeping the feature path in
    numpy is what prevents it, so this asserts the ordering is survivable.
    """
    lgb = pytest.importorskip("lightgbm")
    from sklearn.multioutput import MultiOutputRegressor

    rng = np.random.default_rng(9)
    x = rng.normal(size=(200, 8))
    MultiOutputRegressor(lgb.LGBMRegressor(verbosity=-1)).fit(x, x[:, :2])

    ex = FeatureExtractor("raw", stride=4, grid_shape=(47, 29, 18))
    pooled = pool_time(rng.random((2, 47, 29, 18, 20)), n_t=4)
    assert ex.transform(ex.select(pooled)).shape == (2, 4, 12 * 8 * 5)


def test_raw_extractor_reproduces_the_stride_baseline():
    """The refactor must not move the published number: selecting a stride-4
    boolean mask over the flattened grid has to equal slicing the grid."""
    grid = (47, 29, 18)
    x = np.random.default_rng(10).random((2, *grid, 20))
    pooled = pool_time(x, n_t=4)

    ex = FeatureExtractor("raw", stride=4, grid_shape=grid)
    got = ex.transform(ex.select(pooled))

    ref = x[:, ::4, ::4, ::4, :].reshape(2, -1, 4, 5).mean(axis=3).transpose(0, 2, 1)
    assert np.allclose(got, ref, atol=1e-6)


def test_fold_pca_fits_on_rows_and_transforms_to_k():
    grid = (4, 2, 2)
    mask = np.ones(grid, dtype=bool)
    ex = FeatureExtractor("fold-pca", mask=mask, n_components=3, grid_shape=grid)
    assert ex.needs_fit

    rng = np.random.default_rng(8)
    ex.fit(rng.normal(size=(200, 16)))
    selected = rng.normal(size=(5, 4, 16))
    assert ex.transform(selected).shape == (5, 4, 3)


def test_fold_pca_refuses_to_transform_before_fitting():
    mask = np.ones((4, 2, 2), dtype=bool)
    ex = FeatureExtractor("fold-pca", mask=mask, n_components=2, grid_shape=(4, 2, 2))
    with pytest.raises(RuntimeError):
        ex.transform(np.zeros((1, 2, 16)))


def test_corpus_kinds_require_a_mask():
    with pytest.raises(ValueError):
        FeatureExtractor("corpus-pca", mask=None)


def test_parse_spec_splits_and_validates():
    assert parse_spec("raw") == (("raw", None),)
    assert parse_spec("fold-pca+lr-cca") == (("fold-pca", None), ("lr-cca", None))
    for bad in ("", "nope", "fold-pca+nope", "lr-cca:many"):
        with pytest.raises(ValueError):
            parse_spec(bad)


def test_parse_spec_reads_a_per_part_component_budget():
    """Without this a concatenation is not a fair test: the readout whitens
    every feature, so an unbudgeted corpus block doubles the dimensionality
    under one ridge alpha."""
    assert parse_spec("fold-pca+lr-cca:32") == (("fold-pca", None), ("lr-cca", 32))
    assert parse_spec("corpus-pca:8") == (("corpus-pca", 8),)


def _slice_extractor(dim, offset):
    """A pass-through source selecting ``dim`` columns from ``offset``.

    Kind stays ``raw`` so ``transform`` is the identity -- this exercises the
    composite's concatenation, not any basis.
    """
    ex = FeatureExtractor("raw", stride=1, grid_shape=(2, 2, 2))
    ex.mask_flat = np.zeros(8, dtype=bool)
    ex.mask_flat[offset:offset + dim] = True
    return ex


def test_composite_concatenates_its_parts_in_order():
    a, b = _slice_extractor(3, 0), _slice_extractor(2, 4)
    comp = CompositeExtractor("a+b", [a, b])

    pooled = np.arange(1 * 2 * 8, dtype=float).reshape(1, 2, 8)
    out = comp(pooled)
    assert out.shape == (1, 2, 5)
    assert np.allclose(out[..., :3], a(pooled))
    assert np.allclose(out[..., 3:], b(pooled))
    assert comp.parts == (a, b)


def test_composite_needs_fit_if_any_part_does():
    mask = np.ones((4, 2, 2), dtype=bool)
    fold = FeatureExtractor("fold-pca", mask=mask, n_components=2, grid_shape=(4, 2, 2))
    plain = _slice_extractor(3, 0)
    assert CompositeExtractor("x", [plain]).needs_fit is False
    assert CompositeExtractor("y", [plain, fold]).needs_fit is True


# --- which participants a basis is allowed to see ----------------------------


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


@pytest.mark.parametrize("kind", FEATURE_KINDS)
def test_only_fold_local_sources_are_fold_local(kind):
    """The corpus bases are frozen: if one of them ever reports ``needs_fit``,
    it is being refitted per fold and is no longer an unsupervised transfer.

    The converse matters too, which is why this is keyed on the ``fold-`` prefix
    rather than naming one source. ``fold-srm`` and ``fold-pls`` are fitted per
    fold by construction -- and ``fold-pls`` reads the *targets*, so a version of
    it that stopped reporting ``needs_fit`` would be applying a supervised basis
    fitted somewhere other than inside the training fold. Both directions are
    the same invariant: a source is fold-local exactly when its name says so."""
    mask = np.ones((4, 2, 2), dtype=bool)
    basis = None
    if kind in ("corpus-pca", "diff-pca"):
        basis = {"mean": np.zeros(16), "components": np.eye(16)[:, :2]}
    elif kind == "lr-cca":
        basis = {"mean": np.zeros(16), "left_index": np.arange(8),
                 "right_index": np.arange(8, 16),
                 "left_weights": np.eye(8)[:, :2], "right_weights": np.eye(8)[:, :2]}
    ex = FeatureExtractor(kind, mask=mask, basis=basis, grid_shape=(4, 2, 2))
    assert ex.needs_fit == kind.startswith("fold-")
