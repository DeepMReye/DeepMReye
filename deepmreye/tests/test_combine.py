"""Tests for the per-block readouts (`deepmreye/evaluate/combine.py`).

The property that matters is not accuracy, it is that a **useless block cannot
hurt**. `ridge-cv` on a concatenation fails exactly there -- one alpha over both
blocks means noise columns are penalised as lightly as signal ones -- so that is
what these tests pin, together with the nesting claim (one block => ridge-cv)
and the fitted diagnostics the eval report writes out.
"""
import numpy as np
import pytest

from deepmreye.evaluate.baselines import BLOCK_READOUTS, fit_readout, predict


def _data(n=400, d_signal=8, d_noise=24, seed=0):
    """[signal block | pure-noise block] -> y. The noise block is a distractor."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(n, d_signal))
    w = rng.normal(size=(d_signal, 2))
    y = signal @ w + 0.3 * rng.normal(size=(n, 2))
    noise = rng.normal(size=(n, d_noise))
    groups = np.repeat(np.arange(10), n // 10)
    return np.hstack([signal, noise]), y, groups, [d_signal, d_noise]


def _r(model, x, y):
    p = predict(model, x)
    return float(np.mean([np.corrcoef(p[:, i], y[:, i])[0, 1] for i in range(2)]))


@pytest.mark.parametrize("readout", BLOCK_READOUTS)
def test_block_readout_beats_ridge_cv_when_one_block_is_noise(readout):
    x, y, groups, blocks = _data()
    x_tr, y_tr, g_tr = x[:300], y[:300], groups[:300]
    x_te, y_te = x[300:], y[300:]

    banded = fit_readout(readout, x_tr, y_tr, blocks=blocks, groups=g_tr)
    plain = fit_readout("ridge-cv", x_tr, y_tr)
    assert _r(banded, x_te, y_te) > _r(plain, x_te, y_te)


def test_banded_ridge_shrinks_the_noise_block_hard():
    x, y, groups, blocks = _data()
    model = fit_readout("banded-ridge", x, y, blocks=blocks, groups=groups)
    # The penalty on the noise block must exceed the penalty on the signal one.
    assert model.block_alphas_[1] > model.block_alphas_[0]


def test_stacked_ridge_gives_the_noise_block_almost_no_weight():
    x, y, groups, blocks = _data()
    model = fit_readout("stack-ridge", x, y, blocks=blocks, groups=groups)
    assert model.stack_weights_.shape == (2, 2)
    np.testing.assert_allclose(model.stack_weights_.sum(axis=0), 1.0, atol=1e-6)
    assert model.stack_weights_[1].max() < 0.25


@pytest.mark.parametrize("readout", BLOCK_READOUTS)
def test_single_block_matches_ridge_cv(readout):
    """Nesting: with nothing to weight, these must be ridge with a tuned alpha."""
    x, y, groups, _ = _data(d_noise=0)
    model = fit_readout(readout, x, y, blocks=None, groups=groups)
    plain = fit_readout("ridge-cv", x, y)
    assert _r(model, x, y) == pytest.approx(_r(plain, x, y), abs=0.02)


def test_block_widths_must_match_the_features():
    x, y, groups, _ = _data()
    with pytest.raises(ValueError, match="blocks"):
        fit_readout("banded-ridge", x, y, blocks=[3, 3], groups=groups)


def test_composite_extractor_records_block_widths():
    from deepmreye.evaluate.features import CompositeExtractor

    class Stub:
        needs_fit = False

        def __init__(self, d):
            self.d = d

        def __call__(self, pooled, raw=None, n_t=None, subject_ids=None):
            return np.zeros((2, 3, self.d))

    ex = CompositeExtractor("a+b", [Stub(5), Stub(7)])
    out = ex(None)
    assert out.shape == (2, 3, 12)
    assert ex.block_widths == [5, 7]


def test_dyadic_blocks_are_log_spaced_and_conserve_width():
    from deepmreye.evaluate.combine import dyadic_blocks

    assert dyadic_blocks([64]) == [8, 8, 16, 32]
    assert dyadic_blocks([64, 32]) == [8, 8, 16, 32, 8, 8, 16]
    for widths in ([256], [64, 32], [12], [5], [100]):
        assert sum(dyadic_blocks(widths)) == sum(widths)
    # No runt band: a 4-wide tail is absorbed rather than given its own penalty.
    assert min(dyadic_blocks([68])) >= 8


def test_dyadic_banding_recovers_a_tapered_spectrum():
    """A variance-ordered block whose tail is noise should get a tapered penalty."""
    from deepmreye.evaluate.combine import dyadic_blocks

    rng = np.random.default_rng(1)
    n, d = 600, 64
    # Components 0-7 carry gaze; everything past 16 is noise -- the situation a
    # `:k` truncation handles by guessing k.
    x = rng.normal(size=(n, d))
    y = x[:, :8] @ rng.normal(size=(8, 2)) + 0.5 * rng.normal(size=(n, 2))
    groups = np.repeat(np.arange(10), n // 10)
    banded = fit_readout("banded-ridge", x[:400], y[:400],
                         blocks=dyadic_blocks([d]), groups=groups[:400])
    # The leading band must be penalised less than the trailing one.
    assert banded.block_alphas_[0] < banded.block_alphas_[-1]
    assert _r(banded, x[400:], y[400:]) > _r(
        fit_readout("ridge-cv", x[:400], y[:400]), x[400:], y[400:])


def _fake_corpus(d=60, k=20, seed=3):
    """A corpus-pca-shaped basis dict over `d` voxels."""
    rng = np.random.default_rng(seed)
    q = np.linalg.qr(rng.normal(size=(d, d)))[0][:, :k]
    vals = np.linspace(10.0, 1.0, k)
    return {"mean": np.zeros(d), "components": q, "eigenvalues": vals,
            "total_variance": np.array([vals.sum() + 5.0])}


def test_fold_shrunk_pca_endpoints_reproduce_fold_pca_and_corpus_pca():
    from deepmreye.unsupervised import fit_shrunk_pca

    rng = np.random.default_rng(0)
    d, k = 60, 6
    rows = rng.normal(size=(200, d)) @ np.diag(np.linspace(5, 0.1, d))
    corpus = _fake_corpus(d=d)

    # lam=0: the fold covariance alone, so the retained variances must match a
    # plain PCA of the same rows.
    fold = fit_shrunk_pca(rows, corpus, k, lam=0.0)
    cov = np.cov(rows, rowvar=False)
    want = np.sort(np.linalg.eigvalsh(cov))[::-1][:k] / np.trace(cov)
    np.testing.assert_allclose(fold["eigenvalues"], want, rtol=1e-6)

    # lam=1: the corpus target alone, so the directions must be its own.
    corp = fit_shrunk_pca(rows, corpus, k, lam=1.0)
    overlap = np.abs(corp["components"].T @ corpus["components"][:, :k])
    np.testing.assert_allclose(np.diag(overlap), np.ones(k), atol=1e-5)


def test_fold_shrunk_pca_interpolates_and_keeps_full_rank_directions():
    """A mid lam must not collapse the basis into the corpus subspace."""
    from deepmreye.unsupervised import fit_shrunk_pca

    rng = np.random.default_rng(1)
    d, k_corpus = 60, 20
    corpus = _fake_corpus(d=d, k=k_corpus)
    # Fold variance deliberately placed OUTSIDE the corpus span.
    outside = np.linalg.qr(rng.normal(size=(d, d)))[0]
    outside -= corpus["components"] @ (corpus["components"].T @ outside)
    direction = outside[:, 0] / np.linalg.norm(outside[:, 0])
    rows = (rng.normal(size=(200, 1)) * 20.0) @ direction[None, :]
    rows += 0.01 * rng.normal(size=(200, d))

    mid = fit_shrunk_pca(rows, corpus, 4, lam=0.5)
    # The out-of-corpus direction still has to be recovered: isotropic
    # completion of the target's tail is what keeps lam>0 from truncating.
    assert np.abs(mid["components"][:, 0] @ direction) > 0.9


@pytest.mark.parametrize("block,sig", [(0, slice(0, 8)), (2, slice(16, 32)),
                                       (5, slice(128, 256))])
def test_banded_search_finds_which_band_carries_the_signal(block, sig):
    """The >2-block search is a random simplex sweep, so it needs a test that it
    *responds* rather than returning whatever the seed drew first.

    On the real corpus it selected the same weight vector for three different
    bases, which looks like a stuck search; this is the check that it is a broad
    optimum instead. Signal is placed in one dyadic band at a time and the band
    with the largest weight has to be that one.
    """
    from deepmreye.evaluate.combine import BandedRidge, dyadic_blocks

    blocks = dyadic_blocks([256])
    rng = np.random.default_rng(5)
    x = rng.normal(size=(4000, 256))
    y = x[:, sig] @ rng.normal(size=(x[:, sig].shape[1], 2))
    y += 0.5 * rng.normal(size=(4000, 2))
    groups = np.repeat(np.arange(20), 200)
    m = BandedRidge(blocks=blocks, seed=0).fit(x, y, groups=groups)
    assert int(np.argmax(m.block_weights_)) == block
