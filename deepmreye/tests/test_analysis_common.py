"""The analysis helpers that stand in for the protocol's own pieces must give
the protocol's answers: the covariance ridge is RidgeCV, least squares is
least squares, and the confident-fraction statistic keeps what it says."""
import json

import numpy as np

from deepmreye import probe
from scripts.unlabeled.common import (
    corpus_sizes, keep_most_confident, ols_readout, ridge_by_covariance,
)


def _xy(seed=0, n=400, p=12):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    y = x @ rng.normal(size=(p, 3)) + 0.5 * rng.normal(size=(n, 3))
    return x, y


def test_ridge_by_covariance_is_ridgecv():
    from sklearn.linear_model import RidgeCV

    x, y = _xy()
    q = np.random.default_rng(1).normal(size=(50, x.shape[1]))
    want = RidgeCV(alphas=probe.ALPHAS).fit(x, y).predict(q)
    np.testing.assert_allclose(ridge_by_covariance(x, y)(q), want, rtol=1e-8, atol=1e-8)


def test_ols_readout_is_least_squares():
    x, y = _xy(2)
    q = np.random.default_rng(3).normal(size=(20, x.shape[1]))
    X = np.c_[x, np.ones(len(x))]
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    np.testing.assert_allclose(ols_readout(x, y)(q), np.c_[q, np.ones(len(q))] @ coef, atol=1e-8)


def test_keep_most_confident_keeps_the_most_confident():
    rows = ([{"dataset": "A", "score": s, "r": r} for s, r in ((3, 0.9), (2, 0.8), (1, 0.1), (0, 0.0))]
            + [{"dataset": "B", "score": s, "r": r} for s, r in ((1, 0.6), (0, 0.2))])
    med, mean = keep_most_confident(rows, "score", 0.5)
    # A keeps its top two (0.9, 0.8 -> 0.85), B its top one (0.6).
    assert np.isclose(med, 0.725) and np.isclose(mean, 0.725)
    assert np.isclose(keep_most_confident(rows, "score", 1.0)[0], np.median([0.45, 0.4]))


def test_corpus_sizes_reads_the_sweep_or_the_files(tmp_path):
    for n in (400, 25, 2060):
        (tmp_path / f"basis_n{n}.npz").write_bytes(b"")
    assert corpus_sizes(tmp_path) == [25, 400, 2060]
    (tmp_path / "sweep.json").write_text(json.dumps({"sizes": [25, 2060]}))
    assert corpus_sizes(tmp_path) == [25, 2060]
