"""The DeepMReye 1.0 baseline harness.

The comparison this script exists to make is only valid if it scores the
published model the *same way* ``eval_probe`` scores everything else. That makes
``_reduce`` load-bearing: it has to reproduce
``evaluate.probe.temporal_targets`` exactly. Scored at TR resolution against a
probe number computed on 5-TR means, the published model would look worse for a
reason that has nothing to do with the model -- averaging suppresses noise, and
it would be doing so on only one side.

So the first test here is an equivalence test against the real binner, not a
property test. Nothing about TensorFlow is imported: the script keeps every TF
import inside a function precisely so this file can run in the project venv.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from eval_dme1 import (  # noqa: E402
    CONTAMINATED,
    V1_FILES,
    _reduce,
    find_corpus,
    score_subject,
)

from deepmreye.evaluate.probe import temporal_targets  # noqa: E402


@pytest.mark.parametrize("n_trs,bin_trs", [(100, 5), (60, 5), (40, 4), (97, 5)])
def test_reduce_matches_the_probes_own_binner(n_trs, bin_trs):
    """The equivalence the whole comparison rests on."""
    rng = np.random.default_rng(0)
    labels = rng.normal(size=(n_trs, 10, 2))
    n_t = int(np.ceil(n_trs / bin_trs))
    expected = temporal_targets(labels[None], n_t)[0]
    assert np.allclose(_reduce(labels, bin_trs), expected, equal_nan=True)


def test_reduce_at_bin_one_is_the_per_tr_mean_over_subsamples():
    rng = np.random.default_rng(1)
    labels = rng.normal(size=(30, 10, 2))
    assert np.allclose(_reduce(labels, 1), labels.mean(axis=1))


def test_reduce_pads_a_run_that_does_not_divide_evenly():
    labels = np.ones((7, 10, 2))
    out = _reduce(labels, 5)
    assert out.shape == (2, 2)
    assert np.allclose(out, 1.0)


def test_a_bin_with_no_valid_gaze_stays_nan_rather_than_zero():
    """A NaN bin means no gaze was recorded, and it is masked downstream. Filling
    it with zero would silently invent a fixation at the origin."""
    labels = np.full((10, 10, 2), np.nan)
    labels[5:] = 1.0
    out = _reduce(labels, 5)
    assert np.isnan(out[0]).all()
    assert np.allclose(out[1], 1.0)


def test_reduce_ignores_nans_within_a_bin():
    labels = np.ones((5, 10, 2))
    labels[0, :, :] = np.nan
    assert np.allclose(_reduce(labels, 5), 1.0)


# ------------------------------------------------------------------- scoring

def test_a_perfect_prediction_scores_r_of_one():
    rng = np.random.default_rng(2)
    true = rng.normal(size=(100, 10, 2))
    got = score_subject(true.copy(), true, bin_trs=5)
    assert got["pearson_r_x"] == pytest.approx(1.0, abs=1e-6)
    assert got["pearson_r_y"] == pytest.approx(1.0, abs=1e-6)
    assert got["euclidean_error"] == pytest.approx(0.0, abs=1e-6)


def test_an_unrelated_prediction_scores_near_zero():
    # 4000 TRs -> 800 bins, so the null SD of r is ~1/sqrt(800) = 0.035 and the
    # bound below is >4 sigma. At 400 TRs it is only 80 bins and a chance |r| of
    # 0.28 is an ordinary 2.5-sigma draw, which is a flaky test, not a signal.
    rng = np.random.default_rng(3)
    got = score_subject(rng.normal(size=(4000, 10, 2)),
                        rng.normal(size=(4000, 10, 2)), bin_trs=5)
    assert got["n"] == 800
    assert abs(got["pearson_r_x"]) < 0.15
    assert abs(got["pearson_r_y"]) < 0.15


def test_the_two_gaze_axes_are_scored_separately():
    """dsL06 is the case that matters: the published model reaches r_x 0.95 and
    r_y -0.05 there, and a single pooled number would hide that entirely."""
    rng = np.random.default_rng(4)
    true = rng.normal(size=(300, 10, 2))
    pred = true.copy()
    pred[..., 1] = rng.normal(size=(300, 10))
    got = score_subject(pred, true, bin_trs=5)
    assert got["pearson_r_x"] > 0.95
    assert abs(got["pearson_r_y"]) < 0.3


def test_a_run_with_too_few_valid_bins_is_dropped_not_scored():
    labels = np.full((40, 10, 2), np.nan)
    labels[:10] = 1.0
    assert score_subject(np.zeros((40, 10, 2)), labels, bin_trs=5) is None


def test_nan_targets_are_dropped_rather_than_imputed():
    rng = np.random.default_rng(5)
    true = rng.normal(size=(200, 10, 2))
    pred = true.copy()
    true[:50] = np.nan
    got = score_subject(pred, true, bin_trs=5)
    assert got["n"] == 30                       # 200/5 bins, first 10 all-NaN
    assert got["pearson_r_x"] == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------- guard rails

def test_the_all_data_checkpoint_is_marked_contaminated():
    """``datasets_1to6.h5`` was trained on every labeled participant here, so
    reporting it as held out would be straightforwardly wrong."""
    assert "datasets_1to6.h5" in CONTAMINATED
    assert "datasets_1to5.h5" not in CONTAMINATED


def test_the_vendored_file_list_stays_minimal():
    """Only the architecture and its two helpers. Vendoring v1's package
    ``__init__`` would drag in the whole library, including its ANTs import."""
    assert set(V1_FILES) == {"deepmreye/architecture.py",
                             "deepmreye/util/util.py",
                             "deepmreye/util/model_opts.py"}


def test_find_corpus_honours_an_explicit_path(tmp_path):
    assert find_corpus(str(tmp_path)) == tmp_path


def test_find_corpus_requires_labeled_datasets(tmp_path, monkeypatch):
    """A directory with no ``dsL*`` is not this corpus, and silently accepting
    one would score the published model against nothing."""
    monkeypatch.setenv("DEEPMREYE_DATA", str(tmp_path))
    monkeypatch.setattr("eval_dme1.REPO", tmp_path)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    with pytest.raises(SystemExit):
        find_corpus()

    (tmp_path / "dsL01_x").mkdir()
    (tmp_path / "dsL01_x" / "sub-01.h5").touch()
    assert find_corpus() == tmp_path
