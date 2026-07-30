"""Tests for the probe head and the held-out split protocols.

These pin the two failures that made earlier probe numbers meaningless: the
target being averaged over a whole 100-TR window, and NaN gaze samples deleting
entire datasets from the evaluation.
"""
import numpy as np
import pytest

from deepmreye.data.probe_dataset import PARADIGM_GROUPS, dataset_folds, paradigm_folds
from deepmreye.evaluate.baselines import fit_readout
from deepmreye.evaluate.probe import (
    compute_probe_metrics,
    flatten_valid,
    temporal_targets,
)


def test_targets_are_kept_per_temporal_bin_not_per_window():
    """One target per temporal patch, so within-window gaze motion survives."""
    labels = np.zeros((2, 100, 10, 2), dtype=np.float32)
    # A gaze sweep across the window: the whole point is that this is visible.
    labels[..., 0] = np.linspace(-10, 10, 100)[None, :, None]

    out = temporal_targets(labels, n_t=20)

    assert out.shape == (2, 20, 2)
    # Monotonic across bins, and spanning the real range -- not collapsed to 0.
    assert np.all(np.diff(out[0, :, 0]) > 0)
    assert out[0, 0, 0] < -8 and out[0, -1, 0] > 8


def test_window_mean_would_have_destroyed_that_signal():
    """The regression this replaced: SD of window means vs SD within window."""
    labels = np.zeros((1, 100, 10, 2), dtype=np.float32)
    labels[..., 0] = np.linspace(-10, 10, 100)[None, :, None]

    binned = temporal_targets(labels, n_t=20)[0, :, 0]
    window_mean = np.nanmean(labels[0, ..., 0])

    assert binned.std() > 5.0        # the signal is there per bin
    assert abs(window_mean) < 0.5    # and gone entirely in the window mean


def test_nan_gaze_samples_are_averaged_around_not_propagated():
    """A bin with any valid sample stays usable.

    Windows containing at least one NaN are 100% of two labeled datasets, so
    propagating NaN removed those datasets from the evaluation completely.
    """
    labels = np.full((1, 100, 10, 2), 3.0, dtype=np.float32)
    labels[0, 0, 0, :] = np.nan  # one missing sample in the first bin

    out = temporal_targets(labels, n_t=20)

    assert not np.isnan(out[0, 0]).any()
    assert out[0, 0, 0] == pytest.approx(3.0)


def test_bin_with_no_valid_sample_stays_nan():
    # Masked downstream rather than silently imputed to something wrong.
    labels = np.full((1, 100, 10, 2), 3.0, dtype=np.float32)
    labels[0, :5] = np.nan  # the entire first bin

    out = temporal_targets(labels, n_t=20)

    assert np.isnan(out[0, 0]).all()
    assert not np.isnan(out[0, 1]).any()


def test_uneven_window_pads_like_the_patcher():
    # The patcher pads T up to a multiple of temp_patch_size; targets must bin
    # the same way or block and gaze go out of alignment.
    labels = np.ones((1, 98, 10, 2), dtype=np.float32)
    out = temporal_targets(labels, n_t=20)
    assert out.shape == (1, 20, 2)
    assert not np.isnan(out[0, :19]).any()


def test_flatten_valid_drops_only_nan_target_rows():
    feats = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    targs = np.zeros((2, 3, 2))
    targs[0, 1] = np.nan

    x, y = flatten_valid(feats, targs)

    assert len(x) == 5 and len(y) == 5
    assert not np.isnan(y).any()


def test_probe_recovers_a_linear_signal():
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(40, 20, 16))
    w = rng.normal(size=(16, 2))
    targs = feats @ w

    x, y = flatten_valid(feats, targs)
    model = fit_readout("ridge", x, y)
    metrics = compute_probe_metrics(y, model.predict(x), baseline=y.mean(axis=0))

    assert metrics["r2_vs_baseline"] > 0.95
    assert metrics["n"] == 800


def test_r2_is_measured_against_the_training_mean():
    """R^2 against the test mean flatters a model that only learned the centre."""
    y_true = np.array([[1.0, 1.0], [3.0, 3.0]])
    constant = np.array([[2.0, 2.0], [2.0, 2.0]])

    # Predicting the test set's own mean scores 0 against that mean...
    assert compute_probe_metrics(y_true, constant)["r2_x"] == pytest.approx(0.0)
    # ...and also 0 against an explicit baseline equal to it, as it should.
    out = compute_probe_metrics(y_true, constant, baseline=np.array([2.0, 2.0]))
    assert out["r2_vs_baseline"] == pytest.approx(0.0)


def test_metrics_survive_a_constant_target():
    # A held-out fold can be nearly constant; Pearson is undefined there and
    # must report NaN rather than crash the fold.
    y = np.ones((10, 2))
    out = compute_probe_metrics(y, np.ones((10, 2)) * 1.5)
    assert np.isnan(out["pearson_r_x"])
    assert out["euclidean_error"] == pytest.approx(np.sqrt(2) * 0.5)


def test_leave_one_dataset_out_covers_every_dataset_once():
    datasets = ["dsL01_guided_fixations", "dsL02_pursuit", "dsL03_pursuit"]
    folds = dataset_folds(datasets)

    assert len(folds) == 3
    assert set().union(*(h for _, h in folds)) == set(datasets)


def test_paradigm_folds_hold_out_all_pursuit_together():
    """Holding out one pursuit set alone still trains on two others."""
    datasets = sorted(sum(PARADIGM_GROUPS.values(), []))
    folds = dict(paradigm_folds(datasets))

    assert folds["pursuit"] == {"dsL02_pursuit", "dsL03_pursuit", "dsL04_pursuit"}
    assert folds["fixation"] == {"dsL01_guided_fixations"}


def test_paradigm_folds_skip_a_fold_that_would_leave_no_training_data():
    # Only pursuit present: holding it out leaves nothing to train on.
    assert paradigm_folds(["dsL02_pursuit", "dsL03_pursuit"]) == []
