"""Tests for the probe head and the held-out split protocols.

These pin the two failures that made earlier probe numbers meaningless: the
target being averaged over a whole 100-TR window, and NaN gaze samples deleting
entire datasets from the evaluation.
"""
import numpy as np
import pytest
import torch

from deepmreye.data.probe_dataset import PARADIGM_GROUPS, dataset_folds, paradigm_folds
from deepmreye.evaluate.baselines import fit_readout
from deepmreye.evaluate.probe import (
    collapse_spatial,
    compute_probe_metrics,
    flatten_valid,
    parse_spatial_pool,
    pool_spatial,
    spatial_grid,
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


def test_pool_spatial_keeps_time_and_averages_space():
    n_s, n_t, d = 4, 3, 8
    # Token (s, t) is flattened as s * n_t + t; give each a distinct value.
    reps = torch.zeros(1, n_s * n_t, d)
    for s in range(n_s):
        for t in range(n_t):
            reps[0, s * n_t + t, :] = s * 10 + t

    out = pool_spatial(reps, n_s, n_t)

    assert out.shape == (1, n_t, d)
    # Mean over s of (s*10 + t) = 15 + t for n_s = 4.
    for t in range(n_t):
        assert out[0, t, 0].item() == pytest.approx(15.0 + t)


def test_pool_spatial_rejects_a_token_count_that_does_not_match():
    with pytest.raises(ValueError):
        pool_spatial(torch.zeros(1, 11, 8), 4, 3)


def test_spatial_grid_ceils_because_the_patcher_pads():
    # 47x29x18 at patch size 8 pads to 48x32x24 -> the 6x4x3 grid the whole
    # evaluation is written in terms of.
    assert spatial_grid((47, 29, 18), 8) == (6, 4, 3)


def test_parse_spatial_pool_rejects_a_grid_finer_than_the_encoder_has():
    assert parse_spatial_pool("mean", (6, 4, 3)) == (1, 1, 1)
    assert parse_spatial_pool("none", (6, 4, 3)) == (6, 4, 3)
    assert parse_spatial_pool("2x1x1", (6, 4, 3)) == (2, 1, 1)
    for bad in ("7x4x3", "2x1", "sixby", "0x1x1"):
        with pytest.raises(ValueError):
            parse_spatial_pool(bad, (6, 4, 3))


def _positional_reps(grid, n_t, d):
    """Tokens whose value encodes their spatial position -- the contrast that
    mean-pooling destroys and that gaze direction is actually carried in."""
    n_s = grid[0] * grid[1] * grid[2]
    reps = torch.zeros(1, n_s * n_t, d)
    for s in range(n_s):
        for t in range(n_t):
            reps[0, s * n_t + t, :] = float(s)
    return reps


def test_collapse_spatial_mean_matches_pool_spatial():
    grid, n_t, d = (2, 2, 2), 3, 4
    reps = _positional_reps(grid, n_t, d)
    n_s = 8

    assert torch.allclose(collapse_spatial(reps, n_s, n_t, grid, "mean"),
                          pool_spatial(reps, n_s, n_t))


def test_collapse_spatial_unpooled_keeps_every_token_and_its_position():
    """The 0.45 -> 0.86 correlation gap lives here: 'none' must keep the
    across-orbit contrast, and keep it in a stable feature order."""
    grid, n_t, d = (2, 2, 2), 3, 4
    reps = _positional_reps(grid, n_t, d)

    out = collapse_spatial(reps, 8, n_t, grid, "none")

    assert out.shape == (1, n_t, 8 * d)
    for t in range(n_t):
        # Features are laid out position-major: token s occupies columns s*d.
        assert out[0, t].view(8, d)[:, 0].tolist() == list(range(8))
    # Mean-pooling the same tokens throws all of that away.
    assert collapse_spatial(reps, 8, n_t, grid, "mean").shape == (1, n_t, d)


def test_collapse_spatial_pools_to_a_coarser_grid():
    """2x1x1 is the left-orbit / right-orbit cut."""
    grid, n_t, d = (2, 2, 2), 3, 4
    reps = _positional_reps(grid, n_t, d)

    out = collapse_spatial(reps, 8, n_t, grid, "2x1x1")

    assert out.shape == (1, n_t, 2 * d)
    # Grid index x * 4 + y * 2 + z, so x = 0 holds tokens 0-3 and x = 1 holds
    # 4-7; pooling over y and z leaves their means, 1.5 and 5.5.
    assert out[0, 0].view(2, d)[:, 0].tolist() == pytest.approx([1.5, 5.5])


def test_collapse_spatial_rejects_a_grid_that_does_not_match_the_tokens():
    with pytest.raises(ValueError):
        collapse_spatial(torch.zeros(1, 24, 4), 8, 3, (3, 3, 3), "none")


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
