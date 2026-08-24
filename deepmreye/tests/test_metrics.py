"""Tests for the metrics, and above all for the calibration they depend on.

R-squared and Euclidean error are the two metrics that can be *silently* wrong
here: the protocol hands them predictions in z-units against labels in degrees,
which produces a number rather than an error. So the tests below pin the
invariances that separate the three metrics -- r survives an affine map, R2 does
not, Euclidean error is in label units -- and check that the calibration
recovers a known gain.
"""
import numpy as np
import pytest

from deepmreye import metrics


def _signal(n=400, seed=0):
    rng = np.random.default_rng(seed)
    true = rng.normal(size=(n, 2)) * [4.0, 3.0]
    pred = true + rng.normal(size=(n, 2)) * 0.5
    return pred, true


def test_pearson_is_invariant_to_gain_and_offset():
    """Why r is the metric to compare methods on: it cannot be gamed by scale."""
    pred, true = _signal()
    base = metrics.pearson(pred[:, 0], true[:, 0])
    for gain, offset in ((3.0, 0.0), (0.1, 12.0), (1.0, -7.5)):
        assert metrics.pearson(pred[:, 0] * gain + offset, true[:, 0]) == pytest.approx(base)


def test_r_squared_is_destroyed_by_the_gain_r_ignores():
    """And why R2 needs the calibration: the same prediction, rescaled, collapses.

    The two directions fail differently, which is worth pinning. Shrinking the
    prediction drives R2 toward **zero** -- it degrades into predicting the mean.
    Over-shooting drives it **negative**, because the errors then exceed the
    spread of the target itself. Measured cross-dataset gains span 0.11 to 2.27,
    so both directions occur in practice and neither is visible in r.
    """
    pred, true = _signal()
    assert metrics.r_squared(pred[:, 0], true[:, 0]) > 0.9
    assert 0.0 < metrics.r_squared(pred[:, 0] * 0.11, true[:, 0]) < 0.3
    assert metrics.r_squared(pred[:, 0] * 2.27, true[:, 0]) < -0.5


def test_r_squared_is_not_pearson_squared():
    """A perfectly correlated prediction at the wrong scale: r=1, R2 far below."""
    true = np.linspace(-5, 5, 200)
    pred = 3.0 * true + 2.0
    assert metrics.pearson(pred, true) == pytest.approx(1.0)
    assert metrics.r_squared(pred, true) < -1.0


def test_r_squared_of_the_mean_is_zero():
    true = np.linspace(-5, 5, 200)
    assert metrics.r_squared(np.full_like(true, true.mean()), true) == pytest.approx(0.0)


def test_fit_affine_recovers_a_planted_gain_and_offset():
    rng = np.random.default_rng(1)
    true = rng.normal(size=(500, 2)) * [5.0, 2.0]
    pred = (true - [1.0, -3.0]) / [2.5, 0.4]     # what a z-unit readout looks like
    gain, offset = metrics.fit_affine(pred, true)
    assert gain == pytest.approx([2.5, 0.4], rel=1e-6)
    assert offset == pytest.approx([1.0, -3.0], abs=1e-6)
    assert metrics.apply_affine(pred, gain, offset) == pytest.approx(true)


def test_calibration_turns_an_uninterpretable_r2_into_a_good_one():
    """The whole point of the module, end to end.

    `z` is what the protocol's readout actually emits: unit variance, zero mean,
    against labels at 4 and 3 degrees SD. Its r is already excellent; its
    uncalibrated R2 and its error in degrees describe the unit mismatch and
    nothing else.
    """
    pred, true = _signal(seed=2)
    z = (pred - pred.mean(axis=0)) / pred.std(axis=0)
    raw = metrics.score(z, true)
    gain, offset = metrics.fit_affine(z, true)
    cal = metrics.score(z, true, gain, offset)

    assert raw["r"] == pytest.approx(cal["r"]) and cal["r"] > 0.98   # r is untouched
    assert cal["r2"] > 0.9                                          # meaningful after
    assert cal["r2"] - raw["r2"] > 0.3                              # and much better
    assert cal["euclid_median"] < 0.5 * raw["euclid_median"]         # error in degrees
    assert cal["gain_x"] == pytest.approx(gain[0])


def test_euclidean_is_in_label_units_and_median_resists_outliers():
    true = np.zeros((100, 2))
    pred = np.zeros((100, 2))
    pred[:, 0] = 2.0                       # every sample 2 degrees off in x
    med, mean = metrics.euclidean(pred, true)
    assert med == pytest.approx(2.0)
    assert mean == pytest.approx(2.0)

    pred[0, 0] = 500.0                     # one track-loss sample
    med, mean = metrics.euclidean(pred, true)
    assert med == pytest.approx(2.0)       # median unmoved
    assert mean > 6.0                      # mean now describes the artifact


def test_degenerate_inputs_return_nan_rather_than_a_number():
    """A constant prediction has no correlation; saying 0.0 would be a claim."""
    true = np.linspace(0, 1, 50)
    assert np.isnan(metrics.pearson(np.ones(50), true))
    assert np.isnan(metrics.pearson(true[:5], true[:5]))       # too few rows
    assert np.isnan(metrics.r_squared(true[:5], true[:5]))
    assert all(np.isnan(v) for v in metrics.euclidean(np.zeros((3, 2)), np.zeros((3, 2))))


def test_nanmedian_ignores_missing_folds():
    assert metrics.nanmedian([1.0, np.nan, 3.0, None]) == pytest.approx(2.0)
    assert np.isnan(metrics.nanmedian([np.nan, None]))
