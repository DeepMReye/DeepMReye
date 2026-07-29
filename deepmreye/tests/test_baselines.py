"""The readout zoo and the per-subject aggregation it is scored with."""
import numpy as np
import pytest

from deepmreye.evaluate.baselines import (
    ALL_READOUTS,
    DEFAULT_READOUTS,
    build_readout,
    fit_readout,
    predict,
)
from deepmreye.evaluate.probe import aggregate_by_subject, flatten_valid_groups


def _linear_task(n=300, d=40, noise=0.3, seed=0):
    """Gaze as a noisy linear function of the features."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    w = rng.normal(size=(d, 2))
    y = x @ w + rng.normal(scale=noise, size=(n, 2))
    return x, y


@pytest.mark.parametrize("name", ALL_READOUTS)
def test_every_readout_fits_and_predicts_two_columns(name):
    x, y = _linear_task()
    model = fit_readout(name, x, y)
    assert model is not None
    pred = predict(model, x)
    assert pred.shape == (len(x), 2)
    assert np.isfinite(pred).all()


@pytest.mark.parametrize("name", [n for n in ALL_READOUTS if n != "mean"])
def test_every_readout_beats_the_mean_on_a_learnable_task(name):
    """A readout that cannot beat a constant on clean linear data is broken.

    This is a smoke test of the wiring -- scaler, multi-output handling, the
    PLS/PCA component clamps -- not a claim about the models.
    """
    x, y = _linear_task()
    const = fit_readout("mean", x, y)
    model = fit_readout(name, x, y)

    err = lambda m: np.mean((y - predict(m, x)) ** 2)
    assert err(model) < err(const)


def test_mean_readout_is_actually_constant():
    x, y = _linear_task()
    pred = predict(fit_readout("mean", x, y), x)
    assert np.allclose(pred, pred[0])
    np.testing.assert_allclose(pred[0], y.mean(axis=0), rtol=1e-6)


def test_components_are_clamped_to_what_the_data_supports():
    """PCA and PLS raise rather than degrade when asked for too many components,
    and a small held-out fold is exactly where that happens."""
    for name in ("pca-ridge", "pls"):
        model = build_readout(name, n_samples=8, n_features=5, n_components=64)
        x, y = _linear_task(n=8, d=5)
        model.fit(x, y)          # must not raise
        assert predict(model, x).shape == (8, 2)


def test_ridge_cv_picks_a_different_alpha_than_the_fixed_one():
    """The point of ridge-cv over ridge is that alpha=1.0 is arbitrary."""
    x, y = _linear_task(n=200, d=150, noise=2.0)
    model = fit_readout("ridge-cv", x, y)
    chosen = model[-1].alpha_
    assert chosen != 1.0


def test_fit_returns_none_rather_than_raising_on_too_little_data():
    assert fit_readout("ridge-cv", np.zeros((1, 4)), np.zeros((1, 2))) is None


def test_default_readouts_are_a_subset_of_all():
    assert set(DEFAULT_READOUTS) <= set(ALL_READOUTS)


# --- per-subject aggregation ------------------------------------------------


def test_pooling_across_subjects_rewards_predicting_only_the_subject():
    """The reason metrics are aggregated per participant.

    Two subjects whose gaze sits in different places. The 'model' predicts each
    subject's mean and nothing else -- zero within-subject decoding. Pooled, it
    looks like a strong correlation; per subject, it is correctly nothing.
    """
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(-10, 1, size=(200, 2)),
                        rng.normal(+10, 1, size=(200, 2))])
    subjects = np.array(["sub-a"] * 200 + ["sub-b"] * 200)
    preds = np.concatenate([np.full((200, 2), -10.0), np.full((200, 2), +10.0)])

    from deepmreye.evaluate.probe import compute_probe_metrics
    pooled = compute_probe_metrics(y, preds)
    assert pooled["pearson_r_x"] > 0.9      # the trap

    per_subject = aggregate_by_subject(y, preds, subjects)
    assert per_subject["n_subjects"] == 2
    assert np.isnan(per_subject["pearson_r_x"])   # constant prediction within subject


def test_aggregation_takes_the_median_not_the_mean():
    """One failed participant should not move the headline number."""
    rng = np.random.default_rng(1)
    ys, preds, subs = [], [], []
    for i in range(5):
        y = rng.normal(size=(100, 2))
        # Four good subjects, one that is pure noise.
        pred = y + rng.normal(scale=0.1, size=y.shape) if i < 4 else rng.normal(size=y.shape)
        ys.append(y)
        preds.append(pred)
        subs += [f"sub-{i}"] * 100

    m = aggregate_by_subject(np.concatenate(ys), np.concatenate(preds), np.array(subs))
    assert m["n_subjects"] == 5
    assert m["pearson_r_x"] > 0.9


def test_subjects_with_too_few_rows_are_dropped():
    y = np.random.default_rng(2).normal(size=(105, 2))
    subs = np.array(["sub-big"] * 100 + ["sub-tiny"] * 5)
    m = aggregate_by_subject(y, y, subs, min_rows=20)
    assert set(m["per_subject"]) == {"sub-big"}


# --- row expansion ----------------------------------------------------------


def test_flatten_expands_groups_and_drops_the_same_rows_as_targets():
    """Dataset and subject labels are per window; features and targets are per
    temporal bin. Getting the expansion wrong misattributes predictions."""
    feats = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    targets = np.array([[[1.0, 1.0], [np.nan, 2.0], [3.0, 3.0]],
                        [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]])
    datasets = np.array(["ds_a", "ds_b"])
    subjects = np.array(["sub-1", "sub-2"])

    x, y, ds, sub = flatten_valid_groups(feats, targets, datasets, subjects)

    assert len(x) == len(y) == len(ds) == len(sub) == 5   # one NaN bin dropped
    assert list(ds) == ["ds_a", "ds_a", "ds_b", "ds_b", "ds_b"]
    assert list(sub) == ["sub-1", "sub-1", "sub-2", "sub-2", "sub-2"]
    # The surviving feature rows are the un-dropped bins, in order.
    np.testing.assert_array_equal(x, feats.reshape(-1, 4)[[0, 2, 3, 4, 5]])
