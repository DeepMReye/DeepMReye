"""Tests for the leave-one-dataset-out gaze protocol.

The point of `evaluate` is that there is exactly one implementation of the number,
so these tests check the properties that would let a second one drift back in:
the lag semantics, the cache guard, and that the scoring recovers a known signal
and refuses to score noise.
"""

import numpy as np
import pytest

from deepmreye import evaluate as tp
from deepmreye.model import ALPHAS, make_lags


def _lags_reference(z, lags):
    """Independent reimplementation: block `l` at row `t` is `z[clip(t + l, 0, T-1)]`.

    Written from the semantics rather than copied from the implementation -- comparing two
    copies of the same code tests nothing.
    """
    t_n = len(z)
    if lags == 0:
        return z.copy()
    out = []
    for lag in range(-lags, lags + 1):
        idx = np.clip(np.arange(t_n) + lag, 0, t_n - 1)
        out.append(z[idx])
    return np.concatenate(out, axis=1)


def test_make_lags_matches_an_independent_implementation():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(37, 5))
    for lags in (0, 1, 2, 5):
        assert np.array_equal(make_lags(z, lags), _lags_reference(z, lags))


def test_make_lags_edge_pads_rather_than_zero_pads():
    """The distinction an identity-initialised Conv1d would get wrong."""
    z = np.arange(4, dtype=float).reshape(4, 1)
    out = make_lags(z, 1)
    assert out.shape == (4, 3)
    # lag -1 at row 0 clamps to z[0], not to 0.
    assert out[0, 0] == z[0, 0]
    # lag +1 at the last row clamps to z[-1], not to 0.
    assert out[-1, 2] == z[-1, 0]


def test_make_lags_zero_is_identity():
    z = np.random.default_rng(1).normal(size=(9, 3))
    assert np.array_equal(make_lags(z, 0), z)


def _rec(dataset, subject, t=80, k=4, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(t, 2, k)).astype(np.float32)
    labels = rng.normal(size=(t, 10, 2)).astype(np.float32)
    return {"dataset": dataset, "subject": subject, "z": z, "labels": labels}


def test_fingerprint_changes_with_the_corpus():
    a = [_rec("dsL01", "sub-01"), _rec("dsL02", "sub-01", seed=1)]
    assert tp.corpus_fingerprint(a) == tp.corpus_fingerprint(list(reversed(a)))

    dropped = tp.corpus_fingerprint(a[:1])
    assert dropped != tp.corpus_fingerprint(a)

    longer = [_rec("dsL01", "sub-01", t=81), a[1]]
    assert tp.corpus_fingerprint(longer) != tp.corpus_fingerprint(a)


def test_cache_round_trip_and_guards(tmp_path):
    recs = [_rec("dsL01", "sub-01"), _rec("dsL01", "sub-02", seed=2)]
    path = tmp_path / "cache.npz"
    tp.save_cache(path, recs, "basis_a.npz", 4)

    back = tp.load_cache(path, "basis_a.npz", 4)
    assert [r["subject"] for r in back] == ["sub-01", "sub-02"]
    assert np.array_equal(back[0]["z"], recs[0]["z"])
    assert back[0]["labels"].shape == (80, 10, 2)

    # A different rank or a different basis must not load.
    with pytest.raises(SystemExit):
        tp.load_cache(path, "basis_a.npz", 8)
    with pytest.raises(SystemExit):
        tp.load_cache(path, "basis_b.npz", 4)


def test_cache_refuses_a_corpus_its_fingerprint_does_not_describe(tmp_path):
    """The guard against the failure that is otherwise silent.

    A cache of a retired corpus -- fewer participants, or labels under a
    superseded sign convention -- loads without complaint if only the geometry
    is checked, and answers a question nobody asked.
    """
    recs = [_rec("dsL01", "sub-01"), _rec("dsL01", "sub-02", seed=2)]
    path = tmp_path / "cache.npz"
    tp.save_cache(path, recs, "basis_a.npz", 4)

    d = dict(np.load(path, allow_pickle=False))
    d["fingerprint"] = np.array(tp.corpus_fingerprint(recs[:1]))
    np.savez(path, **d)
    with pytest.raises(SystemExit) as e:
        tp.load_cache(path, "basis_a.npz", 4)
    assert "fingerprint" in str(e.value)


def test_label_convention_is_part_of_the_fingerprint(monkeypatch):
    recs = [_rec("dsL01", "sub-01")]
    before = tp.corpus_fingerprint(recs)
    monkeypatch.setattr(tp, "LABEL_CONVENTION", "y-up-hypothetical")
    assert tp.corpus_fingerprint(recs) != before


def test_lodo_subtr_recovers_a_known_linear_generator():
    """Gaze built as a fixed linear map of z must decode at r ~ 1 across datasets.

    The 10 sub-TR samples vary *smoothly* within a TR, as real gaze does. That matters for
    the assertion: the sub-TR metric pools all 10 slots into one correlation, so a generator
    giving each slot an independent random weight vector caps r near 0.94 no matter how
    exactly the model fits -- an artifact of the generator, not of the harness.
    """
    rng = np.random.default_rng(3)
    k = 6
    w_pos = rng.normal(size=(k, 2))      # gaze position
    w_vel = rng.normal(size=(k, 2))      # within-TR drift, small and smooth
    ramp = np.linspace(-0.5, 0.5, 10).reshape(1, 10, 1)
    recs = []
    for ds in ("dsA", "dsB", "dsC"):
        for s in range(4):
            t = 120
            z = rng.normal(size=(t, 2, k))
            feat = 0.5 * (z[:, 0] + z[:, 1])
            labels = (feat @ w_pos)[:, None, :] + 0.1 * ramp * (feat @ w_vel)[:, None, :]
            recs.append({"dataset": ds, "subject": f"sub-{s}",
                         "z": z.astype(np.float32), "labels": labels.astype(np.float32)})

    out = tp.lodo(recs, lambda r: tp.bilateral(r, k))
    assert out["median_subtr"] > 0.99
    assert out["median_1tr"] > 0.99
    assert set(out["folds"]) == {"dsA", "dsB", "dsC"}


def test_lodo_subtr_reports_both_resolutions_and_all_folds():
    recs = [_rec(ds, f"sub-{i}", seed=10 * j + i)
            for j, ds in enumerate(("dsA", "dsB")) for i in range(3)]
    out = tp.lodo(recs, lambda r: tp.bilateral(r, 4))
    for key in ("participants", "folds", "summary", "median_subtr", "median_1tr"):
        assert key in out
    for res in ("subtr", "1tr"):
        for key in ("median_r", "median_r2", "median_euclid_median", "mean_r"):
            assert key in out["summary"][res]
    assert set(out["folds"]) == {"dsA", "dsB"}


def test_pure_noise_does_not_decode():
    """The guard against a harness that scores well on anything."""
    recs = [_rec(ds, f"sub-{i}", seed=100 + 10 * j + i)
            for j, ds in enumerate(("dsA", "dsB", "dsC")) for i in range(3)]
    out = tp.lodo(recs, lambda r: tp.bilateral(r, 4))
    assert abs(out["median_subtr"]) < 0.35


def test_calibration_targets_are_recorded():
    """The numbers are pinned in code so a drifting harness is caught, not rationalised."""
    assert tp.CALIBRATION["lr-cca:32"] == pytest.approx(0.7412)
    assert tp.CALIBRATION["lr-cca:32+lags1"] == pytest.approx(0.7712)


def test_train_filter_is_a_no_op_when_it_accepts_everything():
    """The default path must not move: every existing number is `train_filter=None`."""
    recs = [_rec(ds, f"sub-{i}", seed=10 * j + i)
            for j, ds in enumerate(("dsA", "dsB", "dsC")) for i in range(3)]
    base = tp.lodo(recs, lambda r: tp.bilateral(r, 4))
    same = tp.lodo(recs, lambda r: tp.bilateral(r, 4), train_filter=lambda r, held: True)
    assert same["median_subtr"] == pytest.approx(base["median_subtr"])
    assert same["median_1tr"] == pytest.approx(base["median_1tr"])


def test_train_filter_restricts_only_the_training_side():
    """A budget must not shrink the test set, or the metric's denominator moves with it.

    Half the participants are dropped from training; every fold must still report
    the same participant count, because each is scored on its whole dataset.
    """
    recs = [_rec(ds, f"sub-{i}", seed=10 * j + i)
            for j, ds in enumerate(("dsA", "dsB", "dsC")) for i in range(4)]
    keep = {"sub-0", "sub-1"}
    out = tp.lodo(recs, lambda r: tp.bilateral(r, 4),
                  train_filter=lambda r, held: r["subject"] in keep)
    assert {d: f["n"] for d, f in out["folds"].items()} == {"dsA": 4, "dsB": 4, "dsC": 4}
    assert len(out["participants"]) == 12


def test_train_filter_that_empties_a_fold_drops_it_rather_than_raising():
    """`lodo` already skips a fold with no usable training rows; a filter must not change that."""
    recs = [_rec(ds, f"sub-{i}", seed=10 * j + i)
            for j, ds in enumerate(("dsA", "dsB")) for i in range(3)]
    out = tp.lodo(recs, lambda r: tp.bilateral(r, 4), train_filter=lambda r, held: False)
    assert out["folds"] == {}


# --- readout and views: parameters of the one implementation, not second ones ---------

def _dataset_specific_recs():
    """Each dataset maps the same features to gaze through its own matrix."""
    rng = np.random.default_rng(7)
    k, recs = 6, []
    for ds in ("dsA", "dsB", "dsC"):
        w = rng.normal(size=(k, 2))
        for s in range(6):
            z = rng.normal(size=(120, 2, k))
            labels = np.repeat((0.5 * (z[:, 0] + z[:, 1]) @ w)[:, None, :], 10, axis=1)
            recs.append({"dataset": ds, "subject": f"sub-{s}",
                         "z": z.astype(np.float32), "labels": labels.astype(np.float32)})
    return recs, k


def test_readout_none_is_ridge_and_groups_name_participants():
    from sklearn.linear_model import RidgeCV

    recs, k = _dataset_specific_recs()
    seen = []

    def ridge(x, y, groups):
        seen.append(len(np.unique(groups)))
        assert len(groups) == len(x) == len(y)
        return RidgeCV(alphas=ALPHAS).fit(x, y).predict

    base = tp.lodo(recs, lambda r: tp.bilateral(r, k))
    same = tp.lodo(recs, lambda r: tp.bilateral(r, k), readout=ridge)
    assert same["median_subtr"] == pytest.approx(base["median_subtr"])
    assert seen == [12, 12, 12]          # two training datasets of six participants per fold


def test_kept_predictions_are_what_was_scored_and_can_be_a_feature():
    """The kept predictions are the scored ones, in the readout's z-units."""
    recs, k = _dataset_specific_recs()
    fn = lambda r: tp.bilateral(r, k)  # noqa: E731
    base = tp.lodo(recs, fn)
    kept = tp.lodo(recs, fn, keep_predictions=True)
    assert "predictions" not in base
    assert kept["median_subtr"] == base["median_subtr"]
    assert set(kept["predictions"]) == {(r["dataset"], r["subject"]) for r in recs}
    assert all(kept["predictions"][(r["dataset"], r["subject"])].shape == (120, 20) for r in recs)


def test_orbit_views_average_to_the_scored_prediction():
    """The readout is linear in the orbits' average, so the two one-orbit
    decodings average to the scored one exactly -- what makes their agreement
    a confidence of that very prediction."""
    recs, k = _dataset_specific_recs()
    out = tp.lodo(recs, tp.features(k, 1), keep_predictions=True, views=tp.orbit_views(k, 1))
    for key, pred in out["predictions"].items():
        mean = 0.5 * (out["views"]["left"][key] + out["views"]["right"][key])
        assert np.allclose(mean, pred, atol=1e-6)   # the cache is float32
