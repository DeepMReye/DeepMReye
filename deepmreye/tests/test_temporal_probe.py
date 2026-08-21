"""Tests for the sub-TR protocol.

The point of `temporal_probe` is that there is exactly one implementation of the sub-TR
number, so these tests check the properties that would let a second one drift back in: the
lag semantics, the cache guard, and that the scoring recovers a known signal.
"""
import json

import numpy as np
import pytest

from deepmreye import temporal_probe as tp


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
        assert np.array_equal(tp.make_lags(z, lags), _lags_reference(z, lags))


def test_make_lags_edge_pads_rather_than_zero_pads():
    """The distinction an identity-initialised Conv1d would get wrong."""
    z = np.arange(4, dtype=float).reshape(4, 1)
    out = tp.make_lags(z, 1)
    assert out.shape == (4, 3)
    # lag -1 at row 0 clamps to z[0], not to 0.
    assert out[0, 0] == z[0, 0]
    # lag +1 at the last row clamps to z[-1], not to 0.
    assert out[-1, 2] == z[-1, 0]


def test_make_lags_zero_is_identity():
    z = np.random.default_rng(1).normal(size=(9, 3))
    assert np.array_equal(tp.make_lags(z, 0), z)


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
    tp.save_subtr_cache(path, recs, "basis_n2000.npz", 4, False)

    back = tp.load_subtr_cache(path, "basis_n2000.npz", 4, False,
                               fingerprint=tp.corpus_fingerprint(recs))
    assert [r["subject"] for r in back] == ["sub-01", "sub-02"]
    assert np.array_equal(back[0]["z"], recs[0]["z"])
    assert back[0]["labels"].shape == (80, 10, 2)

    # A different rank, basis or motion setting must not load.
    with pytest.raises(SystemExit):
        tp.load_subtr_cache(path, "basis_n2000.npz", 8, False)
    with pytest.raises(SystemExit):
        tp.load_subtr_cache(path, "basis_n1039.npz", 4, False)

    # And neither must a different corpus, which is the check the existing labeled cache
    # lacks -- 285 participants of a retired corpus load there without complaint.
    with pytest.raises(SystemExit) as e:
        tp.load_subtr_cache(path, "basis_n2000.npz", 4, False,
                            fingerprint=tp.corpus_fingerprint(recs[:1]))
    assert "different corpus" in str(e.value)


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

    out = tp.lodo_subtr(recs, lambda r: tp.cca_avg(r, k))
    assert out["median_subtr"] > 0.99
    assert out["median_1tr"] > 0.99
    assert set(out["subtr"]) == {"dsA", "dsB", "dsC"}


def test_lodo_subtr_reports_both_resolutions_and_all_folds():
    recs = [_rec(ds, f"sub-{i}", seed=10 * j + i)
            for j, ds in enumerate(("dsA", "dsB")) for i in range(3)]
    out = tp.lodo_subtr(recs, lambda r: tp.cca_avg(r, 4))
    for key in ("subtr", "1tr", "median_subtr", "median_1tr"):
        assert key in out
    assert set(out["subtr"]) == {"dsA", "dsB"}


def test_pure_noise_does_not_decode():
    """The guard against a harness that scores well on anything."""
    recs = [_rec(ds, f"sub-{i}", seed=100 + 10 * j + i)
            for j, ds in enumerate(("dsA", "dsB", "dsC")) for i in range(3)]
    out = tp.lodo_subtr(recs, lambda r: tp.cca_avg(r, 4))
    assert abs(out["median_subtr"]) < 0.35


def test_calibration_targets_are_recorded():
    """The numbers are pinned in code so a drifting harness is caught, not rationalised."""
    assert tp.CALIBRATION["lr-cca:32"] == pytest.approx(0.742)
    assert tp.CALIBRATION["lr-cca:32+lags2"] == pytest.approx(0.759)
