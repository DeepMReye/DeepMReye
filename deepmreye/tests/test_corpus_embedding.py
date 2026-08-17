"""The corpus-embedding measurement.

``proxy_a_distance`` is the number the domain-mismatch conclusion rests on, so
it is tested against the two properties that make it interpretable (0 for
identical populations, 2 for separable ones) and against the two ways it could
silently lie here: a 246-vs-1204 class imbalance, and a classifier that wins by
recognising one acquisition rather than one domain.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from visualize_corpus_embedding import (  # noqa: E402
    _slab_indices,
    describe,
    nearest_unlabeled,
    neighbour_mix,
    probe_deltas,
    proxy_a_distance,
)


def _two_domains(n=120, d=8, shift=0.0, seed=0, n_groups=3):
    """Two populations, each spread over ``n_groups`` datasets."""
    rng = np.random.default_rng(seed)
    x = np.vstack([rng.normal(size=(n, d)), rng.normal(size=(n, d)) + shift])
    y = np.array([0] * n + [1] * n)
    groups = np.array([f"{side}{i % n_groups}"
                       for side in "ab" for i in range(n)])
    return x, y, groups


# ----------------------------------------------------------- proxy A-distance

def test_identical_populations_give_distance_near_zero():
    x, y, g = _two_domains(shift=0.0, seed=1)
    d, grouped = proxy_a_distance(x, y, g)
    assert grouped
    assert abs(d) < 0.35


def test_well_separated_populations_approach_two():
    x, y, g = _two_domains(shift=6.0, seed=2)
    d, _ = proxy_a_distance(x, y, g)
    assert d > 1.8


def test_distance_grows_with_the_shift():
    xs = [proxy_a_distance(*_two_domains(shift=s, seed=3)[:2],
                           _two_domains(shift=s, seed=3)[2])[0]
          for s in (0.0, 1.0, 3.0)]
    assert xs[0] < xs[1] < xs[2]


def test_class_imbalance_does_not_manufacture_a_distance():
    """The real call is 246 labeled against 1204 unlabeled. Plain accuracy would
    read 0.83 for a classifier that always says "unlabeled", i.e. d_A = 1.3 out
    of nothing; balanced error is what stops that."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=(1000, 6))
    y = np.array([1] * 100 + [0] * 900)
    g = np.array([f"g{i % 12}" for i in range(1000)])
    d, _ = proxy_a_distance(x, y, g)
    assert abs(d) < 0.35


def test_a_difference_that_is_only_dataset_identity_is_not_counted():
    """Each dataset has its own offset, but the two domains draw those offsets
    from the same distribution. Grouped folds must therefore see no domain
    difference -- otherwise the metric is reporting acquisition, not domain."""
    rng = np.random.default_rng(5)
    rows, y, groups = [], [], []
    for side in (0, 1):
        for k in range(6):
            offset = rng.normal(size=8) * 4.0
            rows.append(rng.normal(size=(40, 8)) + offset)
            y += [side] * 40
            groups += [f"{side}-{k}"] * 40
    d, grouped = proxy_a_distance(np.vstack(rows), np.array(y),
                                  np.array(groups))
    assert grouped
    assert d < 0.5


def test_a_single_dataset_per_side_is_reported_as_ungrouped():
    """One acquisition per side cannot be held out, so the number is an upper
    bound and the caller has to know that."""
    x, y, _ = _two_domains(shift=1.0, seed=6)
    g = np.array(["only-a"] * (len(y) // 2) + ["only-b"] * (len(y) // 2))
    _, grouped = proxy_a_distance(x, y, g)
    assert not grouped


def test_distance_is_symmetric_in_the_label_coding():
    x, y, g = _two_domains(shift=2.0, seed=7)
    a, _ = proxy_a_distance(x, y, g)
    b, _ = proxy_a_distance(x, 1 - y, g)
    assert a == pytest.approx(b, abs=0.2)


def test_distance_is_invariant_to_feature_scaling():
    """Features here are log-variances and Fisher-z correlations on wildly
    different scales; a metric that moved with a rescaling would be reporting
    units."""
    x, y, g = _two_domains(shift=2.0, seed=8)
    scaled = x * np.array([1e-3, 1e3, 1, 1, 10, 10, 0.1, 0.1])
    assert proxy_a_distance(x, y, g)[0] == pytest.approx(
        proxy_a_distance(scaled, y, g)[0], abs=0.25)


# ----------------------------------------------------------------- neighbours

def test_a_dataset_in_its_own_pocket_has_a_low_unlabeled_neighbour_share():
    rng = np.random.default_rng(9)
    far = rng.normal(size=(60, 5)) + 60.0
    corpus = rng.normal(size=(300, 5))
    x = np.vstack([far, corpus])
    is_lab = np.array([True] * 60 + [False] * 300)
    dsets = np.array(["dsL99_far"] * 60 + ["ds000001"] * 300)
    assert neighbour_mix(x, is_lab, dsets)["dsL99_far"] < 0.1


def test_a_dataset_drawn_from_the_corpus_sits_near_chance():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(360, 5))
    is_lab = np.array([True] * 60 + [False] * 300)
    dsets = np.array(["dsL99_same"] * 60 + ["ds000001"] * 300)
    chance = float((~is_lab).mean())
    assert abs(neighbour_mix(x, is_lab, dsets)["dsL99_same"] - chance) < 0.2


def test_nearest_unlabeled_never_returns_a_labeled_participant():
    """The shortlist is meant to be fitted on as unlabeled data; a labeled
    participant leaking in would quietly make the basis transductive."""
    rng = np.random.default_rng(11)
    x = rng.normal(size=(200, 6))
    is_lab = np.zeros(200, bool)
    is_lab[:40] = True
    keep, sim = nearest_unlabeled(x, is_lab, 50)
    assert len(keep) == 50
    assert not is_lab[keep].any()
    assert len(sim) == 200


def test_nearest_unlabeled_prefers_participants_closer_to_the_labeled_cloud():
    rng = np.random.default_rng(12)
    labeled = rng.normal(size=(40, 6)) + 5.0
    near = rng.normal(size=(30, 6)) * 0.1 + 5.0
    far = rng.normal(size=(30, 6)) - 5.0
    x = np.vstack([labeled, near, far])
    is_lab = np.array([True] * 40 + [False] * 60)
    keep, _ = nearest_unlabeled(x, is_lab, 30)
    assert (keep < 70).mean() > 0.9      # the "near" block is indices 40..69


# ---------------------------------------------------------------- descriptors

def _write_participant(path, n_trs=60, shape=(6, 5, 4), seed=0):
    rng = np.random.default_rng(seed)
    block = rng.normal(size=shape + (n_trs,)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("eye_block", data=block)
    return block


def test_describe_returns_the_expected_widths(tmp_path):
    p = tmp_path / "sub-01.h5"
    _write_participant(p)
    mask = np.ones((6, 5, 4), bool)
    d = int(mask.sum())
    k = 5
    rng = np.random.default_rng(1)
    comps = rng.normal(size=(d, k))
    got = describe(p, mask, comps, np.zeros(d), k, per_slab=8, n_slabs=3)
    assert got is not None
    cov, sd = got
    assert cov.shape == (k + k * (k - 1) // 2,)
    assert sd.shape == (d,)
    assert np.isfinite(cov).all() and np.isfinite(sd).all()


def test_describe_reads_no_gaze_labels(tmp_path):
    """Signature-level guarantee that the embedding is unsupervised: a
    participant with labels must give the identical descriptor to one without."""
    mask = np.ones((6, 5, 4), bool)
    d = int(mask.sum())
    comps = np.random.default_rng(2).normal(size=(d, 4))

    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    block = _write_participant(a, seed=3)
    with h5py.File(b, "w") as f:
        f.create_dataset("eye_block", data=block)
        f.create_dataset("labels", data=np.ones((block.shape[-1], 10, 2)))

    ga = describe(a, mask, comps, np.zeros(d), 4)
    gb = describe(b, mask, comps, np.zeros(d), 4)
    assert np.allclose(ga[0], gb[0]) and np.allclose(ga[1], gb[1])


def test_a_partly_covered_participant_is_rejected(tmp_path):
    """Otherwise the SD descriptor would describe the crop, not the brain."""
    p = tmp_path / "sub-01.h5"
    block = _write_participant(p, seed=4)
    block[:, :, :2] = 0.0
    with h5py.File(p, "w") as f:
        f.create_dataset("eye_block", data=block)
    mask = np.ones(block.shape[:3], bool)
    d = int(mask.sum())
    assert describe(p, mask, np.zeros((d, 3)), np.zeros(d), 3) is None


def test_a_run_too_short_to_summarise_is_rejected(tmp_path):
    p = tmp_path / "sub-01.h5"
    _write_participant(p, n_trs=10)
    mask = np.ones((6, 5, 4), bool)
    d = int(mask.sum())
    assert describe(p, mask, np.zeros((d, 3)), np.zeros(d), 3, min_trs=32) is None


@pytest.mark.parametrize("n_trs,per,slabs", [(100, 24, 3), (30, 24, 3),
                                             (24, 24, 3), (500, 24, 4)])
def test_slabs_stay_inside_the_run_and_are_contiguous(n_trs, per, slabs):
    out = _slab_indices(n_trs, per, slabs)
    assert out
    for a, b in out:
        assert 0 <= a < b <= n_trs
        assert b - a == min(per, n_trs)


# -------------------------------------------------------------- probe deltas

def test_probe_deltas_are_corpus_minus_fold(tmp_path):
    import json

    def arm(rx, ry):
        return {"by_subject": {"pearson_r_x": rx, "pearson_r_y": ry}}

    path = tmp_path / "probe.json"
    path.write_text(json.dumps({
        "dsL01_x": {"fold-pca:64/ridge-cv": arm(0.8, 0.6),
                    "corpus-pca:64/ridge-cv": arm(0.7, 0.5)}}))
    out = probe_deltas(path, ["dsL01_x"])
    assert out["dsL01_x"] == pytest.approx(-0.1)


def test_probe_deltas_degrade_to_none_rather_than_raising(tmp_path):
    """The panel is optional; a missing or stale results file must not take the
    whole figure down with it."""
    assert probe_deltas(tmp_path / "nope.json", ["dsL01_x"]) is None
    bad = tmp_path / "bad.json"
    bad.write_text('{"dsL01_x": {"some-other-arm": {}}}')
    assert probe_deltas(bad, ["dsL01_x"]) is None
