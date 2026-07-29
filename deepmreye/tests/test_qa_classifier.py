"""Tests for the QA triage features and model.

The model only ever ranks and flags -- it never approves a dataset -- so these
check that the features separate the classes and that the plumbing holds, not
that any particular accuracy is reached.
"""
import numpy as np
import pytest

from deepmreye import qa_classifier as qac
from deepmreye.storage import subject_path, write_subject


def _eyes_block(t=120, seed=0):
    """A block with signal concentrated centrally in both halves, as real eyes are."""
    rng = np.random.default_rng(seed)
    block = np.zeros((47, 29, 18, t), dtype=np.float32)
    for cx in (12, 34):  # left and right eye
        block[cx - 5:cx + 5, 10:20, 5:13, :] = rng.normal(size=(10, 10, 8, t))
    return block


def _no_eyes_block(t=120, seed=0):
    """Sparse signal smeared toward the edges, as a failed registration leaves."""
    rng = np.random.default_rng(seed)
    block = np.zeros((47, 29, 18, t), dtype=np.float32)
    block[:4, :4, :3, :] = rng.normal(size=(4, 4, 3, t))
    block[-4:, -4:, -3:, :] = rng.normal(size=(4, 4, 3, t))
    return block


def test_feature_vector_shape_and_finiteness():
    feats = qac.extract_features(_eyes_block())
    assert feats.shape == (len(qac.FEATURE_NAMES),)
    assert np.all(np.isfinite(feats))


def test_empty_block_does_not_crash():
    """All-zero blocks occur when registration fails outright."""
    feats = qac.extract_features(np.zeros((47, 29, 18, 50), np.float32))
    assert feats.shape == (len(qac.FEATURE_NAMES),)
    assert np.all(np.isfinite(feats))


def test_features_separate_eyes_from_no_eyes():
    eyes = qac.extract_features(_eyes_block())
    none = qac.extract_features(_no_eyes_block())

    i_occ = qac.FEATURE_NAMES.index("nonzero_frac")
    i_ctr = qac.FEATURE_NAMES.index("center_edge_ratio")
    assert eyes[i_occ] > none[i_occ]
    assert eyes[i_ctr] > none[i_ctr]


def test_features_ignore_voxels_outside_the_mask():
    """Padding the bounding box must not change the statistics: otherwise the
    model learns crop geometry instead of eye coverage."""
    block = _eyes_block()
    feats = qac.extract_features(block)

    padded = np.zeros((47, 29, 18, block.shape[-1]), dtype=np.float32)
    padded[...] = block
    np.testing.assert_allclose(feats, qac.extract_features(padded), rtol=1e-5)


def test_build_training_set_reads_qa_labels(tmp_path):
    import h5py

    entries = {"ds1": {"sub-01": 1, "sub-02": 0}, "ds2": {"sub-01": 1, "sub-02": -1}}
    with h5py.File(tmp_path / "datasets.h5", "w") as f:
        for ds, subs in entries.items():
            grp = f.create_group(ds)
            for sub, approved in subs.items():
                grp.create_group(sub).attrs["approved"] = approved

    for ds, subs in entries.items():
        for sub, approved in subs.items():
            block = _eyes_block(t=40) if approved == 1 else _no_eyes_block(t=40)
            write_subject(subject_path(tmp_path, ds, sub), block)

    X, y, keys = qac.build_training_set(tmp_path)

    # The unlabeled subject (-1) is excluded; the three labeled ones are kept.
    assert len(X) == 3
    assert sorted(y.tolist()) == [0, 1, 1]
    assert ("ds2", "sub-02") not in keys


def test_train_reports_grouped_cv():
    """CV must group by dataset -- subjects of one dataset share a failure mode."""
    pytest.importorskip("sklearn")

    X, y, groups = [], [], []
    for ds in range(4):
        for i in range(3):
            X.append(qac.extract_features(_eyes_block(t=30, seed=ds * 10 + i)))
            y.append(1)
            groups.append(f"ds{ds}")
            X.append(qac.extract_features(_no_eyes_block(t=30, seed=ds * 10 + i)))
            y.append(0)
            groups.append(f"ds{ds}")

    model, scores = qac.train(np.vstack(X), np.asarray(y), groups=groups)
    assert scores is not None and len(scores) >= 3
    assert model.predict_proba(np.vstack(X)[:1]).shape == (1, 2)


def test_eyes_cut_and_faint_count_as_approved():
    """Labels 3 (cut off) and 4 (faint eyes) must keep dataset in training:
    eyeballs still carry gaze signal, and excluding them would drop whole
    datasets under the all-or-nothing rule."""
    import h5py
    import tempfile
    from pathlib import Path
    from deepmreye.pipeline import is_dataset_approved

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "reg.h5"
        with h5py.File(path, "w") as f:
            g = f.create_group("ds_approved")
            g.create_group("sub-01").attrs["approved"] = 1   # clean eyes
            g.create_group("sub-02").attrs["approved"] = 3   # eyes, cut off
            g.create_group("sub-03").attrs["approved"] = 4   # eyes, faint
            
            g2 = f.create_group("ds_mixed")
            g2.create_group("sub-01").attrs["approved"] = 4  # eyes, faint
            g2.create_group("sub-02").attrs["approved"] = 0  # no eyes -> drops it

        with h5py.File(path, "r") as f:
            assert is_dataset_approved(f["ds_approved"]) is True
            assert is_dataset_approved(f["ds_mixed"]) is False


def test_training_set_includes_faint_and_cut_labels(tmp_path):
    """The model predicts the exact label, so class 3 and 4 must reach training."""
    import h5py
    from deepmreye.storage import subject_path, write_subject

    with h5py.File(tmp_path / "datasets.h5", "w") as f:
        g = f.create_group("ds1")
        for sub, lbl in [("sub-01", 1), ("sub-02", 3), ("sub-03", 4), ("sub-04", 0)]:
            g.create_group(sub).attrs["approved"] = lbl

    for sub, lbl in [("sub-01", 1), ("sub-02", 3), ("sub-03", 4), ("sub-04", 0)]:
        block = _eyes_block(t=40) if lbl in (1, 3, 4) else _no_eyes_block(t=40)
        write_subject(subject_path(tmp_path, "ds1", sub), block)

    X, y, keys = qac.build_training_set(tmp_path)
    assert sorted(y.tolist()) == [0, 1, 3, 4]
