"""Tests for the label round trip between laptop and cluster.

Labels are the expensive artifact of this project, so the merge must be
conservative: filling gaps is fine, silently overwriting work done on the other
machine is not.
"""
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from sync_labels import _merge_registry


def _registry(path, entries):
    """entries: {dataset: {subject: approved}}"""
    with h5py.File(path, "w") as f:
        for ds, subs in entries.items():
            grp = f.create_group(ds)
            for sub, lbl in subs.items():
                grp.create_group(sub).attrs["approved"] = lbl
    return path


def _labels(path):
    out = {}
    with h5py.File(path, "r") as f:
        for ds in f:
            for sub in f[ds]:
                out[f"{ds}/{sub}"] = int(f[ds][sub].attrs["approved"])
    return out


def test_remote_labels_fill_local_gaps(tmp_path):
    remote = _registry(tmp_path / "r.h5", {"ds1": {"sub-01": 1, "sub-02": 0}})
    local = _registry(tmp_path / "l.h5", {"ds1": {"sub-01": -1, "sub-02": -1}})

    assert _merge_registry(remote, local) == 2
    assert _labels(local) == {"ds1/sub-01": 1, "ds1/sub-02": 0}


def test_local_labels_are_never_overwritten(tmp_path):
    """The whole point: a pull must not undo labeling done here."""
    remote = _registry(tmp_path / "r.h5", {"ds1": {"sub-01": 0}})
    local = _registry(tmp_path / "l.h5", {"ds1": {"sub-01": 1}})

    _merge_registry(remote, local)
    assert _labels(local) == {"ds1/sub-01": 1}


def test_unlabeled_remote_does_not_clear_local(tmp_path):
    remote = _registry(tmp_path / "r.h5", {"ds1": {"sub-01": -1}})
    local = _registry(tmp_path / "l.h5", {"ds1": {"sub-01": 1}})

    assert _merge_registry(remote, local) == 0
    assert _labels(local) == {"ds1/sub-01": 1}


def test_subjects_absent_locally_are_ignored(tmp_path):
    """The local corpus may be a subset; that must not crash the merge."""
    remote = _registry(tmp_path / "r.h5", {"ds1": {"sub-01": 1, "sub-99": 1},
                                           "ds_other": {"sub-01": 1}})
    local = _registry(tmp_path / "l.h5", {"ds1": {"sub-01": -1}})

    assert _merge_registry(remote, local) == 1
    assert _labels(local) == {"ds1/sub-01": 1}


def test_dataset_level_skip_propagates(tmp_path):
    """-99 (whole dataset skipped) is a label too and must survive the trip."""
    with h5py.File(tmp_path / "r.h5", "w") as f:
        g = f.create_group("ds1")
        g.attrs["approved"] = -99
        g.create_group("sub-01").attrs["approved"] = -1
    local = _registry(tmp_path / "l.h5", {"ds1": {"sub-01": -1}})

    _merge_registry(tmp_path / "r.h5", local)
    with h5py.File(local, "r") as f:
        assert f["ds1"].attrs["approved"] == -99


def test_merge_is_idempotent(tmp_path):
    remote = _registry(tmp_path / "r.h5", {"ds1": {"sub-01": 1, "sub-02": 3}})
    local = _registry(tmp_path / "l.h5", {"ds1": {"sub-01": -1, "sub-02": -1}})

    assert _merge_registry(remote, local) == 2
    assert _merge_registry(remote, local) == 0  # nothing left to apply
    assert _labels(local) == {"ds1/sub-01": 1, "ds1/sub-02": 3}
