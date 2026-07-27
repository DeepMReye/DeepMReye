"""Tests for the participant index and the validation that gates publishing."""
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build_index import inspect, run_build

from deepmreye.storage import subject_path, write_subject


def _block(t=120, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(size=(47, 29, 18, t)), -5, 5).astype(np.float32)


def test_index_row_has_the_fields_consumers_filter_on(tmp_path):
    path = subject_path(tmp_path, "ds1", "sub-01")
    write_subject(path, _block(), attrs={"repetition_time": 2.0, "normalized": True})

    row = inspect(path, "ds1", "sub-01", deep=True)
    assert "error" not in row
    assert row["n_trs"] == 120
    assert (row["shape_x"], row["shape_y"], row["shape_z"]) == (47, 29, 18)
    assert row["repetition_time"] == 2.0
    assert row["has_labels"] is False
    assert row["dtype"] == "float32"


def test_truncated_file_is_flagged_not_indexed(tmp_path):
    """Truncated uploads open-fail; they must never reach the artifact."""
    path = subject_path(tmp_path, "ds1", "sub-01")
    write_subject(path, _block())
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 3])

    assert "error" in inspect(path, "ds1", "sub-01")


def test_wrong_spatial_shape_is_flagged(tmp_path):
    path = subject_path(tmp_path, "ds1", "sub-01")
    write_subject(path, np.zeros((40, 29, 18, 50), np.float32))
    row = inspect(path, "ds1", "sub-01")
    assert "error" in row and "spatial shape" in row["error"]


def test_all_zero_block_is_flagged(tmp_path):
    """Registration can fail into an empty block; deep validation catches it."""
    path = subject_path(tmp_path, "ds1", "sub-01")
    write_subject(path, np.zeros((47, 29, 18, 50), np.float32))
    row = inspect(path, "ds1", "sub-01", deep=True)
    assert "error" in row and "all-zero" in row["error"]


def test_values_beyond_clip_limit_are_flagged(tmp_path):
    """A file claiming to be normalized must actually be within +/-5."""
    block = _block(t=50)
    block[0, 0, 0, 0] = 42.0
    path = subject_path(tmp_path, "ds1", "sub-01")
    write_subject(path, block, attrs={"normalized": True})

    row = inspect(path, "ds1", "sub-01", deep=True)
    assert "error" in row and "clip limit" in row["error"]


def test_label_length_mismatch_is_flagged(tmp_path):
    """Written directly, bypassing write_subject's own guard."""
    path = subject_path(tmp_path, "ds1", "sub-01")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("eye_block", data=_block(t=50))
        f.create_dataset("labels", data=np.zeros((40, 10, 2), np.float32))

    row = inspect(path, "ds1", "sub-01")
    assert "error" in row and "labels" in row["error"]


def test_run_build_separates_good_from_bad(tmp_path):
    write_subject(subject_path(tmp_path, "ds1", "sub-ok"), _block(t=60))
    broken = subject_path(tmp_path, "ds1", "sub-broken")
    write_subject(broken, _block(t=60))
    data = broken.read_bytes()
    broken.write_bytes(data[: len(data) // 3])

    good, bad = run_build(tmp_path, deep=True)
    assert [r["subject"] for r in good] == ["sub-ok"]
    assert [r["subject"] for r in bad] == ["sub-broken"]


def test_index_carries_qa_labels(tmp_path):
    """Publishing filters on QA status, so it has to be in the index."""
    write_subject(subject_path(tmp_path, "ds1", "sub-01"), _block(t=60))
    write_subject(subject_path(tmp_path, "ds1", "sub-02"), _block(t=60))

    with h5py.File(tmp_path / "datasets.h5", "w") as f:
        grp = f.create_group("ds1")
        grp.create_group("sub-01").attrs["approved"] = 1
        grp.create_group("sub-02").attrs["approved"] = 0

    good, _ = run_build(tmp_path)
    qa = {r["subject"]: r["qa_approved"] for r in good}
    assert qa == {"sub-01": 1, "sub-02": 0}
