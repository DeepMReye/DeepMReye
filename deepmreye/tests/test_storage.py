"""Tests for the per-participant on-disk layout and the registry sidecars."""
import h5py
import numpy as np
import pytest

from deepmreye import registry
from deepmreye.storage import (
    FORMAT_VERSION,
    is_intact,
    iter_subjects,
    read_subject,
    subject_path,
    write_subject,
)


def _block(t=120, x=47, y=29, z=18, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(x, y, z, t)).astype(np.float32)


def test_write_read_roundtrip(tmp_path):
    block = _block()
    path = subject_path(tmp_path, "ds000001", "sub-01")
    write_subject(path, block, attrs={"repetition_time": 2.0})

    out, labels, attrs = read_subject(path)
    np.testing.assert_array_equal(out, block)
    assert labels is None
    assert attrs["has_labels"] is np.False_ or attrs["has_labels"] == False  # noqa: E712
    assert attrs["repetition_time"] == 2.0
    assert attrs["n_trs"] == 120
    assert attrs["format_version"] == FORMAT_VERSION


def test_window_read_matches_full_read(tmp_path):
    """Windowed reads are what the loaders do; they must not shift the data."""
    block = _block()
    path = subject_path(tmp_path, "ds000001", "sub-01")
    write_subject(path, block)

    out, _, _ = read_subject(path, start=30, end=130)
    np.testing.assert_array_equal(out, block[..., 30:130])


def test_labels_roundtrip_preserves_nans(tmp_path):
    """NaNs mark TRs with no valid gaze; dropping them would misalign time."""
    block = _block(t=50)
    labels = np.random.default_rng(1).normal(size=(50, 10, 2)).astype(np.float32)
    labels[3] = np.nan

    path = subject_path(tmp_path, "ds6", "sub-a")
    write_subject(path, block, labels=labels)

    _, out_labels, attrs = read_subject(path)
    np.testing.assert_array_equal(out_labels, labels)  # equal_nan by default here
    assert np.isnan(out_labels[3]).all()
    assert attrs["has_labels"]


def test_label_length_mismatch_rejected(tmp_path):
    block = _block(t=50)
    labels = np.zeros((49, 10, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="mismatch"):
        write_subject(subject_path(tmp_path, "ds", "sub"), block, labels=labels)


def test_non_4d_rejected(tmp_path):
    with pytest.raises(ValueError, match="4D"):
        write_subject(subject_path(tmp_path, "ds", "sub"), np.zeros((10, 10, 10), np.float32))


def test_write_is_atomic(tmp_path):
    """A failed write must not leave a half-written file behind."""
    path = subject_path(tmp_path, "ds", "sub")
    write_subject(path, _block(t=20))
    assert is_intact(path)

    with pytest.raises(ValueError):
        write_subject(path, _block(t=20), labels=np.zeros((5, 10, 2), np.float32))

    # The original survives and no .tmp is orphaned in its place.
    assert is_intact(path)
    assert read_subject(path)[0].shape[-1] == 20


def test_is_intact_detects_truncation(tmp_path):
    """Truncated uploads only fail on open -- the validation pass relies on this."""
    path = subject_path(tmp_path, "ds", "sub")
    write_subject(path, _block(t=60))
    assert is_intact(path)

    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 3])
    assert not is_intact(path)


def test_iter_subjects_finds_all(tmp_path):
    for ds, sub in [("ds1", "sub-01"), ("ds1", "sub-02"), ("ds2", "sub-01")]:
        write_subject(subject_path(tmp_path, ds, sub), _block(t=20))

    found = {(ds, sub) for ds, sub, _ in iter_subjects(tmp_path)}
    assert found == {("ds1", "sub-01"), ("ds1", "sub-02"), ("ds2", "sub-01")}


def test_registry_merge_applies_records(tmp_path):
    registry.record(tmp_path, "ds000001", "sub-01",
                    {"repetition_time": 2.0, "n_trs": 100}, worker_id=1)
    registry.record(tmp_path, "ds000001", "sub-02",
                    {"repetition_time": 2.5, "n_trs": 200}, worker_id=2)

    assert registry.merge_pending(tmp_path) == 2

    with h5py.File(tmp_path / "datasets.h5", "r") as f:
        assert f["ds000001/sub-01"].attrs["repetition_time"] == 2.0
        assert f["ds000001/sub-02"].attrs["n_trs"] == 200
    # Sidecars are cleared so a second merge is a no-op.
    assert registry.merge_pending(tmp_path) == 0


def test_merge_never_clobbers_qa_labels(tmp_path):
    """QA labels are the expensive artifact; a merge must not overwrite them."""
    registry.record(tmp_path, "ds1", "sub-01", {"n_trs": 100}, worker_id=1)
    registry.merge_pending(tmp_path)

    with h5py.File(tmp_path / "datasets.h5", "a") as f:
        f["ds1/sub-01"].attrs["approved"] = 1

    registry.record(tmp_path, "ds1", "sub-01", {"n_trs": 150}, worker_id=1)
    registry.merge_pending(tmp_path)

    with h5py.File(tmp_path / "datasets.h5", "r") as f:
        assert f["ds1/sub-01"].attrs["approved"] == 1  # preserved
        assert f["ds1/sub-01"].attrs["n_trs"] == 150   # updated


def test_merge_tolerates_truncated_sidecar_line(tmp_path):
    """A killed worker leaves a partial final line; the rest must still merge."""
    registry.record(tmp_path, "ds1", "sub-01", {"n_trs": 100}, worker_id=7)
    sidecar = registry.pending_dir(tmp_path) / "worker_7.jsonl"
    with open(sidecar, "a") as f:
        f.write('{"dataset": "ds1", "subject": "sub-02", "n_tr')  # truncated

    assert registry.merge_pending(tmp_path) == 1
