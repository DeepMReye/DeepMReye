"""Tests for the windowed loaders over the per-participant layout."""
import h5py
import numpy as np

from deepmreye.data.jepa_dataset import JEPADataset
from deepmreye.data.probe_dataset import ProbeDataset
from deepmreye.storage import subject_path, write_subject


def _block(t, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(47, 29, 18, t)).astype(np.float32)


def _make_registry(tmp_path, entries):
    """entries: {dataset: {subject: approved_label}}"""
    with h5py.File(tmp_path / "datasets.h5", "w") as f:
        for ds, subs in entries.items():
            grp = f.create_group(ds)
            grp.attrs["approved"] = -1
            for sub, approved in subs.items():
                sg = grp.create_group(sub)
                sg.attrs["approved"] = approved
                sg.attrs["repetition_time"] = 2.0


def test_jepa_only_includes_approved_datasets(tmp_path):
    _make_registry(tmp_path, {
        "ds_good": {"sub-01": 1, "sub-02": 1},
        "ds_bad": {"sub-01": 1, "sub-02": 0},   # one no-eyes drops the dataset
    })
    for ds in ("ds_good", "ds_bad"):
        for sub in ("sub-01", "sub-02"):
            write_subject(subject_path(tmp_path, ds, sub), _block(200))

    ds = JEPADataset(tmp_path, window_size=100)
    assert {s["dataset"] for s in ds.sequences} == {"ds_good"}
    assert len(ds) > 0
    assert tuple(ds[0].shape) == (47, 29, 18, 100)


def test_jepa_skips_runs_shorter_than_window(tmp_path):
    _make_registry(tmp_path, {"ds1": {"sub-short": 1, "sub-long": 1}})
    write_subject(subject_path(tmp_path, "ds1", "sub-short"), _block(50))
    write_subject(subject_path(tmp_path, "ds1", "sub-long"), _block(200))

    ds = JEPADataset(tmp_path, window_size=100)
    assert {s["subject"] for s in ds.sequences} == {"sub-long"}


def test_jepa_window_contents_match_disk(tmp_path):
    _make_registry(tmp_path, {"ds1": {"sub-01": 1}})
    block = _block(200, seed=3)
    write_subject(subject_path(tmp_path, "ds1", "sub-01"), block)

    ds = JEPADataset(tmp_path, window_size=100)
    seq = ds.sequences[1]
    start = seq["start_idx"]
    np.testing.assert_array_equal(ds[1].numpy(), block[..., start:start + 100])


def test_jepa_tolerates_truncated_file(tmp_path):
    """A half-written subject must not abort indexing for everyone else."""
    _make_registry(tmp_path, {"ds1": {"sub-ok": 1, "sub-broken": 1}})
    write_subject(subject_path(tmp_path, "ds1", "sub-ok"), _block(200))
    broken = subject_path(tmp_path, "ds1", "sub-broken")
    write_subject(broken, _block(200))
    data = broken.read_bytes()
    broken.write_bytes(data[: len(data) // 3])

    ds = JEPADataset(tmp_path, window_size=100)
    assert {s["subject"] for s in ds.sequences} == {"sub-ok"}


def test_probe_requires_labels(tmp_path):
    write_subject(subject_path(tmp_path, "ds1", "sub-labeled"), _block(200),
                  labels=np.zeros((200, 10, 2), np.float32))
    write_subject(subject_path(tmp_path, "ds1", "sub-unlabeled"), _block(200))

    ds = ProbeDataset(tmp_path, split="train", split_ratio=1.0, window_size=100)
    assert {s["subject"] for s in ds.samples} == {"sub-labeled"}


def test_probe_splits_are_disjoint_and_nonempty(tmp_path):
    for i in range(6):
        write_subject(subject_path(tmp_path, "ds1", f"sub-{i:02d}"), _block(200),
                      labels=np.zeros((200, 10, 2), np.float32))

    train = ProbeDataset(tmp_path, split="train", window_size=100)
    test = ProbeDataset(tmp_path, split="test", window_size=100)

    train_subs = {s["subject"] for s in train.samples}
    test_subs = {s["subject"] for s in test.samples}
    assert train_subs and test_subs
    assert not (train_subs & test_subs)
    assert len(train_subs | test_subs) == 6


def test_probe_subject_split_covers_every_dataset(tmp_path):
    """Splitting per dataset, not over a pooled shuffle, so a small dataset
    cannot land entirely in one split by chance."""
    for ds in ("ds_a", "ds_b", "ds_c"):
        for i in range(4):
            write_subject(subject_path(tmp_path, ds, f"sub-{i}"), _block(150),
                          labels=np.zeros((150, 10, 2), np.float32))

    train = ProbeDataset(tmp_path, split="train", window_size=100)
    test = ProbeDataset(tmp_path, split="test", window_size=100)

    assert {s["dataset"] for s in train.samples} == {"ds_a", "ds_b", "ds_c"}
    assert {s["dataset"] for s in test.samples} == {"ds_a", "ds_b", "ds_c"}


def test_probe_dataset_split_holds_out_whole_datasets(tmp_path):
    for ds in ("ds_a", "ds_b", "ds_c", "ds_d"):
        for i in range(2):
            write_subject(subject_path(tmp_path, ds, f"sub-{i}"), _block(150),
                          labels=np.zeros((150, 10, 2), np.float32))

    train = ProbeDataset(tmp_path, split="train", split_by="dataset", window_size=100)
    test = ProbeDataset(tmp_path, split="test", split_by="dataset", window_size=100)

    train_ds = {s["dataset"] for s in train.samples}
    test_ds = {s["dataset"] for s in test.samples}
    assert train_ds and test_ds
    assert not (train_ds & test_ds)


def test_probe_returns_aligned_block_and_labels(tmp_path):
    block = _block(150, seed=5)
    labels = np.arange(150 * 10 * 2, dtype=np.float32).reshape(150, 10, 2)
    write_subject(subject_path(tmp_path, "ds1", "sub-01"), block, labels=labels)

    ds = ProbeDataset(tmp_path, split="train", split_ratio=1.0, window_size=100)
    x, y, ds_name = ds[0]
    start = ds.samples[0]["start"]

    assert ds_name == "ds1"
    np.testing.assert_array_equal(x.numpy(), block[..., start:start + 100])
    np.testing.assert_array_equal(y.numpy(), labels[start:start + 100])
