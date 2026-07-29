"""Tests for the windowed loaders over the per-participant layout."""
import h5py
import numpy as np

from deepmreye.data.jepa_dataset import JEPADataset
from deepmreye.data.probe_dataset import ProbeDataset
from deepmreye.storage import subject_path, write_subject


def _block(t, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(47, 29, 18, t)).astype(np.float32)


# ProbeDataset requires a usable TR per subject -- the model conditions on it,
# so a subject without one cannot be scored. Fixtures write a plausible value.
LABEL_ATTRS = {"repetition_time": 2.0}


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
    block, tr = ds[0]
    assert tuple(block.shape) == (47, 29, 18, 100)
    assert float(tr) == 2.0


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
    np.testing.assert_array_equal(ds[1][0].numpy(), block[..., start:start + 100])


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
                  labels=np.zeros((200, 10, 2), np.float32), attrs=LABEL_ATTRS)
    write_subject(subject_path(tmp_path, "ds1", "sub-unlabeled"), _block(200))

    ds = ProbeDataset(tmp_path, split="train", split_ratio=1.0, window_size=100)
    assert {s["subject"] for s in ds.samples} == {"sub-labeled"}


def test_probe_splits_are_disjoint_and_nonempty(tmp_path):
    for i in range(6):
        write_subject(subject_path(tmp_path, "ds1", f"sub-{i:02d}"), _block(200),
                      labels=np.zeros((200, 10, 2), np.float32), attrs=LABEL_ATTRS)

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
                          labels=np.zeros((150, 10, 2), np.float32), attrs=LABEL_ATTRS)

    train = ProbeDataset(tmp_path, split="train", window_size=100)
    test = ProbeDataset(tmp_path, split="test", window_size=100)

    assert {s["dataset"] for s in train.samples} == {"ds_a", "ds_b", "ds_c"}
    assert {s["dataset"] for s in test.samples} == {"ds_a", "ds_b", "ds_c"}


def test_probe_dataset_split_holds_out_whole_datasets(tmp_path):
    for ds in ("ds_a", "ds_b", "ds_c", "ds_d"):
        for i in range(2):
            write_subject(subject_path(tmp_path, ds, f"sub-{i}"), _block(150),
                          labels=np.zeros((150, 10, 2), np.float32), attrs=LABEL_ATTRS)

    train = ProbeDataset(tmp_path, split="train", split_by="dataset", window_size=100)
    test = ProbeDataset(tmp_path, split="test", split_by="dataset", window_size=100)

    train_ds = {s["dataset"] for s in train.samples}
    test_ds = {s["dataset"] for s in test.samples}
    assert train_ds and test_ds
    assert not (train_ds & test_ds)


def test_probe_returns_aligned_block_and_labels(tmp_path):
    block = _block(150, seed=5)
    labels = np.arange(150 * 10 * 2, dtype=np.float32).reshape(150, 10, 2)
    write_subject(subject_path(tmp_path, "ds1", "sub-01"), block, labels=labels,
                  attrs=LABEL_ATTRS)

    ds = ProbeDataset(tmp_path, split="train", split_ratio=1.0, window_size=100)
    x, y, ds_name, sub_name, tr = ds[0]
    start = ds.samples[0]["start"]

    assert ds_name == "ds1"
    assert sub_name == "sub-01"
    assert float(tr) == 2.0
    np.testing.assert_array_equal(x.numpy(), block[..., start:start + 100])
    np.testing.assert_array_equal(y.numpy(), labels[start:start + 100])


def test_jepa_excludes_the_gaze_labeled_datasets(tmp_path):
    """The probe sets are not pretraining data.

    They are also wildly over-represented -- pursuit runs are 2,000-4,200 TRs
    against a ~270 TR median elsewhere -- so leaving them in would let six
    datasets supply 45% of all windows.
    """
    _make_registry(tmp_path, {"ds000001": {"sub-01": 1}, "dsL01_guided_fixations": {"sub-a": 1}})
    write_subject(subject_path(tmp_path, "ds000001", "sub-01"), _block(200))
    write_subject(subject_path(tmp_path, "dsL01_guided_fixations", "sub-a"), _block(200))

    ds = JEPADataset(tmp_path, window_size=100)
    assert {s["dataset"] for s in ds.sequences} == {"ds000001"}

    both = JEPADataset(tmp_path, window_size=100, exclude_labeled=False)
    assert {s["dataset"] for s in both.sequences} == {"ds000001", "dsL01_guided_fixations"}


def test_jepa_recognises_a_labeled_dataset_by_attribute_not_only_name(tmp_path):
    with h5py.File(tmp_path / "datasets.h5", "w") as f:
        for name, labeled in (("ds000001", False), ("gaze_set", True)):
            grp = f.create_group(name)
            grp.attrs["approved"] = -1
            if labeled:
                grp.attrs["labeled"] = True
            sg = grp.create_group("sub-01")
            sg.attrs["approved"] = 1
            sg.attrs["repetition_time"] = 2.0
    for name in ("ds000001", "gaze_set"):
        write_subject(subject_path(tmp_path, name, "sub-01"), _block(200))

    ds = JEPADataset(tmp_path, window_size=100)
    assert {s["dataset"] for s in ds.sequences} == {"ds000001"}


def test_jepa_skips_subjects_whose_tr_cannot_be_trusted(tmp_path):
    """TR conditions the temporal encoding, so a nonsense one is not defaulted.

    0.044 s and 10 s both appear in real OpenNeuro headers.
    """
    with h5py.File(tmp_path / "datasets.h5", "w") as f:
        grp = f.create_group("ds1")
        grp.attrs["approved"] = -1
        for sub, tr in (("sub-ok", 2.0), ("sub-fast", 0.044), ("sub-slow", 10.0)):
            sg = grp.create_group(sub)
            sg.attrs["approved"] = 1
            sg.attrs["repetition_time"] = tr
        sg = grp.create_group("sub-none")   # no repetition_time at all
        sg.attrs["approved"] = 1
    for sub in ("sub-ok", "sub-fast", "sub-slow", "sub-none"):
        write_subject(subject_path(tmp_path, "ds1", sub), _block(200))

    ds = JEPADataset(tmp_path, window_size=100)
    assert {s["subject"] for s in ds.sequences} == {"sub-ok"}
    assert ds.skipped["bad_tr"] == 3


def test_jepa_uses_the_registry_n_trs_without_opening_every_file(tmp_path):
    # At full extraction this index covers tens of thousands of subjects; one
    # HDF5 open apiece is minutes of startup before training begins.
    _make_registry(tmp_path, {"ds1": {"sub-01": 1}})
    write_subject(subject_path(tmp_path, "ds1", "sub-01"), _block(200))
    with h5py.File(tmp_path / "datasets.h5", "a") as f:
        f["ds1"]["sub-01"].attrs["n_trs"] = 200

    ds = JEPADataset(tmp_path, window_size=100)
    assert len(ds.sequences) == 3   # starts 0, 50, 100 at stride 50


def test_probe_leave_one_dataset_out_holds_out_exactly_that_dataset(tmp_path):
    for ds in ("dsL01_guided_fixations", "dsL02_pursuit", "dsL05_free_viewing"):
        for i in range(2):
            write_subject(subject_path(tmp_path, ds, f"sub-{i}"), _block(150),
                          labels=np.zeros((150, 10, 2), np.float32), attrs=LABEL_ATTRS)

    holdout = {"dsL02_pursuit"}
    train = ProbeDataset(tmp_path, split="train", holdout=holdout, window_size=100)
    test = ProbeDataset(tmp_path, split="test", holdout=holdout, window_size=100)

    assert {s["dataset"] for s in test.samples} == holdout
    assert {s["dataset"] for s in train.samples} == {"dsL01_guided_fixations", "dsL05_free_viewing"}


def test_probe_within_subject_split_shares_no_timepoint(tmp_path):
    """The within-subject split cuts the timeline, not the window index.

    Windows overlap by half a window, so splitting the *list* of windows would
    put the same TRs on both sides and report a near-perfect score that is
    entirely leakage.
    """
    n_trs = 400
    for i in range(3):
        write_subject(subject_path(tmp_path, "ds1", f"sub-{i}"), _block(n_trs),
                      labels=np.zeros((n_trs, 10, 2), np.float32), attrs=LABEL_ATTRS)

    train = ProbeDataset(tmp_path, split="train", split_by="time", window_size=100)
    test = ProbeDataset(tmp_path, split="test", split_by="time", window_size=100)
    assert len(train) and len(test)

    # Same participants either side -- that is the point of this protocol.
    assert ({s["subject"] for s in train.samples}
            == {s["subject"] for s in test.samples} == {"sub-0", "sub-1", "sub-2"})

    for sub in ("sub-0", "sub-1", "sub-2"):
        def trs(dataset):
            covered = set()
            for s in dataset.samples:
                if s["subject"] == sub:
                    covered |= set(range(s["start"], s["start"] + 100))
            return covered
        assert not (trs(train) & trs(test)), f"{sub} leaks timepoints across the split"


def test_probe_within_subject_split_keeps_short_runs_on_both_sides(tmp_path):
    """A dataset must not vanish from the test split because of the stride grid.

    Window starts are multiples of ``stride``, so ``start >= cut`` can have no
    solution even when the cut is inside the run. dsL01 is 270 TRs: at
    split_ratio 0.8 the cut wants 216, the last legal start is 170, and the last
    start on the grid is 150. Any cut above 150 silently drops 170 of the 270
    labeled subjects from the evaluation.
    """
    n_trs = 270          # the real dsL01 length
    write_subject(subject_path(tmp_path, "short_ds", "sub-0"), _block(n_trs),
                  labels=np.zeros((n_trs, 10, 2), np.float32), attrs=LABEL_ATTRS)
    write_subject(subject_path(tmp_path, "long_ds", "sub-0"), _block(2000),
                  labels=np.zeros((2000, 10, 2), np.float32), attrs=LABEL_ATTRS)

    train = ProbeDataset(tmp_path, split="train", split_by="time", window_size=100)
    test = ProbeDataset(tmp_path, split="test", split_by="time", window_size=100)

    assert {s["dataset"] for s in train.samples} == {"short_ds", "long_ds"}
    assert {s["dataset"] for s in test.samples} == {"short_ds", "long_ds"}


def test_probe_within_subject_gap_widens_the_separation(tmp_path):
    write_subject(subject_path(tmp_path, "ds1", "sub-0"), _block(600),
                  labels=np.zeros((600, 10, 2), np.float32), attrs=LABEL_ATTRS)

    tight = ProbeDataset(tmp_path, split="train", split_by="time", window_size=100)
    gapped = ProbeDataset(tmp_path, split="train", split_by="time", window_size=100, gap=100)
    assert len(gapped) < len(tight)


def test_probe_skips_labeled_subjects_without_a_usable_tr(tmp_path):
    write_subject(subject_path(tmp_path, "ds1", "sub-ok"), _block(150),
                  labels=np.zeros((150, 10, 2), np.float32), attrs=LABEL_ATTRS)
    write_subject(subject_path(tmp_path, "ds1", "sub-notr"), _block(150),
                  labels=np.zeros((150, 10, 2), np.float32))

    ds = ProbeDataset(tmp_path, split="train", split_ratio=1.0, window_size=100)
    assert {s["subject"] for s in ds.samples} == {"sub-ok"}
