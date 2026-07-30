"""Tests for the windowed loader over the per-participant layout."""
import numpy as np

from deepmreye.data.probe_dataset import ProbeDataset
from deepmreye.storage import subject_path, write_subject


def _block(t, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(47, 29, 18, t)).astype(np.float32)


# ProbeDataset requires a usable TR per subject -- the model conditions on it,
# so a subject without one cannot be scored. Fixtures write a plausible value.
LABEL_ATTRS = {"repetition_time": 2.0}


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
