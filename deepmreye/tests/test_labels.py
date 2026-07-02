import h5py

from deepmreye.labels import (
    append_label_events,
    export_labels,
    restore_labels,
    _latest_labels,
)


def _make_registry(path):
    """A registry with a skipped dataset, an approved dataset, and unlabeled subjects."""
    with h5py.File(path, "w") as f:
        ds1 = f.create_group("ds000001")
        ds1.attrs["approved"] = -1
        ds1.create_group("sub-01").attrs["approved"] = 1
        ds1.create_group("sub-02").attrs["approved"] = 0  # no eyes

        ds2 = f.create_group("ds000002")
        ds2.attrs["approved"] = -99  # skipped whole dataset
        ds2.create_group("sub-01").attrs["approved"] = -1  # unlabeled

        ds3 = f.create_group("ds000003")
        ds3.attrs["approved"] = -1
        ds3.create_group("sub-01").attrs["approved"] = 1
        ds3.create_group("sub-02").attrs["approved"] = 1


def test_export_captures_labels_and_skips_unlabeled(tmp_path):
    h5_path = tmp_path / "datasets.h5"
    csv_path = tmp_path / "labels.csv"
    _make_registry(h5_path)

    n = export_labels(h5_path, csv_path)
    # ds1: 2 subject labels; ds2: 1 dataset-skip (its subject is unlabeled -> skipped);
    # ds3: 2 subject labels. Total = 5.
    assert n == 5
    assert csv_path.exists()

    latest = _latest_labels(csv_path)
    assert latest[("ds000002", "dataset", "")]["label"] == "-99"
    assert latest[("ds000001", "subject", "sub-02")]["label"] == "0"
    assert ("ds000002", "subject", "sub-01") not in latest  # unlabeled not exported


def test_roundtrip_survives_registry_rebuild(tmp_path):
    """The disaster case: labels are wiped from the h5 but restored from the CSV."""
    h5_path = tmp_path / "datasets.h5"
    csv_path = tmp_path / "labels.csv"
    _make_registry(h5_path)
    export_labels(h5_path, csv_path)

    # Simulate a recompile: rebuild the registry with all labels reset to unlabeled.
    _make_registry(h5_path)
    with h5py.File(h5_path, "a") as f:
        f["ds000001"].attrs["approved"] = -1
        f["ds000002"].attrs["approved"] = -1  # skip flag lost on rebuild
        for ds in f.keys():
            for sub in f[ds].keys():
                f[ds][sub].attrs["approved"] = -1

    applied, missing = restore_labels(h5_path, csv_path)
    assert applied == 5
    assert missing == 0

    with h5py.File(h5_path, "r") as f:
        assert f["ds000002"].attrs["approved"] == -99
        assert f["ds000001"]["sub-01"].attrs["approved"] == 1
        assert f["ds000001"]["sub-02"].attrs["approved"] == 0
        assert f["ds000003"]["sub-01"].attrs["approved"] == 1


def test_latest_event_wins(tmp_path):
    """A relabel appended later overrides the earlier value on restore."""
    h5_path = tmp_path / "datasets.h5"
    csv_path = tmp_path / "labels.csv"
    _make_registry(h5_path)

    append_label_events(csv_path, [("ds000001", "subject", "sub-02", 0)])
    append_label_events(csv_path, [("ds000001", "subject", "sub-02", 1)])  # corrected later

    applied, missing = restore_labels(h5_path, csv_path)
    assert applied == 1 and missing == 0
    with h5py.File(h5_path, "r") as f:
        assert f["ds000001"]["sub-02"].attrs["approved"] == 1


def test_restore_skips_datasets_not_in_registry(tmp_path):
    h5_path = tmp_path / "datasets.h5"
    csv_path = tmp_path / "labels.csv"
    _make_registry(h5_path)

    append_label_events(csv_path, [("ds999999", "subject", "sub-01", 1)])  # not in registry
    applied, missing = restore_labels(h5_path, csv_path)
    assert applied == 0 and missing == 1
