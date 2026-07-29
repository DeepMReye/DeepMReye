"""Tests for QA Labeling Flask App and dataset navigation."""
import sys
from pathlib import Path
import h5py
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import label_datasets as ld


@pytest.fixture
def test_h5(tmp_path):
    h5_path = tmp_path / "datasets.h5"
    with h5py.File(h5_path, "w") as f:
        g1 = f.create_group("ds000001")
        s1 = g1.create_group("sub-01")
        s1.attrs["report_html_path"] = str(tmp_path / "report1.html")
        s1.attrs["approved"] = -1

        s2 = g1.create_group("sub-02")
        s2.attrs["report_html_path"] = str(tmp_path / "report2.html")
        s2.attrs["approved"] = -1

        g2 = f.create_group("ds000002")
        s3 = g2.create_group("sub-01")
        s3.attrs["report_html_path"] = str(tmp_path / "report3.html")
        s3.attrs["approved"] = 1

    (tmp_path / "report1.html").write_text("<html>Report 1</html>")
    (tmp_path / "report2.html").write_text("<html>Report 2</html>")
    (tmp_path / "report3.html").write_text("<html>Report 3</html>")

    old_h5 = ld.H5_PATH
    ld.H5_PATH = str(h5_path)
    ld.NO_DOWNLOAD = True
    yield h5_path
    ld.H5_PATH = old_h5


def test_get_dataset_info(test_h5):
    datasets, details, counts = ld.get_dataset_info()
    assert datasets == ["ds000001", "ds000002"]
    assert counts["total"] == 2
    assert counts["unlabeled"] == 1
    assert counts["labeled"] == 1
    assert details["ds000001"]["status"] == "unlabeled"
    assert details["ds000002"]["status"] == "labeled"


def test_index_route(test_h5):
    ld.app.config["TESTING"] = True
    client = ld.app.test_client()

    response = client.get("/")
    assert response.status_code == 200
    assert b"Dataset 1/2" in response.data
    assert b"ds000001" in response.data

    # Jump directly to dataset 2
    response_ds2 = client.get("/?ds=ds000002")
    assert response_ds2.status_code == 200
    assert b"Dataset 2/2" in response_ds2.data
    assert b"ds000002" in response_ds2.data


def test_submit_label_save_next_and_prev(test_h5):
    ld.app.config["TESTING"] = True
    client = ld.app.test_client()

    # Submit labels for ds000001 and save_next (including label 4 for faint eyes)
    res = client.post("/submit", data={
        "dataset": "ds000001",
        "action": "save_next",
        "label_sub-01": "1",
        "label_sub-02": "4"
    }, follow_redirects=True)

    assert res.status_code == 200
    assert b"ds000002" in res.data

    # Verify HDF5 updated
    with h5py.File(test_h5, "r") as f:
        assert f["ds000001"]["sub-01"].attrs["approved"] == 1
        assert f["ds000001"]["sub-02"].attrs["approved"] == 4

    # Submit for ds000002 and save_prev
    res_prev = client.post("/submit", data={
        "dataset": "ds000002",
        "action": "save_prev",
        "label_sub-01": "1"
    }, follow_redirects=True)

    assert res_prev.status_code == 200
    assert b"ds000001" in res_prev.data

    # Submit for ds000001 and jump directly to ds000002 via dropdown
    res_jump = client.post("/submit", data={
        "dataset": "ds000001",
        "action": "save_to_ds",
        "target_ds": "ds000002",
        "label_sub-01": "1",
        "label_sub-02": "1"
    }, follow_redirects=True)

    assert res_jump.status_code == 200
    assert b"ds000002" in res_jump.data


def test_skip_and_unskip(test_h5):
    ld.app.config["TESTING"] = True
    client = ld.app.test_client()

    res = client.post("/submit", data={
        "dataset": "ds000001",
        "action": "skip_next"
    }, follow_redirects=True)

    assert res.status_code == 200
    with h5py.File(test_h5, "r") as f:
        assert f["ds000001"].attrs["approved"] == -99

    res_unskip = client.post("/submit", data={
        "dataset": "ds000001",
        "action": "unskip"
    }, follow_redirects=True)

    assert res_unskip.status_code == 200
    with h5py.File(test_h5, "r") as f:
        assert f["ds000001"].attrs["approved"] == -1


def test_empty_report_dataset_is_skipped(test_h5):
    """Datasets without HTML reports cannot be visually QA'd and are automatically marked as skipped (-99)."""
    with h5py.File(test_h5, "a") as f:
        g = f.create_group("ds_noreports")
        g.create_group("sub-01")  # no report_html_path attribute

    ld.app.config["TESTING"] = True
    client = ld.app.test_client()

    res = client.post("/submit", data={
        "dataset": "ds_noreports",
        "action": "save_next"
    }, follow_redirects=True)

    assert res.status_code == 200
    with h5py.File(test_h5, "r") as f:
        assert f["ds_noreports"].attrs["approved"] == -99


def test_rapid_audit_route_and_api(test_h5):
    ld.app.config["TESTING"] = True
    client = ld.app.test_client()

    # Mark ds000001 subjects as 1 and 4 (eyes present)
    with h5py.File(test_h5, "a") as f:
        f["ds000001"]["sub-01"].attrs["approved"] = 1
        f["ds000001"]["sub-02"].attrs["approved"] = 4

    # GET /rapid
    res_rapid = client.get("/rapid")
    assert res_rapid.status_code == 200
    assert b"Rapid Visual Audit" in res_rapid.data

    # GET /api/rapid_datasets
    res_api = client.get("/api/rapid_datasets")
    assert res_api.status_code == 200
    data = res_api.get_json()
    assert len(data) >= 1
    ds_names = [d["dataset"] for d in data]
    assert "ds000001" in ds_names

    # Toggle ds000001 to removed via POST /api/toggle_dataset_approval
    res_toggle = client.post("/api/toggle_dataset_approval", json={"dataset": "ds000001", "approved": False})
    assert res_toggle.status_code == 200
    assert res_toggle.get_json()["approved"] is False

    with h5py.File(test_h5, "r") as f:
        assert f["ds000001"]["sub-01"].attrs["approved"] == 0
        assert f["ds000001"]["sub-02"].attrs["approved"] == 0
