"""The corpus on disk: the per-participant layout, and where a corpus is found.

Resolution decides whether a command touches the network, so the order is
pinned here: an explicit path, then $DEEPMREYE_DATA, then ./data, then the
cache -- and a download only when the cache lacks what the command needs.
"""
import numpy as np
import pytest

from deepmreye import data
from deepmreye.data import (
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


def test_iter_subjects_honours_a_dataset_pattern(tmp_path):
    for ds in ("ds000001", "dsL01_x", "dsL02_y"):
        write_subject(subject_path(tmp_path, ds, "sub-01"), _block(t=10))
    assert {ds for ds, _s, _p in iter_subjects(tmp_path, "dsL*")} == {"dsL01_x", "dsL02_y"}


# --- resolution ----------------------------------------------------------------


@pytest.fixture
def no_network(monkeypatch):
    calls = []
    monkeypatch.setattr(data, "fetch", lambda *a, **kw: calls.append((a, kw)) or kw.get("target"))
    monkeypatch.delenv("DEEPMREYE_DATA", raising=False)
    return calls


def _corpus(path, datasets=("ds000001",)):
    for ds in datasets:
        write_subject(subject_path(path, ds, "sub-01"), _block(t=10))
    return path


def test_explicit_path_wins_and_is_never_downloaded_into(tmp_path, monkeypatch, no_network):
    monkeypatch.setenv("DEEPMREYE_DATA", str(_corpus(tmp_path / "env")))
    explicit = _corpus(tmp_path / "explicit")
    assert data.resolve(explicit, patterns=data.LABELLED) == explicit
    assert no_network == []


def test_explicit_path_must_exist(tmp_path, no_network):
    with pytest.raises(FileNotFoundError):
        data.resolve(tmp_path / "missing")


def test_env_var_then_cwd_data(tmp_path, monkeypatch, no_network):
    env = _corpus(tmp_path / "env")
    monkeypatch.setenv("DEEPMREYE_DATA", str(env))
    assert data.resolve(quiet=True) == env

    monkeypatch.delenv("DEEPMREYE_DATA")
    monkeypatch.chdir(tmp_path)
    _corpus(tmp_path / "data")
    assert data.resolve(quiet=True) == tmp_path / "data"
    assert no_network == []


def test_cache_is_used_when_it_has_what_is_asked(tmp_path, monkeypatch, no_network):
    monkeypatch.chdir(tmp_path)
    cache = _corpus(tmp_path / "cache", ("ds000001", "dsL01_x"))
    (cache / "index.parquet").write_bytes(b"")
    monkeypatch.setenv("DEEPMREYE_CACHE", str(cache))
    assert data.resolve(patterns=data.LABELLED, quiet=True) == cache
    assert no_network == []


def test_cache_is_topped_up_when_it_lacks_what_is_asked(tmp_path, monkeypatch, no_network):
    monkeypatch.chdir(tmp_path)
    cache = _corpus(tmp_path / "cache", ("ds000001",))       # no labelled folders yet
    monkeypatch.setenv("DEEPMREYE_CACHE", str(cache))
    data.resolve(patterns=data.LABELLED, quiet=True)
    assert len(no_network) == 1 and no_network[0][0][0] == data.LABELLED


def test_no_download_raises_with_a_useful_message(tmp_path, monkeypatch, no_network):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPMREYE_CACHE", str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError, match="DEEPMREYE_DATA"):
        data.resolve(download=False)


# --- gaze labels from an eye tracker ---------------------------------------------


def test_bin_gaze_means_samples_into_ten_bins_per_volume():
    tr, n_vol = 2.0, 3
    times = np.arange(0, n_vol * tr, 0.01)                   # 100 Hz
    x, y = times.copy(), -times
    labels = data.bin_gaze(times, x, y, n_vol, tr)
    assert labels.shape == (3, 10, 2)
    # Bin j of volume t covers [(t + j/10) tr, (t + (j+1)/10) tr): its mean time.
    assert labels[1, 3, 0] == pytest.approx((1 + 3.5 / 10) * tr, abs=0.01)
    assert labels[1, 3, 1] == pytest.approx(-(1 + 3.5 / 10) * tr, abs=0.01)
    # The mean of a volume's bins is its mean gaze.
    assert labels[2, :, 0].mean() == pytest.approx(x[(times >= 4) & (times < 6)].mean(), abs=0.01)


def test_bin_gaze_leaves_unrecorded_bins_nan_and_skips_blinks():
    times = np.arange(0, 4.0, 0.01)
    x = np.ones_like(times)
    x[(times >= 0.2) & (times < 0.4)] = np.nan               # a blink over bin 1 of volume 0
    labels = data.bin_gaze(times, x, x, n_volumes=3, tr=2.0)  # the run outlasts the recording
    assert np.isnan(labels[0, 1]).all() and np.isfinite(labels[0, 0]).all()
    assert np.isnan(labels[2]).all()
    assert np.nanmax(labels) == 1.0


def test_read_gaze_tsv_finds_columns_by_name(tmp_path):
    path = tmp_path / "g.tsv"
    path.write_text("x\ttime\ty\n1.0\t0.0\t2.0\nn/a\t0.5\t3.0\n")
    t, x, y = data.read_gaze_tsv(path)
    assert list(t) == [0.0, 0.5] and x[0] == 1.0 and np.isnan(x[1]) and list(y) == [2.0, 3.0]
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="time"):
        data.read_gaze_tsv(bad)
