"""Tests for the BIDS eye-tracking ingest.

The failure this module exists to prevent is a constant time shift between gaze
and BOLD. It is invisible in every cheap check -- the labels still look like
gaze, the array still has the right shape, the decoder still trains -- so the
tests here are deliberately about *time*, not about shapes.
"""
import gzip
import json

import re

import numpy as np
import pytest

from deepmreye.eyetracking import (
    ANCHOR_EVENTS,
    ANCHOR_INDEXED_MESSAGE,
    ANCHOR_MESSAGE,
    ANCHOR_STARTTIME,
    ANCHOR_TRIGGER,
    N_SUBTR,
    SyncError,
    anchor_seconds,
    bin_to_subtr,
    center_and_scale,
    clean_gaze,
    column_index,
    consistency,
    find_gaze_columns,
    load_sidecar,
    read_asc,
    read_physio_events,
    read_tsv,
    trigger_onsets,
)


# --------------------------------------------------------------------------
# tsv parsing
# --------------------------------------------------------------------------

def test_headerless_tsv_uses_sidecar_columns():
    blob = b"1\t2.5\t3.5\n2\t2.6\t3.6\n"
    arr, cols = read_tsv(blob, columns=["t", "x", "y"])
    assert arr.shape == (2, 3)
    assert cols == ["t", "x", "y"]
    assert arr[1, 1] == pytest.approx(2.6)


def test_header_row_overrides_sidecar_columns():
    """ds005166 ships a header despite BIDS saying headerless; it must win."""
    blob = b"eye_timestamp\teye1_x_coordinate\teye1_y_coordinate\n10\t100\t200\n"
    arr, cols = read_tsv(blob, columns=["wrong", "names", "here"])
    assert cols == ["eye_timestamp", "eye1_x_coordinate", "eye1_y_coordinate"]
    assert arr.shape == (1, 3)
    assert arr[0, 2] == pytest.approx(200)


def test_gzip_payload_is_transparently_decompressed():
    arr, cols = read_tsv(gzip.compress(b"1\t2\t3\n"), columns=["a", "b", "c"])
    assert arr.shape == (1, 3)


def test_na_becomes_nan():
    arr, _ = read_tsv(b"1\tn/a\t3\n", columns=["a", "b", "c"])
    assert np.isnan(arr[0, 1])


def test_find_gaze_columns_prefers_bids_names():
    assert find_gaze_columns(["timestamp", "x_coordinate", "y_coordinate"]) == (1, 2)


def test_find_gaze_columns_handles_binocular():
    cols = ["eye_timestamp", "eye1_x_coordinate", "eye1_y_coordinate",
            "eye1_pupil_size", "eye2_x_coordinate", "eye2_y_coordinate"]
    assert find_gaze_columns(cols) == (1, 2)


def test_find_gaze_columns_returns_none_for_pupil_only():
    """ds006578 records pupil but no gaze; it must not be mistaken for usable."""
    assert find_gaze_columns(["Time", "ConfidenceInterval", "PupilMeasure1"]) is None


def test_column_index_is_case_insensitive():
    assert column_index(["Time", "Trigger", "X"], "trigger") == 1


# --------------------------------------------------------------------------
# blink / sentinel handling
# --------------------------------------------------------------------------

def test_sentinels_and_out_of_range_become_nan():
    x = np.array([100.0, 0.0, 4294967295.0, 1e9, 200.0])
    y = np.array([100.0, 0.0, 4294967295.0, 1e9, 200.0])
    cx, cy = clean_gaze(x, y)
    assert np.isfinite(cx[0]) and np.isfinite(cx[4])
    assert np.isnan(cx[1]) and np.isnan(cx[2]) and np.isnan(cx[3])
    assert np.isnan(cy[1])


def test_zero_on_one_axis_only_is_kept():
    """x=0 with a valid y is a real gaze position on the vertical midline."""
    cx, cy = clean_gaze(np.array([0.0]), np.array([300.0]))
    assert np.isfinite(cx[0]) and np.isfinite(cy[0])


def test_clean_gaze_does_not_mutate_input():
    x = np.array([0.0, 5.0])
    clean_gaze(x, x.copy())
    assert x[0] == 0.0


# --------------------------------------------------------------------------
# binning: the time convention
# --------------------------------------------------------------------------

def test_bin_shape_and_dtype():
    t = np.arange(0, 10, 0.01)
    out = bin_to_subtr(t, t, t, n_trs=5, tr=2.0)
    assert out.shape == (5, N_SUBTR, 2)
    assert out.dtype == np.float32


def test_constant_gaze_recovers_exactly():
    t = np.arange(0, 20, 0.001)
    out = bin_to_subtr(t, np.full_like(t, 3.0), np.full_like(t, -7.0),
                       n_trs=10, tr=2.0)
    assert np.allclose(out[..., 0], 3.0)
    assert np.allclose(out[..., 1], -7.0)


def test_bin_holds_the_mean_of_its_own_window():
    """Sub-bin j of TR t must average [(t+j/10)*tr, (t+(j+1)/10)*tr)."""
    tr, n_trs = 2.0, 3
    t = np.arange(0, n_trs * tr, 0.001)
    x = t.copy()                      # gaze == time, so the answer is arithmetic
    out = bin_to_subtr(t, x, x, n_trs=n_trs, tr=tr)
    width = tr / N_SUBTR
    for tt in range(n_trs):
        for j in range(N_SUBTR):
            lo = (tt + j / N_SUBTR) * tr
            expected = lo + width / 2      # mean of a dense uniform sweep
            assert out[tt, j, 0] == pytest.approx(expected, abs=1e-3)


def test_the_ten_samples_average_to_the_mean_over_the_tr():
    """What temporal_targets assumes when it averages the sub-TR axis."""
    tr = 1.5
    t = np.arange(0, tr * 4, 0.0005)
    x = np.sin(2 * np.pi * t / 3.0)
    out = bin_to_subtr(t, x, x, n_trs=4, tr=tr)
    for tt in range(4):
        m = (t >= tt * tr) & (t < (tt + 1) * tr)
        assert out[tt, :, 0].mean() == pytest.approx(x[m].mean(), abs=1e-3)


def test_sub_bins_do_not_overlap():
    """A single sample lands in exactly one bin, so bins share no endpoint.

    dsL04/dsL05 were built the other way (labels[t,9] == labels[t+1,0]); that
    convention double-counts the boundary. This asserts the choice made here.
    """
    tr = 2.0
    edge = tr / N_SUBTR            # boundary between sub-bin 0 and 1
    out = bin_to_subtr(np.array([edge]), np.array([42.0]), np.array([42.0]),
                       n_trs=1, tr=tr)
    assert np.isnan(out[0, 0, 0]), "sample on the edge leaked back into bin 0"
    assert out[0, 1, 0] == pytest.approx(42.0), "edge sample belongs to bin 1"


def test_sample_at_time_zero_is_in_the_first_bin():
    out = bin_to_subtr(np.array([0.0]), np.array([5.0]), np.array([5.0]),
                       n_trs=1, tr=2.0)
    assert out[0, 0, 0] == pytest.approx(5.0)


def test_samples_before_volume_zero_are_dropped_not_wrapped():
    """A tracker that starts early must not have its pre-scan samples folded in."""
    t = np.array([-5.0, -0.001, 0.05])
    out = bin_to_subtr(t, np.array([1.0, 2.0, 9.0]), np.array([1.0, 2.0, 9.0]),
                       n_trs=1, tr=2.0)
    assert out[0, 0, 0] == pytest.approx(9.0)


def test_samples_past_the_scan_end_are_dropped():
    t = np.array([0.05, 100.0])
    out = bin_to_subtr(t, np.array([3.0, 999.0]), np.array([3.0, 999.0]),
                       n_trs=1, tr=2.0)
    assert out[0, 0, 0] == pytest.approx(3.0)
    assert not np.any(out == 999.0)


def test_empty_bins_are_nan_not_interpolated():
    out = bin_to_subtr(np.array([0.0]), np.array([1.0]), np.array([1.0]),
                       n_trs=2, tr=2.0)
    assert np.isfinite(out[0, 0, 0])
    assert np.isnan(out[0, 1:, :]).all()
    assert np.isnan(out[1]).all()


def test_min_samples_threshold_blanks_thin_bins():
    t = np.array([0.01, 0.02, 0.21])
    out = bin_to_subtr(t, np.ones_like(t), np.ones_like(t), n_trs=1, tr=2.0,
                       min_samples=2)
    assert np.isfinite(out[0, 0, 0])     # two samples
    assert np.isnan(out[0, 1, 0])        # one sample


def test_all_nan_gaze_yields_all_nan_labels():
    t = np.arange(0, 4, 0.01)
    out = bin_to_subtr(t, np.full_like(t, np.nan), np.full_like(t, np.nan),
                       n_trs=2, tr=2.0)
    assert np.isnan(out).all()


def test_track_loss_on_one_axis_nulls_the_whole_sample():
    """A sample with x lost is a track loss; its y is not a gaze position.

    The coordinates are deliberately coupled rather than masked independently.
    An eye tracker that cannot localise the pupil horizontally has not somehow
    measured it vertically -- keeping the surviving axis would feed the probe a
    coordinate the tracker never resolved.
    """
    t = np.array([0.01, 0.02])
    x = np.array([np.nan, 4.0])
    y = np.array([5.0, 7.0])
    out = bin_to_subtr(t, x, y, n_trs=1, tr=2.0)
    assert out[0, 0, 0] == pytest.approx(4.0)
    assert out[0, 0, 1] == pytest.approx(7.0), "the lost sample's y leaked in"


# --------------------------------------------------------------------------
# the shift test -- the property the whole module is for
# --------------------------------------------------------------------------

@pytest.mark.parametrize("shift_trs", [-3, -1, 1, 2, 5])
def test_a_time_shift_shows_up_as_the_same_shift_in_the_labels(shift_trs):
    tr, n_trs = 1.0, 40
    t = np.arange(0, n_trs * tr, 0.002)
    rng = np.random.default_rng(0)
    trace = np.cumsum(rng.normal(size=len(t))) * 0.01

    truth = bin_to_subtr(t, trace, trace, n_trs=n_trs, tr=tr)
    shifted = bin_to_subtr(t + shift_trs * tr, trace, trace, n_trs=n_trs, tr=tr)

    a = truth[..., 0].mean(axis=1)
    b = shifted[..., 0].mean(axis=1)
    lo, hi = max(0, shift_trs), min(n_trs, n_trs + shift_trs)
    overlap = np.arange(lo, hi)
    # atol is loose because shifting the sample times re-assigns the odd sample
    # across a bin edge (one of ~50 per sub-bin), which moves a bin mean by
    # ~2e-4 on a random walk. That is discretisation, not misalignment: a real
    # off-by-one-TR error moves these values by O(1).
    assert np.allclose(b[overlap], a[overlap - shift_trs], atol=1e-3, equal_nan=True)


def test_lag_sweep_recovers_an_injected_offset():
    """The instrument the real sync check relies on must locate a known offset.

    Delaying the recording by ``+4`` TRs makes ``off[k] == ref[k - 4]``, so the
    sweep -- which pairs ``ref[lag + i]`` with ``off[i]`` -- must peak at
    ``lag = -4``. Getting that sign backwards is exactly the mistake the real
    script would make, so it is pinned here.
    """
    tr, n_trs, injected = 1.0, 200, 4
    t = np.arange(0, n_trs * tr, 0.01)
    rng = np.random.default_rng(1)
    trace = np.cumsum(rng.normal(size=len(t))) * 0.05

    ref = bin_to_subtr(t, trace, trace, n_trs=n_trs, tr=tr)[..., 0].mean(axis=1)
    off = bin_to_subtr(t + injected * tr, trace, trace,
                       n_trs=n_trs, tr=tr)[..., 0].mean(axis=1)

    def r_at(lag):
        if lag > 0:
            a, b = ref[lag:], off[:len(off) - lag]
        elif lag < 0:
            a, b = ref[:len(ref) + lag], off[-lag:]
        else:
            a, b = ref, off
        m = np.isfinite(a) & np.isfinite(b)
        return np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 10 else -np.inf

    lags = list(range(-10, 11))
    best = max(lags, key=r_at)
    assert best == -injected, f"lag sweep found {best}, expected {-injected}"
    assert r_at(best) > 0.999


def test_lag_sweep_peak_is_broad_for_an_autocorrelated_trace():
    """Why the real check reports a margin, not just an argmax.

    Gaze is smooth, so neighbouring lags score almost as well as the truth. A
    sync check that only prints the argmax hides how weakly it is determined.
    """
    tr, n_trs = 1.0, 200
    t = np.arange(0, n_trs * tr, 0.01)
    rng = np.random.default_rng(1)
    trace = np.cumsum(rng.normal(size=len(t))) * 0.05
    ref = bin_to_subtr(t, trace, trace, n_trs=n_trs, tr=tr)[..., 0].mean(axis=1)

    m = np.isfinite(ref[1:]) & np.isfinite(ref[:-1])
    neighbour = np.corrcoef(ref[1:][m], ref[:-1][m])[0, 1]
    assert neighbour > 0.9, "a random walk should be strongly autocorrelated"


# --------------------------------------------------------------------------
# anchoring
# --------------------------------------------------------------------------

def test_starttime_anchor_places_volume_zero_correctly():
    """StartTime=-12.27 means the tracker began 12.27 s before volume 0."""
    times = np.arange(1000.0, 1100.0, 0.002)
    t0, info = anchor_seconds(ANCHOR_STARTTIME, sidecar={"StartTime": -12.27},
                              times=times)
    assert t0 == pytest.approx(1000.0 + 12.27)
    assert info["anchor"] == ANCHOR_STARTTIME
    rel = times - t0
    assert rel[0] == pytest.approx(-12.27)


def test_positive_starttime_means_tracker_started_after_the_scan():
    times = np.arange(0.0, 10.0, 0.1)
    t0, _ = anchor_seconds(ANCHOR_STARTTIME, sidecar={"StartTime": 3.0}, times=times)
    assert (times - t0)[0] == pytest.approx(3.0)


def test_starttime_equal_to_first_timestamp_is_rejected():
    """ds006833/ds005166 write the raw tracker clock into StartTime."""
    times = np.arange(3433520.0, 3433600.0)
    with pytest.raises(SyncError, match="raw tracker clock"):
        anchor_seconds(ANCHOR_STARTTIME, sidecar={"StartTime": [3433520]}, times=times)


def test_zero_starttime_on_a_synthesised_grid_is_accepted():
    """ds000113 has no timestamp column and a legitimate StartTime of 0.0.

    Times are synthesised as arange(n)/fs, so times[0] is 0 by construction and
    coincides with StartTime without that meaning anything. The tracker-clock
    guard must not fire here.
    """
    times = np.arange(0.0, 10.0, 0.001)
    t0, info = anchor_seconds(ANCHOR_STARTTIME, sidecar={"StartTime": 0.0},
                              times=times, times_from_column=False)
    assert t0 == pytest.approx(0.0)
    assert info["anchor"] == ANCHOR_STARTTIME


def test_tracker_clock_guard_still_fires_on_a_real_timestamp_column():
    times = np.arange(3433520.0, 3433600.0)
    with pytest.raises(SyncError, match="raw tracker clock"):
        anchor_seconds(ANCHOR_STARTTIME, sidecar={"StartTime": [3433520]},
                       times=times, times_from_column=True)


def test_missing_starttime_raises():
    with pytest.raises(SyncError, match="StartTime absent"):
        anchor_seconds(ANCHOR_STARTTIME, sidecar={}, times=np.arange(3.0))


def test_multi_element_starttime_is_ambiguous_and_raises():
    with pytest.raises(SyncError, match="ambiguous"):
        anchor_seconds(ANCHOR_STARTTIME, sidecar={"StartTime": [1.0, 2.0]},
                       times=np.arange(3.0))


def test_trigger_onsets_finds_rising_edges():
    times = np.arange(10.0)
    trig = np.array([0, 0, 1, 1, 0, 0, 1, 0, 0, 1])
    assert np.allclose(trigger_onsets(trig, times), [2.0, 6.0, 9.0])


def test_trigger_high_at_the_first_sample_counts_as_an_onset():
    times = np.arange(5.0)
    assert trigger_onsets(np.array([1, 0, 0, 1, 0]), times)[0] == 0.0


def test_trigger_anchor_uses_the_first_pulse_and_validates_the_period():
    times = np.arange(0, 100, 0.01)
    trig = np.zeros_like(times)
    trig[(np.round(times, 2) * 100).astype(int) % 200 == 0] = 1   # every 2.0 s
    t0, info = anchor_seconds(ANCHOR_TRIGGER, times=times, trigger=trig, tr=2.0)
    assert t0 == pytest.approx(0.0)
    assert info["median_interval"] == pytest.approx(2.0, abs=1e-6)


def test_trigger_anchor_rejects_a_period_that_is_not_the_tr():
    times = np.arange(0, 100, 0.01)
    trig = np.zeros_like(times)
    trig[(np.round(times, 2) * 100).astype(int) % 300 == 0] = 1   # 3.0 s
    with pytest.raises(SyncError, match="trigger period"):
        anchor_seconds(ANCHOR_TRIGGER, times=times, trigger=trig, tr=2.0)


def test_trigger_anchor_needs_more_than_one_pulse():
    times = np.arange(0, 5, 0.01)
    trig = np.zeros_like(times)
    trig[10] = 1
    with pytest.raises(SyncError, match="trigger edges"):
        anchor_seconds(ANCHOR_TRIGGER, times=times, trigger=trig, tr=2.0)


def test_message_anchor_returns_the_matching_onset():
    events = [(3433591.0, "RECORD_START"),
              (3492113.0, "trial 1 mri_trigger val = -8"),
              (3492114.0, "trial 1 started")]
    t0, info = anchor_seconds(ANCHOR_MESSAGE, events=events,
                              message_pattern=r"mri_trigger")
    assert t0 == pytest.approx(3492113.0)
    assert "mri_trigger" in info["message"]


def test_message_anchor_raises_when_absent():
    with pytest.raises(SyncError, match="no event matched"):
        anchor_seconds(ANCHOR_MESSAGE, events=[(1.0, "RECORD_START")],
                       message_pattern=r"mri_trigger")


def test_unknown_anchor_strategy_raises():
    with pytest.raises(ValueError, match="unknown anchor"):
        anchor_seconds("vibes", sidecar={"StartTime": 0.0})


# --------------------------------------------------------------------------
# consistency
# --------------------------------------------------------------------------

def test_consistency_accepts_full_coverage():
    ok, rep = consistency(np.arange(0, 300, 0.01), n_trs=150, tr=2.0)
    assert ok
    assert rep["covered_fraction"] == pytest.approx(1.0, abs=1e-3)


def test_consistency_rejects_a_recording_that_covers_half_the_scan():
    ok, rep = consistency(np.arange(0, 150, 0.01), n_trs=150, tr=2.0)
    assert not ok
    assert rep["covered_fraction"] == pytest.approx(0.5, abs=1e-3)


def test_consistency_rejects_a_recording_that_ends_before_the_scan_starts():
    ok, _ = consistency(np.arange(-500, -400, 0.01), n_trs=150, tr=2.0)
    assert not ok


def test_consistency_tolerates_a_tracker_that_overruns_the_scan():
    """ds006833 records 249 s around a 185 s scan; that is fine."""
    ok, rep = consistency(np.arange(-58.0, 190.0, 0.01), n_trs=154, tr=1.2)
    assert ok
    assert rep["covered_fraction"] == pytest.approx(1.0, abs=1e-3)


def test_consistency_on_empty_input():
    ok, rep = consistency(np.array([]), n_trs=10, tr=2.0)
    assert not ok and "reason" in rep


# --------------------------------------------------------------------------
# units and orientation
# --------------------------------------------------------------------------

def test_center_subtracts_the_screen_middle():
    lab = np.zeros((2, N_SUBTR, 2), dtype=np.float32)
    lab[..., 0] = 960.0
    lab[..., 1] = 540.0
    out = center_and_scale(lab, center=(960, 540))
    assert np.allclose(out, 0.0)


def test_flip_y_inverts_only_the_vertical_axis():
    lab = np.ones((1, N_SUBTR, 2), dtype=np.float32)
    out = center_and_scale(lab, flip_y=True)
    assert np.allclose(out[..., 0], 1.0)
    assert np.allclose(out[..., 1], -1.0)


def test_degrees_per_unit_scales_after_centering():
    lab = np.full((1, N_SUBTR, 2), 100.0, dtype=np.float32)
    out = center_and_scale(lab, center=(50, 50), degrees_per_unit=0.034)
    assert np.allclose(out, 50 * 0.034)


def test_center_and_scale_preserves_nan():
    lab = np.full((1, N_SUBTR, 2), np.nan, dtype=np.float32)
    assert np.isnan(center_and_scale(lab, center=(1, 1), degrees_per_unit=2.0)).all()


def test_center_and_scale_does_not_mutate_input():
    lab = np.ones((1, N_SUBTR, 2), dtype=np.float32)
    center_and_scale(lab, center=(5, 5), flip_y=True)
    assert np.allclose(lab, 1.0)


# --------------------------------------------------------------------------
# physioevents
# --------------------------------------------------------------------------

def test_read_physio_events_keeps_only_real_messages():
    blob = ("3433570\t10\tfixation\t0\tn/a\n"
            "3433591\tn/a\tn/a\tn/a\tRECORD_START\n"
            "3492113\tn/a\tn/a\tn/a\ttrial 1 mri_trigger val = -8\n")
    ev = read_physio_events(blob)
    assert len(ev) == 2
    assert ev[0] == (3433591.0, "RECORD_START")
    assert ev[1][0] == 3492113.0


def test_read_physio_events_handles_gzip_and_short_rows():
    blob = gzip.compress(b"1\t2\n3433591\tn/a\tn/a\tn/a\tRECORD_START\n")
    assert read_physio_events(blob) == [(3433591.0, "RECORD_START")]


def test_load_sidecar_accepts_bytes():
    assert load_sidecar(json.dumps({"StartTime": 1.0}).encode())["StartTime"] == 1.0


# --------------------------------------------------------------------------
# end-to-end on a synthetic run with a known answer
# --------------------------------------------------------------------------

def test_full_path_recovers_a_known_trace_through_a_trigger_anchor():
    tr, n_trs, fs = 2.0, 60, 60.0
    clock0 = 12345.678                       # arbitrary tracker epoch
    n = int(n_trs * tr * fs) + 600           # 10 s of pre-scan recording
    times = clock0 + np.arange(n) / fs

    scan_start = clock0 + 10.0
    rel = times - scan_start
    x = 300.0 * np.sin(2 * np.pi * rel / 20.0) + 960.0
    y = 200.0 * np.cos(2 * np.pi * rel / 30.0) + 540.0

    trig = np.zeros(n)
    for v in range(n_trs):
        trig[np.argmin(np.abs(times - (scan_start + v * tr)))] = 1

    t0, info = anchor_seconds(ANCHOR_TRIGGER, times=times, trigger=trig, tr=tr)
    assert t0 == pytest.approx(scan_start, abs=1 / fs)

    ok, _ = consistency(times - t0, n_trs, tr)
    assert ok

    lab = bin_to_subtr(times - t0, x, y, n_trs=n_trs, tr=tr)
    lab = center_and_scale(lab, center=(960, 540), flip_y=True)

    per_tr = np.nanmean(lab[..., 0], axis=1)
    expect = [300.0 * np.sin(2 * np.pi * ((v + 0.5) * tr) / 20.0) for v in range(n_trs)]
    assert np.corrcoef(per_tr, expect)[0, 1] > 0.999
    assert np.nanmax(np.abs(lab[..., 1])) == pytest.approx(200.0, rel=0.05)


# --------------------------------------------------------------------------
# the ingest script's dataset configuration
# --------------------------------------------------------------------------

def _configs():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.fetch_eyetracking import DATASETS, bold_for
    return DATASETS, bold_for


def test_every_dataset_config_declares_a_known_anchor():
    from deepmreye.eyetracking import ANCHORS

    DATASETS, _ = _configs()
    for ds, cfg in DATASETS.items():
        assert cfg["anchor"] in ANCHORS, ds


def test_message_anchored_configs_carry_a_pattern_and_events_suffix():
    DATASETS, _ = _configs()
    for ds, cfg in DATASETS.items():
        if cfg["anchor"] in (ANCHOR_MESSAGE, ANCHOR_INDEXED_MESSAGE):
            assert cfg.get("message_pattern"), ds
            # A .asc carries its own messages, so no separate events file.
            if not cfg["et_pattern"].endswith(r"\.asc\.gz$"):
                assert cfg.get("events_suffix"), ds


def test_indexed_message_configs_capture_the_index():
    """Without a capture group the anchor cannot know which volume a pulse is."""
    import re as _re

    DATASETS, _ = _configs()
    for ds, cfg in DATASETS.items():
        if cfg["anchor"] == ANCHOR_INDEXED_MESSAGE:
            assert _re.compile(cfg["message_pattern"]).groups >= 1, ds


def test_trigger_anchored_configs_name_their_trigger_column():
    DATASETS, _ = _configs()
    for ds, cfg in DATASETS.items():
        if cfg["anchor"] == ANCHOR_TRIGGER:
            assert cfg.get("trigger_col"), ds


def test_corpus_names_are_unique():
    DATASETS, _ = _configs()
    names = [c["corpus_name"] for c in DATASETS.values()]
    assert len(set(names)) == len(names)


def test_active_configs_use_dsL_and_excluded_ones_must_not():
    """`dsL*/*.h5` is the glob that selects the labeled subset.

    A config whose key starts with ``_`` failed verification and is kept only
    for the record. Its data must sit *outside* that glob, or an unalignable
    dataset silently rejoins the probe set -- which is the whole failure the
    exclusion exists to prevent.
    """
    DATASETS, _ = _configs()
    for key, cfg in DATASETS.items():
        excluded = key.startswith("_")
        starts_with_dsl = cfg["corpus_name"].startswith("dsL")
        if excluded:
            assert not starts_with_dsl, (
                f"{key} is excluded but {cfg['corpus_name']!r} is still in the "
                f"dsL* probe glob")
        else:
            assert starts_with_dsl, key


def test_excluded_configs_are_not_offered_on_the_command_line():
    """`--dataset` must not let an excluded dataset be re-ingested by accident."""
    import subprocess
    import sys as _sys
    from pathlib import Path as _Path
    out = subprocess.run(
        [_sys.executable, str(_Path(__file__).resolve().parents[2]
                              / "scripts" / "fetch_eyetracking.py"), "--list"],
        capture_output=True, text=True, timeout=120)
    for key in ("_ds007532_excluded", "_ds001242_excluded"):
        assert key not in out.stdout, (
            f"{key} is offered on the command line. Configs keyed with a "
            "leading underscore failed verification and are kept only as "
            "documentation -- an excluded dataset must not be re-ingested "
            "because someone tab-completed it.")


def test_only_documented_geometry_claims_degrees():
    """label_units must not say degrees unless a conversion actually exists."""
    DATASETS, _ = _configs()
    for ds, cfg in DATASETS.items():
        if cfg["label_units"] == "degrees_visual_angle":
            assert cfg.get("degrees_per_unit") is not None, ds
        else:
            assert cfg.get("degrees_per_unit") is None, ds


def test_configs_with_no_timestamp_column_can_synthesise_times():
    """A TSV needs a clock from somewhere; an EyeLink file brings its own.

    `.asc` and `.edf` carry sample timestamps in the file, so `timestamp_col`
    is meaningless for them -- `build_labels` never reads it on those paths.
    Only the TSV branch has to synthesise times, and only it needs a sampling
    frequency or a sidecar to do so.
    """
    DATASETS, _ = _configs()
    for ds, cfg in DATASETS.items():
        if re.search(r"\\.(asc|edf)(\\.gz)?\$", cfg["et_pattern"], re.IGNORECASE):
            continue
        if cfg.get("timestamp_col") is None:
            assert cfg.get("sidecar_key") or cfg.get("sampling_frequency"), ds


def test_bold_for_strips_the_recording_entity():
    _, bold_for = _configs()
    et = ("ds006833/sub-01/ses-02/func/"
          "sub-01_ses-02_task-DeepMReyeCalib_run-01_recording-eye1_physio.tsv.gz")
    assert bold_for(et) == ("ds006833/sub-01/ses-02/func/"
                            "sub-01_ses-02_task-DeepMReyeCalib_run-01_bold.nii.gz")


def test_bold_for_handles_the_eyetrack_spelling():
    _, bold_for = _configs()
    et = "ds008366/sub-29/func/sub-29_task-learn_run-13_eyetrack.tsv.gz"
    assert bold_for(et).endswith("sub-29_task-learn_run-13_bold.nii.gz")


def test_bold_for_keeps_the_run_entity():
    """Dropping run- would pair gaze with the wrong run of the same task."""
    _, bold_for = _configs()
    a = bold_for("d/sub-01/func/sub-01_task-x_run-01_recording-eye_physio.tsv.gz")
    b = bold_for("d/sub-01/func/sub-01_task-x_run-02_recording-eye_physio.tsv.gz")
    assert a != b and "run-01" in a and "run-02" in b


# --------------------------------------------------------------------------
# EyeLink ASCII (.asc) reading
# --------------------------------------------------------------------------

ASC_MONO = """** CONVERTED FROM data/x.EDF using edfapi
MSG\t7095056 GAZE_COORDS 0.00 0.00 1920.00 1080.00
START\t7095057 \tRIGHT\tEVENTS
SAMPLES\tGAZE\tRIGHT\tRATE\t1000.00
SFIX R   7095076
7095057\t  930.0\t  611.6\t  461.0\t.....
7095058\t  931.0\t  612.6\t  462.0\t.....
7095059\t   .\t   .\t    0.0\t...
EFIX R   7095076\t7095077\t2\t 1553.2\t  417.4\t    598
MSG\t7095060 TTLPulse_10
END\t7095061
"""

ASC_BINO = """MSG\t5818431 RECCFG CR 1000 2 1 LR
START\t5818453 \tLEFT\tRIGHT\tSAMPLES\tEVENTS
5818453\t  930.0\t  611.6\t  461.0\t  920.6\t  589.3\t  532.0\t.....
5818454\t  931.0\t  612.6\t  462.0\t  921.6\t  590.3\t  533.0\t.....
"""


def test_read_asc_monocular_samples_and_messages():
    t, x, y, msgs = read_asc(ASC_MONO)
    assert len(t) == 3
    assert t[0] == pytest.approx(7095057)
    assert x[0] == pytest.approx(930.0)
    assert y[1] == pytest.approx(612.6)
    assert np.isnan(x[2]) and np.isnan(y[2])
    assert (7095060.0, "TTLPulse_10") in msgs


def test_read_asc_skips_event_lines():
    """SFIX/EFIX/START/END must not be mistaken for samples."""
    t, _, _, _ = read_asc(ASC_MONO)
    assert len(t) == 3, "an event line was parsed as a sample"


def test_read_asc_binocular_picks_one_eye_not_the_average():
    t, x, y, _ = read_asc(ASC_BINO, eye="left")
    assert x[0] == pytest.approx(930.0)
    tr, xr, yr, _ = read_asc(ASC_BINO, eye="right")
    assert xr[0] == pytest.approx(920.6)
    assert x[0] != pytest.approx((930.0 + 920.6) / 2)


def test_read_asc_auto_eye_prefers_the_one_with_data():
    bad_left = ASC_BINO.replace("  930.0\t  611.6", "   .\t   .").replace(
        "  931.0\t  612.6", "   .\t   .")
    _, x, _, _ = read_asc(bad_left, eye="auto")
    assert np.isfinite(x).all(), "auto should have fallen back to the right eye"


def test_read_asc_handles_gzip():
    t, _, _, _ = read_asc(gzip.compress(ASC_MONO.encode()))
    assert len(t) == 3


def test_read_asc_on_an_events_only_export_returns_no_samples():
    """ds008366 exports events without samples; that must be visible, not empty-ish."""
    events_only = "\n".join(l for l in ASC_MONO.splitlines()
                            if not _ASC_SAMPLE_TEST.match(l))
    t, x, y, msgs = read_asc(events_only)
    assert len(t) == 0 and len(x) == 0
    assert msgs, "messages should still be recovered"


_ASC_SAMPLE_TEST = __import__("re").compile(r"^\d{4,}\s")


# --------------------------------------------------------------------------
# indexed-message anchor
# --------------------------------------------------------------------------

def _pulse_events(n=100, tr=1.5, t0=5851.392, first_index=10):
    return [(t0 + (i - 1) * tr, f"TTLPulse_{i}")
            for i in range(first_index, first_index + n)]


def test_indexed_message_anchor_extrapolates_back_to_pulse_one():
    """The log starts at pulse 10; volume 0 is still located exactly."""
    tr, t0 = 1.5, 5851.392
    ev = _pulse_events(n=200, tr=tr, t0=t0, first_index=10)
    got, info = anchor_seconds(ANCHOR_INDEXED_MESSAGE, events=ev,
                               message_pattern=r"TTLPulse_(\d+)", tr=tr)
    assert got == pytest.approx(t0, abs=1e-6)
    assert info["first_index"] == 10
    assert info["median_interval"] == pytest.approx(tr, rel=1e-6)


def test_indexed_message_anchor_recovers_the_tr_as_its_slope():
    ev = _pulse_events(n=300, tr=2.0, t0=100.0, first_index=1)
    _, info = anchor_seconds(ANCHOR_INDEXED_MESSAGE, events=ev,
                             message_pattern=r"TTLPulse_(\d+)", tr=2.0)
    assert info["median_interval"] == pytest.approx(2.0, rel=1e-6)
    assert info["max_residual"] < 1e-6


def test_indexed_message_anchor_rejects_a_video_frame_pulse_train():
    """ds006642 logs a 24 Hz PULSE_ alongside the 1.5 s TTLPulse_."""
    ev = [(i * 0.042, f"PULSE_{i}") for i in range(1, 500)]
    with pytest.raises(SyncError, match="pulse spacing"):
        anchor_seconds(ANCHOR_INDEXED_MESSAGE, events=ev,
                       message_pattern=r"PULSE_(\d+)", tr=1.5)


def test_indexed_message_anchor_needs_enough_pulses():
    ev = _pulse_events(n=5)
    with pytest.raises(SyncError, match="only 5 indexed messages"):
        anchor_seconds(ANCHOR_INDEXED_MESSAGE, events=ev,
                       message_pattern=r"TTLPulse_(\d+)", tr=1.5)


def test_indexed_message_anchor_requires_a_capture_group():
    ev = _pulse_events(n=50)
    with pytest.raises(SyncError, match="indexed messages matched"):
        anchor_seconds(ANCHOR_INDEXED_MESSAGE, events=ev,
                       message_pattern=r"TTLPulse_\d+", tr=1.5)


def test_indexed_message_anchor_tolerates_jitter_but_reports_it():
    rng = np.random.default_rng(0)
    tr, t0 = 1.5, 0.0
    ev = [(t0 + (i - 1) * tr + rng.normal(0, 0.02), f"TTLPulse_{i}")
          for i in range(1, 400)]
    got, info = anchor_seconds(ANCHOR_INDEXED_MESSAGE, events=ev,
                               message_pattern=r"TTLPulse_(\d+)", tr=tr)
    assert got == pytest.approx(t0, abs=0.02)
    assert info["max_residual"] > 0


# --------------------------------------------------------------------------
# the vertical convention
#
# Three ingested datasets (ds000113, ds001242, ds004158) shipped with y negated
# because `center_and_scale`'s docstring had the corpus's vertical direction
# backwards. Nothing caught it: a lag sweep is blind to a sign error, so every
# one of them verified at lag 0 with a healthy margin while decoding at a
# *negative* vertical correlation. These pin the convention so the next config
# cannot be written against a guess.
# --------------------------------------------------------------------------

def _ingest_configs():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from fetch_eyetracking import DATASETS
    return DATASETS


def test_corpus_vertical_convention_is_y_downward():
    """No flip for a top-left tracker -- the corpus is in screen coordinates.

    Established from anatomy rather than from another dataset: the template is
    stored L, A, S and the eyeball's dark lens sits at its anterior pole, so
    looking up moves that lens to higher z. `verify_gaze_sync.py --convention`
    measures the resulting dipole and six datasets vote positive unanimously.
    """
    lab = np.ones((1, N_SUBTR, 2), dtype=np.float32)
    out = center_and_scale(lab, flip_y=False)
    assert np.allclose(out[..., 1], 1.0), "flip_y=False must leave y untouched"


@pytest.mark.parametrize("accession", ["ds006833", "ds006642", "ds004158",
                                       "ds000113", "_ds001242_excluded"])
def test_every_ingest_config_keeps_the_screen_vertical_convention(accession):
    cfg = _ingest_configs()[accession]
    assert cfg.get("flip_y", False) is False, (
        f"{accession} sets flip_y=True. Every tracker here is top-left origin "
        "and so is the corpus, so a flip inverts the dataset -- which is what "
        "happened to ds000113, ds001242 and ds004158. See the note in "
        "deepmreye/eyetracking.center_and_scale before changing this.")


def test_ds004158_reads_its_gaze_columns_y_first():
    """The TSV is (y, x, pupil, time) and nothing in the dataset says so.

    There is no root sidecar, so this override is the only column information
    available. The README's "keep fixation to a fixation dot at the screen
    center" is what settles it: read this way, 20 subjects' medians land 10 px
    and 2 px from a 1920x1080 centre; read the other way, 422 px and 430 px off.
    """
    cols = _ingest_configs()["ds004158"]["columns"]
    assert find_gaze_columns(cols) == (1, 0)


# --------------------------------------------------------------------------
# ANCHOR_EVENTS and the EDF reader
#
# Three datasets (ds001840, ds004283, ds007305) ship gaze only as EyeLink's
# binary .edf and carry no scanner pulse anywhere in the recording. What they do
# carry is stimulus messages on the tracker clock plus a BIDS events.tsv whose
# onsets are already relative to volume 0, so the origin is recovered by fitting
# one against the other. That fit is the only anchor here that validates itself,
# and these pin the two checks that make it worth trusting.
# --------------------------------------------------------------------------

def _events_anchor(t0, *, clock=1.0, jitter=0.0, n=60, seed=0):
    rng = np.random.default_rng(seed)
    onsets = np.sort(rng.uniform(0, 700, n))
    msg_t = clock * onsets + t0 + rng.normal(0, jitter, n)
    return onsets, [(t, "stim_onset") for t in msg_t]


def test_events_anchor_recovers_the_origin():
    onsets, events = _events_anchor(267.886, jitter=0.0003)
    got, info = anchor_seconds(ANCHOR_EVENTS, events=events,
                               message_pattern=r"^stim_onset$",
                               bids_onsets=onsets)
    assert got == pytest.approx(267.886, abs=0.01)
    assert info["n_events"] == 60
    assert info["clock_ratio"] == pytest.approx(1.0, abs=1e-3)


def test_events_anchor_rejects_a_clock_that_is_not_the_same_rate():
    """A slope away from 1 means the messages are not these events.

    ds001840 fails exactly here: its events.tsv onsets are the *design*, and the
    stimulus actually ran ~2.5% fast, so the fit returns a slope of 0.975. Over
    an 859 s run that is 21 s of drift -- no single origin can align it, which
    is why it is not ingested on this anchor.
    """
    onsets, events = _events_anchor(2.30, clock=0.975)
    with pytest.raises(SyncError, match="clock ratio"):
        anchor_seconds(ANCHOR_EVENTS, events=events,
                       message_pattern=r"^stim_onset$", bids_onsets=onsets)


def test_events_anchor_rejects_scattered_residuals():
    onsets, events = _events_anchor(100.0, jitter=0.5)
    with pytest.raises(SyncError, match="residual"):
        anchor_seconds(ANCHOR_EVENTS, events=events,
                       message_pattern=r"^stim_onset$", bids_onsets=onsets)


def test_events_anchor_rejects_a_mismatched_event_count():
    onsets, events = _events_anchor(10.0)
    with pytest.raises(SyncError, match="not the\n?\\s*same event list|same event list"):
        anchor_seconds(ANCHOR_EVENTS, events=events[:-3],
                       message_pattern=r"^stim_onset$", bids_onsets=onsets)


def test_events_anchor_needs_onsets():
    _, events = _events_anchor(10.0)
    with pytest.raises(SyncError, match="events.tsv onsets"):
        anchor_seconds(ANCHOR_EVENTS, events=events,
                       message_pattern=r"^stim_onset$", bids_onsets=None)


def test_read_edf_rejects_european_data_format():
    """EEG's European Data Format shares the extension and must not parse.

    Silently reading one as gaze would produce plausible-looking numbers with no
    relationship to where anyone was looking.
    """
    from deepmreye.eyetracking import read_edf
    edf_plus = b"0       " + b" " * 80 + b"EDF+C" + b"\x00" * 100
    with pytest.raises(SyncError, match="SR_RESEARCH|European Data Format"):
        read_edf(edf_plus)


def test_ds004283_uses_the_self_validating_anchor():
    DATASETS, _ = _configs()
    cfg = DATASETS["ds004283"]
    assert cfg["anchor"] == ANCHOR_EVENTS
    assert cfg.get("center") is None, (
        "ds004283 must take its screen centre from the .edf header's "
        "DISPLAY_COORDS, not from a number typed into the config -- that guess "
        "is what hid ds004158's transposed columns.")
