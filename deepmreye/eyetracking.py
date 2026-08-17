"""Align BIDS eye-tracking recordings to BOLD volumes and bin them to ``[T, 10, 2]``.

This is the ingest path for gaze labels that already exist on OpenNeuro. The
corpus stores gaze as ``labels[T, 10, 2]`` -- ``T`` TRs, 10 sub-TR samples, x and
y -- so the job here is to turn a continuous eye-tracker stream into that array
against the right volume.

**The whole problem is the time origin.** A tracker samples on its own clock and
that clock has no fixed relationship to the scanner's. Getting it wrong shifts
every label by a constant and is nearly invisible downstream: the labels still
look like plausible gaze, the decoder still trains, it just scores lower. So the
origin is never assumed -- it is recovered by one of three explicit strategies,
recorded in the output attrs, and checked (:func:`consistency`) against the
scan's own duration:

- :data:`ANCHOR_STARTTIME` -- the BIDS-compliant case. ``StartTime`` in the
  sidecar is seconds from the first volume's onset to the first sample, negative
  when the tracker started first. ds007532 does this correctly (``-12.27``).
- :data:`ANCHOR_TRIGGER` -- a column carrying the scanner pulse. The first
  rising edge is volume 0. ds001242 does this, and the pulse train is its own
  verification: the median inter-pulse interval must equal the TR.
- :data:`ANCHOR_MESSAGE` -- a message logged into the tracker stream when the
  scanner triggered, found in a BIDS ``physioevents`` file. ds006833 logs
  ``trial 1 mri_trigger val = -8``.

Note what is *not* a strategy: reading ``StartTime`` when it holds a raw tracker
clock. ds006833 and ds005166 both put the tracker's own first timestamp there,
which is self-referential and carries no sync information at all. Taking it at
face value would silently place volume 0 at the start of the recording -- which
for ds006833 is 58.5 s early. :func:`anchor_seconds` raises rather than guess.

Units are recorded, never invented. Only ds001242 ships a complete geometry
(``degreePerPixel``); the rest give a viewing distance but no physical screen
size, which is not enough to reach degrees of visual angle. Those datasets are
stored in their native screen units with ``label_units`` saying so. Pearson r --
the corpus's headline metric -- is invariant to that affine, and cross-dataset
R^2 was already established to be unidentifiable, so nothing downstream is lost
by refusing to fabricate a conversion.
"""
import gzip
import io
import json
import re

import numpy as np

# Sub-TR samples per volume. Fixed by the corpus layout, not a tuning knob.
N_SUBTR = 10

ANCHOR_STARTTIME = "starttime"
ANCHOR_TRIGGER = "trigger"
ANCHOR_MESSAGE = "message"
ANCHOR_INDEXED_MESSAGE = "indexed_message"
ANCHORS = (ANCHOR_STARTTIME, ANCHOR_TRIGGER, ANCHOR_MESSAGE, ANCHOR_INDEXED_MESSAGE)

# EyeLink writes blinks and track loss as 0 or as a large sentinel; some exports
# use the uint32 max. Anything outside a generous screen box is not a gaze
# position. Kept deliberately wide -- real overshoots past the screen edge are
# genuine samples and the calibration can legitimately place gaze off-screen.
SENTINELS = (0.0, 4294967295.0, -32768.0)
PLAUSIBLE_ABS = 1e4


class SyncError(ValueError):
    """The recording could not be placed on the scanner's clock."""


def read_tsv(blob, columns=None):
    """Parse a BIDS ``.tsv`` / ``.tsv.gz`` payload into ``(array, columns)``.

    BIDS physio files are headerless with column names in the sidecar, but
    several datasets ship a header row anyway (ds005166). Both are accepted:
    a first row that fails to parse as numbers is treated as the header and
    overrides ``columns``.
    """
    if isinstance(blob, (bytes, bytearray)):
        if blob[:2] == b"\x1f\x8b":
            blob = gzip.decompress(blob)
        text = blob.decode("utf8", "replace")
    else:
        text = blob

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty tsv")

    header = lines[0].split("\t")
    has_header = any(re.search(r"[A-Za-z]{2}", c) and c.strip().lower() != "n/a"
                     for c in header)
    if has_header:
        columns = [c.strip() for c in header]
        lines = lines[1:]

    arr = np.genfromtxt(io.StringIO("\n".join(lines)), delimiter="\t",
                        missing_values=("n/a", "N/A", ""), filling_values=np.nan)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1) if len(lines) == 1 else arr.reshape(-1, 1)
    return arr, columns


def column_index(columns, *names):
    """Index of the first of ``names`` present in ``columns`` (case-insensitive)."""
    if not columns:
        return None
    low = [str(c).strip().lower() for c in columns]
    for n in names:
        if n.lower() in low:
            return low.index(n.lower())
    return None


def find_gaze_columns(columns):
    """``(ix, iy)`` for the gaze coordinate columns, or ``None``.

    Prefers the monocular BIDS names, then the binocular ``eye1_*`` form
    (ds005166 records both eyes; eye1 is taken and eye2 ignored rather than
    averaged, because the two can disagree wildly during track loss on one eye).
    """
    for xs, ys in (("x_coordinate", "y_coordinate"),
                   ("eye1_x_coordinate", "eye1_y_coordinate"),
                   ("x", "y"), ("gaze_x", "gaze_y"), ("X", "Y")):
        ix, iy = column_index(columns, xs), column_index(columns, ys)
        if ix is not None and iy is not None:
            return ix, iy
    return None


def clean_gaze(x, y):
    """Blank blinks, track loss and sentinels to NaN, in place-safe fashion."""
    x = np.asarray(x, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64).copy()
    bad = ~np.isfinite(x) | ~np.isfinite(y)
    for s in SENTINELS:
        bad |= (x == s) & (y == s)
    bad |= (np.abs(x) > PLAUSIBLE_ABS) | (np.abs(y) > PLAUSIBLE_ABS)
    x[bad] = np.nan
    y[bad] = np.nan
    return x, y


def trigger_onsets(trigger, times):
    """Times of rising edges in a scanner-pulse column."""
    t = np.asarray(trigger, dtype=np.float64)
    hi = (np.nan_to_num(t) > 0.5).astype(np.int8)
    idx = np.flatnonzero(np.diff(hi) == 1) + 1
    if hi[0] == 1:
        idx = np.r_[0, idx]
    return np.asarray(times, dtype=np.float64)[idx]


def anchor_seconds(strategy, *, sidecar=None, times=None, trigger=None,
                   events=None, message_pattern=None, tr=None,
                   times_from_column=True, n_trs=None):
    """Tracker-clock time (in the units of ``times``) of the first volume's onset.

    Returns ``(t0, info)``. ``t0`` is subtracted from the sample times to put
    them on the scanner's clock. ``info`` carries what the choice was based on,
    so the decision survives into the output attrs and can be audited later.

    ``times_from_column`` says whether ``times`` came from a real timestamp
    column or was synthesised as ``arange(n) / fs``. It matters only for the
    tracker-clock guard below: a synthesised grid starts at 0 by construction,
    so ``StartTime == times[0] == 0`` is the legitimate "tracker and scanner
    started together" case (ds000113) rather than the pathological one.
    """
    if strategy == ANCHOR_STARTTIME:
        st = (sidecar or {}).get("StartTime")
        if st is None:
            raise SyncError("StartTime absent from sidecar")
        if isinstance(st, (list, tuple)):
            if len(st) != 1:
                raise SyncError(f"StartTime is a list of {len(st)}; ambiguous")
            st = st[0]
        st = float(st)
        # A StartTime that coincides with the recording's own first timestamp is
        # the tracker clock written into a field that is defined as an offset.
        # It says nothing about the scanner. This is the ds006833 / ds005166
        # failure and it must not be silently accepted.
        if (times_from_column and times is not None and len(times)
                and abs(st - float(times[0])) < 1e-6):
            raise SyncError(
                f"StartTime ({st}) equals the first sample timestamp -- it is a "
                f"raw tracker clock, not a BIDS offset, and carries no sync "
                f"information. Use a trigger or message anchor instead.")
        # The BIDS offset is in seconds and is the time of the FIRST SAMPLE
        # relative to volume 0, so volume 0 sits at first_sample_time - StartTime.
        t0 = float(times[0]) - st if times is not None and len(times) else -st
        return t0, {"anchor": ANCHOR_STARTTIME, "start_time": st}

    if strategy == ANCHOR_TRIGGER:
        if trigger is None or times is None:
            raise SyncError("trigger anchor needs a trigger column and times")
        onsets = trigger_onsets(trigger, times)
        if len(onsets) < 2:
            raise SyncError(f"only {len(onsets)} trigger edges")
        gaps = np.diff(onsets)
        info = {"anchor": ANCHOR_TRIGGER, "n_pulses": int(len(onsets)),
                "median_interval": float(np.median(gaps))}
        # The pulse train verifies itself: its period must be the TR. A mismatch
        # means the column is not the scanner pulse (or the TR is wrong), and
        # anchoring on it would be worse than not anchoring at all.
        if tr is not None and not np.isclose(info["median_interval"], tr, rtol=0.02):
            raise SyncError(
                f"trigger period {info['median_interval']:.4f}s != TR {tr}s")
        return float(onsets[0]), info

    if strategy == ANCHOR_INDEXED_MESSAGE:
        # A pulse train logged as numbered messages ("TTLPulse_10" ...
        # "TTLPulse_1608"). Better than taking the first one, for two reasons:
        # the numbering says which volume each pulse belongs to, so a log that
        # starts at 10 rather than 1 still locates volume 0; and fitting the
        # whole train gives the acquisition's own TR back as the slope, which
        # is a validation the first-pulse approach cannot do.
        if events is None or message_pattern is None:
            raise SyncError("indexed_message anchor needs events and a pattern")
        rx = re.compile(message_pattern)
        pairs = []
        for onset, msg in events:
            m = rx.search(str(msg))
            if m and m.groups():
                try:
                    pairs.append((float(onset), int(m.group(1))))
                except (ValueError, IndexError):
                    continue
        if len(pairs) < 10:
            raise SyncError(f"only {len(pairs)} indexed messages matched "
                            f"{message_pattern!r}")
        t = np.array([p[0] for p in pairs], dtype=np.float64)
        i = np.array([p[1] for p in pairs], dtype=np.float64)
        slope, intercept = np.polyfit(i, t, 1)
        resid = float(np.max(np.abs(t - (slope * i + intercept))))
        info = {"anchor": ANCHOR_INDEXED_MESSAGE, "n_pulses": len(pairs),
                "median_interval": float(slope), "max_residual": resid,
                "first_index": int(i.min()), "last_index": int(i.max())}
        if tr is not None and not np.isclose(slope, tr, rtol=0.02):
            raise SyncError(
                f"pulse spacing {slope:.4f} != TR {tr}s -- these are not volume "
                f"triggers (ds006642 also logs a 24 Hz video-frame pulse)")
        # Determine whether pulse indexing is 0-based or 1-based.
        # In ds006642, the eye-tracker logging starts at the 10th volume.
        # When 1-indexed, the first recorded pulse is 10 and last is n_trs.
        # When 0-indexed, the first recorded pulse is 9 and last is n_trs - 1.
        first_idx = int(i.min())
        max_idx = int(i.max())
        if first_idx <= 0 or first_idx == 9 or (n_trs is not None and max_idx == int(n_trs) - 1):
            first_vol_pulse_idx = 0.0
        else:
            first_vol_pulse_idx = 1.0
        info["first_vol_pulse_idx"] = float(first_vol_pulse_idx)
        return float(intercept + slope * first_vol_pulse_idx), info

    if strategy == ANCHOR_MESSAGE:
        if events is None or message_pattern is None:
            raise SyncError("message anchor needs events and a pattern")
        rx = re.compile(message_pattern)
        for onset, msg in events:
            if rx.search(str(msg)):
                return float(onset), {"anchor": ANCHOR_MESSAGE,
                                      "message": str(msg)[:120]}
        raise SyncError(f"no event matched {message_pattern!r}")

    raise ValueError(f"unknown anchor strategy {strategy!r}")


def bin_to_subtr(times, x, y, n_trs, tr, n_sub=N_SUBTR, min_samples=1):
    """Bin a gaze stream to ``[n_trs, n_sub, 2]``.

    ``times`` are seconds from the onset of volume 0. Sub-bin ``j`` of volume
    ``t`` covers ``[(t + j/n_sub) * tr, (t + (j+1)/n_sub) * tr)`` and holds the
    **mean** of the samples falling in it -- non-overlapping, so a sample is
    counted exactly once and the mean over the 10 is the mean gaze during that
    volume. That last property is what ``evaluate.probe.temporal_targets``
    assumes when it averages them.

    Bins with fewer than ``min_samples`` valid samples become NaN; the evaluation
    masks those, and inventing a value there would create labels the tracker
    never recorded.
    """
    times = np.asarray(times, dtype=np.float64)
    x, y = clean_gaze(x, y)

    n_bins = int(n_trs) * int(n_sub)
    width = float(tr) / float(n_sub)
    out = np.full((n_bins, 2), np.nan, dtype=np.float64)

    valid = np.isfinite(times) & (np.isfinite(x) | np.isfinite(y))
    if not valid.any():
        return out.reshape(int(n_trs), int(n_sub), 2).astype(np.float32)

    t, xv, yv = times[valid], x[valid], y[valid]
    # floor, so a sample exactly on a bin edge belongs to the bin it opens.
    idx = np.floor(t / width).astype(np.int64)
    keep = (idx >= 0) & (idx < n_bins)
    idx, xv, yv = idx[keep], xv[keep], yv[keep]

    for col, vals in ((0, xv), (1, yv)):
        ok = np.isfinite(vals)
        if not ok.any():
            continue
        counts = np.bincount(idx[ok], minlength=n_bins)
        sums = np.bincount(idx[ok], weights=vals[ok], minlength=n_bins)
        enough = counts >= min_samples
        out[enough, col] = sums[enough] / counts[enough]

    return out.reshape(int(n_trs), int(n_sub), 2).astype(np.float32)


def consistency(times, n_trs, tr, tol=0.10):
    """Does the recording plausibly cover the scan? ``(ok, report)``.

    Cheap, and it catches the failure that matters: an anchor off by tens of
    seconds usually leaves the recording covering only part of the run, or
    starting after it ended. ``tol`` is a fraction of the scan duration.
    """
    times = np.asarray(times, dtype=np.float64)
    scan = float(n_trs) * float(tr)
    if not len(times):
        return False, {"reason": "no samples"}
    lo, hi = float(np.nanmin(times)), float(np.nanmax(times))
    covered = max(0.0, min(hi, scan) - max(lo, 0.0))
    frac = covered / scan if scan > 0 else 0.0
    rep = {"scan_duration": scan, "record_start": lo, "record_end": hi,
           "covered_fraction": float(frac)}
    return (frac >= 1.0 - tol), rep


def center_and_scale(labels, *, degrees_per_unit=None, center=None, flip_y=False):
    """Centre gaze on the screen middle and optionally convert to degrees.

    ``center`` is the screen-centre coordinate in the tracker's own units; it is
    subtracted so 0 is straight ahead, matching the corpus convention. With
    ``EnvironmentCoordinates: top-left`` the tracker's y grows downward while the
    corpus's grows upward, which ``flip_y`` corrects -- get this wrong and the
    vertical axis decodes at a *negative* correlation, which is the one sign
    error that still looks like a working model.
    """
    out = np.asarray(labels, dtype=np.float64).copy()
    if center is not None:
        out[..., 0] -= float(center[0])
        out[..., 1] -= float(center[1])
    if flip_y:
        out[..., 1] *= -1.0
    if degrees_per_unit is not None:
        out *= float(degrees_per_unit)
    return out.astype(np.float32)


def read_physio_events(blob, columns=None):
    """``[(onset, message), ...]`` from a BIDS ``physioevents`` file."""
    if isinstance(blob, (bytes, bytearray)):
        if blob[:2] == b"\x1f\x8b":
            blob = gzip.decompress(blob)
        text = blob.decode("utf8", "replace")
    else:
        text = blob
    out = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        msg = parts[4].strip()
        if not msg or msg == "n/a":
            continue
        try:
            out.append((float(parts[0]), msg))
        except ValueError:
            continue
    return out


_ASC_SAMPLE = re.compile(r"^(\d{4,})\s")


def read_asc(blob, eye="auto"):
    """Parse an EyeLink ASCII export into ``(times, x, y, messages)``.

    ``edf2asc`` output is line-oriented: ``MSG <t> <text>`` for messages, event
    lines (``SFIX``/``EFIX``/``ESACC``/``EBLINK``...) that start with letters,
    and sample lines that start with the tracker timestamp. Only samples and
    messages are read; the event lines are redundant with the samples here.

    Monocular samples are ``t x y pupil``, binocular ``t lx ly lp rx ry rp``.
    ``eye`` selects which to return for a binocular file: ``"left"``,
    ``"right"``, or ``"auto"`` (the first eye with any valid data). The two are
    *not* averaged -- during track loss on one eye they disagree wildly, and an
    average of a good and a bad estimate is neither.

    Missing samples are written as ``.`` and become NaN. Times are returned in
    the tracker's own units (milliseconds), matching :func:`read_physio_events`,
    so a caller scales both together.
    """
    if isinstance(blob, (bytes, bytearray)):
        if blob[:2] == b"\x1f\x8b":
            blob = gzip.decompress(blob)
        text = blob.decode("utf8", "replace")
    else:
        text = blob

    times, cols, messages = [], [], []
    for line in text.splitlines():
        if line.startswith("MSG"):
            parts = line.split(None, 2)
            if len(parts) == 3:
                try:
                    messages.append((float(parts[1]), parts[2].strip()))
                except ValueError:
                    pass
            continue
        if not _ASC_SAMPLE.match(line):
            continue
        parts = line.split()
        try:
            t = float(parts[0])
        except ValueError:
            continue
        vals = []
        for p in parts[1:7]:
            if p in (".", "..."):
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(p))
                except ValueError:
                    vals.append(np.nan)
        if len(vals) < 2:
            continue
        times.append(t)
        cols.append(vals)

    if not times:
        return (np.zeros(0), np.zeros(0), np.zeros(0), messages)

    width = max(len(c) for c in cols)
    arr = np.full((len(cols), width), np.nan)
    for i, c in enumerate(cols):
        arr[i, :len(c)] = c

    # >=6 numeric fields after the timestamp means both eyes were recorded.
    binocular = width >= 6
    if not binocular:
        x, y = arr[:, 0], arr[:, 1]
    else:
        left, right = (arr[:, 0], arr[:, 1]), (arr[:, 3], arr[:, 4])
        if eye == "left":
            x, y = left
        elif eye == "right":
            x, y = right
        else:
            x, y = left if np.isfinite(left[0]).sum() >= np.isfinite(right[0]).sum() else right

    return np.asarray(times, dtype=np.float64), x, y, messages


def load_sidecar(blob):
    """Parse a JSON sidecar from bytes or str."""
    if isinstance(blob, (bytes, bytearray)):
        blob = blob.decode("utf8", "replace")
    return json.loads(blob)
