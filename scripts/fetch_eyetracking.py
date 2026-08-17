#!/usr/bin/env python3
"""Ingest OpenNeuro datasets that ship eye tracking alongside BOLD.

The labeled half of this corpus was six datasets. It does not have to be: a scan
of all 2409 OpenNeuro accessions found 382 participants across 18 datasets with a
gaze recording paired to a functional run. This script turns the usable ones into
``dsL##_<name>`` participants in the corpus's own format, so they are
indistinguishable from the original six downstream.

Why that matters more than the participant count: every leave-one-dataset-out
claim in this project currently rests on **six** folds, and the temporal-envelope
law on twelve (dataset, axis) cells. Independent acquisitions are the scarce
resource, not subjects.

Each dataset needs a configuration below because **the time origin is not
discoverable automatically**. Getting it wrong shifts every label by a constant,
which is close to invisible: the labels still look like gaze and the decoder
still trains, it just scores lower. See ``deepmreye/eyetracking.py`` for the
three anchor strategies and why reading ``StartTime`` blindly is not one of them.

    python scripts/fetch_eyetracking.py --list
    python scripts/fetch_eyetracking.py --dataset ds006833 --limit 2 --dry-run
    python scripts/fetch_eyetracking.py --dataset ds006833

``--dry-run`` does everything except the BOLD download and registration: it
resolves the anchor, bins the gaze and reports coverage. Run it first. It costs
seconds per subject and catches every sync problem that is knowable without the
imaging data.
"""
import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling scripts
from deepmreye.eyetracking import (  # noqa: E402
    ANCHOR_INDEXED_MESSAGE,
    ANCHOR_MESSAGE,
    ANCHOR_STARTTIME,
    ANCHOR_TRIGGER,
    SyncError,
    anchor_seconds,
    bin_to_subtr,
    center_and_scale,
    consistency,
    find_gaze_columns,
    column_index,
    load_sidecar,
    read_asc,
    read_physio_events,
    read_tsv,
)
from deepmreye.pipeline import BUCKET_NAME, make_s3_client  # noqa: E402
from deepmreye.storage import is_intact, subject_path, write_subject  # noqa: E402

# --------------------------------------------------------------------------
# Per-dataset configuration.
#
# `center` is the screen middle in the tracker's own coordinate units. It only
# shifts the labels, so an imperfect value costs nothing measurable -- Pearson r
# is translation invariant and every readout fits an intercept. `degrees_per_unit`
# is left None wherever the dataset does not actually document enough geometry to
# derive it (viewing distance without physical screen size is not enough), and
# `label_units` then says "pixel" rather than pretending otherwise.
# --------------------------------------------------------------------------
DATASETS = {
    "ds006833": dict(
        corpus_name="dsL07_deepmreye_calib",
        # A calibration protocol run for DeepMReye itself: fixation, pursuit and
        # free viewing. The companion `DeepMReyeClosed` task is excluded -- it is
        # deliberately eyes-closed, so there is no gaze to decode.
        et_pattern=r"/func/.*_task-DeepMReyeCalib_.*_recording-eye1_physio\.tsv\.gz$",
        events_suffix="_recording-eye1_physioevents.tsv.gz",
        anchor=ANCHOR_MESSAGE,
        message_pattern=r"mri_trigger",
        timestamp_col="timestamp",
        time_scale=1e-3,            # EyeLink clock is milliseconds
        center=(960.0, 540.0),      # CalibrationPosition centre -> 1920x1080
        flip_y=False,               # Match corpus vertical convention
        degrees_per_unit=18.0 / 1080.0, # Kling et al. 2026: 18 dva square calibration over 1080px active area
        label_units="degrees_visual_angle",
        valid_box=(-400.0, 2320.0, -400.0, 1480.0),
    ),
    "ds006642": dict(
        corpus_name="dsL11_backtothefuture",
        # EyeLink ASCII rather than BIDS physio: samples and messages sit in one
        # file, so the anchor and the gaze are on the same clock by construction.
        et_pattern=r"/sourcedata/.*_task-backtothefuture_eyelinkraw\.asc\.gz$",
        # The recordings live under `sourcedata/` and order their entities the
        # other way round (`run-003_task-x`) from the BOLD (`task-x_run-003`),
        # so the default side-by-side rule cannot reach the volume.
        bold_rewrite=[
            (r"^ds006642/sourcedata/", "ds006642/"),
            (r"_run-(\d+)_task-([A-Za-z0-9]+)_eyelinkraw\.asc\.gz$",
             r"_task-\2_run-\1_bold.nii.gz"),
        ],
        anchor=ANCHOR_INDEXED_MESSAGE,
        # NOT `PULSE_`: that is the 24 Hz video frame counter (42 ms spacing,
        # ~60000 per run). `TTLPulse_` is the volume trigger at 1490 ms against
        # a 1.5 s TR, and its count matches the volume count exactly (1608
        # pulses for a 1608-volume run), which validates the run pairing too.
        message_pattern=r"TTLPulse_(\d+)",
        timestamp_col="asc",        # unused for .asc; times come from the samples
        time_scale=1e-3,
        center=(960.0, 600.0),      # GAZE_COORDS 0 0 1919 1199
        flip_y=False,
        degrees_per_unit=28.9 / 1920.0, # NNDb-3T+: 28.9 dva horizontal display extent over 1920px
        label_units="degrees_visual_angle",
        valid_box=(-400.0, 2320.0, -400.0, 1600.0),
    ),
    "ds004158": dict(
        corpus_name="dsL12_rest",
        # Szinte et al. 2022 (INT Marseille): Fast multi-band TR = 0.80s resting state.
        # Screen: 77.3 x 44.5 cm at 1.2m viewing distance (35.71 dva over 1920px).
        et_pattern=r"/func/.*_task-rest_.*_recording-eye1_physio\.tsv\.gz$",
        events_suffix="_recording-eye1_physioevents.tsv.gz",
        anchor=ANCHOR_INDEXED_MESSAGE,
        message_pattern=r"TR num (\d+) onset",
        timestamp_col="timestamp",
        columns=["x_coordinate", "y_coordinate", "pupil_size", "timestamp"],
        time_scale=1e-3,            # Milliseconds
        center=(960.0, 540.0),      # ScreenResolution: 1920x1080
        flip_y=True,                # EnvironmentCoordinates: 'top-left'
        degrees_per_unit=35.71 / 1920.0, # 35.71 dva horizontal extent
        label_units="degrees_visual_angle",
        valid_box=(-400.0, 2320.0, -400.0, 1480.0),
    ),
    "_ds007532_excluded": dict(
        corpus_name="dsX10_visseq_unaligned",
        et_pattern=r"/func/.*_recording-eye1_physio\.tsv\.gz$",
        # NOT `starttime`, despite every run having one. This dataset mixes both
        # conventions run by run -- sub-01 alone has proper offsets (-12.27,
        # -7.38, -15.62) on some runs and raw tracker clocks (2351691, 1331388,
        # 3200988) on others -- and even the plausible-looking values are wrong:
        # the offsets actually used spanned -89.6 to -0.7 s, which would have
        # the tracker stopping a minute before the scan ended. Anchoring on them
        # produced per-subject peak lags scattered from -5 to +4 and a FAILED
        # verdict, which is how the problem was found.
        #
        # `TRIGGER SENT` in the physioevents is right and self-checking: the two
        # occurrences per run bracket the scan (468.75 s apart against a 470 s
        # acquisition), and the first sits 21-35 s into the recording, which
        # places the run comfortably inside it.
        anchor=ANCHOR_MESSAGE,
        message_pattern=r"TRIGGER SENT",
        events_suffix="_recording-eye1_physioevents.tsv.gz",
        timestamp_col="timestamp",
        time_scale=1e-3,
        center=(512.0, 384.0),      # ScreenAOIDefinition circle at (384,384) r384
        flip_y=True,
        degrees_per_unit=None,
        label_units="pixel",
        valid_box=(-600.0, 1700.0, -600.0, 1700.0),
    ),
}

# Configs keyed with a leading underscore failed verification and are kept only
# as documentation. They are not offered on the command line: an excluded
# dataset must not be re-ingested because someone tab-completed it.
ACTIVE = {k: v for k, v in DATASETS.items() if not k.startswith("_")}

# Coverage below this fraction of the scan means the recording does not actually
# span the run; those participants are reported and skipped rather than written
# with mostly-NaN labels.
MIN_COVERAGE = 0.60


def s3_list(s3, prefix, pattern):
    pg = s3.get_paginator("list_objects_v2")
    rx = re.compile(pattern)
    out = []
    for page in pg.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        for o in page.get("Contents", []):
            if rx.search(o["Key"]):
                out.append(o["Key"])
    return sorted(out)


def s3_get(s3, key):
    return s3.get_object(Bucket=BUCKET_NAME, Key=key)["Body"].read()


def bold_for(et_key, cfg=None):
    """The BOLD run an eye-tracking file belongs to.

    The default assumes the two sit side by side and differ only in suffix,
    which is true for well-formed BIDS. ``cfg["bold_rewrite"]`` overrides that
    with an ordered list of ``(pattern, replacement)`` applied to the whole key
    -- ds006642 keeps its recordings under ``sourcedata/`` *and* orders the
    entities differently (``run-003_task-x`` against the BOLD's
    ``task-x_run-003``), so no suffix rule can reach it.
    """
    if cfg and cfg.get("bold_rewrite"):
        key = et_key
        for pattern, repl in cfg["bold_rewrite"]:
            key = re.sub(pattern, repl, key)
        return key
    name = et_key.split("/")[-1]
    stem = re.sub(r"_recording-[A-Za-z0-9-]+_physio\.tsv(\.gz)?$", "", name)
    stem = re.sub(r"_eyetrack\w*\.(tsv|tsv\.gz|asc)$", "", stem)
    return et_key.rsplit("/", 1)[0] + "/" + stem + "_bold.nii.gz"


def nifti_dims(s3, key):
    """``(nvol, tr)`` from a .nii.gz header via a ranged read (no full download)."""
    import gzip
    import io
    import struct

    raw = s3.get_object(Bucket=BUCKET_NAME, Key=key, Range="bytes=0-65535")["Body"].read()
    hdr = gzip.GzipFile(fileobj=io.BytesIO(raw)).read(352)
    if len(hdr) < 352:
        raise ValueError("short NIfTI header")
    endian = "<" if struct.unpack("<i", hdr[:4])[0] == 348 else ">"
    dim = struct.unpack(endian + "8h", hdr[40:56])
    pixdim = struct.unpack(endian + "8f", hdr[76:108])
    units = struct.unpack(endian + "B", hdr[123:124])[0] & 0x38
    tr = float(pixdim[4])
    if units == 16:      # milliseconds
        tr /= 1000.0
    elif units == 24:    # microseconds
        tr /= 1e6
    return int(dim[4]), tr


def build_labels(s3, cfg, et_key, n_trs, tr):
    """Gaze for one run as ``[n_trs, 10, 2]``, plus a record of how it was aligned."""
    sidecar = {}
    # BIDS inheritance: merge run-level, task-level, and root-level sidecars
    parts = et_key.split("/")
    ds_root = parts[0]
    task_m = re.search(r"task-([A-Za-z0-9]+)", et_key)
    task_name = task_m.group(1) if task_m else ""
    rec_m = re.search(r"recording-([A-Za-z0-9]+)", et_key)
    rec_name = rec_m.group(1) if rec_m else ""

    cand_sidecars = [
        cfg.get("sidecar_key"),
        re.sub(r"\.tsv\.gz$", ".json", et_key),
        re.sub(r"\.tsv$", ".json", et_key),
        f"{ds_root}/task-{task_name}_recording-{rec_name}_physio.json" if task_name and rec_name else None,
        f"{ds_root}/task-{task_name}_physio.json" if task_name else None,
    ]
    for cand in cand_sidecars:
        if not cand:
            continue
        try:
            loaded = load_sidecar(s3_get(s3, cand))
            if isinstance(loaded, dict):
                for k, v in loaded.items():
                    if k not in sidecar:
                        sidecar[k] = v
        except Exception:
            continue

    scale = float(cfg.get("time_scale", 1.0))
    trigger, events, from_column = None, None, True

    if et_key.endswith((".asc", ".asc.gz")):
        # EyeLink ASCII: samples and messages come from the same file, so the
        # anchor's events and the gaze are guaranteed to be on one clock.
        raw_t, x, y, messages = read_asc(s3_get(s3, et_key), eye=cfg.get("eye", "auto"))
        if not len(raw_t):
            raise SyncError("no samples in the .asc (events-only export)")
        times = raw_t * scale
        events = [(t, m) for t, m in messages]
    else:
        columns = cfg.get("columns") or sidecar.get("Columns")
        arr, columns = read_tsv(s3_get(s3, et_key), columns=columns)
        if columns is None:
            raise SyncError("no column names in sidecar or header")

        gaze = find_gaze_columns(columns)
        if gaze is None:
            raise SyncError(f"no gaze columns among {columns}")
        ix, iy = gaze
        x = arr[:, ix].astype(np.float64)
        y = arr[:, iy].astype(np.float64)

        tcol = cfg.get("timestamp_col")
        if tcol is None:
            fs = float(sidecar.get("SamplingFrequency") or cfg["sampling_frequency"])
            times = np.arange(len(arr), dtype=np.float64) / fs
            from_column = False
        else:
            it = column_index(columns, tcol)
            if it is None:
                raise SyncError(f"timestamp column {tcol!r} not in {columns}")
            times = arr[:, it].astype(np.float64) * scale

        if cfg.get("trigger_col"):
            itr = column_index(columns, cfg["trigger_col"])
            if itr is None:
                raise SyncError(
                    f"trigger column {cfg['trigger_col']!r} not in {columns}")
            trigger = arr[:, itr]

    if events is None and cfg["anchor"] in (ANCHOR_MESSAGE, ANCHOR_INDEXED_MESSAGE):
        ev_key = re.sub(r"_recording-.*$", "", et_key) + cfg["events_suffix"]
        events = read_physio_events(s3_get(s3, ev_key))
    if events is not None and scale != 1.0:
        # Put the messages on the same clock as `times`, which was already
        # scaled. Mixing a millisecond message with a second-scale sample stream
        # is a 1000x error that looks like a wildly out-of-range anchor.
        events = [(t * scale, m) for t, m in events]

    t0, info = anchor_seconds(
        cfg["anchor"], sidecar=sidecar, times=times, trigger=trigger,
        events=events, message_pattern=cfg.get("message_pattern"), tr=tr,
        times_from_column=from_column, n_trs=n_trs)
    # `t0` needs no scaling here: the messages were converted to the same clock
    # as `times` before the anchor ran. Scaling again would multiply the origin
    # by 1000 for a millisecond tracker -- an error large enough that coverage
    # would catch it, but only by accident.
    #
    # `time_offset` moves the assumed onset of volume 0 later by that many
    # seconds. It exists so the sub-TR sweep in `verify_gaze_sync.py --sub-tr`
    # can locate an offset finer than the integer lag sweep can resolve, using
    # this exact code path rather than a reimplementation.
    rel = times - t0 - float(cfg.get("time_offset", 0.0))

    x = np.asarray(x, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64).copy()
    box = cfg.get("valid_box")
    if box:
        xmin, xmax, ymin, ymax = box
        off = (x < xmin) | (x > xmax) | (y < ymin) | (y > ymax)
        x[off] = np.nan
        y[off] = np.nan

    ok, cov = consistency(rel, n_trs, tr)
    labels = bin_to_subtr(rel, x, y, n_trs=n_trs, tr=tr)
    labels = center_and_scale(labels, center=cfg.get("center"),
                              flip_y=cfg.get("flip_y", False),
                              degrees_per_unit=cfg.get("degrees_per_unit"))

    info.update(cov)
    info["coverage_ok"] = bool(ok)
    info["nan_fraction"] = float(np.isnan(labels).mean())
    info["n_samples"] = int(len(times))
    return labels, info


def extract_block(s3, bold_key, masks):
    """Download, coregister and extract one run's eye block."""
    from deepmreye.pipeline import DEFAULT_TRANSFORMS
    from deepmreye.preprocess import normalize_img, run_participant

    eyemask_small, eyemask_big, dme_template, x_edges, y_edges, z_edges = masks
    with tempfile.TemporaryDirectory() as tmp:
        local = str(Path(tmp) / bold_key.split("/")[-1])
        s3.download_file(BUCKET_NAME, bold_key, local)
        masked_eye, _, _ = run_participant(
            fp_func=local, dme_template=dme_template,
            eyemask_big=eyemask_big, eyemask_small=eyemask_small,
            x_edges=x_edges, y_edges=y_edges, z_edges=z_edges,
            replace_with=0, transforms=DEFAULT_TRANSFORMS,
            save_path=tmp, as_pickle=False, save_overview=False,
            dataset_name="run", save_blocks=False, thumbnail_path=None)
    return normalize_img(np.asarray(masked_eye, dtype=np.float32))


def relabel_dataset(ds, out_dir, subjects=None):
    """Recompute labels in place, reusing the eye blocks already on disk.

    Registration is the expensive half and it does not depend on the gaze at
    all, so a change to the alignment (a ``time_offset``, a coordinate
    convention) must not cost another ANTs run. Each file's own attrs record
    which recording it was built from, so the re-derivation uses exactly the
    same inputs as the original write.
    """
    cfg = DATASETS[ds]
    s3 = make_s3_client()
    paths = sorted((Path(out_dir) / cfg["corpus_name"]).glob("*.h5"))
    if subjects:
        paths = [p for p in paths if p.stem in set(subjects)]
    print(f"[*] relabelling {len(paths)} participants of {cfg['corpus_name']} "
          f"(time_offset {cfg.get('time_offset', 0.0):+.2f}s)")

    done = 0
    for p in paths:
        with h5py.File(p, "r") as f:
            block = f["eye_block"][...]
            attrs = dict(f.attrs)
        et_key, bold_key = attrs["eyetracking_key"], attrs["source_key"]
        tr = float(attrs["repetition_time"])
        try:
            n_trs, _ = nifti_dims(s3, bold_key)
            labels, info = build_labels(s3, cfg, et_key, n_trs, tr)
        except Exception as e:
            print(f"    [!] {p.stem}: {str(e)[:100]}")
            continue

        n_have = block.shape[-1]
        if labels.shape[0] > n_have:
            labels = labels[:n_have]
        elif labels.shape[0] < n_have:
            pad = np.full((n_have - labels.shape[0],) + labels.shape[1:], np.nan,
                          dtype=np.float32)
            labels = np.concatenate([labels, pad], axis=0)

        # Refresh everything the alignment decided, not just the labels. An
        # earlier version updated only coverage, so a file relabelled under a
        # new anchor still advertised the old one in `gaze_anchor` -- provenance
        # that silently described a run that no longer existed.
        attrs.update({
            "gaze_time_offset": float(cfg.get("time_offset", 0.0)),
            "gaze_anchor": info["anchor"],
            "gaze_anchor_detail": json.dumps(
                {k: v for k, v in info.items()
                 if k in ("start_time", "message", "n_pulses", "median_interval")}),
            "gaze_coverage": float(info["covered_fraction"]),
            "gaze_nan_fraction": float(info["nan_fraction"]),
            "label_units": cfg["label_units"],
        })
        for k in ("format_version", "n_trs", "has_labels"):
            attrs.pop(k, None)
        write_subject(p, block, labels=labels, attrs=attrs)
        done += 1
    print(f"[+] relabelled {done}")
    return done


def run_dataset(ds, out_dir, limit=None, dry_run=False, force=False, subjects=None,
                register=True):
    cfg = DATASETS[ds]
    s3 = make_s3_client()
    print(f"[*] {ds} -> {cfg['corpus_name']}   anchor={cfg['anchor']}")

    et_keys = s3_list(s3, f"{ds}/", cfg["et_pattern"])
    print(f"[*] {len(et_keys)} eye-tracking runs")

    # One participant file per subject: take the first run that aligns cleanly.
    by_sub = {}
    for k in et_keys:
        m = re.search(r"(sub-[A-Za-z0-9]+)", k)
        if m:
            by_sub.setdefault(m.group(1), []).append(k)
    if subjects:
        by_sub = {s: v for s, v in by_sub.items() if s in set(subjects)}
    subs = sorted(by_sub)[:limit] if limit else sorted(by_sub)
    print(f"[*] {len(subs)} subjects\n")

    masks = None
    if not dry_run:
        from deepmreye.preprocess import get_masks
        m = get_masks()
        masks = (m[0], m[1], m[2], m[4], m[5], m[6])

    report, written, skipped = [], 0, 0
    for i, sub in enumerate(subs, 1):
        out_path = subject_path(out_dir, cfg["corpus_name"], sub)
        if not force and not dry_run and is_intact(out_path):
            print(f"  [{i}/{len(subs)}] {sub}: already extracted")
            continue

        chosen = None
        for et_key in by_sub[sub]:
            bold_key = bold_for(et_key, cfg)
            try:
                n_trs, tr = nifti_dims(s3, bold_key)
            except Exception as e:
                report.append({"subject": sub, "run": et_key, "status": "no_bold",
                               "error": str(e)[:80]})
                continue
            try:
                labels, info = build_labels(s3, cfg, et_key, n_trs, tr)
            except SyncError as e:
                report.append({"subject": sub, "run": et_key, "status": "sync_error",
                               "error": str(e)[:140]})
                print(f"  [{i}/{len(subs)}] {sub}: SYNC {e}")
                continue
            info.update(subject=sub, run=et_key, bold=bold_key, n_trs=n_trs, tr=tr)
            if info["covered_fraction"] < MIN_COVERAGE:
                info["status"] = "low_coverage"
                report.append(info)
                continue
            chosen = (et_key, bold_key, labels, info, n_trs, tr)
            break

        if chosen is None:
            skipped += 1
            print(f"  [{i}/{len(subs)}] {sub}: no usable run")
            continue

        et_key, bold_key, labels, info, n_trs, tr = chosen
        print(f"  [{i}/{len(subs)}] {sub}: {n_trs} TRs @ {tr}s  "
              f"coverage {info['covered_fraction']:.2f}  "
              f"NaN {info['nan_fraction']:.2f}  ({info['anchor']})")

        if dry_run:
            info["status"] = "dry_run_ok"
            report.append(info)
            continue

        t0 = time.time()
        try:
            block = extract_block(s3, bold_key, masks)
        except Exception as e:
            info.update(status="extract_failed", error=str(e)[:200])
            report.append(info)
            print(f"      [!] extraction failed: {str(e)[:120]}")
            continue

        # The extracted block is the authority on length: registration can drop
        # volumes. Labels are truncated or NaN-padded to match rather than the
        # write being allowed to fail on a length mismatch.
        n_have = block.shape[-1]
        if labels.shape[0] > n_have:
            labels = labels[:n_have]
        elif labels.shape[0] < n_have:
            pad = np.full((n_have - labels.shape[0],) + labels.shape[1:], np.nan,
                          dtype=np.float32)
            labels = np.concatenate([labels, pad], axis=0)

        write_subject(out_path, block, labels=labels, attrs={
            "dataset": cfg["corpus_name"], "subject": sub,
            "normalized": True, "source_key": bold_key,
            "eyetracking_key": et_key, "repetition_time": float(tr),
            "label_units": cfg["label_units"],
            "gaze_anchor": info["anchor"],
            "gaze_anchor_detail": json.dumps(
                {k: v for k, v in info.items()
                 if k in ("start_time", "message", "n_pulses", "median_interval")}),
            "gaze_coverage": float(info["covered_fraction"]),
            "gaze_nan_fraction": float(info["nan_fraction"]),
        })
        info.update(status="written", seconds=round(time.time() - t0, 1))
        report.append(info)
        written += 1
        print(f"      wrote {out_path.name}  block {block.shape}  "
              f"{time.time() - t0:.0f}s")

    out = Path("results") / f"eyetracking_ingest_{ds}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"dataset": ds, "config_name": cfg["corpus_name"],
                               "written": written, "skipped": skipped,
                               "runs": report}, indent=1, default=str))
    print(f"\n[+] {written} written, {skipped} skipped -> {out}")

    if written and not dry_run and register:
        # Same path the original six went through, so the new datasets are
        # registry citizens on identical terms: approved, flagged `labeled` so
        # the audit grid cannot mark ground truth as no-eyes, and mirrored into
        # labels.csv. Gaze was recorded during the scan, so the eyeballs are in
        # frame by construction -- this is a statement about the data, not a
        # model output gating anything.
        from convert_labeled_to_h5 import register as register_labeled  # noqa: E402

        subs_written = [r["subject"] for r in report if r.get("status") == "written"]
        registry = Path(out_dir) / "datasets.h5"
        if registry.exists():
            n = register_labeled(out_dir, {cfg["corpus_name"]: subs_written})
            print(f"[+] registered {n} subjects in {registry.name}")
        else:
            print(f"[!] no registry at {registry} -- skipped registration")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", choices=sorted(ACTIVE))
    p.add_argument("--data-dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve sync and bin gaze, but do not touch BOLD.")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-register", action="store_true",
                   help="Write participant files but leave datasets.h5 alone.")
    p.add_argument("--labels-only", action="store_true",
                   help="Recompute labels for participants already extracted, "
                        "reusing their eye blocks. Use after changing an "
                        "alignment; registration does not depend on the gaze.")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if args.list or not args.dataset:
        print(f"{'accession':<12}{'corpus name':<28}{'anchor':<12}units")
        for k, c in sorted(ACTIVE.items()):
            print(f"{k:<12}{c['corpus_name']:<28}{c['anchor']:<12}{c['label_units']}")
        return

    from deepmreye.datasource import resolve
    data_dir = args.data_dir or resolve(None, download=False, quiet=True)
    print(f"[*] corpus {data_dir}")
    if args.labels_only:
        relabel_dataset(args.dataset, data_dir, subjects=args.subjects)
        return
    run_dataset(args.dataset, data_dir, limit=args.limit, dry_run=args.dry_run,
                force=args.force, subjects=args.subjects,
                register=not args.no_register)


if __name__ == "__main__":
    main()
