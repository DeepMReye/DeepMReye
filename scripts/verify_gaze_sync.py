#!/usr/bin/env python3
"""Prove that ingested gaze labels are aligned to the BOLD they were written with.

Coverage checks in ``fetch_eyetracking.py`` establish that a recording *spans*
the scan. They cannot establish that it is aligned to it: an anchor off by a
constant still yields full coverage, plausible-looking gaze, and a model that
trains. The error only shows up as a lower score, which is indistinguishable
from the dataset simply being harder.

The test that separates them is a **lag sweep**. Decode gaze from the eye block
at every shift in a window and find where the correlation peaks. The eyeball
signal is not hemodynamic -- it is the orbit moving within the imaged volume --
so a correctly aligned recording peaks at **lag 0**, with no BOLD delay to
account for. A peak at lag *k* means every label is *k* TRs off, and the sweep
says by how much.

Two things make the read honest:

- **A positive control.** The same sweep runs on the six original ``dsL0*``
  datasets, whose alignment predates this work. If the instrument does not put
  those at 0, it is not measuring what it claims and no verdict on the new data
  is worth anything.
- **A margin, not just an argmax.** Gaze is smooth, so neighbouring lags score
  nearly as well and the argmax alone overstates how determined it is. The
  margin (peak minus best competing lag at distance >= 2) is what says the peak
  is real.

    python scripts/verify_gaze_sync.py --datasets dsL07_deepmreye_calib
    python scripts/verify_gaze_sync.py --control          # the original six
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import RidgeCV

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.datasource import resolve  # noqa: E402

MAX_LAG = 8
ALPHAS = np.logspace(0, 5, 11)
# Stride-4 voxel subsample -- the published baseline's feature set. Enough to
# decode with, cheap enough to sweep 17 lags per subject.
STRIDE = 4


def features(block):
    """``[T, V]`` from ``[X, Y, Z, T]``, stride-subsampled and finite."""
    sub = block[::STRIDE, ::STRIDE, ::STRIDE, :]
    x = sub.reshape(-1, sub.shape[-1]).T.astype(np.float64)
    keep = np.isfinite(x).all(axis=0) & (x.std(axis=0) > 1e-8)
    return x[:, keep]


def per_tr_gaze(labels):
    """``[T, 2]`` -- the mean over the 10 sub-TR samples, as the probe uses."""
    with np.errstate(invalid="ignore"):
        return np.nanmean(labels.astype(np.float64), axis=1)


def decode_r(x, y, train_frac=0.6, min_rows=60):
    """Within-subject held-out mean Pearson r over the two axes."""
    ok = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    x, y = x[ok], y[ok]
    if len(x) < min_rows:
        return np.nan
    cut = int(len(x) * train_frac)
    if cut < 20 or len(x) - cut < 20:
        return np.nan
    if y[:cut].std(axis=0).min() < 1e-9:
        return np.nan
    model = RidgeCV(alphas=ALPHAS).fit(x[:cut], y[:cut])
    pred = model.predict(x[cut:])
    rs = []
    for j in range(y.shape[1]):
        if y[cut:, j].std() > 1e-9 and pred[:, j].std() > 1e-9:
            rs.append(np.corrcoef(pred[:, j], y[cut:, j])[0, 1])
    return float(np.mean(rs)) if rs else np.nan


def sweep(block, labels, max_lag=MAX_LAG):
    """``{lag: r}``. Positive lag = labels shifted later than the volumes."""
    x_all = features(block)
    y_all = per_tr_gaze(labels)
    n = min(len(x_all), len(y_all))
    x_all, y_all = x_all[:n], y_all[:n]
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            x, y = x_all[:n - lag], y_all[lag:]
        elif lag < 0:
            x, y = x_all[-lag:], y_all[:n + lag]
        else:
            x, y = x_all, y_all
        out[lag] = decode_r(x, y)
    return out


def summarise(curve):
    """Peak lag, peak r, and the margin over lags at distance >= 2."""
    valid = {k: v for k, v in curve.items() if np.isfinite(v)}
    if not valid:
        return {"peak_lag": None, "peak_r": np.nan, "margin": np.nan}
    peak = max(valid, key=valid.get)
    far = [v for k, v in valid.items() if abs(k - peak) >= 2]
    return {"peak_lag": int(peak), "peak_r": float(valid[peak]),
            "margin": float(valid[peak] - max(far)) if far else np.nan}


def run(data_dir, datasets, per_dataset, max_lag):
    rows = []
    for ds in datasets:
        paths = sorted((Path(data_dir) / ds).glob("*.h5"))[:per_dataset]
        if not paths:
            print(f"[!] {ds}: nothing on disk")
            continue
        print(f"\n[*] {ds}  ({len(paths)} participants)")
        curves = []
        for p in paths:
            with h5py.File(p, "r") as f:
                if "labels" not in f:
                    continue
                block, labels = f["eye_block"][...], f["labels"][...]
            c = sweep(block, labels, max_lag)
            s = summarise(c)
            curves.append(c)
            rows.append({"dataset": ds, "subject": p.stem, **s,
                         "curve": {str(k): (None if not np.isfinite(v) else round(v, 4))
                                   for k, v in c.items()}})
            flag = "" if s["peak_lag"] == 0 else f"   <-- PEAK AT {s['peak_lag']}"
            print(f"    {p.stem:<24} peak lag {str(s['peak_lag']):>3}  "
                  f"r {s['peak_r']:+.3f}  margin {s['margin']:+.3f}{flag}")

        if curves:
            lags = sorted(curves[0])
            mean = {l: float(np.nanmean([c[l] for c in curves])) for l in lags}
            ms = summarise(mean)
            print(f"    {'MEAN':<24} peak lag {str(ms['peak_lag']):>3}  "
                  f"r {ms['peak_r']:+.3f}  margin {ms['margin']:+.3f}")
            print("      " + "  ".join(f"{l:+d}:{mean[l]:+.2f}" for l in lags
                                       if abs(l) <= 4))
            verdict = "PASS" if ms["peak_lag"] == 0 else "FAIL"
            print(f"      verdict: {verdict}")
            rows.append({"dataset": ds, "subject": "__mean__", **ms,
                         "verdict": verdict,
                         "curve": {str(k): round(v, 4) for k, v in mean.items()}})
    return rows


def sub_tr_sweep(data_dir, corpus_name, n_subjects=4, span=1.5, step=0.25):
    """Locate the onset of volume 0 to finer than one TR.

    The integer sweep can only say "lag 0 wins". When the profile is lopsided --
    ds000113 scores +0.46 one TR early against +0.12 one TR late -- the true
    origin is probably not exactly on a volume boundary, and no amount of
    integer resolution will say so. This re-bins the *raw* gaze at fractional
    offsets through the ingest's own code path (``time_offset``), so what is
    measured is the shipped pipeline rather than a reimplementation of it.

    A peak away from 0 here does not invalidate the labels -- they are stored
    per TR and a sub-TR shift cannot be represented differently -- but it says
    how much accuracy the discretisation costs.
    """
    import h5py as _h5
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fetch_eyetracking import DATASETS, build_labels, nifti_dims  # noqa: E402
    from deepmreye.pipeline import make_s3_client

    accession = next((k for k, c in DATASETS.items()
                      if c["corpus_name"] == corpus_name), None)
    if accession is None:
        print(f"[!] {corpus_name} has no ingest config; cannot re-bin")
        return []
    cfg = DATASETS[accession]
    s3 = make_s3_client()
    offsets = np.round(np.arange(-span, span + 1e-9, step), 3)

    paths = sorted((Path(data_dir) / corpus_name).glob("*.h5"))[:n_subjects]
    # The sweep overrides cfg["time_offset"], so what it reports is the
    # *absolute* best offset, not a residual on top of the configured one. A
    # correctly configured dataset therefore peaks at its own configured value,
    # not at zero -- print both so that is never read the wrong way round.
    configured = float(cfg.get("time_offset", 0.0))
    print(f"\n[*] sub-TR sweep {corpus_name} ({len(paths)} participants, "
          f"{offsets[0]:+.2f}..{offsets[-1]:+.2f}s step {step})"
          f"\n    configured offset {configured:+.2f}s -- the sweep reports "
          f"absolute offsets, so a correct config peaks here")
    per_subject = []
    for p in paths:
        with _h5.File(p, "r") as f:
            block = f["eye_block"][...]
            et_key = f.attrs["eyetracking_key"]
            bold_key = f.attrs["source_key"]
            tr = float(f.attrs["repetition_time"])
        try:
            n_trs, _ = nifti_dims(s3, bold_key)
        except Exception as e:
            print(f"    {p.stem}: {str(e)[:60]}")
            continue
        x_all = features(block)
        row = {}
        for off in offsets:
            try:
                labels, _ = build_labels(s3, {**cfg, "time_offset": float(off)},
                                         et_key, n_trs, tr)
            except Exception:
                row[float(off)] = np.nan
                continue
            y = per_tr_gaze(labels)
            n = min(len(x_all), len(y))
            row[float(off)] = decode_r(x_all[:n], y[:n])
        per_subject.append({"subject": p.stem, "curve": row})
        best = max((v, k) for k, v in row.items() if np.isfinite(v))[1]
        print(f"    {p.stem:<24} best offset {best:+.2f}s  r {row[best]:+.3f}")

    if per_subject:
        mean = {o: float(np.nanmean([s["curve"][o] for s in per_subject]))
                for o in per_subject[0]["curve"]}
        best = max(mean, key=lambda k: (mean[k] if np.isfinite(mean[k]) else -9))
        print(f"    {'MEAN':<24} best offset {best:+.2f}s  r {mean[best]:+.3f}")
        print("      " + "  ".join(f"{o:+.2f}:{mean[o]:+.2f}" for o in sorted(mean)))
    return per_subject


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--sub-tr", nargs="*", default=None, metavar="DATASET",
                   help="Re-bin raw gaze at fractional offsets to locate the "
                        "onset of volume 0 finer than one TR.")
    p.add_argument("--sub-tr-subjects", type=int, default=4)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--control", action="store_true",
                   help="Also sweep the six original labeled datasets, whose "
                        "alignment is independent of this ingest.")
    p.add_argument("--per-dataset", type=int, default=5)
    p.add_argument("--max-lag", type=int, default=MAX_LAG)
    p.add_argument("--out", default="results/gaze_sync_verification.json")
    args = p.parse_args()

    data_dir = Path(args.data_dir or resolve(None, download=False, quiet=True))

    if args.sub_tr:
        for ds in args.sub_tr:
            sub_tr_sweep(data_dir, ds, n_subjects=args.sub_tr_subjects)
        return 0

    datasets = list(args.datasets or [])
    if args.control or not datasets:
        datasets = sorted({d.name for d in data_dir.glob("dsL0[1-6]*")}) + datasets
    print(f"[*] corpus {data_dir}")

    rows = run(data_dir, datasets, args.per_dataset, args.max_lag)
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"\n[+] wrote {out}")

    bad = [r for r in rows if r["subject"] == "__mean__" and r.get("verdict") == "FAIL"]
    if bad:
        print("\n[!] MISALIGNED: " + ", ".join(
            f"{r['dataset']} (peak {r['peak_lag']})" for r in bad))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
