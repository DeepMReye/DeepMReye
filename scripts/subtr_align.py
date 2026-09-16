#!/usr/bin/env python3
"""Is the corpus aligned to gaze at sub-TR resolution, or only at TR resolution?

`scripts/slice_time.py` set out to find slice timing in the z axis and found
something else. Every z band -- and the pooled basis -- predicts the **late**
gaze samples inside a TR better than the early ones, by a clean monotone ramp
(0.92 -> 1.03 of a band's own mean), and the effect does **not** order with z.
So it is not a slice sweep. It is a global offset: the eye block lines up with
gaze slightly later than the label convention puts it.

That matters because the sync verification this corpus passed
(`verify_gaze_sync.py`) works at **TR** resolution. A sub-TR misalignment is
invisible to it by construction -- exactly the same blind spot that let three
datasets ship with a flipped y axis, where "every lag scores the same
magnitude" hid the error.

Two things are measured here, in this order, because the first can produce the
second as an artefact:

`--control`  The same profile at `lags0`. With `lags+-1` the readout sees the
             volumes at `t-1`, `t` and `t+1`, so it can predict late samples of
             TR `t` from the volume at `t+1` and the ramp would be a property of
             the lag stack rather than of the alignment. If the ramp survives at
             `lags0`, it is real.

`--sweep`    Shift the unfolded `[T*10, 2]` gaze trace by `d` sub-TR samples and
             re-score. A peak away from `d = 0` is a measured misalignment.
             Reported with nested selection -- choose `d` on eight folds, read
             the ninth -- because this file has twice recorded a tuned peak that
             did not survive that.

Note what a shifted target is and is not: the same gaze trace re-assigned to
TRs, so the comparison across `d` is of *alignments*, not of models. Read a peak
as "this is where the data says the volumes sit", not as an accuracy gain.

    python scripts/subtr_align.py --voxels <voxdir> --control --sweep
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmreye import probe  # noqa: E402
from scripts.improve_lrcca import (  # noqa: E402
    Corpus, Eigs, cca_to_block, fit_cca, project, slicer, summarise,
)
from scripts.slice_time import sub_tr_profile  # noqa: E402

SHIFTS = [-4, -3, -2, -1, 0, 1, 2, 3, 4]


def shift_labels(labels, d):
    """Move the gaze trace `d` sub-TR samples along its own unfolded time axis.

    Unfold `[T, 10, 2]` to `[T * 10, 2]` first: shifting inside a TR without
    unfolding would wrap samples between neighbouring TRs and call it a lag.
    Edge padding, for the reason `probe.make_lags` uses it -- zeros would inject
    a jump to the origin at the two rows a readout sees as extreme.
    """
    if d == 0:
        return labels
    t = len(labels)
    flat = np.asarray(labels, dtype=np.float32).reshape(t * 10, 2)
    if d > 0:
        flat = np.concatenate([flat[d:], np.repeat(flat[-1:], d, axis=0)])
    else:
        flat = np.concatenate([np.repeat(flat[:1], -d, axis=0), flat[:d]])
    return flat.reshape(t, 10, 2)


def nested(rows, folds, res):
    """Choose the shift on eight folds, read the ninth."""
    picked, vals, base = [], [], []
    for h in folds:
        others = [f for f in folds if f != h]
        best = max(rows, key=lambda d: np.mean([rows[d]["folds"][f][res]
                                                for f in others]))
        picked.append(best)
        vals.append(rows[best]["folds"][h][res])
        base.append(rows[0]["folds"][h][res])
    return np.array(vals), np.array(base), picked


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--voxels", required=True)
    p.add_argument("--moments", default="results/lrcca_variants/moments_n2000.npz")
    p.add_argument("--out", default="results/lrcca_variants/subtr_align.json")
    p.add_argument("--control", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--k", type=int, default=32)
    a = p.parse_args()

    corpus = Corpus(a.moments)
    li, ri = corpus.orbits()
    n_vox = int(corpus.mask.sum())
    fit = fit_cca(corpus.cov("total"), li, ri, 64, 256, 1e-3, eigs=Eigs(), key="total")
    blocks = [("full",) + cca_to_block(fit, n_vox, corpus.mu) + (64,)]
    del corpus._cache
    recs, index = project(a.voxels, blocks)
    got = {}

    if a.control:
        print("[*] r(within-TR sample), normalised -- is the ramp the lag stack?")
        for lags in (0, 1):
            r = np.array(sub_tr_profile(recs, slicer(index, "full", k=a.k, lags=lags)))
            got[f"profile_lags{lags}"] = r.tolist()
            w = np.clip(r - np.nanmin(r), 0, None)
            c = float((w * np.arange(10)).sum() / max(w.sum(), 1e-9))
            row = "".join(f"{v:>7.3f}" for v in r / np.nanmean(r))
            print(f"  lags{lags}  {row}   centroid {c:.2f}", flush=True)
        Path(a.out).write_text(json.dumps(got, indent=1))

    if a.sweep:
        original = {id(r): r["labels"] for r in recs}
        rows = {}
        print(f"\n{'shift (sub-TR samples)':<26}{'sub-TR':>9}{'1-TR':>9}")
        for d in SHIFTS:
            for r in recs:
                r["labels"] = shift_labels(original[id(r)], d)
            rows[d] = summarise(probe.lodo(
                recs, slicer(index, "full", k=a.k, lags=1)))
            got[f"shift{d}"] = rows[d]
            print(f"{d:<26}{rows[d]['subtr']:>9.4f}{rows[d]['1tr']:>9.4f}", flush=True)
            Path(a.out).write_text(json.dumps(got, indent=1))
        for r in recs:
            r["labels"] = original[id(r)]

        folds = sorted(rows[0]["folds"])
        print()
        for res in ("subtr", "1tr"):
            vals, base, picked = nested(rows, folds, res)
            d = vals - base
            print(f"  nested {res:<6} median {np.median(vals):.4f} "
                  f"(d {np.median(vals) - np.median(base):+.4f})  mean d {d.mean():+.4f}"
                  f"  folds {int((d > 0).sum())}/{len(folds)}  chosen {picked}",
                  flush=True)
            got[f"nested_{res}"] = {"median": float(np.median(vals)),
                                    "mean_delta": float(d.mean()),
                                    "folds": int((d > 0).sum()),
                                    "chosen": [int(x) for x in picked]}
        Path(a.out).write_text(json.dumps(got, indent=1))
    print(f"\n[+] {a.out}")


if __name__ == "__main__":
    main()
