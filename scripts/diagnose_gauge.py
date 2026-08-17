#!/usr/bin/env python3
"""Step 0 go/no-go: is there a fixed corpus component that tracks gaze?

The zero-label plan rests on one assumption, and this tests it before anything is
built. `analyze_identifiability.py` recovers gaze per run at |r| ~ 0.75 with no
labels in the fit -- but it refits CCA on every run, and CCA is invariant to
permuting and negating its components, so *which* variate is horizontal and what
sign it carries has to come from somewhere. Today it comes from labels, and the
script reports `median(|r|)`, which hides that.

The way out is that the **frozen corpus basis has no gauge freedom**. It is one
fixed set of filters applied to everybody, so component `j` means the same thing
in every participant and its sign is fixed by construction. If some component
tracks gaze consistently across subjects *and* datasets, the gauge is solved for
free and a genuinely zero-label decoder exists:

    gaze_x_hat(t) = sign * project(lr-cca, corpus_basis, voxels)[t, j]

with no fitting of any kind on the target participant.

So this measures, per component and per gaze axis, the **signed** correlation
across participants -- signed, because sign consistency is the whole question.
A component with |r| 0.8 that flips sign between subjects is worthless; a
component with r 0.5 and the same sign everywhere is a method.

Reports three things per (component, axis):

    mean r      signed, across participants. Consistency shows up here.
    mean |r|    the upper bound the sign costs us. mean r << mean |r| means the
                component tracks gaze but flips, i.e. the gauge is NOT free.
    sign rate   fraction of participants agreeing with the majority sign.

    python scripts/diagnose_gauge.py --max-subjects 12
    python scripts/diagnose_gauge.py --basis results/scaling/basis_n25.npz
"""
import argparse
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.datasource import resolve
from deepmreye.unsupervised import load_basis, project

MIN_TR = 120
# Datasets deliberately kept out of every number on this project: `dsL11` is a
# pending ingest that has never passed verify_gaze_sync.py.
EXCLUDE = ("dsL11_backtothefuture",)


def run_variates(path, mask_flat, basis, k, null=False, rng=None):
    """Corpus lr-cca variates and gaze for one participant, or None.

    ``null`` circularly shifts the gaze trace, which preserves its
    autocorrelation and its marginal exactly and destroys only its alignment to
    the BOLD. That is the control this diagnostic cannot do without: gaze and
    eye-block signals are both strongly autocorrelated, so the effective degrees
    of freedom are far below ``T`` and a correlation of 0.5 between two slow
    series is not by itself evidence of anything.
    """
    with h5py.File(path, "r") as f:
        if "labels" not in f:
            return None
        block = f["eye_block"][:]
        gaze = np.nanmean(f["labels"][:], axis=1)          # [T, 2]
    t = block.shape[-1]
    if t < MIN_TR or not np.isfinite(gaze).any():
        return None
    if null:
        rng = rng or np.random.default_rng(0)
        gaze = np.roll(gaze, int(rng.integers(t // 4, 3 * t // 4)), axis=0)
    x = block.reshape(-1, t).T[:, mask_flat].astype(np.float64)   # [T, 14236]
    return project("lr-cca", basis, x, k=k), gaze


def signed_corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or np.std(a[ok]) < 1e-9 or np.std(b[ok]) < 1e-9:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--basis", default="results/scaling/basis_n1039.npz")
    p.add_argument("--k", type=int, default=32,
                   help="Corpus components examined. 32 is the best-scoring "
                        "lr-cca budget at this corpus size.")
    p.add_argument("--max-subjects", type=int, default=12,
                   help="Per dataset, for runtime. 0 = all.")
    p.add_argument("--top", type=int, default=6, help="Components printed.")
    p.add_argument("--null", action="store_true",
                   help="Control: circularly shift each participant's gaze.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    root = Path(resolve(args.data_dir, download=False, quiet=True))
    mask, bases, meta = load_basis(args.basis)
    mask_flat = mask.reshape(-1)
    basis = bases["lr-cca"]
    rng = np.random.default_rng(args.seed)
    mode = "NULL CONTROL (gaze circularly shifted)" if args.null else "real gaze"
    print(f"[*] basis {Path(args.basis).name}: {meta['n_subjects']} subjects, "
          f"{meta['datasets']} datasets, k={args.k} -- {mode}")

    per_ds = {}
    for ds_dir in sorted(root.glob("dsL*")):
        if ds_dir.name in EXCLUDE:
            continue
        paths = sorted(ds_dir.glob("*.h5"))
        if args.max_subjects:
            paths = paths[: args.max_subjects]
        rows = []
        for path in paths:
            try:
                got = run_variates(path, mask_flat, basis, args.k, args.null, rng)
            except Exception as e:
                print(f"  [!] {path.name}: {e.__class__.__name__}: {e}")
                continue
            if got is None:
                continue
            var, gaze = got
            rows.append(np.array([[signed_corr(var[:, j], gaze[:, ax])
                                   for ax in (0, 1)]
                                  for j in range(var.shape[1])]))
        if rows:
            per_ds[ds_dir.name] = np.stack(rows)        # [n_sub, k, 2]
            print(f"  {ds_dir.name:<26}{len(rows):>4} participants")

    if not per_ds:
        raise SystemExit("[!] no usable participants")

    allr = np.concatenate(list(per_ds.values()))         # [N, k, 2]
    print(f"\n[*] {len(allr)} participants across {len(per_ds)} datasets\n")

    for ax, name in enumerate(("gaze x (horizontal)", "gaze y (vertical)")):
        m = np.nanmean(allr[:, :, ax], axis=0)
        ma = np.nanmean(np.abs(allr[:, :, ax]), axis=0)
        maj = np.sign(m)
        rate = np.nanmean(np.sign(allr[:, :, ax]) == maj[None, :], axis=0)
        order = np.argsort(-np.abs(m))[: args.top]
        print(f"=== {name}: best components by |mean signed r|")
        print(f"{'comp':>5}{'mean r':>9}{'mean |r|':>10}{'sign rate':>11}"
              f"{'   per-dataset mean r'}")
        for j in order:
            per = "  ".join(f"{ds[3:6]}:{np.nanmean(v[:, j, ax]):+.2f}"
                            for ds, v in per_ds.items())
            print(f"{j:>5}{m[j]:>+9.3f}{ma[j]:>10.3f}{rate[j]:>11.2f}   {per}")
        print()

    # The verdict the plan's Step 0 asks for, stated as a number rather than
    # left to the reader: a gauge is free exactly when the signed mean is close
    # to the absolute mean, i.e. subjects agree on the sign.
    print("=== verdict")
    for ax, name in enumerate(("x", "y")):
        m = np.nanmean(allr[:, :, ax], axis=0)
        ma = np.nanmean(np.abs(allr[:, :, ax]), axis=0)
        j = int(np.nanargmax(np.abs(m)))
        rate = np.nanmean(np.sign(allr[:, j, ax]) == np.sign(m[j]))
        print(f"  gaze {name}: best fixed component {j}, signed r {m[j]:+.3f}, "
              f"|r| {ma[j]:.3f}, sign agreement {rate:.1%}")
    print("\nA fixed component with high signed r and >90% sign agreement means "
          "the gauge is\nfree and a zero-label decoder exists. Signed r far "
          "below |r| means it flips\nbetween participants and the gauge must be "
          "recovered some other way.")

    # The component above was chosen by looking at every dataset, including the
    # ones its number is then quoted on -- 32 candidates against 7 datasets is
    # mild selection, but it is selection. Choosing it on the *other* datasets
    # and scoring the held-out one is the honest form, and it is also exactly
    # how the method would be deployed: the component index is decided once,
    # elsewhere, and then applied to a study nobody has labels for.
    print("\n=== leave-one-dataset-out gauge (component chosen on the others)")
    print(f"{'held-out dataset':<26}{'comp x':>7}{'r_x':>8}{'comp y':>8}{'r_y':>8}"
          f"{'  agrees with pooled'}")
    pooled = [int(np.nanargmax(np.abs(np.nanmean(allr[:, :, ax], axis=0))))
              for ax in (0, 1)]
    lodo = {0: [], 1: []}
    for held in per_ds:
        others = np.concatenate([v for ds, v in per_ds.items() if ds != held])
        cells = []
        for ax in (0, 1):
            m = np.nanmean(others[:, :, ax], axis=0)
            j = int(np.nanargmax(np.abs(m)))
            sign = np.sign(m[j]) or 1.0
            # Signed by the *training* datasets' sign, then scored held out.
            r = float(np.nanmean(sign * per_ds[held][:, j, ax]))
            lodo[ax].append(r)
            cells.append((j, r))
        same = all(c[0] == pooled[ax] for ax, c in enumerate(cells))
        print(f"{held:<26}{cells[0][0]:>7}{cells[0][1]:>8.3f}"
              f"{cells[1][0]:>8}{cells[1][1]:>8.3f}   {'yes' if same else 'NO'}")
    print(f"{'mean':<26}{'':>7}{np.mean(lodo[0]):>8.3f}{'':>8}"
          f"{np.mean(lodo[1]):>8.3f}")


if __name__ == "__main__":
    main()
