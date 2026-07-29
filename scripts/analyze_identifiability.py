#!/usr/bin/env python3
"""How much gaze is recoverable from a single run, with no cross-run transfer?

This is an **analysis, not a baseline**, and the distinction matters. The
estimator here is fitted on the first half of the very run it is scored on, and
it still needs labels on that half to decide which canonical variate is the
horizontal axis, which is the vertical, and what sign each carries. Nothing
about it could be deployed on a new subject with no eye tracker -- which is the
entire problem. Do not put it in the baseline table; it answers a different
question:

    Is the cross-dataset gap a *representation* failure or a *readout transfer*
    failure?

The argument it tests. Left and right orbit crops are two views of one latent.
Gaze is conjugate, so it is shared between them; anatomy is not shared; local
noise, ghosting and susceptibility artefacts are largely not shared. Two views
of a shared latent with independent per-view noise is the textbook setting where
CCA is the *identifying* estimator. So if the top canonical variates track gaze
with no labels used in the fit, the gaze latent is recoverable per run, and
anything lost across datasets was lost in the readout, not the representation.

Three arms, all scored on the **second half** of each run:

  cca    unsupervised. CCA between the left and right orbit. The variate index
         and its sign are chosen on the fit half only.
  ridge  supervised, within run. The per-run ceiling.
  null   the control. Identical protocol on circularly-shifted gaze, which
         preserves the autocorrelation but destroys the alignment. This is what
         says the best-of-N variate selection is not manufacturing the effect.

Compare the output against the cross-dataset numbers from
``scripts/eval_probe.py --protocol dataset``.

    python scripts/analyze_identifiability.py --data-dir data
    python scripts/analyze_identifiability.py --data-dir data --null

The eye block is stored as ``np.concatenate((right, left))`` along X, so
``X[:24]`` is the right orbit and ``X[24:]`` the left -- confirmed by the
occupancy dip at X=23-24 (95 and 52 live voxels against 200+ either side).
"""
import argparse
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.datasource import resolve

# The X index the two orbits are concatenated at. Fixed by the mask crop.
X_SPLIT = 24

# Number of canonical components. This is not a free knob to tune for a nice
# number -- it was measured. At 3 components the horizontal-gaze variate is
# simply not in the set for several datasets (dsL02 r_x 0.209, dsL05 0.374);
# at 8 the same datasets give 0.917 and 0.871. The shared latent between the
# orbits is not one-dimensional, and gaze is not its leading direction.
N_CC = 8
N_PCA = 20
MIN_TR = 120


def views(block, x_split=X_SPLIT):
    """Right and left orbit as ``[T, n_voxels]``, live voxels only."""
    out = []
    for part in (block[:x_split], block[x_split:]):
        flat = part.reshape(-1, part.shape[-1]).T
        live = flat.std(axis=0) > 1e-6
        out.append(flat[:, live])
    return out


def motion_proxy(block):
    """Stand-in for realignment parameters, which the corpus does not store.

    Rigid head motion translates the whole crop coherently, so the mean signal
    over all orbit voxels and its temporal derivative capture some of it. This
    is markedly weaker than true 6-DOF parameters and is a stated limitation of
    this analysis, not a fix: a canonical variate could still be partly motion.
    Ablating it changes the result very little, which is itself ambiguous --
    either motion is not dominant, or the proxy is too weak to remove it.
    """
    flat = block.reshape(-1, block.shape[-1]).T
    live = flat.std(axis=0) > 1e-6
    g = flat[:, live].mean(axis=1)
    return np.column_stack([g, np.gradient(g)])


def regress_out(x, confounds):
    """Project the confound subspace out of every column of ``x``."""
    c = np.column_stack([np.ones(len(x)), confounds])
    beta, *_ = np.linalg.lstsq(c, x, rcond=None)
    return x - c @ beta


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or np.std(a[ok]) < 1e-9 or np.std(b[ok]) < 1e-9:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def run_subject(path, deconfound=True, null=False, n_cc=N_CC, rng=None):
    with h5py.File(path, "r") as f:
        block = f["eye_block"][:]
        gaze = np.nanmean(f["labels"][:], axis=1)          # [T, 2]

    t = block.shape[-1]
    if t < MIN_TR:
        return None
    right, left = views(block)
    if right.shape[1] < N_PCA or left.shape[1] < N_PCA:
        return None

    if null:
        # Circular shift: keeps each axis's autocorrelation and marginal exactly,
        # destroys only its alignment to the BOLD. Applied before the split, so
        # the shifted series is used for *both* variate selection and scoring --
        # otherwise the control would be easier than the real thing.
        rng = rng or np.random.default_rng(0)
        gaze = np.roll(gaze, int(rng.integers(t // 4, 3 * t // 4)), axis=0)

    if deconfound:
        conf = motion_proxy(block)
        right, left = regress_out(right, conf), regress_out(left, conf)

    half = t // 2
    fit, test = slice(0, half), slice(half, t)

    pr, pl = PCA(N_PCA).fit(right[fit]), PCA(N_PCA).fit(left[fit])
    rf, lf = pr.transform(right[fit]), pl.transform(left[fit])
    rt, lt = pr.transform(right[test]), pl.transform(left[test])

    out = {}

    # --- unsupervised: CCA between the two orbits
    cca = CCA(n_components=n_cc, max_iter=1000).fit(rf, lf)
    uf, vf = cca.transform(rf, lf)
    ut, vt = cca.transform(rt, lt)
    # Average the two views' variates: both are estimates of the same shared
    # latent, so averaging halves the independent per-view noise.
    var_fit, var_test = (uf + vf) / 2, (ut + vt) / 2

    for axis in (0, 1):
        scores = [corr(var_fit[:, k], gaze[fit][:, axis]) for k in range(n_cc)]
        if not np.any(np.isfinite(scores)):
            continue
        best = int(np.nanargmax(np.abs(scores)))
        sign = np.sign(scores[best]) or 1.0
        out[f"cca_{'xy'[axis]}"] = corr(sign * var_test[:, best], gaze[test][:, axis])

    # --- supervised within run: the per-run ceiling
    both_fit = np.column_stack([rf, lf])
    both_test = np.column_stack([rt, lt])
    g_fit = gaze[fit]
    ok = np.isfinite(g_fit).all(axis=1)
    if ok.sum() >= N_PCA:
        pred = Ridge(alpha=1.0).fit(both_fit[ok], g_fit[ok]).predict(both_test)
        for axis in (0, 1):
            out[f"ridge_{'xy'[axis]}"] = corr(pred[:, axis], gaze[test][:, axis])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--n-cc", type=int, default=N_CC)
    p.add_argument("--max-subjects", type=int, default=15,
                   help="Per dataset, for runtime. 0 = all.")
    p.add_argument("--raw", action="store_true", help="Skip the confound regression.")
    p.add_argument("--null", action="store_true",
                   help="Control: circularly shift gaze, destroying alignment.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    root = Path(resolve(args.data_dir, download=False, quiet=True))
    rng = np.random.default_rng(args.seed)

    mode = "NULL CONTROL (gaze circularly shifted)" if args.null else "real gaze"
    print(f"Within-run identifiability -- {mode}")
    print(f"CCA components {args.n_cc}, confound regression "
          f"{'off' if args.raw else 'on'}, fit on first half, scored on second\n")
    print(f"{'dataset':<24} {'n':>4} {'CCA r_x':>9} {'CCA r_y':>9} "
          f"{'ridge r_x':>10} {'ridge r_y':>10}")
    print("-" * 72)

    for ds_dir in sorted(root.glob("dsL*")):
        paths = sorted(ds_dir.glob("*.h5"))
        if args.max_subjects:
            paths = paths[: args.max_subjects]
        rows = []
        for path in paths:
            try:
                r = run_subject(path, not args.raw, args.null, args.n_cc, rng)
            except Exception as e:
                print(f"  [!] {path.name}: {e.__class__.__name__}: {e}")
                continue
            if r:
                rows.append(r)
        if not rows:
            print(f"{ds_dir.name:<24} {0:>4}   (no usable runs)")
            continue

        def med(key):
            vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
            return np.median(np.abs(vals)) if vals else np.nan

        print(f"{ds_dir.name:<24} {len(rows):>4} {med('cca_x'):>9.3f} {med('cca_y'):>9.3f} "
              f"{med('ridge_x'):>10.3f} {med('ridge_y'):>10.3f}")

    print("\nMedian |r| across subjects. Compare against "
          "`eval_probe.py --protocol dataset`:\nif within-run r is high where "
          "cross-dataset r is low, that dataset is a transfer\nfailure, not a data failure.")


if __name__ == "__main__":
    main()
