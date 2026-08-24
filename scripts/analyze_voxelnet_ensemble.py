"""Does the voxel network carry gaze information the linear incumbent does not?

The single-model comparison ("net 0.75 against incumbent 0.75") cannot answer this: two
models can tie and still be right about different participants, different timepoints, or
different axes. What settles it is whether COMBINING them beats either alone, and whether
the residuals are correlated.

Reads the `.npz` files written by `train_voxelnet_scratch.py --save-preds` (one per seed,
each holding `<subject>|net`, `<subject>|inc` and `<subject>|lab`) and scores, with the same
`subject_scores` both arms are scored by:

  * each arm alone,
  * the net ensembled across seeds (attacks the ~0.03 seed SD),
  * net + incumbent,
  * the seed ensemble + incumbent.

Streams are z-scored per participant and per output column before averaging. Pearson r is
invariant to that, so it cannot flatter any arm; it only stops a difference in predicted
SCALE from deciding the mixture.
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepmreye.temporal_probe import fold_median, subject_scores


def zc(a):
    """Per-column z-score over time; constant columns pass through un-scaled."""
    mu = a.mean(axis=0, keepdims=True)
    sd = a.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (a - mu) / sd



def _score_group(files, weights):
    packs = [dict(np.load(f)) for f in files]
    subs = sorted({k.split("|")[0] for k in packs[0] if k.endswith("|lab")})
    inc, ens, per, mix = [], [], [[] for _ in packs], {w: [] for w in weights}
    for s in subs:
        lab = packs[0][f"{s}|lab"]
        i_p = zc(packs[0][f"{s}|inc"])
        nets = [zc(pk[f"{s}|net"]) for pk in packs]
        for j, n_p in enumerate(nets):
            per[j].append(subject_scores(n_p, lab)[0])
        inc.append(subject_scores(i_p, lab)[0])
        n_ens = zc(np.mean(nets, axis=0))
        ens.append(subject_scores(n_ens, lab)[0])
        for w in weights:
            mix[w].append(subject_scores(w * n_ens + (1 - w) * i_p, lab)[0])
    return (fold_median(inc), [fold_median(p) for p in per], fold_median(ens),
            {w: fold_median(v) for w, v in mix.items()}, len(subs))


def by_fold(args):
    """One row per fold, then the median over folds -- the only quotable aggregate."""
    files = sorted(glob.glob(args.preds))
    groups = {}
    for f in files:
        groups.setdefault(Path(f).stem.rsplit("_seed", 1)[0], []).append(f)
    if not groups:
        raise SystemExit(f"[!] nothing matched {args.preds}")
    hdr = f"{'fold':<26} {'n':>3} {'incumb':>8} {'best 1':>8} {'ens':>8} " + \
          " ".join(f"{'mix' + str(w):>8}" for w in args.weights)
    print(hdr)
    print("-" * len(hdr))
    I, E, S, M = [], [], [], {w: [] for w in args.weights}
    for name in sorted(groups):
        if len(groups[name]) < 2:
            print(f"{name:<26} only {len(groups[name])} seed(s) -- skipped")
            continue
        i_r, per, e_r, mix, n = _score_group(groups[name], args.weights)
        I.append(i_r); E.append(e_r); S.append(max(per))
        for w in args.weights: M[w].append(mix[w])
        print(f"{name:<26} {n:>3} {i_r:>8.4f} {max(per):>8.4f} {e_r:>8.4f} " +
              " ".join(f"{mix[w]:>8.4f}" for w in args.weights))
    print("-" * len(hdr))
    print(f"{'MEDIAN over folds':<26} {'':>3} {np.median(I):>8.4f} {np.median(S):>8.4f} "
          f"{np.median(E):>8.4f} " + " ".join(f"{np.median(M[w]):>8.4f}" for w in args.weights))
    print(f"\nensemble - incumbent, per fold: "
          f"{[round(e - i, 4) for e, i in zip(E, I)]}")
    print(f"folds won by the ensemble: {sum(e > i for e, i in zip(E, I))}/{len(I)}")
    print("\n`best 1` is the single best seed per fold and is NOT an achievable arm -- it is")
    print("chosen with the test score. It is shown only to bound what seed luck is worth.")
    return 0

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--preds", default="results/subtr/preds/best_seed*.npz")
    p.add_argument("--weights", type=float, nargs="*", default=[0.25, 0.5, 0.75],
                   help="Net weight w in w*net + (1-w)*incumbent.")
    p.add_argument("--by-fold", action="store_true",
                   help="Group files by the name before `_seed` and report one row per fold "
                        "plus the 9-fold median, which is the only quotable number.")
    args = p.parse_args()

    if args.by_fold:
        return by_fold(args)

    files = sorted(glob.glob(args.preds))
    if not files:
        raise SystemExit(f"[!] nothing matched {args.preds}")
    packs = [dict(np.load(f)) for f in files]
    print(f"[*] {len(files)} seed(s): {[Path(f).stem for f in files]}")

    subs = sorted({k.split("|")[0] for k in packs[0] if k.endswith("|lab")})
    print(f"[*] {len(subs)} test participants\n")

    per_seed = [[] for _ in packs]
    inc, ens, mix = [], [], {w: [] for w in args.weights}
    ens_mix = {w: [] for w in args.weights}
    resid_r = []
    for s in subs:
        lab = packs[0][f"{s}|lab"]
        i_p = zc(packs[0][f"{s}|inc"])
        nets = [zc(pk[f"{s}|net"]) for pk in packs]
        for j, n_p in enumerate(nets):
            per_seed[j].append(subject_scores(n_p, lab)[0])
        inc.append(subject_scores(i_p, lab)[0])
        n_ens = zc(np.mean(nets, axis=0))
        ens.append(subject_scores(n_ens, lab)[0])
        for w in args.weights:
            mix[w].append(subject_scores(w * nets[0] + (1 - w) * i_p, lab)[0])
            ens_mix[w].append(subject_scores(w * n_ens + (1 - w) * i_p, lab)[0])
        # How much of what the net gets is NOT in the incumbent. The residual must be taken
        # against a target on the SAME scale as the predictions: both prediction streams are
        # z-scored, so comparing them to raw-unit labels makes every residual approximately
        # -y and correlates them at ~1.0 by construction, which says nothing about the models.
        t = min(len(i_p), len(lab))
        y = lab[:t].reshape(t, 20)
        ok = np.isfinite(y).all(axis=1)
        if ok.sum() > 10:
            y_z = zc(y[ok])
            rn = (nets[0][:t][ok] - y_z).ravel()
            ri = (i_p[:t][ok] - y_z).ravel()
            good = np.isfinite(rn) & np.isfinite(ri)
            if good.sum() > 10:
                resid_r.append(float(np.corrcoef(rn[good], ri[good])[0, 1]))

    print(f"{'arm':<34} {'median r':>9}")
    print("-" * 45)
    print(f"{'incumbent (lr-cca:32 + lags -> ridge)':<34} {fold_median(inc):>9.4f}")
    for j, f in enumerate(files):
        print(f"{'net  ' + Path(f).stem:<34} {fold_median(per_seed[j]):>9.4f}")
    print(f"{f'net  {len(packs)}-seed ensemble':<34} {fold_median(ens):>9.4f}")
    print()
    for w in args.weights:
        print(f"{f'mix  {w:.2f}*net(seed0) + {1-w:.2f}*inc':<34} {fold_median(mix[w]):>9.4f}")
    for w in args.weights:
        print(f"{f'mix  {w:.2f}*ens + {1-w:.2f}*inc':<34} {fold_median(ens_mix[w]):>9.4f}")
    if resid_r:
        print(f"\nmean residual correlation net vs incumbent: {np.mean(resid_r):.4f}")
        print("(1.0 = the two make the same errors and an ensemble cannot help;")
        print(" lower = genuinely complementary information)")


if __name__ == "__main__":
    main()
