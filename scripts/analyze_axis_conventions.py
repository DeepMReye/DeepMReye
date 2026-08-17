#!/usr/bin/env python3
"""Is a failing fold decoding the wrong *axis*, or no axis at all?

``dsL03_pursuit`` transfers at r ~ 0.20 while decoding at r ~ 0.88 within its own
runs. Two very different things produce that, and the median-r table cannot tell
them apart:

- **a convention mismatch** -- this dataset's gaze x is the others' gaze y, or
  its sign is flipped. Then the cross-dataset predictor is right but pointed the
  wrong way, and the fix is a relabelling, not a model.
- **a genuine transfer failure** -- the voxel-to-gaze map really differs, and no
  permutation of the axes recovers it.

The 2x2 matrix of correlations between (pred_x, pred_y) and (true_x, true_y)
separates them. A convention mismatch puts the mass **off** the diagonal, or on
the diagonal with a negative sign; a real failure leaves the whole matrix small.

Pearson r is used throughout because it is invariant to gain, which is exactly
the nuisance ``analyze_calibration.py`` covers and which must not be confused
with this.

    python scripts/analyze_axis_conventions.py --max-windows 600
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.data.probe_dataset import ProbeDataset, dataset_folds
from deepmreye.datasource import resolve
from deepmreye.evaluate.baselines import fit_readout, predict
from deepmreye.evaluate.features import FeatureExtractor, pool_time
from deepmreye.evaluate.probe import flatten_valid_groups, temporal_targets
from deepmreye.unsupervised import corpus_mask

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_probe import cap, extract  # noqa: E402


def per_subject_corr(y, pred, subs, min_rows=20):
    """Median over participants of the 2x2 correlation matrix.

    Per participant, not pooled: pooled correlations mix between-subject offsets
    into a number that is supposed to describe within-subject decoding.
    """
    mats = []
    for s in np.unique(subs):
        m = subs == s
        if m.sum() < min_rows:
            continue
        c = np.empty((2, 2))
        for i in range(2):
            for j in range(2):
                a, b = pred[m, i], y[m, j]
                c[i, j] = (np.corrcoef(a, b)[0, 1]
                           if a.std() > 1e-9 and b.std() > 1e-9 else np.nan)
        mats.append(c)
    return np.nanmedian(np.stack(mats), axis=0) if mats else np.full((2, 2), np.nan)


def features_for(ds, args, mask, desc):
    loader = DataLoader(cap(ds, args.max_windows), batch_size=args.batch_size,
                        num_workers=args.num_workers)
    ex = {"fold": FeatureExtractor("raw", stride=args.voxel_stride)}
    n_t = args.window_size // args.temp_patch_size
    got = extract(loader, n_t, ex, desc)
    if got is None:
        return None
    return flatten_valid_groups(got[0]["fold"], got[1], got[2], got[3])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--window-size", type=int, default=100)
    p.add_argument("--temp-patch-size", type=int, default=5)
    p.add_argument("--voxel-stride", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-components", type=int, default=32)
    p.add_argument("--max-windows", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Run only for specific dataset folds (default: all).")
    args = p.parse_args()

    data_dir = resolve(args.data_dir, download=False, quiet=True)
    mask = corpus_mask(data_dir)
    present = sorted({s.dataset for s in ProbeDataset(
        labeled_data_dir=data_dir, split="train",
        window_size=args.window_size)._discover()})
    if args.datasets:
        present = [d for d in present if d in set(args.datasets)]
    print(f"[*] data {data_dir}\n[*] datasets {', '.join(present)}\n")

    common = dict(labeled_data_dir=data_dir, window_size=args.window_size)
    print(f"{'fold':<24} {'setting':<14} "
          f"{'px~tx':>8} {'px~ty':>8} {'py~tx':>8} {'py~ty':>8}   verdict")
    print("-" * 92)

    for name, holdout in dataset_folds(present):
        tr = features_for(ProbeDataset(split="train", holdout=holdout, **common),
                          args, mask, f"{name} train")
        te = features_for(ProbeDataset(split="test", holdout=holdout, **common),
                          args, mask, f"{name} test")
        if tr is None or te is None:
            continue
        x_tr, y_tr, _, _ = tr
        x_te, y_te, _, sub_te = te

        model = fit_readout("ridge-cv", x_tr, y_tr, args.n_components, args.seed)
        cross = per_subject_corr(y_te, predict(model, x_te), sub_te)

        # Within the held-out dataset itself, subjects split. This is the
        # reference: it says how much signal is there at all, so a small
        # cross-dataset matrix can be read as transfer rather than absence.
        w_tr = features_for(ProbeDataset(split="train", split_by="subject",
                                         **common), args, mask, f"{name} w-train")
        w_te = features_for(ProbeDataset(split="test", split_by="subject",
                                         **common), args, mask, f"{name} w-test")
        within = np.full((2, 2), np.nan)
        if w_tr is not None and w_te is not None:
            keep_tr = np.isin(w_tr[2], list(holdout))
            keep_te = np.isin(w_te[2], list(holdout))
            if keep_tr.sum() > 50 and keep_te.sum() > 20:
                m = fit_readout("ridge-cv", w_tr[0][keep_tr], w_tr[1][keep_tr],
                                args.n_components, args.seed)
                within = per_subject_corr(w_te[1][keep_te],
                                          predict(m, w_te[0][keep_te]),
                                          w_te[3][keep_te])

        for setting, c in (("cross-dataset", cross), ("within-dataset", within)):
            diag = np.nanmean([c[0, 0], c[1, 1]])
            off = np.nanmean([abs(c[0, 1]), abs(c[1, 0])])
            if np.isnan(diag):
                verdict = "-"
            elif off > abs(diag) + 0.15:
                verdict = "AXES SWAPPED?"
            elif diag < -0.15:
                verdict = "SIGN FLIPPED?"
            elif abs(diag) < 0.35 and off < 0.35:
                verdict = "no signal recovered"
            else:
                verdict = "ok"
            print(f"{name:<24} {setting:<14} "
                  + " ".join(f"{v:>8.3f}" for v in
                             (c[0, 0], c[0, 1], c[1, 0], c[1, 1]))
                  + f"   {verdict}")
        print()

    print("px~tx = corr(predicted x, true x). Mass off the diagonal means the "
          "axes are\nmismatched; a negative diagonal means a sign convention "
          "differs. Both would be\ndata-provenance bugs, not modelling failures.")


if __name__ == "__main__":
    main()
