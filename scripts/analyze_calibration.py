#!/usr/bin/env python3
"""Why cross-dataset gaze correlates well but scores R^2 near zero.

The symptom, measured on the leave-one-dataset-out folds: dsL04 reaches Pearson
r ~ 0.5-0.6 while its R^2 against the training-mean gaze is 0.03. The map
transfers in *shape* and not in *scale*. This script quantifies that and tests
whether anything unsupervised fixes it.

For each held-out dataset, a ridge is fitted on the others and scored under:

  raw            predictions as they come out.
  oracle-affine  the best per-axis ``a * pred + b``, fitted **on the test
                 labels**. Not a method -- an upper bound. It says how much of
                 the gap is purely affine and therefore how much a perfect
                 calibration could ever recover.
  shift          subtract the prediction mean, add the training label mean.
  z-match        z-score the predictions, rescale to the training label spread.
  feat-std       standardise features per dataset before the ridge.
  quantile       map prediction quantiles onto the training label distribution.

The last four are unsupervised and could be deployed. The measured result is
that none of them work, and the reason is identifiable rather than incidental:
the required gain is approximately ``test_gaze_SD / train_gaze_SD``, and the
target's marginal spread is exactly what differs between paradigms -- a fixation
task and a free-viewing task do not have the same gaze distribution. Degrees of
visual angle depend on screen size and viewing distance, which are not in the
BOLD signal at all. No unsupervised correction can recover a scale factor that
the input does not contain.

This is the argument for reporting Pearson r as the headline (see
``scripts/eval_probe.py``) and for treating calibration as a separate problem --
one that a handful of labeled timepoints per run, or known stimulus geometry,
would solve directly.

    python scripts/analyze_calibration.py --data-dir data --max-windows 300
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.data.probe_dataset import ProbeDataset, dataset_folds
from deepmreye.datasource import resolve
from deepmreye.evaluate.probe import flatten_valid_groups

from eval_probe import cap, voxel_features


def r2(y, pred, const):
    """R^2 against a constant prediction, pooled over both axes."""
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - const) ** 2)


def quantile_map(pred, ref):
    """Map ``pred`` onto ``ref``'s marginal, per axis, monotonically."""
    out = np.empty_like(pred)
    for i in range(pred.shape[1]):
        rank = np.argsort(np.argsort(pred[:, i]))
        out[:, i] = np.quantile(ref[:, i], (rank + 0.5) / len(rank))
    return out


def standardise_within(feats, groups):
    out = feats.copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] = (out[m] - out[m].mean(0)) / (out[m].std(0) + 1e-6)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--window-size", type=int, default=100)
    p.add_argument("--temp-patch-size", type=int, default=5)
    p.add_argument("--voxel-stride", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-windows", type=int, default=300)
    args = p.parse_args()

    data_dir = resolve(args.data_dir, download=False, quiet=True)
    n_t = args.window_size // args.temp_patch_size

    present = sorted({s.dataset for s in ProbeDataset(
        labeled_data_dir=data_dir, split="train", window_size=args.window_size)._discover()})

    print(f"{'held out':<24} {'raw':>8} {'oracle':>8} {'shift':>8} {'z-match':>8} "
          f"{'feat-std':>9} {'quantile':>9}   {'gain x,y':>12} {'offset x,y':>12}")
    print("-" * 112)

    rows = []
    for name, holdout in dataset_folds(present):
        common = dict(labeled_data_dir=data_dir, holdout=holdout,
                      window_size=args.window_size)
        tr_ds = cap(ProbeDataset(split="train", **common), args.max_windows)
        te_ds = cap(ProbeDataset(split="test", **common), args.max_windows)
        if not len(tr_ds) or not len(te_ds):
            continue

        tr = voxel_features(DataLoader(tr_ds, batch_size=args.batch_size),
                            n_t, args.voxel_stride, f"{name} train")
        te = voxel_features(DataLoader(te_ds, batch_size=args.batch_size),
                            n_t, args.voxel_stride, f"{name} test")

        x_tr, y_tr, d_tr, _ = flatten_valid_groups(tr[0], tr[1], tr[2], tr[3])
        x_te, y_te, d_te, _ = flatten_valid_groups(te[0], te[1], te[2], te[3])
        if len(x_te) < 10:
            continue

        const = y_tr.mean(0)
        pred = Ridge(alpha=1.0).fit(x_tr, y_tr).predict(x_te)

        oracle = np.empty_like(pred)
        gains, offsets = [], []
        for i in range(2):
            a, b = np.polyfit(pred[:, i], y_te[:, i], 1)
            oracle[:, i] = a * pred[:, i] + b
            gains.append(a)
            offsets.append(b)

        shift = pred - pred.mean(0) + const
        zmatch = (pred - pred.mean(0)) / (pred.std(0) + 1e-9) * y_tr.std(0) + const
        qmap = quantile_map(pred, y_tr)
        fstd = Ridge(alpha=1.0).fit(standardise_within(x_tr, d_tr), y_tr) \
            .predict(standardise_within(x_te, d_te))

        vals = [r2(y_te, v, const) for v in (pred, oracle, shift, zmatch, fstd, qmap)]
        print(f"{name:<24} " + " ".join(f"{v:>8.3f}" for v in vals[:4]) +
              f" {vals[4]:>9.3f} {vals[5]:>9.3f}   "
              f"{gains[0]:>5.2f},{gains[1]:>5.2f}  {offsets[0]:>5.2f},{offsets[1]:>5.2f}")
        rows.append(vals)

    if rows:
        a = np.array(rows)
        print("-" * 112)
        print(f"{'MEAN':<24} " + " ".join(f"{v:>8.3f}" for v in a[:, :4].mean(0)) +
              f" {a[:, 4].mean():>9.3f} {a[:, 5].mean():>9.3f}")
        print("\nGain far from 1.0 with offset near 0 is a pure scale mismatch. "
              "If `oracle`\nis high while every unsupervised column is not, the "
              "information needed to\ncalibrate is not in the input.")


if __name__ == "__main__":
    main()
