#!/usr/bin/env python3
"""Is gaze *non-linearly* accessible from the canonical coordinates at all?

This is the ceiling measurement for the whole non-linear program on this corpus,
and it should be run before any further effort is spent on a non-linear encoder.

The argument. `eval_probe`'s readout is linear (`ridge-cv`). A non-linear encoder
placed in front of a linear readout can therefore only help if gaze depends
non-linearly on the encoder's input -- if the dependence is linear, the best a
non-linear branch can do is reproduce the linear map, and anything else it adds
is variance. So the question "can an Orbit-JEPA beat `lr-cca`" is upper-bounded
by "can a *supervised* non-linear readout beat a supervised linear one on the
same features". Supervised is the generous case: it gets the labels the encoder
never sees, and it optimises the exact quantity being scored.

If the supervised non-linear readout does not win, no unsupervised non-linear
objective can be expected to, and a negative Orbit-JEPA result is a property of
the signal rather than of the objective or the tuning.

Arms, all leave-one-dataset-out on the 7 verified folds, per-subject median r:

``ridge``        linear readout on the canonical coordinates. The baseline.
``mlp``          a 2-layer MLP readout, same features.
``gbt``          gradient boosting, same features.
``poly-ridge``   ridge on the features plus their squares and pairwise products
                 of the leading few -- an explicit, interpretable non-linearity.
``ridge-k256``   linear readout on all 256 canonical directions, which separates
                 "non-linearity helps" from "more linear directions help".

Usage
-----
    python scripts/analyze_nonlinear_ceiling.py
    python scripts/analyze_nonlinear_ceiling.py --k 32 --arms ridge mlp
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.sweep_orbitjepa import MAX_TRAIN_ROWS, load_labeled_cache

ARMS = ("ridge", "ridge-k256", "poly-ridge", "mlp", "gbt")


def make_readout(arm, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    alphas = np.logspace(-3, 5, 17)
    if arm in ("ridge", "ridge-k256", "poly-ridge"):
        return RidgeCV(alphas=alphas)
    if arm == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-2,
                         learning_rate_init=1e-3, max_iter=300,
                         early_stopping=True, n_iter_no_change=15,
                         random_state=seed))
    if arm == "gbt":
        return MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                          early_stopping=True, random_state=seed))
    raise ValueError(arm)


def expand(x, arm, n_cross=12):
    """Feature map. `poly-ridge` gets squares plus leading pairwise products.

    Kept explicit rather than `PolynomialFeatures(2)`: at k=32 the full quadratic
    map is 560 columns, which would confound "non-linearity helps" with "more
    columns help" through the ridge penalty. Squares are free (k more columns)
    and the cross terms are capped at the leading `n_cross` directions, where the
    canonical correlations say the shared signal actually is.
    """
    if arm != "poly-ridge":
        return x
    c = min(n_cross, x.shape[1])
    idx = np.triu_indices(c, k=1)
    cross = x[:, :c][:, idx[0]] * x[:, :c][:, idx[1]]
    return np.concatenate([x, x ** 2, cross], axis=1)


def lodo(recs, arm, k, seed=0):
    feats = {}
    for r in recs:
        z = r["z"].astype(np.float64)
        width = z.shape[-1] if arm == "ridge-k256" else k
        feats[id(r)] = expand(0.5 * (z[:, 0, :width] + z[:, 1, :width]), arm)

    datasets = sorted({r["dataset"] for r in recs})
    per_fold = {}
    for held in datasets:
        train = [r for r in recs if r["dataset"] != held]
        xs, ys = [], []
        for ds in sorted({r["dataset"] for r in train}):
            g = np.concatenate([r["gaze"] for r in train if r["dataset"] == ds])
            x = np.concatenate([feats[id(r)] for r in train if r["dataset"] == ds])
            ok = np.isfinite(g).all(axis=1) & np.isfinite(x).all(axis=1)
            if ok.sum() < 10:
                continue
            g, x = g[ok], x[ok]
            sd = g.std(axis=0)
            sd[sd < 1e-9] = 1.0
            ys.append((g - g.mean(axis=0)) / sd)
            xs.append(x)
        x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
        if len(x_tr) > MAX_TRAIN_ROWS:
            sel = np.random.default_rng(seed).choice(len(x_tr), MAX_TRAIN_ROWS, replace=False)
            x_tr, y_tr = x_tr[sel], y_tr[sel]

        model = make_readout(arm, seed).fit(x_tr, y_tr)

        per_sub = []
        for r in (r for r in recs if r["dataset"] == held):
            x, g = feats[id(r)], r["gaze"]
            ok = np.isfinite(g).all(axis=1) & np.isfinite(x).all(axis=1)
            if ok.sum() < 10:
                continue
            pred = np.asarray(model.predict(x[ok]))
            per_sub.append([
                np.corrcoef(pred[:, ax], g[ok][:, ax])[0, 1]
                if np.std(pred[:, ax]) > 1e-12 and np.std(g[ok][:, ax]) > 1e-12 else np.nan
                for ax in (0, 1)])
        if per_sub:
            med = np.nanmedian(np.array(per_sub, dtype=float), axis=0)
            per_fold[held] = {"r_x": float(med[0]), "r_y": float(med[1]),
                              "mean": float(np.nanmean(med))}
    median = float(np.median([v["mean"] for v in per_fold.values()]))
    return median, per_fold


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cache", default="results/jepa/labeled_cache.npz")
    p.add_argument("--basis", default="results/scaling/basis_n1039.npz")
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/jepa/nonlinear_ceiling.json")
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    recs = load_labeled_cache(args.cache, args.basis, args.m, False)
    print(f"[*] {len(recs)} labeled participants, k={args.k} canonical coordinates")
    print(f"[*] every arm is SUPERVISED -- this is the ceiling an unsupervised "
          f"non-linear encoder would have to reach\n")

    out = {}
    folds = None
    for arm in args.arms:
        med, per_fold = lodo(recs, arm, args.k, seed=args.seed)
        out[arm] = {"median_r": med, "per_fold": per_fold}
        folds = folds or list(per_fold)
        print(f"  {arm:<12} median r = {med:.3f}", flush=True)

    base = out.get("ridge", {}).get("median_r")
    print("\n" + "=" * 88)
    print(f"{'readout':<14}{'median r':>10}{'vs ridge':>10}   per-fold")
    print("-" * 88)
    for arm in args.arms:
        row = out[arm]
        delta = f"{row['median_r'] - base:+.3f}" if base is not None else "--"
        per = " ".join(f"{row['per_fold'][f]['mean']:.2f}" for f in folds)
        print(f"{arm:<14}{row['median_r']:>10.3f}{delta:>10}   {per}")
    print("=" * 88)
    if base is not None:
        best_nl = max((out[a]["median_r"] for a in args.arms if a != "ridge"),
                      default=float("nan"))
        verdict = ("non-linear readouts DO beat linear -- a non-linear encoder has "
                   "headroom" if best_nl > base + 0.02 else
                   "no non-linear readout beats linear by more than the ~0.02 noise "
                   "floor, so gaze is linearly accessible here and a non-linear "
                   "encoder in front of a linear readout has nothing to add")
        print(f"\nVerdict: {verdict}.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"k": args.k, "arms": out}, indent=2))
    print(f"[*] -> {args.out}")


if __name__ == "__main__":
    main()
