#!/usr/bin/env python3
"""Zero-label gaze decoding, measured against everything that uses labels.

The claim under test is not "we decode gaze better". It is that gaze can be
decoded **without any labels from the target study**, which is the constraint
DeepMReye 1.0 could not drop: it needed 250 labeled participants to train the
CNN that decodes the 251st.

`deepmreye/gauge.py` explains the mechanism. In short, per-run CCA between the
two orbits recovers gaze with no labels but with an arbitrary component order
and sign, and the frozen corpus basis -- one fixed filter set applied to
everybody -- has no such freedom, so it can *name* the run's variates.

Every arm below sees the **same features** (corpus lr-cca variates and/or the
same orbit voxels), the same per-TR resolution, the same first-half/second-half
split, and the same aggregation. The only thing that varies is how many labels
the arm was allowed:

| arm | labels used | what it establishes |
|---|---|---|
| `fixed` | none from any study but the 9 bits below | the floor: 2 components, no fitting at all |
| `adapted` | same 9 bits | **the method** |
| `oracle-gauge` | the target run's own | upper bound on `adapted` -- what the label-free gauge gives up |
| `random-gauge` | none | control: must be ~0, or the per-run CCA is decoding by itself |
| `supervised-xds` | every other dataset's | the honest label-using comparison, same features |
| `supervised-within` | the target run's own first half | the per-run ceiling |

The 9 bits are two component indices and two signs, chosen **leave-one-dataset-
out** so the gauge applied to a fold is never selected on it. That is the whole
label cost of the method, it is paid once, and `--fixed-gauge` pins it to the
published constants instead.

`--null` circularly shifts every gaze trace, preserving its autocorrelation and
marginal exactly while destroying alignment. Gaze and eye-block signals are both
slow, so effective degrees of freedom are far below T and this control is not
optional.

    python scripts/eval_zero_label.py
    python scripts/eval_zero_label.py --null
    python scripts/eval_zero_label.py --scaling      # gauge vs corpus size
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.datasource import resolve
from deepmreye.gauge import (
    DEFAULT_GAUGE,
    MIN_TR,
    N_CC,
    N_PCA,
    as_rows,
    corr,
    gauge_by_teacher,
    motion_proxy,
    oracle_gauge,
    orbit_views,
    regress_out,
    run_cca,
    select_gauge,
)
from deepmreye.unsupervised import load_basis, project

EXCLUDE = ("dsL11_backtothefuture",)
ARMS = ("fixed", "adapted", "oracle-gauge", "random-gauge",
        "supervised-xds", "supervised-within")


def prepare(path, mask, basis, k, n_cc, n_pca, null, rng):
    """One participant: corpus variates, per-run CCA variates, gaze."""
    with h5py.File(path, "r") as f:
        if "labels" not in f:
            return None
        block = f["eye_block"][:]
        gaze = np.nanmean(f["labels"][:], axis=1).astype(np.float64)
    t = block.shape[-1]
    if t < MIN_TR or not np.isfinite(gaze).any():
        return None
    if null:
        gaze = np.roll(gaze, int(rng.integers(t // 4, 3 * t // 4)), axis=0)

    rows = as_rows(block, mask)
    corpus = project("lr-cca", basis, rows, k=k)
    left, right = orbit_views(rows, basis)
    conf = motion_proxy(rows)
    left, right = regress_out(left, conf), regress_out(right, conf)

    fit = slice(0, t // 2)
    variates = run_cca(left, right, fit, n_cc, n_pca)
    return {"corpus": corpus, "variates": variates, "gaze": gaze,
            "n_t": t, "subject": path.stem}


def score(pred, gaze, test):
    """Signed r per axis on the held-out half. Signed, never |r|."""
    return [corr(pred[test, ax], gaze[test, ax]) for ax in (0, 1)]


def evaluate_fold(rec, gauge, rng, ridge_xds):
    """Every arm for one participant. Returns {arm: [r_x, r_y]}."""
    t, gaze = rec["n_t"], rec["gaze"]
    fit, test = slice(0, t // 2), slice(t // 2, t)
    corpus, variates = rec["corpus"], rec["variates"]
    teachers = {ax: gauge[ax][1] * corpus[:, gauge[ax][0]] for ax in ("x", "y")}
    out = {}

    out["fixed"] = score(np.column_stack([teachers["x"], teachers["y"]]),
                         gaze, test)

    cols = []
    for ax in ("x", "y"):
        idx, sign, _q = gauge_by_teacher(variates, teachers[ax])
        cols.append(sign * variates[:, idx])
    out["adapted"] = score(np.column_stack(cols), gaze, test)

    # Upper bound: the gauge chosen from the run's own labels, on the fit half
    # only (so it is the fairest possible label-using version of `adapted`).
    orc = oracle_gauge(variates, gaze, fit)
    out["oracle-gauge"] = score(
        np.column_stack([orc[ax][1] * variates[:, orc[ax][0]]
                         for ax in ("x", "y")]), gaze, test)

    # Control. If per-run CCA decoded gaze regardless of which variate you
    # picked, every number above would be meaningless.
    out["random-gauge"] = score(
        np.column_stack([rng.choice([-1.0, 1.0])
                         * variates[:, rng.integers(variates.shape[1])]
                         for _ in ("x", "y")]), gaze, test)

    out["supervised-xds"] = (score(ridge_xds.predict(corpus), gaze, test)
                             if ridge_xds is not None else [np.nan, np.nan])

    from sklearn.linear_model import Ridge

    ok = np.isfinite(gaze[fit]).all(axis=1)
    if ok.sum() >= N_PCA:
        model = Ridge(alpha=1.0).fit(corpus[fit][ok], gaze[fit][ok])
        out["supervised-within"] = score(model.predict(corpus), gaze, test)
    else:
        out["supervised-within"] = [np.nan, np.nan]
    return out


def fit_xds_ridge(records):
    """Supervised ridge on pooled other-dataset participants, same features.

    Gaze is z-scored per dataset before pooling, exactly as
    `eval_probe --standardize-targets dataset` does: the labeled sets are in
    different units (degrees against screen pixels), so an unstandardised pool
    simply follows whichever dataset has the largest target variance. Pearson r
    is invariant to it.
    """
    from sklearn.linear_model import Ridge

    xs, ys = [], []
    for ds_records in records.values():
        g = np.concatenate([r["gaze"] for r in ds_records])
        x = np.concatenate([r["corpus"] for r in ds_records])
        ok = np.isfinite(g).all(axis=1)
        if ok.sum() < 10:
            continue
        g = g[ok]
        sd = g.std(axis=0)
        sd[sd < 1e-9] = 1.0
        ys.append((g - g.mean(axis=0)) / sd)
        xs.append(x[ok])
    if not xs:
        return None
    return Ridge(alpha=1.0).fit(np.concatenate(xs), np.concatenate(ys))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--basis", default="results/scaling/basis_n1039.npz")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--n-cc", type=int, default=N_CC)
    p.add_argument("--n-pca", type=int, default=N_PCA)
    p.add_argument("--max-subjects", type=int, default=0,
                   help="Per dataset. 0 = all.")
    p.add_argument("--fixed-gauge", action="store_true",
                   help="Use the published constants instead of selecting "
                        "leave-one-dataset-out.")
    p.add_argument("--null", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/zero_label/eval.json")
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    root = Path(resolve(args.data_dir, download=False, quiet=True))
    mask, bases, meta = load_basis(args.basis)
    basis = bases["lr-cca"]
    rng = np.random.default_rng(args.seed)
    mode = "NULL CONTROL (gaze circularly shifted)" if args.null else "real gaze"
    print(f"[*] basis {Path(args.basis).name}: {meta['n_subjects']} subjects, "
          f"{meta['datasets']} datasets, k={args.k}, n_cc={args.n_cc} -- {mode}")

    records = {}
    for ds_dir in sorted(root.glob("dsL*")):
        if ds_dir.name in EXCLUDE:
            continue
        paths = sorted(ds_dir.glob("*.h5"))
        if args.max_subjects:
            paths = paths[: args.max_subjects]
        rows = []
        for path in paths:
            try:
                rec = prepare(path, mask, basis, args.k, args.n_cc, args.n_pca,
                              args.null, rng)
            except Exception as e:
                print(f"  [!] {path.name}: {e.__class__.__name__}: {e}")
                continue
            if rec:
                rows.append(rec)
        if rows:
            records[ds_dir.name] = rows
            print(f"  {ds_dir.name:<26}{len(rows):>4} participants", flush=True)
    if not records:
        raise SystemExit("[!] no usable participants")

    results, gauges = {}, {}
    for held in records:
        others = {d: v for d, v in records.items() if d != held}
        if args.fixed_gauge:
            gauge = DEFAULT_GAUGE
        else:
            gauge = select_gauge({d: [(r["corpus"], r["gaze"]) for r in v]
                                  for d, v in others.items()},
                                 k=args.k)
        gauges[held] = gauge
        ridge_xds = fit_xds_ridge(others)
        per_sub = [evaluate_fold(rec, gauge, rng, ridge_xds)
                   for rec in records[held]]
        results[held] = {
            arm: [float(np.nanmedian([s[arm][ax] for s in per_sub]))
                  for ax in (0, 1)]
            for arm in ARMS
        }
        results[held]["n_subjects"] = len(per_sub)

    width = 26 + 9 * len(ARMS)
    print("\n" + "=" * width)
    print("Signed median r across participants, held-out half of each run")
    print(f"{'dataset':<20}{'n':>4}" + "".join(f"{a[:8]:>9}" for a in ARMS))
    print("-" * width)
    for held, res in results.items():
        print(f"{held.replace('dsL', ''):<20}{res['n_subjects']:>4}"
              + "".join(f"{np.mean(res[a]):>9.3f}" for a in ARMS))
    print("-" * width)
    summary = {a: float(np.median([np.mean(r[a]) for r in results.values()]))
               for a in ARMS}
    print(f"{'median over folds':<24}" + "".join(f"{summary[a]:>9.3f}" for a in ARMS))
    for ax, name in ((0, "x"), (1, "y")):
        print(f"{'  axis ' + name:<24}"
              + "".join(f"{np.median([r[a][ax] for r in results.values()]):>9.3f}"
                        for a in ARMS))

    print("\ngauge selected per held-out fold (component, sign):")
    for held, g in gauges.items():
        print(f"  {held:<26} x={g['x']}  y={g['y']}")
    stable = len({(g["x"][0], g["y"][0]) for g in gauges.values()}) == 1
    print(f"  -> {'STABLE' if stable else 'UNSTABLE'} across folds")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"results": results, "summary": summary,
         "gauges": {k: {a: list(v) for a, v in g.items()} for k, g in gauges.items()},
         "basis": str(args.basis), "null": args.null,
         "n_cc": args.n_cc, "k": args.k}, indent=2))
    print(f"\n[*] -> {out}")


if __name__ == "__main__":
    main()
