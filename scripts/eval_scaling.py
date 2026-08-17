#!/usr/bin/env python3
"""Step 3: Measure zero-label gauge performance across corpus sizes n={25..1039}.

Tests the hypothesis that a larger unlabeled corpus yields a better-estimated
canonical reference frame, improving the zero-label gauge accuracy and signed r.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.datasource import resolve
from deepmreye.gauge import (
    N_CC,
    N_PCA,
    select_gauge,
)

# We import prepare, evaluate_fold, fit_xds_ridge from eval_zero_label
from scripts.eval_zero_label import EXCLUDE, ARMS, evaluate_fold, fit_xds_ridge, prepare
from deepmreye.unsupervised import load_basis

BASES = [
    ("n25", "results/scaling/basis_n25.npz"),
    ("n50", "results/scaling/basis_n50.npz"),
    ("n100", "results/scaling/basis_n100.npz"),
    ("n200", "results/scaling/basis_n200.npz"),
    ("n400", "results/scaling/basis_n400.npz"),
    ("n800", "results/scaling/basis_n800.npz"),
    ("n1039", "results/scaling/basis_n1039.npz"),
]


def eval_basis(basis_path, root, rng, k=32):
    mask, bases, meta = load_basis(basis_path)
    basis = bases["lr-cca"]

    records = {}
    for ds_dir in sorted(root.glob("dsL*")):
        if ds_dir.name in EXCLUDE:
            continue
        paths = sorted(ds_dir.glob("*.h5"))
        rows = []
        for path in paths:
            try:
                rec = prepare(path, mask, basis, k, N_CC, N_PCA, False, rng)
            except Exception:
                continue
            if rec:
                rows.append(rec)
        if rows:
            records[ds_dir.name] = rows

    results, gauges = {}, {}
    for held in records:
        others = {d: v for d, v in records.items() if d != held}
        gauge = select_gauge({d: [(r["corpus"], r["gaze"]) for r in v]
                              for d, v in others.items()},
                             k=k)
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

    summary = {a: float(np.median([np.mean(r[a]) for r in results.values()]))
               for a in ARMS}
    stable = len({(g["x"][0], g["y"][0]) for g in gauges.values()}) == 1
    sample_gauge = next(iter(gauges.values()))

    return {
        "summary": summary,
        "n_subjects": meta["n_subjects"],
        "gauge_stable": stable,
        "sample_gauge": {a: list(v) for a, v in sample_gauge.items()},
        "by_dataset": results,
    }


def main():
    warnings.filterwarnings("ignore")
    root = Path(resolve(None, download=False, quiet=True))
    rng = np.random.default_rng(0)

    scaling_results = {}
    print("==========================================================================")
    print("Scaling Curve: Zero-Label Gauge across Corpus Sizes n={25..1039}")
    print(f"{'n_corpus':<10}{'fixed':>10}{'adapted':>10}{'oracle-g':>10}{'supervis-xds':>14}{'gauge x,y':>14}{'stable':>9}")
    print("--------------------------------------------------------------------------")

    for name, basis_path in BASES:
        res = eval_basis(basis_path, root, rng)
        scaling_results[name] = res
        s = res["summary"]
        gx, gy = res["sample_gauge"]["x"][0], res["sample_gauge"]["y"][0]
        stable_str = "YES" if res["gauge_stable"] else "NO"
        print(f"{name:<10}{s['fixed']:>10.3f}{s['adapted']:>10.3f}{s['oracle-gauge']:>10.3f}{s['supervised-xds']:>14.3f}{f'({gx},{gy})':>14}{stable_str:>9}", flush=True)

    print("--------------------------------------------------------------------------")
    out_path = Path("results/zero_label/scaling.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scaling_results, indent=2))
    print(f"\n[*] -> {out_path}")


if __name__ == "__main__":
    main()
