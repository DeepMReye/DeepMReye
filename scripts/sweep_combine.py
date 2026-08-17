#!/usr/bin/env python3
"""Three ways of combining a corpus basis with a fold-local one, measured.

`fold-pca:64` beats every frozen corpus basis by ~0.02 median r and the two win
*different* folds, which is the classic signature of complementary feature
spaces. Every combination attempt on this project so far concatenated them and
fitted `ridge-cv`, which cannot express the combination anyone actually wants:
one alpha over both blocks means the corpus block is penalised exactly as much as
the fold-local one. So "concatenation loses" was never a test of complementarity.

Three arms, in increasing order of how deep in the pipeline they combine:

1. **readout** -- `banded-ridge` / `stack-ridge` on the concatenation. One
   penalty per block, or a convex combination of per-block predictions
   (Nunez-Elizalde 2019; Dupre la Tour 2022; Lin 2024). Fitted weights are
   reported, so a redundant block says so directly instead of being inferred
   from a score.
2. **prior** -- `--dyadic-blocks` on a single 256-component basis. Both bases are
   variance-ordered, so a penalty that grows down the spectrum is the prior the
   data implies, and `:k` truncation is its crudest form -- a step function. Asks
   whether truncation is the wrong *prior* rather than the wrong budget.
3. **covariance** -- `fold-shrunk-pca` at a sweep of lambda. PCA of
   `(1-lam) C_fold + lam C_corpus`, so `fold-pca` and `corpus-pca` are the two
   endpoints of one curve. This is the only arm that can beat both, because it is
   the only one that fixes what is actually wrong with `fold-pca`: it is a noisy
   covariance estimate from a few hundred labeled windows, and the corpus is a
   well-estimated shrinkage target that keeps improving as the corpus grows.

    python scripts/sweep_combine.py --stage readout prior covariance
    python scripts/sweep_combine.py --stage covariance --lambdas 0.25 0.5 0.75
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

BASIS = "results/scaling/basis_n1039.npz"
EXCLUDE = ["dsL11_backtothefuture"]


def run(features, readouts, out, budget, extra=()):
    out = Path(out)
    if out.exists():
        print(f"[=] {out} exists, skipping")
        return out
    cmd = [".venv/bin/python", "scripts/eval_probe.py",
           "--protocol", "dataset", "--standardize-targets", "dataset",
           "--basis", BASIS, "--features", *features,
           "--readouts", *readouts,
           "--max-train-windows", str(budget),
           "--exclude-datasets", *EXCLUDE,
           "--out", str(out), *extra]
    print(f"[*] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # Loud, not swallowed: a silently skipped cell reads as a missing
        # measurement later and nobody remembers which.
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"[!] eval_probe failed for {out}")
    return out


def mean_r(entry):
    ps = entry["by_subject"]["per_subject"]
    if not ps:
        return float("nan")
    return float(np.mean([
        np.median([v["pearson_r_x"] for v in ps.values()]),
        np.median([v["pearson_r_y"] for v in ps.values()])]))


def summarise(path, label=None):
    data = json.loads(Path(path).read_text())
    table = {}
    for fold, arms in data.items():
        for arm, entry in arms.items():
            table.setdefault(arm, {})[fold] = mean_r(entry)
    print(f"\n=== {label or path}")
    print(f"{'arm':<48}{'folds':>6}{'median r':>10}{'mean r':>9}")
    for arm, folds in sorted(table.items(),
                             key=lambda kv: -np.median(list(kv[1].values()))):
        vals = [v for v in folds.values() if np.isfinite(v)]
        print(f"{arm:<48}{len(vals):>6}{np.median(vals):>10.3f}{np.mean(vals):>9.3f}")
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", nargs="+", default=["readout", "prior", "covariance"],
                   choices=["readout", "prior", "covariance"])
    p.add_argument("--out-dir", default="results/combine")
    p.add_argument("--budget", type=int, default=1000,
                   help="Labeled training windows. 1000 is the budget every "
                        "scaling number in STATE.md was measured at.")
    p.add_argument("--lambdas", nargs="+", type=float,
                   default=[0.1, 0.25, 0.5, 0.75, 0.9])
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "readout" in args.stage:
        summarise(run(["fold-pca:64", "lr-cca:32", "fold-pca:64+lr-cca:32"],
                      ["ridge-cv", "banded-ridge", "stack-ridge"],
                      out_dir / "probe_combine.json", args.budget),
                  "1. readout-level: per-block penalties and stacking")

    if "prior" in args.stage:
        # 256 components with a learned taper against 64 with a hard cut. The
        # `:64` arms are carried along as the reference, since the comparison is
        # "taper over the whole spectrum" vs "truncate".
        summarise(run(["fold-pca:256", "corpus-pca:256", "lr-cca:256"],
                      ["banded-ridge"], out_dir / "probe_dyadic.json",
                      args.budget, extra=["--dyadic-blocks"]),
                  "2. prior-level: dyadic taper over 256 components")

    if "covariance" in args.stage:
        tables = {}
        for lam in args.lambdas:
            path = out_dir / f"probe_shrunk_lam{lam:g}.json"
            run(["fold-shrunk-pca:64"], ["ridge-cv"], path, args.budget,
                extra=["--shrink-lambda", str(lam)])
            tables[lam] = summarise(path, f"3. covariance shrinkage lam={lam:g}")
        print("\n=== shrinkage curve (median r across folds)")
        print(f"{'lambda':>8}{'median r':>10}{'mean r':>9}")
        for lam, table in tables.items():
            vals = [v for folds in table.values() for v in folds.values()
                    if np.isfinite(v)]
            print(f"{lam:>8.2f}{np.median(vals):>10.3f}{np.mean(vals):>9.3f}")
        print("lam=0 is fold-pca and lam=1 is corpus-pca by construction; only "
              "an interior peak is a result.")


if __name__ == "__main__":
    main()
