#!/usr/bin/env python3
"""Probe every corpus size against every basis, and report the scaling law.

Pairs with `sweep_corpus_scaling.py`, which writes one basis file per corpus
size. This runs the gaze probe against each of them and answers the question the
project has never actually measured: **does the unlabeled corpus pay, and does
its payment grow?**

Two axes are crossed on purpose.

- **N**, unlabeled participants in the basis fit. If a frozen corpus basis is
  as good at N=25 as at N=800, the unlabeled half is redundant and no amount of
  it will help -- which is the pessimistic reading of every result on this
  project so far.
- **k**, components kept. The reason to cross them rather than fix k=64: the
  covariance is 14236x14236 and a checkpoint at N contributes ~48N rows, so
  below N~300 the estimate is **rank-deficient**. A richer basis should
  therefore only become usable once there is data to estimate it, and if that is
  what happens then the optimal k *grows with N* -- a scaling law rather than a
  single number.

`fold-pca:64` is refitted per fold and is independent of N, so it appears in
every cell as a constant reference line: any claim that the corpus basis is
improving has to be a claim about the gap to it closing.

The labeled budget is **capped and held fixed** across every cell
(`--max-train-windows`). That is deliberate rather than a shortcut: it keeps the
comparison controlled, and scarce labels are the regime where basis quality is
supposed to matter most.

    python scripts/sweep_probe_scaling.py
    python scripts/sweep_probe_scaling.py --report-only      # re-aggregate
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# dsL11 is excluded everywhere: its gaze/BOLD alignment fails per subject
# (see STATE.md, 2026-08-12), so its rows would be noise in every cell.
EXCLUDE = ["dsL11_backtothefuture"]

DEFAULT_FEATURES = [
    "corpus-pca:32", "corpus-pca:64", "corpus-pca:128", "corpus-pca:256",
    "lr-cca:64", "band-pca:64", "gev-fast:64", "gev-slow:64",
    "nuis-pca8:64", "nuis-pca32:64",
    "fold-pca:64",          # the N-independent reference line
]


def mean_r(entry):
    """Per-subject median r, averaged over the two axes."""
    ps = entry["by_subject"]["per_subject"]
    if not ps:
        return float("nan")
    rx = np.median([v["pearson_r_x"] for v in ps.values()])
    ry = np.median([v["pearson_r_y"] for v in ps.values()])
    return float(np.mean([rx, ry]))


def run_one(basis, out, features, budget, data_dir, readout="ridge-cv"):
    cmd = [".venv/bin/python", "scripts/eval_probe.py",
           "--protocol", "dataset", "--readouts", readout,
           "--standardize-targets", "dataset",
           "--basis", str(basis), "--features", *features,
           "--exclude-datasets", *EXCLUDE,
           "--out", str(out)]
    if budget:
        cmd += ["--max-train-windows", str(budget)]
    if data_dir:
        cmd += ["--data-dir", data_dir]
    print(f"    $ {' '.join(cmd[2:])}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        # Loud, not swallowed. `eval_contrastive_low_sample.py` swallowed its
        # failures into a bare except and wrote a 56/56-NaN summary that read as
        # a completed sweep; that is the mistake being avoided here.
        print(f"    [!] FAILED rc={r.returncode}\n{r.stdout[-1500:]}\n"
              f"{r.stderr[-1500:]}", flush=True)
        return False
    return True


def aggregate(out_dir, sizes, features):
    """{feature: {N: {fold: r}}} from the per-N result files."""
    table = {}
    for n in sizes:
        path = Path(out_dir) / f"probe_n{n}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for fold, arms in data.items():
            for key, entry in arms.items():
                feat = key.split("/")[0]
                table.setdefault(feat, {}).setdefault(str(n), {})[fold] = mean_r(entry)
    return table


def report(table, sizes, features):
    folds = sorted({f for feat in table.values() for n in feat.values() for f in n})
    print("\n" + "=" * 78)
    print("median r across folds, by basis and unlabeled corpus size")
    print("=" * 78)
    header = f"{'feature':<18}" + "".join(f"{('N=' + str(n)):>9}" for n in sizes)
    print(header)
    print("-" * len(header))
    for feat in features:
        if feat not in table:
            continue
        row = f"{feat:<18}"
        for n in sizes:
            vals = [v for v in table[feat].get(str(n), {}).values()
                    if np.isfinite(v)]
            row += f"{np.median(vals):>9.3f}" if vals else f"{'--':>9}"
        print(row)
    print("-" * len(header))
    print(f"n folds: {len(folds)} ({', '.join(folds)})")
    print("\nfold-pca:64 is refitted per fold and does not depend on N; any\n"
          "variation in its row is run-to-run noise and sets the scale for\n"
          "reading the others.")


def aggregate_budgets(out_dir, sizes, budgets):
    """{feature: {(N, budget): {fold: r}}} for the label-efficiency grid."""
    table = {}
    for n in sizes:
        for b in budgets:
            path = Path(out_dir) / f"probe_n{n}_b{b}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            for fold, arms in data.items():
                for key, entry in arms.items():
                    feat = key.split("/")[0]
                    cell = table.setdefault(feat, {}).setdefault(f"{n}|{b}", {})
                    cell[fold] = mean_r(entry)
    return table


def report_budgets(table, sizes, budgets, features):
    """Label efficiency: does a bigger unlabeled corpus buy cheaper labels?

    The question a scaling claim actually needs. A frozen basis that merely ties
    at full labels is uninteresting; one that reaches the same score from a
    quarter of the labels is a result, and one whose label requirement *falls* as
    the unlabeled corpus grows is a scaling law.
    """
    print("\n" + "=" * 86)
    print("median r by labeled-window budget (rows) and unlabeled corpus size")
    print("=" * 86)
    for feat in features:
        if feat not in table:
            continue
        print(f"\n  {feat}")
        header = f"    {'budget':<10}" + "".join(f"{('N=' + str(n)):>10}"
                                                for n in sizes)
        print(header)
        for b in budgets:
            row = f"    {(b if b else 'all'):<10}"
            for n in sizes:
                vals = [v for v in table[feat].get(f"{n}|{b}", {}).values()
                        if np.isfinite(v)]
                row += f"{np.median(vals):>10.3f}" if vals else f"{'--':>10}"
            print(row)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--basis-dir", default="results/scaling")
    p.add_argument("--out-dir", default="results/scaling")
    p.add_argument("--sizes", nargs="+", type=int,
                   default=[25, 50, 100, 200, 400, 800])
    p.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    p.add_argument("--max-train-windows", type=int, default=1000,
                   help="Held fixed across cells; 0 for the full budget.")
    p.add_argument("--budgets", nargs="*", type=int, default=None,
                   help="Cross corpus size with LABELED budget instead of "
                        "sweeping N alone. 0 means the full budget. This is the "
                        "label-efficiency grid: `--sizes 25 800 --budgets 100 "
                        "250 500 1000 0`.")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = list(args.sizes)

    if args.budgets is not None:
        budgets = list(args.budgets)
        if not args.report_only:
            for n in sizes:
                basis = Path(args.basis_dir) / f"basis_n{n}.npz"
                if not basis.exists():
                    print(f"[!] no basis for N={n} at {basis}")
                    continue
                for b in budgets:
                    print(f"\n[*] === N={n} unlabeled, budget={b or 'all'} ===",
                          flush=True)
                    run_one(basis, out_dir / f"probe_n{n}_b{b}.json",
                            args.features, b, args.data_dir)
        table = aggregate_budgets(out_dir, sizes, budgets)
        (out_dir / "probe_budget_summary.json").write_text(
            json.dumps(table, indent=2, default=float))
        report_budgets(table, sizes, budgets, args.features)
        print(f"\n[*] -> {out_dir / 'probe_budget_summary.json'}")
        return

    if not args.report_only:
        for n in sizes:
            basis = Path(args.basis_dir) / f"basis_n{n}.npz"
            if not basis.exists():
                print(f"[!] no basis for N={n} at {basis}; run "
                      f"sweep_corpus_scaling.py first")
                continue
            print(f"\n[*] === N={n} unlabeled participants ===", flush=True)
            run_one(basis, out_dir / f"probe_n{n}.json", args.features,
                    args.max_train_windows, args.data_dir)

    table = aggregate(out_dir, sizes, args.features)
    (out_dir / "probe_scaling_summary.json").write_text(
        json.dumps(table, indent=2, default=float))
    report(table, sizes, args.features)
    print(f"\n[*] -> {out_dir / 'probe_scaling_summary.json'}")


if __name__ == "__main__":
    main()
