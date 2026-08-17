#!/usr/bin/env python3
"""Compare Orbit-JEPA checkpoints against the linear arms on the verified folds.

A thin driver over `scripts/eval_probe.py`, deliberately: the probe harness is
the audited one, it is what produced every number in `STATE.md`, and a second
implementation of the split/pooling/readout logic is exactly how the old
`eval_orbitjepa.py` came to report `0.221` against a `0.847` that had been
measured a different way (per-TR targets, `Ridge(alpha=1.0)`, half-run splits --
none of them comparable). Everything here runs through the same code path as
`fold-pca:64`.

Each checkpoint contributes two arms:

``jepa``         the trained model.
``jepa-random``  the same architecture untrained, which by construction is
                 `lr-cca:k` **exactly** (`test_untrained_jepa_reproduces_lr_cca
                 _exactly`). So the control is the linear corpus baseline, and
                 ``jepa - jepa-random`` is the margin over it on identical
                 folds, windows, targets and readout.

Usage
-----
    python scripts/eval_orbitjepa.py --checkpoints results/jepa/*.pt
    python scripts/eval_orbitjepa.py --checkpoints results/jepa/k32_base.pt \\
        --baselines fold-pca:64 lr-cca:32
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The protocol every scaling and combine number in `STATE.md` was measured at.
# Changing any of these makes the comparison against 0.847 / 0.825 invalid.
PROTOCOL = ["--protocol", "dataset", "--readouts", "ridge-cv",
            "--standardize-targets", "dataset",
            "--exclude-datasets", "dsL11_backtothefuture",
            "--basis", "results/scaling/basis_n1039.npz",
            "--max-train-windows", "1000", "--basis-fit-windows", "400"]


def run_probe(features, out_path, jepa_checkpoint=None, extra=()):
    cmd = [sys.executable, str(ROOT / "scripts" / "eval_probe.py"),
           *PROTOCOL, "--features", *features, "--out", str(out_path), *extra]
    if jepa_checkpoint:
        cmd += ["--jepa-checkpoint", str(jepa_checkpoint)]
    print(f"    $ {' '.join(cmd[1:])}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise SystemExit(f"[!] eval_probe failed ({proc.returncode}):\n{tail}")
    return json.loads(Path(out_path).read_text())


def median_r(results, key):
    """Per-fold mean of (r_x, r_y), then median across folds -- the headline.

    Read off `by_subject`, which is the per-participant aggregation, never
    `pooled`. `CLAUDE.md`: pooling every row of every subject into one
    correlation rewards a model that only predicts *which subject this is*.
    """
    per_fold = []
    for fold, arms in results.items():
        if key not in arms:
            continue
        sub = arms[key]["by_subject"]
        per_fold.append(float(np.mean([sub["pearson_r_x"], sub["pearson_r_y"]])))
    return (float(np.median(per_fold)) if per_fold else float("nan")), per_fold


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--baselines", nargs="*", default=["fold-pca:64"],
                   help="Linear arms to measure in the same run. `lr-cca:k` is "
                        "redundant with `jepa-random` and only worth adding as a "
                        "cross-check that the identity still holds.")
    p.add_argument("--results-dir", default="results/jepa")
    p.add_argument("--out", default="results/jepa/summary.json")
    args = p.parse_args()

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table, folds_seen = {}, []

    if args.baselines:
        print(f"[*] baselines: {', '.join(args.baselines)}")
        res = run_probe(args.baselines, out_dir / "baselines.json")
        for spec in args.baselines:
            m, per = median_r(res, f"{spec}/ridge-cv")
            table[spec] = {"median_r": m, "per_fold": per, "kind": "linear"}
            folds_seen = list(res)

    for ckpt in args.checkpoints:
        ckpt = Path(ckpt)
        name = ckpt.stem
        print(f"[*] {name}")
        res = run_probe(["jepa", "jepa-random"], out_dir / f"probe_{name}.json",
                        jepa_checkpoint=ckpt)
        trained, per_t = median_r(res, "jepa/ridge-cv")
        control, per_c = median_r(res, "jepa-random/ridge-cv")
        wins = int(sum(1 for a, b in zip(per_t, per_c) if a > b))
        table[name] = {"median_r": trained, "per_fold": per_t,
                       "control_median_r": control, "control_per_fold": per_c,
                       "margin": trained - control, "folds_won": wins,
                       "n_folds": len(per_t), "kind": "jepa"}
        folds_seen = folds_seen or list(res)

    print("\n" + "=" * 96)
    print(f"{'arm':<22}{'median r':>10}{'control':>10}{'margin':>9}{'folds won':>11}")
    print("-" * 96)
    for name, row in sorted(table.items(), key=lambda kv: -kv[1]["median_r"]):
        if row["kind"] == "linear":
            print(f"{name:<22}{row['median_r']:>10.3f}{'--':>10}{'--':>9}{'--':>11}")
        else:
            won = f"{row['folds_won']}/{row['n_folds']}"
            print(f"{name:<22}{row['median_r']:>10.3f}{row['control_median_r']:>10.3f}"
                  f"{row['margin']:>+9.3f}{won:>11}")
    print("=" * 96)
    print("`control` is the same architecture untrained, which equals lr-cca:k exactly.")

    Path(args.out).write_text(json.dumps({"table": table, "folds": folds_seen,
                                          "protocol": PROTOCOL}, indent=2))
    print(f"\n[*] -> {args.out}")


if __name__ == "__main__":
    main()
