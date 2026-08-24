"""Read the 9-fold promotion runs: single-network LODO medians, per seed and per fold.

Reports the 9-fold median AND the median over the six folds never used for screening.
`dsL03`, `dsL05` and `dsL07` selected the configuration, so a tuned arm is optimistically
biased on exactly those three and a 9-fold median quoted alone would carry that bias
silently. Every arm is compared to its own per-fold incumbent, which is recomputed inside
each run, so the comparison is paired at the fold level by construction.

No ensembling anywhere: each (fold, seed) cell is one network, and the across-seed spread is
reported as a spread, never averaged into a prediction.
"""
import argparse, glob, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

SCREEN = ("dsL03_pursuit", "dsL05_free_viewing", "dsL07_deepmreye_calib")


def load(pattern):
    net, inc = defaultdict(dict), defaultdict(dict)
    for f in sorted(glob.glob(pattern)):
        m = re.match(r"(.+)__(.+)__s(\d+)$", Path(f).stem)
        if not m:
            continue
        try:
            d = json.loads(Path(f).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for _fd, r in d.get("results", {}).items():
            net[m.group(1)][(m.group(2), int(m.group(3)))] = r["net"]
            inc[m.group(1)][(m.group(2), int(m.group(3)))] = r["incumbent"]
    return net, inc


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--glob", default="results/subtr/promote/*.json")
    args = p.parse_args()
    net, inc = load(args.glob)
    if not net:
        raise SystemExit(f"[!] nothing matched {args.glob}")

    for tag in sorted(net):
        cells = sorted(net[tag])
        folds = sorted({c[0] for c in cells})
        seeds = sorted({c[1] for c in cells})
        unseen = [f for f in folds if f not in SCREEN]
        print(f"\n=== {tag} === {len(cells)} cells, {len(folds)} folds, {len(seeds)} seeds")
        print(f"{'fold':<28} {'incumbent':>10} {'net (per seed)':>34} {'mean d':>8}")
        for fd in folds:
            row = [net[tag][(fd, s)] for s in seeds if (fd, s) in net[tag]]
            ic = np.mean([inc[tag][(fd, s)] for s in seeds if (fd, s) in inc[tag]])
            mark = "" if fd in unseen else "  (screened on)"
            print(f"{fd:<28} {ic:>10.4f} {' '.join(f'{x:7.4f}' for x in row):>34} "
                  f"{np.mean(row) - ic:>+8.4f}{mark}")

        for name, sel in (("9-fold", folds), (f"{len(unseen)} unseen folds", unseen)):
            per_seed_net = [np.median([net[tag][(f, s)] for f in sel if (f, s) in net[tag]])
                            for s in seeds]
            per_seed_inc = [np.median([inc[tag][(f, s)] for f in sel if (f, s) in inc[tag]])
                            for s in seeds]
            d = np.array(per_seed_net) - np.array(per_seed_inc)
            print(f"  {name:<18} median net {np.mean(per_seed_net):.4f} "
                  f"(seeds {', '.join(f'{x:.4f}' for x in per_seed_net)})  "
                  f"incumbent {np.mean(per_seed_inc):.4f}  d {d.mean():+.4f}")

        cd = np.array([net[tag][c] - inc[tag][c] for c in cells])
        cu = np.array([net[tag][c] - inc[tag][c] for c in cells if c[0] in unseen])
        print(f"  per-cell vs incumbent: all {cd.mean():+.4f} ({int((cd>0).sum())}/{len(cd)}, "
              f"p={wilcoxon(cd).pvalue:.3f})   unseen only {cu.mean():+.4f} "
              f"({int((cu>0).sum())}/{len(cu)}, p={wilcoxon(cu).pvalue:.3f})")

    tags = sorted(net)
    if len(tags) > 1:
        print("\n=== head to head (paired on shared cells) ===")
        for i, a in enumerate(tags):
            for b in tags[i + 1:]:
                cs = sorted(set(net[a]) & set(net[b]))
                if not cs:
                    continue
                d = np.array([net[a][c] - net[b][c] for c in cs])
                print(f"  {a} - {b}: {d.mean():+.4f}  {int((d>0).sum())}/{len(d)}  "
                      f"p={wilcoxon(d).pvalue:.3f}")


if __name__ == "__main__":
    main()
