"""Rank voxelnet hyperparameter configurations from the screening sweep.

Selection is on **best_val**, the median sub-TR r over validation datasets drawn from each
fold's TRAINING pool. That never touches the held-out dataset, so tuning on it does not
contaminate the final leave-one-dataset-out number the way tuning on `test_r` would.

Every configuration runs the same (fold, seed) cells, so comparisons are **paired**: the
seed sets the model init, the training subsample AND the validation-dataset draw, and it is
the dominant variance component (SD ~0.028 on a single cell). Unpaired means across seeds
hide effects that paired differences show, so a `vs base` column and a win count are printed
rather than a bare mean.
"""
import argparse, glob, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--glob", default="results/subtr/sweep/*.json")
    p.add_argument("--base", default=None, help="Tag to compute paired differences against.")
    p.add_argument("--sort", default="val", choices=("val", "test"))
    args = p.parse_args()

    val, test, cells = defaultdict(dict), defaultdict(dict), set()
    for f in sorted(glob.glob(args.glob)):
        m = re.match(r"(.+)__(.+)__s(\d+)$", Path(f).stem)
        if not m:
            continue
        tag, fold, seed = m.group(1), m.group(2), int(m.group(3))
        try:
            d = json.loads(Path(f).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for _fold, r in d.get("results", {}).items():
            h = r.get("history") or []
            if not h:
                continue
            key = (fold, seed)
            val[tag][key] = max(x["val_r"] for x in h)
            test[tag][key] = (r["net"], r["incumbent"])
            cells.add(key)

    if not val:
        raise SystemExit(f"[!] nothing matched {args.glob}")
    # The reference cell set is the base config's, or -- with no base -- the largest set any
    # config has. It is deliberately NOT the intersection over all configs: a single crashed
    # job would then empty it and every row would silently fall back to an UNPAIRED mean over
    # whatever cells that config happens to hold, which is the one number this table must
    # never print. A config missing a cell now degrades its own row and is marked `*`.
    if args.base in val:
        full = sorted(val[args.base])
    else:
        full = sorted(max(val.values(), key=len))
    print(f"[*] {len(val)} configs, {len(cells)} cells seen, "
          f"{len(full)} reference cells ({'base' if args.base in val else 'largest config'})")

    rows = []
    for tag, v in val.items():
        common = [c for c in full if c in v]
        mv = float(np.mean([v[c] for c in common])) if common else float("nan")
        td = [test[tag][c][0] - test[tag][c][1] for c in common if c in test[tag]]
        rows.append((tag, mv, float(np.mean(td)) if td else float("nan"), len(common)))

    base = args.base if args.base in val else None
    key = 1 if args.sort == "val" else 2
    rows.sort(key=lambda r: -(r[key] if np.isfinite(r[key]) else -9))
    hdr = f"{'config':<34} {'n':>3} {'mean val':>9} {'mean dTest':>11}"
    if base:
        hdr += f" {'dVal vs base':>13} {'win':>6}"
    print(hdr); print("-" * len(hdr))
    for tag, mv, mt, n in rows:
        flag = " " if n == len(full) else "*"
        line = f"{tag:<34} {n:>2}{flag} {mv:>9.4f} {mt:>+11.4f}"
        if base:
            common = [c for c in full if c in val[tag] and c in val[base]]
            d = [val[tag][c] - val[base][c] for c in common]
            if d:
                line += f" {np.mean(d):>+13.4f} {sum(x > 0 for x in d)}/{len(d):<4}"
        print(line + ("   <-- base" if tag == base else ""))


if __name__ == "__main__":
    main()
