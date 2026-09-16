#!/usr/bin/env python3
"""Rank every candidate against the incumbent, by fold rather than by median.

The 9-fold median is a poor instrument below ~0.02: an arm can move it 0.018
while winning four folds of nine. Every table this project has ranked by median
alone has had to be re-read for that, so this one reports the paired fold-level
picture beside the median -- how many of the nine held-out datasets improved,
the mean change across them, and a sign-test p-value -- and sorts by nothing, so
the incumbent stays first and the deltas stay readable.

    python scripts/report_lrcca.py results/lrcca_variants/*.json
"""
import argparse
import json
import sys
from math import comb
from pathlib import Path


def sign_test(deltas, tol=1e-9):
    """Two-sided exact binomial on the signs, ties dropped."""
    d = [x for x in deltas if abs(x) > tol]
    if not d:
        return len(deltas) - len(d), 1.0
    up = sum(1 for x in d if x > 0)
    n = len(d)
    tail = sum(comb(n, i) for i in range(0, min(up, n - up) + 1)) / 2 ** n
    return up, min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--baseline", default="lr-cca:32+lags1  [incumbent]")
    ap.add_argument("--res", default="subtr", choices=("subtr", "1tr"))
    a = ap.parse_args()

    rows = {}
    for f in a.files:
        for name, row in json.loads(Path(f).read_text()).items():
            if name.startswith("_") or "folds" not in row:
                continue
            rows.setdefault(f"{name}", row)

    if a.baseline not in rows:
        sys.exit(f"[!] baseline {a.baseline!r} not among {len(rows)} arms")
    base = rows[a.baseline]
    folds = sorted(base["folds"])

    print(f"{'arm':<36}{'median':>8}{'d':>9}{'mean':>8}{'d':>9}"
          f"{'folds':>7}{'p':>7}")
    print("-" * 84)
    for name, row in rows.items():
        if sorted(row["folds"]) != folds:
            continue
        d = [row["folds"][f][a.res] - base["folds"][f][a.res] for f in folds]
        up, p = sign_test(d)
        print(f"{name:<36}{row[a.res]:>8.4f}{row[a.res] - base[a.res]:>+9.4f}"
              f"{row[a.res + '_mean']:>8.4f}"
              f"{row[a.res + '_mean'] - base[a.res + '_mean']:>+9.4f}"
              f"{up:>4}/{len(folds):<2}{p:>7.3f}")


if __name__ == "__main__":
    main()
