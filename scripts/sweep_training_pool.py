"""How the training pool is composed -- the last untouched degree of freedom.

Every arm in this project has been fitted on the same pool, built two ways nobody has
questioned:

- **`MAX_TRAIN_ROWS = 20000`**, out of **303,733** valid rows. 93% of the labeled data is
  discarded before the ridge ever sees it. That constant has never been swept.
- **Unbalanced pooling.** `dsL04` supplies **37.4%** of the valid rows and `dsL06` **0.2%**
  (`dsL07` 0.7%), so the single pooled ridge behind every leave-one-dataset-out number is
  fitted overwhelmingly on one paradigm. `--standardize-targets dataset` equalises each
  dataset's target *scale*; it does nothing about its *count*.

Both are plausibly load-bearing and neither is a model change: same basis, same features, same
readout. Reported at sub-TR and 1-TR, each at its own optimal lag.

The two are crossed rather than swept separately, because a per-dataset cap and a global cap
interact by construction -- capping datasets lowers the total, which can put it under the
global cap and make that one inert.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sweep_temporal_prior import TemporalPriorRidge          # noqa: E402

from deepmreye.temporal_probe import (calibrate, cca_avg, corpus_fingerprint,  # noqa: E402
                                      load_subtr_cache, lodo_subtr, make_lags)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--basis", default="results/scaling/basis_n2000.npz")
    p.add_argument("--cache", default="results/subtr/labeled_subtr_cache.npz")
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--out", default="results/scaling/training_pool.json")
    args = p.parse_args()

    recs = load_subtr_cache(args.cache, Path(args.basis), args.m, False)
    print(f"[*] {len(recs)} participants, fingerprint {corpus_fingerprint(recs)[:12]}",
          flush=True)
    if not calibrate(recs):
        raise SystemExit("[!] calibration failed")
    print("[+] calibrated\n", flush=True)
    k = args.k

    rows = []
    print(f"{'max_rows':>9} {'per-ds cap':>11} {'sub-TR':>9} {'d':>8} {'1-TR':>9} {'d':>8} {'w':>4}")
    print("-" * 62)
    base = {}
    for mx in (20000, 50000, 100000, 300000):
        for cap in (None, 2000, 5000, 10000, 20000):
            s = lodo_subtr(recs, lambda r: make_lags(cca_avg(r, k), 1),
                           max_train_rows=mx, balance_rows=cap)
            o = lodo_subtr(recs, lambda r: make_lags(cca_avg(r, k), 0),
                           max_train_rows=mx, balance_rows=cap)
            sub, one = s["median_subtr"], o["median_1tr"]
            if not base:
                base = {"subtr": sub, "1tr": one}
            wins = sum(1 for d in s["subtr"]
                       if s["subtr"][d] > rows[0]["per_fold_subtr"][d]) if rows else 0
            rows.append({"max_rows": mx, "cap": cap, "subtr": sub, "1tr": one,
                         "per_fold_subtr": s["subtr"]})
            print(f"{mx:>9} {str(cap):>11} {sub:>9.4f} {sub - base['subtr']:>+8.4f} "
                  f"{one:>9.4f} {one - base['1tr']:>+8.4f} {wins:>3}/9", flush=True)

    best = max(rows, key=lambda r: r["subtr"])
    print(f"\nbest pool: max_rows={best['max_rows']} cap={best['cap']} -> "
          f"sub-TR {best['subtr']:.4f} ({best['subtr'] - base['subtr']:+.4f})")

    # Compose the winner with the robust narrow filter (9/9 folds, p=0.004).
    def ro():
        return TemporalPriorRidge("savgol", 9)
    s = lodo_subtr(recs, lambda r: make_lags(cca_avg(r, k), 1), readout=ro,
                   max_train_rows=best["max_rows"], balance_rows=best["cap"])
    d = np.array([s["subtr"][f] - rows[0]["per_fold_subtr"][f] for f in sorted(s["subtr"])])
    print(f"  + savgol w=9 filter -> sub-TR {s['median_subtr']:.4f}   "
          f"vs incumbent {base['subtr']:.4f} ({s['median_subtr'] - base['subtr']:+.4f}), "
          f"{int((d > 0).sum())}/9 folds")
    rows.append({"max_rows": best["max_rows"], "cap": best["cap"], "filter": "savgol9",
                 "subtr": s["median_subtr"], "per_fold_subtr": s["subtr"]})

    Path(args.out).write_text(json.dumps({"k": k, "rows": rows}, indent=2))
    print(f"\n[+] {args.out}")


if __name__ == "__main__":
    main()
