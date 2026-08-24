"""The headline table with NO training-row subsampling. Every valid labeled row is used.

`MAX_TRAIN_ROWS = 20000` is a bare constant in `temporal_probe.py` with no comment and no
recorded justification, copied into eight scripts. It discards ~93% of the 303,733 valid
labeled rows before the ridge is fitted. Sweeping it showed the effect is negligible
(mean d -0.0001 over 9 folds, p = 0.820), but "negligible" is a measurement, not a reason,
and an unexplained subsample has no place in a published number. This script sets the cap
above the total row count so it can never trigger, and reports what that gives.

Reports sub-TR and 1-TR, each at its own optimal lag, per fold, with mean and median across
folds and the paired test against the capped run.
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

NO_CAP = 10 ** 9          # above any possible pooled row count -> the branch never fires


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--basis", default="results/scaling/basis_n2000.npz")
    p.add_argument("--cache", default="results/subtr/labeled_subtr_cache.npz")
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--out", default="results/scaling/publication_table.json")
    args = p.parse_args()

    recs = load_subtr_cache(args.cache, Path(args.basis), args.m, False)
    print(f"[*] {len(recs)} participants, {len({r['dataset'] for r in recs})} datasets, "
          f"fingerprint {corpus_fingerprint(recs)[:12]}", flush=True)
    if not calibrate(recs):
        raise SystemExit("[!] calibration failed")
    print("[+] calibrated\n", flush=True)
    k = args.k

    def run(lag, cap, ro=None):
        return lodo_subtr(recs, lambda r: make_lags(cca_avg(r, k), lag),
                          max_train_rows=cap, readout=ro)

    def savgol9():
        return TemporalPriorRidge("savgol", 9)

    arms = {
        "capped 20k, lags1":      run(1, 20000),
        "ALL ROWS, lags1":        run(1, NO_CAP),
        "ALL ROWS, lags1+savgol": run(1, NO_CAP, savgol9),
        "capped 20k, lags0":      run(0, 20000),
        "ALL ROWS, lags0":        run(0, NO_CAP),
        "ALL ROWS, lags0+savgol": run(0, NO_CAP, savgol9),
    }
    folds = sorted(arms["ALL ROWS, lags1"]["subtr"])

    print("=== sub-TR (lags +-1) ===")
    print(f"{'fold':<28} {'capped 20k':>11} {'ALL ROWS':>10} {'+savgol':>10}")
    for f in folds:
        print(f"{f:<28} {arms['capped 20k, lags1']['subtr'][f]:>11.4f} "
              f"{arms['ALL ROWS, lags1']['subtr'][f]:>10.4f} "
              f"{arms['ALL ROWS, lags1+savgol']['subtr'][f]:>10.4f}")
    print("\n=== 1-TR (lags 0) ===")
    print(f"{'fold':<28} {'capped 20k':>11} {'ALL ROWS':>10} {'+savgol':>10}")
    for f in folds:
        print(f"{f:<28} {arms['capped 20k, lags0']['1tr'][f]:>11.4f} "
              f"{arms['ALL ROWS, lags0']['1tr'][f]:>10.4f} "
              f"{arms['ALL ROWS, lags0+savgol']['1tr'][f]:>10.4f}")

    from scipy.stats import wilcoxon
    print(f"\n{'arm':<26} {'median':>8} {'mean':>8} {'vs capped':>10} {'wins':>6} {'p':>7}")
    for res, key, ref in (("capped 20k, lags1", "subtr", "capped 20k, lags1"),
                          ("ALL ROWS, lags1", "subtr", "capped 20k, lags1"),
                          ("ALL ROWS, lags1+savgol", "subtr", "capped 20k, lags1"),
                          ("capped 20k, lags0", "1tr", "capped 20k, lags0"),
                          ("ALL ROWS, lags0", "1tr", "capped 20k, lags0"),
                          ("ALL ROWS, lags0+savgol", "1tr", "capped 20k, lags0")):
        v = np.array([arms[res][key][f] for f in folds])
        b = np.array([arms[ref][key][f] for f in folds])
        d = v - b
        pv = wilcoxon(d).pvalue if np.any(d) else float("nan")
        print(f"{res:<26} {np.median(v):>8.4f} {v.mean():>8.4f} {d.mean():>+10.4f} "
              f"{int((d > 0).sum()):>4}/9 {pv:>7.3f}")

    Path(args.out).write_text(json.dumps(
        {"k": k, "basis": args.basis, "n_participants": len(recs),
         "fingerprint": corpus_fingerprint(recs),
         "arms": {a: {"subtr": r["subtr"], "1tr": r["1tr"],
                      "median_subtr": r["median_subtr"], "median_1tr": r["median_1tr"]}
                  for a, r in arms.items()}}, indent=2))
    print(f"\n[+] {args.out}")


if __name__ == "__main__":
    main()
