"""Retune the canonical-component count `k` for `lr-cca` on the N=2000 corpus basis.

`CLAUDE.md` records a law: **the optimal `k` FALLS as the unlabeled corpus grows** --
`corpus-pca` peaks at 256 when N=25 and at 64 when N=800; `lr-cca` peaks at 64 at N=800 and
at 32 at N=1039 -- because a well-estimated basis is compact while a noisy one needs many
components for ridge to recombine. It also says to retune `k` whenever the corpus size
changes.

**`lr-cca:32` was tuned at N=1039 and is being used at N=2000**, so the law predicts the
current setting is too wide. This script is that check. It is cheap: the sub-TR cache already
stores 256 canonical coordinates per participant, so every `k` is a column slice of data
already on disk -- no basis refit, no corpus pass, no GPU.

Both resolutions are reported because `CLAUDE.md` records that the optimal lag count differs
between them (`lags+-1` at sub-TR, `lags+-0` at 1-TR), so `k` may too, and an arm that quotes
one while implying the other is exactly the failure `temporal_probe` exists to prevent.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deepmreye.temporal_probe import (calibrate, cca_avg, corpus_fingerprint, load_subtr_cache,
                                      lodo_subtr, make_lags)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--basis", default="results/scaling/basis_n2000.npz")
    p.add_argument("--cache", default="results/subtr/labeled_subtr_cache.npz")
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--ks", type=int, nargs="*",
                   default=[8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 96, 128])
    p.add_argument("--lags", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--out", default="results/scaling/k_sweep_n2000.json")
    p.add_argument("--skip-calibrate", action="store_true")
    args = p.parse_args()

    recs = load_subtr_cache(args.cache, Path(args.basis), args.m, False)
    print(f"[*] {len(recs)} participants, "
          f"{len({r['dataset'] for r in recs})} datasets, "
          f"fingerprint {corpus_fingerprint(recs)[:12]}", flush=True)

    # The ordering this script produces is only worth reading if the protocol still
    # reproduces the headline it is supposed to be tuning.
    if not args.skip_calibrate:
        if not calibrate(recs):
            raise SystemExit("[!] calibration failed -- fix that before trusting any k here")
        print("[+] calibrated", flush=True)

    rows = []
    for lag in args.lags:
        for k in args.ks:
            def fn(r, k=k, lag=lag):
                return make_lags(cca_avg(r, k), lag)
            res = lodo_subtr(recs, fn)
            rows.append({"k": k, "lags": lag, **res})
            print(f"  lr-cca:{k:<4} lags{lag}   sub-TR {res['median_subtr']:.4f}   "
                  f"1-TR {res['median_1tr']:.4f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"basis": args.basis, "m": args.m, "n_participants": len(recs),
         "fingerprint": corpus_fingerprint(recs), "rows": rows}, indent=2))

    for key, label in (("median_subtr", "sub-TR"), ("median_1tr", "1-TR")):
        best = max(rows, key=lambda r: r[key] if np.isfinite(r[key]) else -9)
        cur = [r for r in rows if r["k"] == 32 and r["lags"] == (1 if key == "median_subtr" else 0)]
        cur = cur[0][key] if cur else float("nan")
        print(f"\n[{label}] best k={best['k']} lags{best['lags']} -> {best[key]:.4f}"
              f"   (k=32 incumbent {cur:.4f}, delta {best[key] - cur:+.4f})")
    print(f"\n[+] {args.out}")


if __name__ == "__main__":
    main()
