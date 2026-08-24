"""Refine the sub-TR temporal prior: filter shape x polynomial order x input lags.

The first pass found Savitzky-Golay w=21 worth +0.0080 sub-TR, beating a Gaussian of matched
support (+0.0067). That ordering is predicted by the readout's own structure: the DCT sweep
showed the model's within-TR prediction is a **mean plus a slope**, and a polynomial filter
preserves a slope where a Gaussian attenuates it.

Two questions this pass answers.

- **Does the smoother make the input lags redundant?** `lags+-1` and an output filter are both
  temporal mixing, so they may be doing the same job twice. If they are, `lags 0 + filter` is
  the cheaper model and should match; if the filter is doing something lags cannot, the two
  should add. Nothing here can be inferred from the first pass, which held lags fixed at its
  own optimum.
- **What is the right polynomial order and width?** w=21 was the coarse grid's best with two
  neighbours 12 and 20 samples away, so the optimum is only located to within a factor of two.
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


class SavGolRidge(TemporalPriorRidge):
    def __init__(self, width=21, poly=2, alphas=None):
        super().__init__("savgol", width, alphas if alphas is not None else __import__(
            "deepmreye.temporal_probe", fromlist=["ALPHAS"]).ALPHAS)
        self.poly = poly

    def _filter(self, fine):
        from scipy.signal import savgol_filter
        w = int(self.param) | 1
        if w <= self.poly + 1 or w >= len(fine):
            return fine
        return savgol_filter(fine, w, self.poly, axis=0, mode="nearest")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--basis", default="results/scaling/basis_n2000.npz")
    p.add_argument("--cache", default="results/subtr/labeled_subtr_cache.npz")
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--out", default="results/scaling/temporal_prior_refine.json")
    args = p.parse_args()

    recs = load_subtr_cache(args.cache, Path(args.basis), args.m, False)
    print(f"[*] {len(recs)} participants, fingerprint {corpus_fingerprint(recs)[:12]}",
          flush=True)
    if not calibrate(recs):
        raise SystemExit("[!] calibration failed")
    print("[+] calibrated\n", flush=True)
    k = args.k

    rows = []
    print(f"{'lags':>4} {'poly':>4} {'width':>6} {'sub-TR':>9} {'1-TR':>9}")
    print("-" * 36)
    for lag in (0, 1, 2):
        # width 0 = no filter, the honest per-lag reference
        for poly, width in [(0, 0)] + [(pp, ww) for pp in (1, 2, 3)
                                       for ww in (9, 13, 17, 21, 25, 31, 41)]:
            if poly == 0:
                ro = None
            else:
                def ro(width=width, poly=poly):
                    return SavGolRidge(width, poly)
            s = lodo_subtr(recs, lambda r, l=lag: make_lags(cca_avg(r, k), l), readout=ro)
            rows.append({"lags": lag, "poly": poly, "width": width,
                         "subtr": s["median_subtr"], "1tr": s["median_1tr"],
                         "per_fold_subtr": s["subtr"]})
            print(f"{lag:>4} {poly:>4} {width:>6} {s['median_subtr']:>9.4f} "
                  f"{s['median_1tr']:>9.4f}", flush=True)

    Path(args.out).write_text(json.dumps({"k": k, "rows": rows}, indent=2))
    for key in ("subtr", "1tr"):
        best = max(rows, key=lambda r: r[key])
        ref = [r for r in rows if r["poly"] == 0 and r["lags"] == (1 if key == "subtr" else 0)][0]
        print(f"\n[{key}] best lags{best['lags']} poly{best['poly']} w{best['width']} "
              f"-> {best[key]:.4f}   (unfiltered reference {ref[key]:.4f}, "
              f"delta {best[key] - ref[key]:+.4f})")
    print(f"\n[+] {args.out}")


if __name__ == "__main__":
    main()
