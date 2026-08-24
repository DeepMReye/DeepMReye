"""A temporal viewing prior on the predicted gaze trajectory, estimated without labels.

**The spatial half of a "viewing prior" cannot help this metric, and that is a theorem, not a
result.** Pearson r is invariant to any affine map of the prediction, and a per-dataset gaze
mean and scale IS an affine map -- which is also why `analyze_calibration.py` finds
mis-calibration destroys R^2 while leaving r intact. Shrinking toward a dataset's centre or
matching its variance therefore changes nothing here. Only a *non-affine* spatial prior could
(dsL01's labels are a 9-point fixation grid, so snapping to it would help) and that requires
knowing the held-out dataset's paradigm, which leave-one-dataset-out forbids.

**The temporal half is exploitable, and it is exploitable with no labels at all.** Gaze is
strongly autocorrelated -- `analyze_temporal_ceiling.py` makes lag-1 autocorrelation the
single best predictor of decodability across the corpus -- while the readout's error is much
whiter, because each TR is predicted from its own voxels. That asymmetry is the entire
opening: a smoother suppresses the white part and keeps the correlated part, and the
strength can be read off the *prediction's own* autocovariance, never off the labels.

Why this is not the lag sweep again. `lags+-L` lets ridge learn a temporal filter, but it
must FIT `k` new coefficients per lag, which is why `lags+-2` already loses and `lags+-3`
loses badly. A smoother imposes the same prior with **one** parameter and a support of tens
of samples instead of two.

And it runs on the **sub-TR axis**: predictions are `[T, 10, 2]`, which unfolds to a
`[T*10, 2]` trajectory at 10x the resolution the lag stack ever sees. Nothing in this project
has operated on that axis.

The prediction the envelope law makes is testable and is the real check: the gain must track
each dataset's gaze autocorrelation -- large on `dsL01`/`dsL02`, near zero on `dsL03`, whose
gaze moves faster than its acquisition resolves.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deepmreye.temporal_probe import (ALPHAS, calibrate, cca_avg, corpus_fingerprint,
                                      load_subtr_cache, lodo_subtr, make_lags)


class TemporalPriorRidge:
    """RidgeCV whose predictions are filtered along the unfolded sub-TR time axis."""

    def __init__(self, mode="none", param=0.0, alphas=ALPHAS):
        self.mode, self.param, self.alphas = mode, param, alphas

    def fit(self, x, y):
        from sklearn.linear_model import RidgeCV
        self.base_ = RidgeCV(alphas=self.alphas).fit(x, y)
        return self

    def _filter(self, fine):
        if self.mode == "none":
            return fine
        if self.mode == "gauss":
            from scipy.ndimage import gaussian_filter1d
            return gaussian_filter1d(fine, self.param, axis=0, mode="nearest")
        if self.mode == "savgol":
            from scipy.signal import savgol_filter
            w = int(self.param) | 1                      # window must be odd
            if w <= 3 or w >= len(fine):
                return fine
            return savgol_filter(fine, w, 2, axis=0, mode="nearest")
        if self.mode == "wiener":
            return self._wiener(fine)
        raise ValueError(self.mode)

    @staticmethod
    def _wiener(fine):
        """Empirical Wiener gain from the signal's own spectrum -- no labels anywhere.

        Gaze is autocorrelated and the readout's error is close to white, so the high
        frequencies of the *predicted* trace are almost pure noise. Their mean power is
        therefore a usable noise estimate, and the Wiener gain (P - N) / P follows from it.
        """
        n = len(fine)
        if n < 32:
            return fine
        out = np.empty_like(fine)
        for j in range(fine.shape[1]):
            col = fine[:, j]
            mu = col.mean()
            spec = np.fft.rfft(col - mu)
            power = np.abs(spec) ** 2
            noise = np.median(power[len(power) // 2:])    # top half of the band
            gain = np.clip((power - noise) / np.maximum(power, 1e-12), 0.0, 1.0)
            out[:, j] = mu + np.fft.irfft(spec * gain, n)
        return out

    def predict(self, x):
        raw = self.base_.predict(x)
        return self._filter(raw.reshape(-1, 2)).reshape(len(raw), 20)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--basis", default="results/scaling/basis_n2000.npz")
    p.add_argument("--cache", default="results/subtr/labeled_subtr_cache.npz")
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--out", default="results/scaling/temporal_prior.json")
    args = p.parse_args()

    recs = load_subtr_cache(args.cache, Path(args.basis), args.m, False)
    print(f"[*] {len(recs)} participants, fingerprint {corpus_fingerprint(recs)[:12]}",
          flush=True)
    if not calibrate(recs):
        raise SystemExit("[!] calibration failed")
    print("[+] calibrated\n", flush=True)

    k = args.k
    arms = [("baseline", "none", 0.0)]
    arms += [(f"gauss sd={s}", "gauss", s) for s in (1, 2, 3, 5, 8, 12, 20, 32)]
    arms += [(f"savgol w={w}", "savgol", w) for w in (9, 21, 41, 81)]
    arms += [("wiener (adaptive)", "wiener", 0.0)]

    rows, folds_sub = [], {}
    print(f"{'arm':<20} {'sub-TR':>9} {'d':>8}   {'1-TR':>9} {'d':>8}")
    print("-" * 60)
    base = {}
    for name, mode, par in arms:
        def ro(mode=mode, par=par):
            return TemporalPriorRidge(mode, par)
        s = lodo_subtr(recs, lambda r: make_lags(cca_avg(r, k), 1), readout=ro)
        o = lodo_subtr(recs, lambda r: make_lags(cca_avg(r, k), 0), readout=ro)
        sub, one = s["median_subtr"], o["median_1tr"]
        if not base:
            base = {"subtr": sub, "1tr": one}
        folds_sub[name] = s["subtr"]
        rows.append({"arm": name, "subtr": sub, "1tr": one, "per_fold_subtr": s["subtr"]})
        print(f"{name:<20} {sub:>9.4f} {sub - base['subtr']:>+8.4f}   "
              f"{one:>9.4f} {one - base['1tr']:>+8.4f}", flush=True)

    # Per-fold detail for the best fixed width, against the envelope law's prediction.
    best = max((r for r in rows if r["arm"] != "baseline"), key=lambda r: r["subtr"])
    print(f"\nper-fold sub-TR, baseline -> {best['arm']}:")
    b0 = folds_sub["baseline"]
    for ds in sorted(b0):
        d = best["per_fold_subtr"].get(ds, float("nan")) - b0[ds]
        print(f"  {ds:<26} {b0[ds]:.4f} -> "
              f"{best['per_fold_subtr'].get(ds, float('nan')):.4f}  {d:+.4f}")

    # An oracle that picks the width per held-out dataset. NOT deployable -- it is the
    # ceiling this family could reach if the width were chosen perfectly, and it says
    # whether a smarter adaptive rule is worth building.
    gauss = [r for r in rows if r["arm"].startswith("gauss")]
    oracle = {ds: max(r["per_fold_subtr"].get(ds, -9) for r in gauss) for ds in b0}
    print(f"\noracle per-fold width  median sub-TR "
          f"{np.median(list(oracle.values())):.4f}  (baseline {np.median(list(b0.values())):.4f})")

    Path(args.out).write_text(json.dumps({"k": k, "rows": rows, "oracle": oracle}, indent=2))
    print(f"\n[+] {args.out}")


if __name__ == "__main__":
    main()
