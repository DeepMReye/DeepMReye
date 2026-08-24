"""Four structured, label-free-basis-preserving ways to read gaze out of `lr-cca` better.

`k` and corpus size are both closed (2026-08-22): 32 is exactly optimal and the corpus
saturates at N~800. Non-linear encoders and non-linear readouts are closed too
(`analyze_nonlinear_ceiling.py`). What has NOT been examined is the *structure* of the linear
readout itself, which has always been a plain `RidgeCV` on `0.5 * (z_left + z_right)`. Every
arm here stays linear and keeps the frozen unsupervised basis untouched.

Only sub-TR and 1-TR are reported -- the 5-TR binning `eval_probe` uses is not a resolution
anyone deploys at.

1. **avg+diff.** `cca_avg` averages the two orbits and throws `z_left - z_right` away. CCA
   builds the pair to maximise agreement, so the average is the agreed part -- but the two
   orbits' weight vectors are fitted independently, and the antisymmetric mode is nowhere
   shown to be nuisance. One line to test, and it doubles the feature count rather than
   changing the basis.

2. **Reduced-rank ridge.** The 20 targets are 10 sub-TR samples x 2 axes and are massively
   correlated, yet ridge fits all 20 independently. Constraining the coefficient matrix to
   rank r is the classic structured regulariser for exactly this and has never been tried
   here. Implemented as the projection of the fitted values onto their own top-r subspace,
   which is the standard RRR solution.

3. **Smooth sub-TR output (DCT).** Gaze inside one TR is a smooth trajectory, so predicting a
   few DCT coefficients per axis instead of 10 independent samples imposes a smoothness prior
   on the *target* rather than on the weights. This is the one arm aimed squarely at the
   sub-TR metric.

4. **Canonical-correlation weighting.** Ridge's penalty is isotropic after standardisation,
   so every canonical direction is treated as equally trustworthy -- but the basis ships
   `canonical_correlations`, which is exactly a per-direction reliability the corpus already
   estimated with no labels. Scaling feature j by rho_j gives direction j an effective penalty
   alpha / rho_j^2. Parameter-free at p=1.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deepmreye.temporal_probe import (ALPHAS, calibrate, cca_avg, corpus_fingerprint,
                                      load_subtr_cache, lodo_subtr, make_lags)


def cca_avg_diff(rec, k=32):
    """Orbit mean AND orbit difference: the part CCA agreed on, plus the part it did not."""
    z = rec["z"]
    return np.concatenate([0.5 * (z[:, 0, :k] + z[:, 1, :k]),
                           0.5 * (z[:, 0, :k] - z[:, 1, :k])], axis=1).astype(np.float64)


class ReducedRankRidge:
    """RidgeCV whose 20 outputs are constrained to an r-dimensional subspace."""

    def __init__(self, rank=4, alphas=ALPHAS):
        self.rank, self.alphas = rank, alphas

    def fit(self, x, y):
        from sklearn.linear_model import RidgeCV
        self.base_ = RidgeCV(alphas=self.alphas).fit(x, y)
        yh = self.base_.predict(x)
        self.mu_ = yh.mean(axis=0)
        # Top-r right singular vectors of the fitted values are the RRR output subspace.
        _u, _s, vt = np.linalg.svd(yh - self.mu_, full_matrices=False)
        self.p_ = vt[:self.rank].T @ vt[:self.rank]
        return self

    def predict(self, x):
        yh = self.base_.predict(x)
        return self.mu_ + (yh - self.mu_) @ self.p_


class SmoothOutputRidge:
    """RidgeCV on the leading `n_coef` DCT coefficients of the 10 sub-TR samples per axis."""

    def __init__(self, n_coef=4, alphas=ALPHAS):
        self.n_coef, self.alphas = n_coef, alphas

    @staticmethod
    def _fwd(y, c):
        from scipy.fft import dct
        return dct(y.reshape(len(y), 10, 2), axis=1, norm="ortho")[:, :c, :].reshape(len(y), -1)

    @staticmethod
    def _inv(coef, c):
        from scipy.fft import idct
        full = np.zeros((len(coef), 10, 2))
        full[:, :c, :] = coef.reshape(len(coef), c, 2)
        return idct(full, axis=1, norm="ortho").reshape(len(coef), 20)

    def fit(self, x, y):
        from sklearn.linear_model import RidgeCV
        self.base_ = RidgeCV(alphas=self.alphas).fit(x, self._fwd(y, self.n_coef))
        return self

    def predict(self, x):
        return self._inv(self.base_.predict(x), self.n_coef)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--basis", default="results/scaling/basis_n2000.npz")
    p.add_argument("--cache", default="results/subtr/labeled_subtr_cache.npz")
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--out", default="results/scaling/cca_readout_sweep.json")
    args = p.parse_args()

    from deepmreye.unsupervised import load_basis

    recs = load_subtr_cache(args.cache, Path(args.basis), args.m, False)
    print(f"[*] {len(recs)} participants, fingerprint {corpus_fingerprint(recs)[:12]}",
          flush=True)
    if not calibrate(recs):
        raise SystemExit("[!] calibration failed -- fix that before trusting anything here")
    print("[+] calibrated\n", flush=True)

    _m, bases, _meta = load_basis(Path(args.basis))
    rho = np.asarray(bases["lr-cca"]["canonical_correlations"], dtype=np.float64)
    k = args.k

    def rho_weighted(rec, power):
        return cca_avg(rec, k) * (rho[:k] ** power)

    arms = {
        "baseline  lr-cca:32": (lambda r: cca_avg(r, k), None),
        "avg+diff  (2x32)": (lambda r: cca_avg_diff(r, k), None),
        "rho-weighted p=1": (lambda r: rho_weighted(r, 1.0), None),
        "rho-weighted p=2": (lambda r: rho_weighted(r, 2.0), None),
        "rho-weighted p=-1": (lambda r: rho_weighted(r, -1.0), None),
    }
    for rank in (2, 3, 4, 6, 8):
        arms[f"reduced-rank r={rank}"] = (lambda r: cca_avg(r, k),
                                          lambda rank=rank: ReducedRankRidge(rank))
    for c in (2, 3, 4, 6):
        arms[f"dct-smooth c={c}"] = (lambda r: cca_avg(r, k),
                                     lambda c=c: SmoothOutputRidge(c))

    rows = []
    print(f"{'arm':<24} {'sub-TR':>9} {'d':>8}   {'1-TR':>9} {'d':>8}")
    print("-" * 64)
    base = {}
    for name, (fn, ro) in arms.items():
        # lags 1 is the sub-TR optimum, lags 0 the 1-TR optimum; run each at its own.
        sub = lodo_subtr(recs, lambda r, fn=fn: make_lags(fn(r), 1), readout=ro)["median_subtr"]
        one = lodo_subtr(recs, lambda r, fn=fn: make_lags(fn(r), 0), readout=ro)["median_1tr"]
        if not base:
            base = {"subtr": sub, "1tr": one}
        rows.append({"arm": name, "subtr": sub, "1tr": one})
        print(f"{name:<24} {sub:>9.4f} {sub - base['subtr']:>+8.4f}   "
              f"{one:>9.4f} {one - base['1tr']:>+8.4f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"k": k, "basis": args.basis, "rows": rows}, indent=2))
    print(f"\n[+] {args.out}")


if __name__ == "__main__":
    main()
