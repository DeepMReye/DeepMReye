#!/usr/bin/env python3
"""The temporal ceiling: does ANY supervised model beat a linear lag stack at sub-TR gaze?

`analyze_nonlinear_ceiling.py` answered the *spatial* version of this question and closed the
non-linear program: no supervised non-linear readout beats ridge on the same per-bin features.
But every arm there is a static map from ONE bin's canonical coordinates to that bin's gaze --
no lags, no window, no sequence model. The temporal version has never been measured, and the
incumbent (`lr-cca:32 + lags+-2`) is itself a *linear* temporal model, so the gap is live.

Same logic as its spatial sibling, which is what makes it a ceiling rather than a baseline:
the probe readout is linear, so a temporal encoder in front of it can only pay if gaze depends
on the window in a way a linear map cannot express. That is upper-bounded by what a
*supervised* model gets on the same window -- generous, since it sees the labels the encoder
never does and optimises the exact quantity scored. If supervised loses here, no unsupervised
temporal objective can win.

Three blocks, one process, one protocol (`temporal_probe.lodo_subtr`, calibrated):

  A. Is the incumbent under-tuned? `results/temporal_lag_sweep.json` reports an inverted U
     peaking at L=1, but it is within-subject `KFold` at a FIXED `Ridge(alpha=10.0)`
     (`sweep_temporal_lags.py:157`), so at L=5 it is 352 columns at a fixed penalty and the
     decline is confounded with under-regularisation. Re-measured here under LODO with
     RidgeCV, plus a per-lag-penalty arm.
  B. Is there gaze outside the k=32 canonical span? This is what a voxel-level model buys.
  C. The actual ceiling: non-linear readouts over the window.

Do NOT extend `analyze_nonlinear_ceiling.py` in place -- its table is quoted verbatim in
CLAUDE.md and STATE.md, and editing it re-points those numbers at a different corpus and a
different resolution. This imports its readouts so the estimators are literally the same
objects.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_nonlinear_ceiling import make_readout  # noqa: E402
from deepmreye.temporal_probe import (  # noqa: E402
    ALPHAS, CALIBRATION, calibrate, cca_avg, corpus_fingerprint, load_subtr_cache, make_lags,
)

NOISE_FLOOR = 0.02


def cross_time_poly(x, k, n_lags, n_cross=8):
    """Squares plus cross-*time* products of the leading directions.

    The interpretable version of what a temporal conv could learn: `z_j(t) * z_j(t+-1)` is a
    product across time within one canonical direction. Restricted to the leading `n_cross`
    directions and to time-pairs, so this measures temporal non-linearity rather than "more
    columns" -- the same discipline `analyze_nonlinear_ceiling.expand` applies spatially.
    """
    blocks = [x[:, i * k:(i + 1) * k] for i in range(n_lags)]
    c = min(n_cross, k)
    cross = []
    for a in range(n_lags - 1):
        for b in range(a + 1, n_lags):
            cross.append(blocks[a][:, :c] * blocks[b][:, :c])
    return np.concatenate([x, x ** 2] + cross, axis=1)


class TCNRegressor:
    """A weight-shared 1D conv over the lag axis, with an sklearn-shaped interface.

    This is the arm that tests the actual hypothesis behind a temporal network: a lag stack
    pays `k` new ridge columns per lag, while a conv reuses one kernel at every offset, so it
    can afford a longer window. Rows arriving here are already `make_lags` windows, so the
    temporal axis is inside the row -- which is what lets a sequence model be scored through
    the same `lodo_subtr` as every other arm rather than a second protocol.

    Deliberately small (a few thousand parameters) and heavily regularised: the question is
    whether weight sharing buys anything at all, not whether a large model can be tuned into
    winning.
    """

    def __init__(self, k, n_lags, hidden=64, dropout=0.2, epochs=60, lr=1e-3, seed=0):
        self.k, self.n_lags = k, n_lags
        self.hidden, self.dropout, self.epochs, self.lr, self.seed = hidden, dropout, epochs, lr, seed

    def _build(self, n_out):
        import torch.nn as nn
        pad = 1
        return nn.Sequential(
            nn.Conv1d(self.k, self.hidden, 3, padding=pad), nn.GELU(), nn.Dropout(self.dropout),
            nn.Conv1d(self.hidden, self.hidden, 3, padding=pad, dilation=1), nn.GELU(),
            nn.Dropout(self.dropout),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(self.hidden, n_out))

    def _shape(self, x):
        import torch
        t = torch.as_tensor(np.asarray(x, dtype=np.float32))
        return t.view(len(t), self.n_lags, self.k).transpose(1, 2)   # [N, k, n_lags]

    def fit(self, x, y):
        import torch
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        self.mu_, self.sd_ = x.mean(0), x.std(0) + 1e-9
        xb = self._shape((x - self.mu_) / self.sd_)
        yb = torch.as_tensor(np.asarray(y, dtype=np.float32))
        self.net_ = self._build(yb.shape[1])
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=1e-2)
        n_val = max(int(0.1 * len(xb)), 1)
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(len(xb), generator=g)
        va, tr = perm[:n_val], perm[n_val:]
        best, best_state, bad = float("inf"), None, 0
        for _ in range(self.epochs):
            self.net_.train()
            idx = tr[torch.randperm(len(tr), generator=g)]
            for i in range(0, len(idx), 512):
                b = idx[i:i + 512]
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(self.net_(xb[b]), yb[b])
                loss.backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                v = float(torch.nn.functional.mse_loss(self.net_(xb[va]), yb[va]))
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= 10:
                    break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        return self

    def predict(self, x):
        import torch
        self.net_.eval()
        with torch.no_grad():
            return self.net_(self._shape((x - self.mu_) / self.sd_)).numpy()


def tcn_readout(k, n_lags):
    def factory():
        return TCNRegressor(k, n_lags)
    return factory


def banded_readout(blocks):
    from deepmreye.evaluate.combine import BandedRidge

    def factory():
        return BandedRidge(blocks=list(blocks))
    return factory


def build_arms(args):
    """(name, block, feature_fn, readout_factory)."""
    arms = []

    # --- Block A: the linear lag curve, properly regularised, under LODO -------------
    for lag in range(0, args.max_lag + 1):
        arms.append((f"ridge-L{lag}", "A",
                     lambda r, L=lag: make_lags(cca_avg(r, 32), L), None))
    for lag in (2, 4):
        n = 2 * lag + 1
        arms.append((f"banded-L{lag}", "A",
                     lambda r, L=lag: make_lags(cca_avg(r, 32), L),
                     banded_readout([32] * n)))

    # --- Block B: is there signal outside the k=32 canonical span? -------------------
    for k in (64, 128, 256):
        arms.append((f"ridge-k{k}-L2", "B",
                     lambda r, K=k: make_lags(cca_avg(r, K), 2), None))
    arms.append(("banded-k128-L2", "B",
                 lambda r: make_lags(cca_avg(r, 128), 2),
                 banded_readout([128] * 5)))
    # Concatenating the two orbits instead of averaging them: the L/R average is a choice
    # nobody has revisited, and it halves the rank available to the readout.
    arms.append(("ridge-lr-concat32-L2", "B",
                 lambda r: make_lags(
                     np.concatenate([r["z"][:, 0, :32], r["z"][:, 1, :32]],
                                    axis=1).astype(np.float64), 2), None))

    # --- Block C: the temporal non-linearity ceiling ---------------------------------
    for lag in (2, 4):
        arms.append((f"mlp-L{lag}", "C",
                     lambda r, L=lag: make_lags(cca_avg(r, 32), L),
                     lambda: make_readout("mlp", 0)))
    arms.append(("poly-time-L2", "C",
                 lambda r: cross_time_poly(make_lags(cca_avg(r, 32), 2), 32, 5), None))
    for lag in (2, 4):
        arms.append((f"tcn-L{lag}", "C",
                     lambda r, L=lag: make_lags(cca_avg(r, 32), L),
                     tcn_readout(32, 2 * lag + 1)))
    if args.gbt:
        arms.append(("gbt-L2", "C",
                     lambda r: make_lags(cca_avg(r, 32), 2),
                     lambda: make_readout("gbt", 0)))

    if args.only:
        keep = set(args.only)
        arms = [a for a in arms if a[0] in keep]
    return arms


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cache", default="results/subtr/labeled_subtr_cache.npz")
    p.add_argument("--basis", default="results/scaling/basis_n2000.npz")
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--max-lag", type=int, default=5)
    p.add_argument("--gbt", action="store_true", help="Include the slow gradient-boosting arm.")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--skip-calibration", action="store_true")
    p.add_argument("--out", default="results/subtr/temporal_ceiling_supervised.json")
    args = p.parse_args()

    from deepmreye.temporal_probe import lodo_subtr

    recs = load_subtr_cache(args.cache, Path(args.basis), args.m, False)
    datasets = sorted({r["dataset"] for r in recs})
    print(f"[*] {len(recs)} participants, {len(datasets)} datasets, "
          f"fingerprint {corpus_fingerprint(recs)[:12]}", flush=True)

    if not args.skip_calibration:
        print("[*] calibrating:", flush=True)
        if not calibrate(recs):
            raise SystemExit("[!] calibration failed -- no ordering from this run is quotable")

    arms = build_arms(args)
    print(f"\n[*] {len(arms)} arms\n", flush=True)
    results = {}
    for name, block, feat_fn, readout in arms:
        t0 = time.time()
        out = lodo_subtr(recs, feat_fn, readout=readout)
        results[name] = {"block": block, **out}
        print(f"  [{block}] {name:<24} sub-TR {out['median_subtr']:.4f}   "
              f"1-TR {out['median_1tr']:.4f}   ({time.time() - t0:.0f}s)", flush=True)

    # ---- verdict -------------------------------------------------------------------
    def best(block):
        cand = {k: v for k, v in results.items() if v["block"] == block}
        if not cand:
            return None, float("nan")
        k = max(cand, key=lambda n: cand[n]["median_subtr"])
        return k, cand[k]["median_subtr"]

    a_name, a_val = best("A")
    print("\n" + "=" * 96)
    print(f"{'arm':<26}{'block':>6}{'sub-TR':>10}{'1-TR':>10}{'vs best linear':>16}{'folds won':>12}")
    print("-" * 96)
    for name, v in sorted(results.items(), key=lambda kv: -kv[1]["median_subtr"]):
        won = ""
        if a_name and name != a_name:
            ref = results[a_name]["subtr"]
            won = f"{sum(1 for d in datasets if v['subtr'].get(d, np.nan) > ref.get(d, np.nan))}/{len(datasets)}"
        delta = v["median_subtr"] - a_val
        print(f"{name:<26}{v['block']:>6}{v['median_subtr']:>10.4f}{v['median_1tr']:>10.4f}"
              f"{delta:>+16.4f}{won:>12}")
    print("=" * 96)

    print(f"\nbest linear (block A): {a_name} at {a_val:.4f}")
    for block in ("B", "C"):
        name, val = best(block)
        if name is None:
            continue
        ref = results[a_name]["subtr"]
        folds = sum(1 for d in datasets
                    if results[name]["subtr"].get(d, np.nan) > ref.get(d, np.nan))
        clears = (val - a_val) > NOISE_FLOOR and folds >= 6
        print(f"best block {block}:        {name} at {val:.4f} "
              f"({val - a_val:+.4f}, {folds}/{len(datasets)} folds) -> "
              f"{'PASS' if clears else 'does not clear the gate'}")

    print(f"\nGate: > +{NOISE_FLOOR} median AND >= 6/{len(datasets)} folds over the best "
          f"linear arm, in this run.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"datasets": datasets, "n_participants": len(recs),
         "fingerprint": corpus_fingerprint(recs), "calibration": CALIBRATION,
         "noise_floor": NOISE_FLOOR, "results": results}, indent=1))
    print(f"[+] {args.out}")


if __name__ == "__main__":
    main()
