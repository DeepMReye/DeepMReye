#!/usr/bin/env python3
"""Bilateral agreement is a spatial contrast. What does a temporal one buy?

`lr-cca` maximises `corr(z_L(t), z_R(t))`: two orbits, same instant. In the
linear limit that already *is* a contrastive objective -- it is exactly what you
get from cross-view InfoNCE when the negatives are drawn at random times. So the
arm this project ships has a temporal story it has never told: **its negatives
are easy ones.**

The hard-negative version is the question here. "Agree across the orbits at
*this* TR and **not** at the neighbouring one" asks for directions that are
bilateral *and* fast, and it separates gaze from the bilaterally coherent slow
nuisance -- motion, drift, physiology -- that `ocon` showed the cross-orbit
constraint inherits by construction. Two linear forms, both fitted from moments
this repository already has:

`lag-deflate`   Replace the cross-covariance `C_LR` with
                `C_LR - lam * (C_LR^(1) + C_LR^(1)T) / 2`, the lag-1 cross-orbit
                moment deflated out, **while the whitening stays on the ordinary
                covariance**. That last clause is the whole point: the `fast`
                arm in `improve_lrcca.py` differenced the whitening as well, so
                it changed the metric and the criterion at once and could not
                say which one cost it 0.03. Here only the criterion moves.
                `lam=0` is the incumbent; `lam=1` is the cross-covariance of the
                temporal difference.

`multilag`      CCA between a **3-TR window** of the left orbit and a 3-TR
                window of the right, so the temporal mixing is chosen by the
                bilateral criterion instead of being a fixed `lags +-1` stack
                bolted onto the readout. The corpus lag stack is 768-dimensional
                per orbit in PC space, which is free; in voxel space it would be
                a 42708^2 accumulator, which is why it was never run.

    python scripts/corpus_pc_timeseries.py
    python scripts/temporal_contrast.py --voxels <voxdir>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmreye import probe  # noqa: E402
from scripts.improve_lrcca import (  # noqa: E402
    Corpus, Eigs, K_STORE, _whiten, cca_to_block, project, slicer, summarise,
)

LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]


# --------------------------------------------------------------------------- #
# 1. Hard negatives at lag 1, with the whitening left alone
# --------------------------------------------------------------------------- #

def fit_lag_deflated(corpus, li, ri, lam, k=K_STORE, n_reduce=256, shrinkage=1e-3,
                     eigs=None):
    """CCA whose cross block is `C_LR` minus `lam` times its lag-1 version."""
    cov = corpus.cov("total")
    eigs = eigs or Eigs()
    wl = _whiten(*eigs.get(cov, ("total", "l"), li), n_reduce, shrinkage)
    wr = _whiten(*eigs.get(cov, ("total", "r"), ri), n_reduce, shrinkage)
    cross = cov[np.ix_(li, ri)]
    if lam:
        cross = cross - lam * corpus.lag[np.ix_(li, ri)]
    u, s, vt = np.linalg.svd(wl.T @ cross @ wr, full_matrices=False)
    k = int(min(k, u.shape[1], vt.shape[0]))
    return {"left_index": li, "right_index": ri, "signs": np.ones(k),
            "left_weights": wl @ u[:, :k], "right_weights": wr @ vt[:k].T,
            "rho": s[:k]}


# --------------------------------------------------------------------------- #
# 2. Cross-orbit CCA over a temporal window, in corpus-PC space
# --------------------------------------------------------------------------- #

def stack_lagged(z, slab, lags):
    """`[T, D]` -> `[T', D * len(lags)]`, never crossing a slab boundary."""
    out, keep = [], None
    order = np.arange(len(z))
    edges = np.flatnonzero(np.diff(slab)) + 1
    groups = np.split(order, edges)
    lo, hi = min(lags), max(lags)
    rows = [g[hi:len(g) + lo] if len(g) > hi - lo else g[:0] for g in groups]
    keep = np.concatenate(rows)
    for lag in lags:
        out.append(z[keep + lag])
    return np.concatenate(out, axis=1), keep


def fit_multilag(pcs, lags, k=K_STORE, n_reduce=256, shrinkage=1e-3):
    """CCA between the two orbits' lagged windows, solved in PC space.

    Returns per-lag voxel weight maps, so the feature at time `t` is the sum of
    three projections of `x(t + lag)` -- i.e. the basis carries the temporal
    filter and the readout does not have to.
    """
    z, slab = pcs["z"], pcs["slab"]
    m = int(pcs["m"][0])
    stacked, _keep = stack_lagged(np.asarray(z, dtype=np.float64), slab, lags)
    stacked -= stacked.mean(0)
    n_l = len(lags)

    # Column order after stacking is [lag0 L|R, lag1 L|R, ...]; pull the two
    # orbits back apart.
    lcols = np.concatenate([i * 2 * m + np.arange(m) for i in range(n_l)])
    rcols = np.concatenate([i * 2 * m + m + np.arange(m) for i in range(n_l)])
    xl, xr = stacked[:, lcols], stacked[:, rcols]
    n = len(stacked)
    cll = xl.T @ xl / n
    crr = xr.T @ xr / n
    clr = xl.T @ xr / n

    def white(c):
        vals, vecs = np.linalg.eigh(c)
        order = np.argsort(-vals)
        vals, vecs = vals[order][:n_reduce], vecs[:, order][:, :n_reduce]
        return vecs / np.sqrt(np.maximum(vals, 0) + shrinkage * float(vals.max()))

    wl, wr = white(cll), white(crr)
    u, s, vt = np.linalg.svd(wl.T @ clr @ wr, full_matrices=False)
    k = int(min(k, u.shape[1], vt.shape[0]))
    al, ar = wl @ u[:, :k], wr @ vt[:k].T          # [m * n_lags, k]

    lpcs, rpcs = pcs["left_pcs"], pcs["right_pcs"]
    per_lag = []
    for i in range(n_l):
        per_lag.append((lpcs @ al[i * m:(i + 1) * m].astype(np.float32),
                        rpcs @ ar[i * m:(i + 1) * m].astype(np.float32)))
    return per_lag, s[:k]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--voxels", required=True)
    p.add_argument("--moments", default="results/lrcca_variants/moments_n2000.npz")
    p.add_argument("--lag-moments", default="results/lrcca_variants/lag1_n2000.npz")
    p.add_argument("--pcs", default="results/lrcca_variants/pcs_n2000.npz")
    p.add_argument("--out", default="results/lrcca_variants/temporal_contrast.json")
    p.add_argument("--k", type=int, default=32)
    a = p.parse_args()

    corpus = Corpus(a.moments, a.lag_moments)
    li, ri = corpus.orbits()
    n_vox = int(corpus.mask.sum())
    eigs = Eigs()

    blocks, t0 = [], time.time()
    for lam in LAMBDAS:
        fit = fit_lag_deflated(corpus, li, ri, lam, eigs=eigs)
        name = f"deflate_l{lam:g}"
        blocks.append((name,) + cca_to_block(fit, n_vox, corpus.mu) + (K_STORE,))
        print(f"  {name:<22} rho1 {fit['rho'][0]:.3f}  rho32 {fit['rho'][31]:.3f}"
              f"  [{time.time() - t0:5.0f}s]", flush=True)
    del corpus._cache

    pcs = np.load(a.pcs, allow_pickle=False)
    lag_sets = {"multilag1": (-1, 0, 1), "multilag0": (0,), "multilag2": (-2, -1, 0, 1, 2)}
    lag_index = {}
    for name, lags in lag_sets.items():
        per_lag, rho = fit_multilag(pcs, list(lags))
        keys = []
        for lag, (lw, rw) in zip(lags, per_lag):
            w = np.zeros((n_vox, K_STORE), dtype=np.float32)
            w[li] = 0.5 * lw
            w[ri] += 0.5 * rw
            key = f"{name}_lag{lag}"
            blocks.append((key, w, np.zeros(n_vox), K_STORE))
            keys.append((lag, key))
        lag_index[name] = keys
        print(f"  {name:<22} rho1 {rho[0]:.3f}  rho32 {rho[31]:.3f}"
              f"  [{time.time() - t0:5.0f}s]", flush=True)

    recs, index = project(a.voxels, blocks)
    report(recs, index, lag_index, a)


def multilag_feature(index, keys, k):
    """Sum the per-lag projections, edge-padded exactly as `probe.make_lags` is."""
    spans = [(lag, index[key][0]) for lag, key in keys]

    def fn(rec):
        z = rec["z"]
        acc = np.zeros((len(z), k))
        for lag, lo in spans:
            part = z[:, lo:lo + k].astype(np.float64)
            if lag < 0:
                part = np.pad(part[:lag], ((-lag, 0), (0, 0)), mode="edge")
            elif lag > 0:
                part = np.pad(part[lag:], ((0, lag), (0, 0)), mode="edge")
            acc += part
        return acc
    return fn


def report(recs, index, lag_index, a):
    got = {}
    print(f"\n{'arm':<34}{'sub-TR':>9}{'1-TR':>9}")
    for lam in LAMBDAS:
        name = f"lag-deflate  lambda={lam:g}"
        got[name] = summarise(probe.lodo(
            recs, slicer(index, f"deflate_l{lam:g}", k=a.k, lags=1)))
        print(f"{name:<34}{got[name]['subtr']:>9.4f}{got[name]['1tr']:>9.4f}", flush=True)
        Path(a.out).write_text(json.dumps(got, indent=1))
    for name, keys in lag_index.items():
        for extra in (0, 1):
            label = f"{name}  readout lags{extra}"
            fn = multilag_feature(index, keys, a.k)
            got[label] = summarise(probe.lodo(
                recs, lambda r, f=fn, e=extra: probe.make_lags(f(r), e)))
            print(f"{label:<34}{got[label]['subtr']:>9.4f}{got[label]['1tr']:>9.4f}",
                  flush=True)
            Path(a.out).write_text(json.dumps(got, indent=1))
    print(f"\n[+] {a.out}")


if __name__ == "__main__":
    main()
