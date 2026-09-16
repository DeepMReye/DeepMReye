#!/usr/bin/env python3
"""Bases that are not a covariance decomposition at all.

Every arm this project has compared -- `corpus-pca`, `lr-cca`, `band-pca`,
`gev-*`, `diff-pca`, `nuis-pca` -- is a decomposition of a second moment. They
differ in *which* second moment and in what they maximise, but none of them can
see past second order, so "independence", "sparsity" and "non-Gaussianity" have
never been tried as selection criteria on this corpus.

The framing that makes the comparison fair: **a linear readout sees only the
subspace.** Rotating k features among themselves is invisible to ridge, so ICA
cannot help by rotating -- only by *choosing a different k-dimensional
subspace*. Each arm below is therefore scored as a subspace selection out of the
same 512 orbit principal directions, against the same ridge, with a random
orthogonal subspace of the same rank as the floor.

    `ica-kurtosis`  FastICA, components ranked by |excess kurtosis|. Gaze is
                    saccadic and bimodal where physiology and drift are closer
                    to Gaussian, so non-Gaussianity is a real candidate
                    criterion rather than an arbitrary one.
    `dict`          MiniBatch dictionary learning, atoms used as a linear
                    projection (the sparse *coding* step is non-linear and is
                    deliberately not used -- it would change the readout, not
                    the basis, and the non-linear readout ceiling is closed).
    `variance`      the top-k of the same 512, i.e. `corpus-pca` -- the control
                    that says what the subspace is worth with no criterion at
                    all beyond variance.

    python scripts/beyond_cca.py --voxels <voxdir>
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
from scripts.improve_lrcca import K_STORE, project, slicer, summarise  # noqa: E402


def load_pcs(path):
    d = np.load(path, allow_pickle=False)
    m = int(d["m"][0])
    n_vox = len(d["mean"])
    proj = np.zeros((n_vox, 2 * m), dtype=np.float32)
    proj[d["left_index"], :m] = d["left_pcs"]
    proj[d["right_index"], m:] = d["right_pcs"]
    return d["z"], proj, d["mean"].astype(np.float64), m


def fit_ica(z, k, seed=0, n_sources=64, max_rows=60000):
    """FastICA on the corpus PC timeseries, ranked by |excess kurtosis|."""
    from sklearn.decomposition import FastICA

    rng = np.random.default_rng(seed)
    rows = z if len(z) <= max_rows else z[rng.choice(len(z), max_rows, replace=False)]
    # FastICA on 512 whitened inputs does not reach a tight tolerance in a few
    # hundred sweeps; the arm is only meaningful if it converged, so the budget
    # is generous and non-convergence is reported rather than warned about.
    ica = FastICA(n_components=n_sources, random_state=seed, whiten="unit-variance",
                  max_iter=5000, tol=1e-4)
    s = ica.fit_transform(np.asarray(rows, dtype=np.float64))
    print(f"    FastICA: {ica.n_iter_} iterations", flush=True)
    s = (s - s.mean(0)) / np.maximum(s.std(0), 1e-12)
    kurt = np.abs((s ** 4).mean(0) - 3.0)
    order = np.argsort(-kurt)[:k]
    return ica.components_[order].T, kurt[order]


def fit_dict(z, k, seed=0, max_rows=40000):
    from sklearn.decomposition import MiniBatchDictionaryLearning

    rng = np.random.default_rng(seed)
    rows = z if len(z) <= max_rows else z[rng.choice(len(z), max_rows, replace=False)]
    d = MiniBatchDictionaryLearning(n_components=k, alpha=1.0, batch_size=256,
                                    max_iter=200, random_state=seed)
    d.fit(np.asarray(rows, dtype=np.float64))
    return d.components_.T, np.zeros(k)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--voxels", required=True)
    p.add_argument("--pcs", default="results/lrcca_variants/pcs_n2000.npz")
    p.add_argument("--out", default="results/lrcca_variants/beyond.json")
    p.add_argument("--k", type=int, default=32)
    a = p.parse_args()

    z, proj, mu, m = load_pcs(a.pcs)
    print(f"[*] corpus PC timeseries {z.shape}, {m} PCs per orbit", flush=True)

    blocks, t0 = [], time.time()
    for name, fn in (("ica-kurtosis", fit_ica), ("dict", fit_dict)):
        w, stat = fn(z, K_STORE)
        blocks.append((name, (proj @ w).astype(np.float32), mu, K_STORE))
        print(f"  {name:<16} {w.shape}  [{time.time() - t0:5.0f}s]", flush=True)
    # Variance control: the leading PCs of the same two orbits, same rank.
    order = np.concatenate([np.arange(K_STORE // 2), m + np.arange(K_STORE // 2)])
    w = np.zeros((2 * m, K_STORE), dtype=np.float32)
    w[order, np.arange(K_STORE)] = 1.0
    blocks.append(("orbit-variance", (proj @ w).astype(np.float32), mu, K_STORE))

    recs, index = project(a.voxels, blocks)
    got = {}
    print(f"\n{'arm':<28}{'sub-TR':>9}{'1-TR':>9}")
    for name, _w, _mu, _k in blocks:
        for k in (a.k, K_STORE):
            row = f"{name}:{k}"
            got[row] = summarise(probe.lodo(recs, slicer(index, name, k=k, lags=1)))
            print(f"{row:<28}{got[row]['subtr']:>9.4f}{got[row]['1tr']:>9.4f}", flush=True)
            Path(a.out).write_text(json.dumps(got, indent=1))
    print(f"\n[+] {a.out}")


if __name__ == "__main__":
    main()
