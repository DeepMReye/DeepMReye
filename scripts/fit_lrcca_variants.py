#!/usr/bin/env python3
"""One corpus pass that keeps enough moments to fit *every* cross-orbit variant.

`fit_lr_cca` has always been fitted from the **total** voxel covariance: one
global mean is subtracted and every TR of every participant is pooled. That
covariance is dominated by *between-participant* structure -- orbit size,
registration offset, susceptibility -- and that structure is bilaterally
coherent, so a cross-orbit CCA is entitled to spend its leading canonical
directions on it. Gaze, by contrast, is a *within-participant* signal: the probe
correlates a prediction against a trace inside one run, so any direction that
only separates participants is a wasted column of the k=32 budget.

The decomposition is exact and costs one extra rank-1 update per participant::

    C_total = C_within + C_between
    C_between = sum_i n_i (mu_i - mu)(mu_i - mu)^T / n

so this pass stores the second moment, the per-participant means and the
per-slab means, and every covariance below is a difference of those. Nothing
here is fitted -- `--fit` does that afterwards from the saved moments, in
seconds, which is what makes a shrinkage sweep affordable.

    python scripts/fit_lrcca_variants.py --accumulate --n 2000
    python scripts/fit_lrcca_variants.py --fit
"""
import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmreye.datasource import resolve  # noqa: E402
from deepmreye.unsupervised import (  # noqa: E402
    Moments, _slabs, corpus_mask, unlabeled_subjects,
)


def accumulate_with_means(subjects, mask, trs_per_subject=48, n_slabs=4,
                          progress=100):
    """`Moments` plus the per-participant and per-slab means it centred away."""
    flat = mask.reshape(-1)
    moments = Moments(int(flat.sum()))
    sub_mu, sub_n, slab_mu, slab_n = [], [], [], []
    t0 = time.time()
    for i, (_ds, _sub, path, n_trs) in enumerate(subjects):
        try:
            with h5py.File(path, "r") as f:
                block = f["eye_block"]
                rows = []
                for start, stop in _slabs(n_trs, trs_per_subject, n_slabs):
                    slab = block[..., start:stop]
                    x = slab.reshape(-1, slab.shape[-1])[flat].T
                    if len(x) < 2:
                        continue
                    moments.add(x)
                    slab_mu.append(x.mean(axis=0, dtype=np.float64))
                    slab_n.append(len(x))
                    rows.append(x)
        except Exception as e:
            print(f"    [!] skipping {path}: {e}", flush=True)
            continue
        if not rows:
            continue
        allx = np.concatenate(rows)
        sub_mu.append(allx.mean(axis=0, dtype=np.float64))
        sub_n.append(len(allx))
        moments.n_subjects += 1
        if progress and i % progress == 0:
            print(f"  [{i + 1}/{len(subjects)}] {moments.n} TRs from "
                  f"{moments.n_subjects} subjects ({time.time() - t0:.0f}s)", flush=True)
    moments.symmetrise()
    return moments, (np.asarray(sub_mu, dtype=np.float32), np.asarray(sub_n)), \
        (np.asarray(slab_mu, dtype=np.float32), np.asarray(slab_n))


def accumulate_lag1(subjects, mask, trs_per_subject=48, n_slabs=4, progress=100):
    """`sum_t x_t x_{t+1}^T`, the one-TR-lagged cross moment.

    With the plain second moment this spans the whole temporal family without a
    third pass: the covariance of temporal *differences* is ``2C - (L + L^T)``
    and the covariance of temporal *sums* is ``2C + (L + L^T)``, both up to the
    edge terms a slab of twelve TRs makes negligible. That matters because the
    recorded reason every predictive objective failed on this corpus is that the
    *predictable* part of an eye block is the nuisance -- motion, drift, global
    signal -- while gaze is nearly white between TRs. A cross-orbit basis fitted
    on differences asks the bilateral question about the fast part only.

    Pairs are taken **inside** a slab, never across the gap between two slabs,
    which would pair TRs minutes apart and call the result a lag of one.
    """
    from scipy.linalg.blas import sgemm

    flat = mask.reshape(-1)
    d = int(flat.sum())
    # Fortran order and `overwrite_c` for the same reason `Moments` needs them:
    # a 14236^2 temporary per slab is 810 MB of pure allocation, and buffering
    # amortises each pass over the accumulator (see `Moments`' docstring).
    lag = np.zeros((d, d), dtype=np.float32, order="F")
    buf0, buf1, n_buf, n_pairs = [], [], 0, 0
    t0 = time.time()

    def flush():
        nonlocal buf0, buf1, n_buf
        if not buf0:
            return
        a0 = np.asfortranarray(np.concatenate(buf0))
        a1 = np.asfortranarray(np.concatenate(buf1))
        out = sgemm(alpha=1.0, a=a0, b=a1, trans_a=1, beta=1.0, c=lag, overwrite_c=1)
        if out is not lag:
            raise RuntimeError("sgemm did not accumulate in place")
        buf0, buf1, n_buf = [], [], 0
    for i, (_ds, _sub, path, n_trs) in enumerate(subjects):
        try:
            with h5py.File(path, "r") as f:
                block = f["eye_block"]
                for start, stop in _slabs(n_trs, trs_per_subject, n_slabs):
                    slab = block[..., start:stop]
                    x = np.ascontiguousarray(
                        slab.reshape(-1, slab.shape[-1])[flat].T, dtype=np.float32)
                    if len(x) < 3:
                        continue
                    buf0.append(x[:-1])
                    buf1.append(x[1:])
                    n_buf += len(x) - 1
                    n_pairs += len(x) - 1
                    if n_buf >= 1024:
                        flush()
        except Exception as e:
            print(f"    [!] skipping {path}: {e}", flush=True)
            continue
        if progress and i % progress == 0:
            print(f"  [{i + 1}/{len(subjects)}] {n_pairs} pairs "
                  f"({time.time() - t0:.0f}s)", flush=True)
    flush()
    return lag, n_pairs


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out", default="results/lrcca_variants/moments_n2000.npz")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--trs-per-subject", type=int, default=48)
    p.add_argument("--n-slabs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0,
                   help="Must match sweep_corpus_scaling.py's shuffle, or the "
                        "corpus this fits is not the corpus basis_n2000 saw.")
    p.add_argument("--lag1", action="store_true",
                   help="Accumulate the lag-1 cross moment instead of the "
                        "second moment and the group means.")
    a = p.parse_args()

    data_dir = resolve(a.data_dir, download=False, quiet=True)
    mask = corpus_mask(data_dir)
    subjects = unlabeled_subjects(data_dir)
    rng = np.random.default_rng(a.seed)
    subjects = [subjects[i] for i in rng.permutation(len(subjects))][:a.n]
    print(f"[*] {len(subjects)} unlabeled participants, seed {a.seed}", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if a.lag1:
        t0 = time.time()
        lag, n_pairs = accumulate_lag1(subjects, mask, a.trs_per_subject, a.n_slabs)
        print(f"[*] {n_pairs} lagged pairs in {time.time() - t0:.0f}s", flush=True)
        np.savez(out, lag=lag, n_pairs=np.array([n_pairs]), mask=mask)
        print(f"[+] {out} ({out.stat().st_size / 1e9:.2f} GB)")
        return

    t0 = time.time()
    moments, (smu, sn), (bmu, bn) = accumulate_with_means(
        subjects, mask, a.trs_per_subject, a.n_slabs)
    print(f"[*] {moments.n} TRs from {moments.n_subjects} subjects "
          f"in {time.time() - t0:.0f}s", flush=True)

    np.savez(out, c=moments.c, s=moments.s, n=np.array([moments.n]),
             n_subjects=np.array([moments.n_subjects]), mask=mask,
             subject_mean=smu, subject_n=sn, slab_mean=bmu, slab_n=bn,
             meta=np.array(json.dumps({
                 "n_subjects": moments.n_subjects, "n_trs": int(moments.n),
                 "trs_per_subject": a.trs_per_subject, "n_slabs": a.n_slabs,
                 "seed": a.seed})))
    print(f"[+] {out} ({out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
