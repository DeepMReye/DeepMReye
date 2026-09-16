#!/usr/bin/env python3
"""The unlabelled corpus as a small timeseries, so criteria past second order can run.

Everything fitted from `moments_n2000.npz` is a decomposition of a covariance,
which fixes the family: variance, cross-covariance, correlation, and the
temporal versions of those. Anything that needs **higher-order** statistics --
independence, sparsity, non-Gaussianity -- cannot be computed from a covariance
at all, and that is why no arm in this project has ever used one.

A covariance is also all you need to make the corpus *small*. Projected onto the
top 256 principal directions of each orbit, 96,000 corpus TRs are a 200 MB array
instead of 5 GB of HDF5, and a method that wants the actual timeseries becomes a
few seconds of work. The reduction is lossless for anything that then lives in
that subspace, which every basis compared here does.

Note what this does *not* buy. A linear readout sees only the **subspace** a
basis spans: rotating k features among themselves is invisible to ridge. So a
method like ICA can only matter here through which directions it *selects*,
never through the rotation it applies -- and it must be scored that way.

    python scripts/corpus_pc_timeseries.py --out results/lrcca_variants/pcs_n2000.npz
"""
import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmreye.datasource import resolve  # noqa: E402
from deepmreye.unsupervised import _slabs, corpus_mask, unlabeled_subjects  # noqa: E402
from scripts.improve_lrcca import Corpus, Eigs  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--moments", default="results/lrcca_variants/moments_n2000.npz")
    p.add_argument("--out", default="results/lrcca_variants/pcs_n2000.npz")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--m", type=int, default=256, help="PCs kept per orbit.")
    p.add_argument("--trs-per-subject", type=int, default=48)
    p.add_argument("--n-slabs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    corpus = Corpus(a.moments)
    li, ri = corpus.orbits()
    eigs = Eigs(n_reduce=a.m)
    cov = corpus.cov("total")
    vl, _ = eigs.get(cov, ("total", "l"), li)
    vr, _ = eigs.get(cov, ("total", "r"), ri)
    del cov, corpus._cache

    n_vox = int(corpus.mask.sum())
    proj = np.zeros((n_vox, 2 * a.m), dtype=np.float32)
    proj[li, :a.m] = vl[:, :a.m]
    proj[ri, a.m:] = vr[:, :a.m]
    off = (corpus.mu @ proj.astype(np.float64)).astype(np.float32)

    data_dir = resolve(a.data_dir, download=False, quiet=True)
    mask = corpus_mask(data_dir)
    flat = mask.reshape(-1)
    subjects = unlabeled_subjects(data_dir)
    rng = np.random.default_rng(a.seed)
    subjects = [subjects[i] for i in rng.permutation(len(subjects))][:a.n]

    chunks, owner, slab_id, t0 = [], [], [], time.time()
    n_slab = 0
    for i, (_ds, _sub, path, n_trs) in enumerate(subjects):
        try:
            with h5py.File(path, "r") as f:
                block = f["eye_block"]
                for start, stop in _slabs(n_trs, a.trs_per_subject, a.n_slabs):
                    slab = block[..., start:stop]
                    x = slab.reshape(-1, slab.shape[-1])[flat].T.astype(np.float32)
                    if len(x) < 2:
                        continue
                    chunks.append(x @ proj - off)
                    owner.append(np.full(len(x), i, dtype=np.int32))
                    # Slab id, not just subject id: a temporal model must never
                    # pair the last TR of one slab with the first of the next,
                    # which are minutes apart in the run.
                    slab_id.append(np.full(len(x), n_slab, dtype=np.int32))
                    n_slab += 1
        except Exception as e:
            print(f"    [!] skipping {path}: {e}", flush=True)
            continue
        if i % 200 == 0:
            print(f"  [{i + 1}/{len(subjects)}] {time.time() - t0:.0f}s", flush=True)

    z = np.concatenate(chunks)
    subj = np.concatenate(owner)
    slab = np.concatenate(slab_id)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, z=z, subject=subj, slab=slab, left_pcs=vl[:, :a.m].astype(np.float32),
             right_pcs=vr[:, :a.m].astype(np.float32),
             left_index=li, right_index=ri, mean=corpus.mu.astype(np.float32),
             m=np.array([a.m]))
    print(f"[+] {out}  z {z.shape} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
