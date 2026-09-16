#!/usr/bin/env python3
"""The cross-orbit basis a fold is actually entitled to fit.

`lr-cca` is deliberately *inductive*: it is fitted on the unlabelled corpus and
never sees a gaze dataset, which is the claim the package ships. But a
leave-one-dataset-out fold is entitled to more than that. The eight training
datasets' **voxels** are available to it -- their labels are already used to fit
the readout -- and the fold-local reference (`fold-pca`) is built from exactly
those voxels. So the honest comparison is three-way, not two:

    fold-pca          8 labelled datasets' voxels, variance ordering
    lr-cca            2000 unlabelled participants, cross-orbit ordering
    lr-cca hybrid     both, one basis per fold

and the arm that has never been run is the third. It is still label-free (a
covariance cannot read a label) and still leave-one-dataset-out (the held-out
dataset is excluded from its own fold's accumulator), so it is quotable under
the same protocol.

`lr-cca-labelled` is the control that separates the two ingredients: the same
cross-orbit fit on the *labelled* voxels alone. If it beats `fold-pca` at
identical data then the bilateral criterion is worth something on its own; if
only the hybrid wins, the corpus is what carries it.

    python scripts/hybrid_lrcca.py --voxels <voxdir>
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
from deepmreye.unsupervised import _slabs, _top_eigenvectors  # noqa: E402
from scripts.improve_lrcca import (  # noqa: E402
    Corpus, Eigs, K_STORE, cca_to_block, concat, fit_cca, project, slicer, summarise,
)


def dataset_moments(voxdir, trs_per_subject, n_slabs=4, chunk=8192):
    """Second moment and row sum per labelled dataset, off the voxel memmap.

    A fixed TR budget per participant for the same reason the corpus pass uses
    one: `dsL03` runs 3.5x longer than `dsL07`, and an unbudgeted accumulator
    would describe whichever study scans longest rather than whichever study is
    there.
    """
    from scipy.linalg.blas import ssyrk

    voxdir = Path(voxdir)
    plan = json.loads((voxdir / "index.json").read_text())
    mm = np.load(voxdir / "voxels.npy", mmap_mode="r")
    d = mm.shape[1]
    out, t0 = {}, time.time()
    for ds in sorted({r["dataset"] for r in plan}):
        c = np.zeros((d, d), dtype=np.float32, order="F")
        s, n, buf, n_buf = np.zeros(d, dtype=np.float64), 0, [], 0

        def flush(buf, n_buf, c):
            if not buf:
                return [], 0
            x = np.asfortranarray(np.concatenate(buf))
            got = ssyrk(alpha=1.0, a=x, trans=1, beta=1.0, c=c, overwrite_c=1)
            if got is not c:
                raise RuntimeError("syrk did not accumulate in place")
            return [], 0

        for r in (p for p in plan if p["dataset"] == ds):
            t = r["stop"] - r["start"]
            for a, b in _slabs(t, trs_per_subject, n_slabs):
                x = np.asarray(mm[r["start"] + a:r["start"] + b], dtype=np.float32)
                if len(x) < 2:
                    continue
                buf.append(x)
                n_buf += len(x)
                s += x.sum(axis=0, dtype=np.float64)
                n += len(x)
                if n_buf >= chunk:
                    buf, n_buf = flush(buf, n_buf, c)
        flush(buf, n_buf, c)
        iu = np.triu_indices_from(c, k=1)
        c[(iu[1], iu[0])] = c[iu]
        out[ds] = (c, s, n)
        print(f"  {ds:<28}{n:>7} TRs  [{time.time() - t0:5.0f}s]", flush=True)
    return out


def combined_cov(parts):
    """`sum` of `(second moment, row sum, n)` triples -> mean-centred covariance."""
    c = sum(p[0].astype(np.float64) for p in parts)
    s = sum(p[1] for p in parts)
    n = sum(p[2] for p in parts)
    mu = s / n
    return c / n - np.outer(mu, mu), mu


def score_per_fold_fn(recs, folds, make_fn):
    """Each fold under its own feature function, reduced by `probe._summarise`."""
    rows = []
    for held in folds:
        res = probe.lodo(recs, make_fn(held))
        rows += [p for p in res["participants"] if p["dataset"] == held]
    return probe._summarise(rows, folds)


def score_per_fold(recs, index, folds, k=32, lags=1):
    """Each fold under its own basis, reduced by the shipped `_summarise`.

    `probe.lodo` applies one feature function to every record, so a per-fold
    basis has to be run fold by fold and filtered down to that fold's
    participants. The reduction is still `probe._summarise`.
    """
    rows = []
    for held in folds:
        res = probe.lodo(recs, slicer(index, held, k=k, lags=lags))
        rows += [p for p in res["participants"] if p["dataset"] == held]
    return probe._summarise(rows, folds)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--voxels", required=True)
    p.add_argument("--moments", default="results/lrcca_variants/moments_n2000.npz")
    p.add_argument("--out", default="results/lrcca_variants/hybrid.json")
    p.add_argument("--trs", type=int, nargs="+", default=[48, 200],
                   help="Labelled TRs per participant folded in. 48 matches the "
                        "corpus pass; 200 matches what fold-pca is given.")
    p.add_argument("--shrinkage", type=float, default=1e-3)
    p.add_argument("--k", type=int, default=32)
    a = p.parse_args()

    corpus = Corpus(a.moments)
    li, ri = corpus.orbits()
    n_vox = int(corpus.mask.sum())
    corpus_part = (corpus.c.astype(np.float32), corpus.s, corpus.n)

    # The corpus basis itself, constant across folds: the second block of the
    # concatenation arms, and the thing they are asking fold-local features to
    # be insufficient without.
    cov_corpus = corpus.c.astype(np.float64) / corpus.n - np.outer(corpus.mu, corpus.mu)
    corpus_fit = fit_cca(cov_corpus, li, ri, K_STORE, 256, a.shrinkage,
                         eigs=Eigs(), key="corpus")
    blocks = [("corpus",) + cca_to_block(corpus_fit, n_vox, corpus.mu) + (K_STORE,)]
    del cov_corpus
    eigs = Eigs()
    for trs in a.trs:
        print(f"[*] labelled moments at {trs} TRs/participant", flush=True)
        per_ds = dataset_moments(a.voxels, trs)
        folds = sorted(per_ds)
        # `transductive` is NOT a leave-one-dataset-out number and must never be
        # reported as one: it folds the held-out dataset's own voxels into its
        # own basis. It is here because it is the deployment case -- you have
        # the participant's scan, you have no eye tracker -- and because it
        # upper-bounds what domain adaptation could ever buy on this corpus.
        for tag, use_corpus in (("hybrid", True), ("labelled", False),
                                ("transductive", True)):
            for held in folds:
                keep = folds if tag == "transductive" else [d for d in folds if d != held]
                parts = [per_ds[d] for d in keep]
                if use_corpus:
                    parts.append(corpus_part)
                cov, mu = combined_cov(parts)
                key = f"{tag}{trs}_{held}"
                fit = fit_cca(cov, li, ri, K_STORE, 256, a.shrinkage, eigs=eigs, key=key)
                blocks.append((key,) + cca_to_block(fit, n_vox, mu) + (K_STORE,))
                # The variance-ordered basis over the same rows: `fold-pca`, refit
                # here rather than quoted, so the comparison is one protocol.
                if not use_corpus:
                    vecs, _ = _top_eigenvectors(cov, K_STORE)
                    blocks.append((f"foldpca{trs}_{held}", vecs.astype(np.float32),
                                   mu, K_STORE))
                print(f"    {key:<34} rho1 {fit['rho'][0]:.3f}", flush=True)
            eigs._c.clear()
        del per_ds

    recs, index = project(a.voxels, blocks)
    folds = sorted({r["dataset"] for r in recs})
    got = {}
    print(f"\n{'arm':<36}{'sub-TR':>9}{'1-TR':>9}")
    for trs in a.trs:
        for tag, k, lags in (("hybrid", a.k, 1), ("labelled", a.k, 1),
                             ("foldpca", 64, 1), ("transductive", a.k, 1)):
            keys = [f"{tag}{trs}_{h}" for h in folds]
            if not all(x in index for x in keys):
                continue
            name = f"{tag}  {trs} labelled TRs/sub"
            got[name] = summarise(score_per_fold(
                recs, {h: index[f"{tag}{trs}_{h}"] for h in folds}, folds, k, lags))
            print(f"{name:<36}{got[name]['subtr']:>9.4f}{got[name]['1tr']:>9.4f}",
                  flush=True)
            Path(a.out).write_text(json.dumps(got, indent=1))

        # Does the corpus carry anything the fold's own labelled voxels cannot?
        # Concatenation answers that directly: if `fold-pca + corpus` beats
        # `fold-pca` the corpus is adding information rather than replacing it,
        # which is a different and stronger claim than a higher median.
        for tag, k in (("foldpca", 64), ("labelled", a.k)):
            if f"{tag}{trs}_{folds[0]}" not in index:
                continue
            name = f"{tag} + lr-cca:32 corpus  {trs} TRs/sub"
            got[name] = summarise(score_per_fold_fn(recs, folds, lambda held, t=tag, kk=k:
                                                    concat([
                                                        slicer(index, f"{t}{trs}_{held}",
                                                               k=kk, lags=1),
                                                        slicer(index, "corpus", k=a.k,
                                                               lags=1)])))
            print(f"{name:<36}{got[name]['subtr']:>9.4f}{got[name]['1tr']:>9.4f}",
                  flush=True)
            Path(a.out).write_text(json.dumps(got, indent=1))
    got["lr-cca:32 corpus"] = summarise(probe.lodo(recs, slicer(index, "corpus", k=a.k,
                                                                lags=1)))
    print(f"{'lr-cca:32 corpus':<36}{got['lr-cca:32 corpus']['subtr']:>9.4f}"
          f"{got['lr-cca:32 corpus']['1tr']:>9.4f}", flush=True)
    Path(a.out).write_text(json.dumps(got, indent=1))
    print(f"\n[+] {a.out}")


if __name__ == "__main__":
    main()
