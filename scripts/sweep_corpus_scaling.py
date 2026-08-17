#!/usr/bin/env python3
"""Does the *unlabeled* corpus actually buy anything as it grows?

This is the question every "unsupervised pretraining" claim on this project has
assumed an answer to without measuring it. `corpus-pca` was fitted once, on 1005
participants, and compared against a fold-local PCA. Nobody has ever asked
whether it would have been just as good on 50 -- which is the difference between
"unlabeled data helps" and "unlabeled data is redundant after the first
hundred", and those are opposite papers.

The sweep is **incremental**, deliberately. `Moments` is additive, so one pass
over a shuffled subject order with a snapshot at each checkpoint gives every
corpus size for the price of the largest, instead of re-reading the corpus once
per size. Shuffled rather than sorted because the sorted order is by path, i.e.
grouped by dataset: a prefix of it would confound "more participants" with
"fewer acquisitions", and acquisitions are the scarce resource here.

Fitted at each checkpoint, all from the same two accumulators:

- `corpus-pca`  variance ordering, the incumbent.
- `lr-cca`      the cross-orbit constraint, best-behaved arm at k=64.
- `band-pca`    variance ordering *after* dropping directions too slow to be
                gaze (see `unsupervised.fit_band_pca`). The candidate that
                should scale, because a per-direction temporal statistic over
                512 directions needs far more data than a few leading
                eigenvectors do.
- `gev-fast` / `gev-slow`
                the extremes of the same temporal axis, as each other's control.

    python scripts/sweep_corpus_scaling.py --checkpoints 25 50 100 200 400 800
    python scripts/sweep_corpus_scaling.py --checkpoints 100 --report-only

Writes `results/scaling/basis_n{N}.npz` plus a `lag1_spectrum.json` recording,
per checkpoint, the measured lag-1 autocorrelation of every principal direction
-- which is what says where the nuisance/gaze/noise boundaries actually sit
rather than where they were guessed.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.datasource import resolve
from deepmreye.unsupervised import (
    Moments,
    _slabs,
    corpus_mask,
    fit_band_pca,
    fit_gev,
    fit_lr_cca,
    fit_nuisance_projected_pca,
    fit_pca,
    save_basis,
    unlabeled_subjects,
)


def add_subject(moments, path, n_trs, flat, trs_per_subject, n_slabs):
    import h5py

    try:
        with h5py.File(path, "r") as f:
            block = f["eye_block"]
            for start, stop in _slabs(n_trs, trs_per_subject, n_slabs):
                slab = block[..., start:stop]
                moments.add(slab.reshape(-1, slab.shape[-1])[flat].T)
    except Exception as e:
        print(f"    [!] skipping {path}: {e}", flush=True)
        return False
    moments.n_subjects += 1
    return True


def fit_all(moments, mask, k, cca_reduce, rho_hi, rho_lo, n_pool, seed):
    """Every basis from the current accumulator state.

    The two covariances are built **once** and shared. Each fit function will
    happily rebuild them, but ``covariance()`` promotes a 14236^2 matrix to
    float64 and subtracts an outer product every call -- 1.6 GB of allocation and
    memory traffic per basis, six times per checkpoint, for a matrix that has not
    changed.
    """
    from deepmreye.unsupervised import _top_eigenvectors

    cov, mu = moments.covariance(diff=False)
    dcov, _ = moments.covariance(diff=True)
    pool_vecs, pool_vals = _top_eigenvectors(cov, max(n_pool, 512), seed)

    bases = {}
    bases["corpus-pca"] = fit_pca(moments, k, diff=False, seed=seed)
    bases["diff-pca"] = fit_pca(moments, k, diff=True, seed=seed)
    bases["band-pca"] = fit_band_pca(moments, k, rho_lo=rho_lo, rho_hi=rho_hi,
                                     n_pool=n_pool, seed=seed,
                                     cached=(cov, mu, dcov, pool_vecs, pool_vals))
    cached = (cov, mu, dcov, pool_vecs[:, :512], pool_vals[:512])
    bases["gev-fast"] = fit_gev(moments, k, mode="fast", seed=seed, cached=cached)
    bases["gev-slow"] = fit_gev(moments, k, mode="slow", seed=seed, cached=cached)
    # Two nuisance budgets rather than one: 8 is the band the next-TR entry
    # measured as the predictable nuisance, 32 asks whether it reaches further.
    for j in (8, 32):
        bases[f"nuis-pca{j}"] = fit_nuisance_projected_pca(
            moments, k, n_nuisance=j, n_pool=n_pool, seed=seed,
            cached=(cov, mu, dcov, pool_vecs, pool_vals))
    bases["lr-cca"] = fit_lr_cca(moments, mask, k, n_reduce=cca_reduce, seed=seed,
                                 cached=(cov, mu))
    return bases


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default="results/scaling")
    p.add_argument("--checkpoints", nargs="+", type=int,
                   default=[25, 50, 100, 200, 400, 800])
    p.add_argument("--k", type=int, default=256,
                   help="Components stored. Probe by truncation, so fit the "
                        "superset once.")
    p.add_argument("--trs-per-subject", type=int, default=48)
    p.add_argument("--n-slabs", type=int, default=4)
    p.add_argument("--cca-reduce", type=int, default=256)
    p.add_argument("--n-pool", type=int, default=512,
                   help="Directions band-pca selects from.")
    p.add_argument("--rho-hi", type=float, default=0.95,
                   help="Drop directions slower than this (the nuisance cut).")
    p.add_argument("--rho-lo", type=float, default=-1.0,
                   help="Drop directions faster than this (the noise cut).")
    p.add_argument("--max-tr", type=float, default=None,
                   help="Keep only participants with repetition time <= this. "
                        "The corpus median TR is 2.00 s against the labeled "
                        "half's 0.80 s, and lag-1 autocorrelation depends "
                        "directly on sampling rate -- so an unmatched basis "
                        "measures temporal structure at the wrong TR. Compare "
                        "against an unmatched fit at the SAME N, not at the full "
                        "corpus size, or the filter is confounded with N.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data_dir = resolve(args.data_dir, download=False, quiet=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] data {data_dir}")

    mask = corpus_mask(data_dir)
    flat = mask.reshape(-1)
    subjects = unlabeled_subjects(data_dir)
    if args.max_tr:
        import h5py

        kept = []
        for rec in subjects:
            try:
                with h5py.File(rec[2], "r") as f:
                    tr = float(f.attrs.get("repetition_time", np.nan))
            except Exception:
                continue
            if np.isfinite(tr) and tr <= args.max_tr:
                kept.append(rec)
        print(f"[*] TR filter <= {args.max_tr}s: {len(kept)} of "
              f"{len(subjects)} participants")
        subjects = kept
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(subjects))
    subjects = [subjects[i] for i in order]
    print(f"[*] {len(subjects)} unlabeled participants "
          f"({len({s[0] for s in subjects})} datasets), shuffled")

    checkpoints = sorted({min(c, len(subjects)) for c in args.checkpoints})
    print(f"[*] checkpoints {checkpoints}; one pass, snapshot at each")

    moments = Moments(int(flat.sum()))
    spectrum, i = {}, 0
    t0 = time.time()
    for target in checkpoints:
        while moments.n_subjects < target and i < len(subjects):
            _ds, _sub, path, n_trs = subjects[i]
            add_subject(moments, path, n_trs, flat, args.trs_per_subject,
                        args.n_slabs)
            i += 1
        # Safe to call repeatedly: syrk only ever writes the upper triangle, so
        # re-mirroring after further adds is what keeps the matrix consistent.
        moments.symmetrise()
        n = moments.n_subjects
        print(f"\n[*] === n={n} participants, {moments.n} TRs "
              f"({time.time() - t0:.0f}s) ===", flush=True)

        bases = fit_all(moments, mask, args.k, args.cca_reduce, args.rho_hi,
                        args.rho_lo, args.n_pool, args.seed)
        # Key names follow `fit_corpus_basis.py`: `eval_probe.load_bases_for`
        # reads `meta['datasets']` directly and KeyErrors on anything else.
        meta = {"n_subjects": n, "n_trs": int(moments.n),
                "datasets": len({s[0] for s in subjects[:i]}),
                "k": args.k, "rho_hi": args.rho_hi, "rho_lo": args.rho_lo,
                "labeled": False, "shuffled_seed": args.seed}
        path = out_dir / f"basis_n{n}.npz"
        save_basis(path, mask, bases, meta)

        bp = bases["band-pca"]
        rho_all = np.asarray(bp["lag1_all"], dtype=float)
        finite = rho_all[np.isfinite(rho_all)]
        spectrum[str(n)] = {
            "n_trs": int(moments.n),
            "rho_percentiles": {str(q): float(np.percentile(finite, q))
                                for q in (1, 5, 25, 50, 75, 95, 99)},
            "rho_leading_16": [float(x) for x in rho_all[:16]],
            "n_dropped_slow": int(bp["n_dropped_slow"]),
            "n_dropped_fast": int(bp["n_dropped_fast"]),
            "cca_top10": [float(x) for x in
                          bases["lr-cca"]["canonical_correlations"][:10]],
            "gev_fast_top5": [float(x) for x in bases["gev-fast"]["eigenvalues"][:5]],
            "gev_slow_top5": [float(x) for x in bases["gev-slow"]["eigenvalues"][:5]],
        }
        print(f"    lag-1 autocorrelation of principal directions: "
              f"median {np.median(finite):+.3f}, "
              f"p5 {np.percentile(finite, 5):+.3f}, "
              f"p95 {np.percentile(finite, 95):+.3f}")
        print(f"    leading 8: "
              f"{', '.join(f'{x:+.2f}' for x in rho_all[:8])}")
        print(f"    band-pca dropped {bp['n_dropped_slow']} too-slow, "
              f"{bp['n_dropped_fast']} too-fast of {args.n_pool}")
        print(f"    -> {path}")

    (out_dir / "lag1_spectrum.json").write_text(json.dumps(spectrum, indent=2))
    print(f"\n[*] spectrum -> {out_dir / 'lag1_spectrum.json'}")
    print(f"[*] {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
