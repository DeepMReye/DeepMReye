#!/usr/bin/env python3
"""Fit the two unsupervised bases on the unlabeled corpus, at every corpus size.

The corpus is :func:`deepmreye.unsupervised.corpus_subjects`: every eligible
participant (2,060), never a gaze-labelled dataset or the accession one was
ingested from, in a seed-``--seed`` shuffle. One streaming pass accumulates a
mean and a 14236 x 14236 second moment over the masked voxels (``Moments`` is
additive), and at each checkpoint both bases are decomposed from the running
state -- so every corpus size costs one pass, and the smaller corpora are
nested in the larger ones. Another seed is another random draw of each size.

    python scripts/fit_corpus_basis.py                       # the full sweep, seed 0
    python scripts/fit_corpus_basis.py --checkpoints all     # the shipped basis only
    for s in $(seq 1 19); do                                 # 19 more draws of each size
      python scripts/fit_corpus_basis.py --seed $s --k 64 --shipped '' \
        --checkpoints 10 100 200 400 800 1600 --out-dir results/corpus_bases/draws/s$s
    done

Writes ``<out-dir>/basis_n{N}.npz`` per checkpoint, each carrying the list of
participants it was fitted on, and copies the largest to ``results/basis.npz``,
which is what :func:`deepmreye.probe.load_or_build` reads. Cost is dominated by
the rank-k update (~14236^2 x 48 flops per participant): about three minutes
for the full corpus on a 12-core laptop.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.datasource import resolve  # noqa: E402
from deepmreye.unsupervised import (  # noqa: E402
    Moments, _slabs, corpus_mask, corpus_subjects, fit_lr_cca, fit_pca, save_basis,
)

CHECKPOINTS = ["10", "100", "200", "400", "800", "1600", "all"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default="results/corpus_bases")
    p.add_argument("--checkpoints", nargs="+", default=CHECKPOINTS,
                   help="Corpus sizes to snapshot at; 'all' is the whole corpus.")
    p.add_argument("--k", type=int, default=256, help="Components stored per basis.")
    p.add_argument("--trs-per-subject", type=int, default=48)
    p.add_argument("--n-slabs", type=int, default=4,
                   help="Contiguous TR slabs per run the budget is split over.")
    p.add_argument("--cca-reduce", type=int, default=256,
                   help="PCA dimensions per orbit before CCA whitening.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shipped", default="results/basis.npz",
                   help="Where the largest checkpoint is copied; '' to skip.")
    args = p.parse_args()

    import h5py

    data_dir = resolve(args.data_dir, download=False, quiet=True)
    mask = corpus_mask(data_dir)
    flat = mask.reshape(-1)
    subjects = corpus_subjects(data_dir, seed=args.seed)
    sizes = sorted({min(len(subjects), len(subjects) if c == "all" else int(c))
                    for c in args.checkpoints})
    print(f"[*] {len(subjects)} participants from {len({s[0] for s in subjects})} datasets; "
          f"checkpoints {sizes}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    moments, t0, path = Moments(int(flat.sum())), time.time(), None
    for i, (ds, sub, file, n_trs) in enumerate(subjects, start=1):
        try:
            with h5py.File(file, "r") as f:
                block = f["eye_block"]
                for start, stop in _slabs(n_trs, args.trs_per_subject, args.n_slabs):
                    slab = block[..., start:stop]
                    moments.add(slab.reshape(-1, slab.shape[-1])[flat].T)
        except Exception as e:                       # a truncated file is skipped, loudly
            print(f"    [!] skipping {file}: {e}", flush=True)
            continue
        moments.n_subjects += 1
        if i not in sizes:
            continue

        moments.symmetrise()
        cov, mu = moments.covariance()
        bases = {"corpus-pca": fit_pca(moments, args.k, seed=args.seed, cached=(cov, mu)),
                 "lr-cca": fit_lr_cca(moments, mask, args.k, args.cca_reduce, seed=args.seed,
                                      cached=(cov, mu))}
        del cov
        used = subjects[:i]
        meta = {"n_subjects": moments.n_subjects, "n_trs": moments.n, "k": args.k,
                "trs_per_subject": args.trs_per_subject, "seed": args.seed,
                "datasets": len({s[0] for s in used}),
                "subjects": [f"{d}/{s}" for d, s, *_ in used],
                "fitted": time.strftime("%Y-%m-%d %H:%M:%S")}
        path = save_basis(out_dir / f"basis_n{i}.npz", mask, bases, meta)
        cc = bases["lr-cca"]["canonical_correlations"]
        print(f"  N={i:<5} {meta['datasets']} datasets, {moments.n} TRs; canonical r "
              f"1st {cc[0]:.3f} 32nd {cc[31]:.3f}  -> {path}  [{time.time() - t0:.0f}s]", flush=True)
        if i == sizes[-1]:
            break
    if path is not None and args.shipped:
        shutil.copy(path, args.shipped)
        print(f"[+] {path} -> {args.shipped}")
    (out_dir / "sweep.json").write_text(json.dumps(
        {"sizes": sizes, "seed": args.seed, "participants": len(subjects)}, indent=1))


if __name__ == "__main__":
    main()
