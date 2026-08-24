#!/usr/bin/env python3
"""Fit the two unsupervised feature bases on the unlabeled corpus.

One streaming pass over the unlabeled participants accumulates a mean and a
14236x14236 second moment over the masked voxels; both bases are decompositions
of that. See ``deepmreye/unsupervised.py`` for what each is and why.

The gaze-labeled datasets (``dsL*``) are excluded outright -- this must not see
the evaluation set -- as are participants whose eye mask is not fully covered,
whose zeros would otherwise steer the basis toward their missingness.

    python scripts/fit_corpus_basis.py --k 256
    python scripts/fit_corpus_basis.py --k 256 --max-subjects 200 --out results/basis_small.npz

Cost is dominated by the rank-k update: ~14236^2 * T flops over the TRs kept,
so ``--trs-per-subject`` is the runtime dial. The default (48 TRs from each of
~1900 subjects) is a few minutes on a laptop and ~1.3 GB of accumulators.
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
    accumulate,
    corpus_mask,
    fit_lr_cca,
    fit_pca,
    save_basis,
    unlabeled_subjects,
)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out", default="results/corpus_basis.npz")
    p.add_argument("--k", type=int, default=256, help="Components kept per basis.")
    p.add_argument("--trs-per-subject", type=int, default=48)
    p.add_argument("--n-slabs", type=int, default=4,
                   help="Contiguous TR slabs per run the budget is split over.")
    p.add_argument("--max-subjects", type=int, default=None)
    p.add_argument("--cca-reduce", type=int, default=256,
                   help="PCA dimensions per orbit before CCA whitening.")
    p.add_argument("--include-labeled", action="store_true",
                   help="Also use the gaze-labeled datasets' VOXELS (never their "
                        "labels). Without --exclude-datasets this is transductive "
                        "and is no longer a leave-one-dataset-out basis.")
    p.add_argument("--exclude-datasets", nargs="*", default=(),
                   help="Datasets to keep out. With --include-labeled, naming the "
                        "held-out fold gives an honest per-fold basis.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data_dir = resolve(args.data_dir, download=False, quiet=True)
    print(f"[*] data {data_dir}")

    mask = corpus_mask(data_dir)
    n_vox = int(mask.sum())
    print(f"[*] eye mask: {n_vox} voxels")

    subjects = unlabeled_subjects(data_dir, include_labeled=args.include_labeled,
                                  exclude_datasets=args.exclude_datasets)
    n_labeled = sum(1 for s in subjects if s[0].startswith("dsL"))
    if args.include_labeled:
        scope = ("TRANSDUCTIVE (labeled voxels included, nothing held out)"
                 if not args.exclude_datasets
                 else f"domain-adapted (labeled voxels included, "
                      f"excluding {sorted(args.exclude_datasets)})")
        print(f"[*] scope: {scope}; {n_labeled} labeled participants folded in")
    if args.max_subjects:
        # Evenly spaced rather than the first N: the corpus is sorted by
        # accession, so a prefix is a biased sample of scanners and eras.
        idx = np.linspace(0, len(subjects) - 1, args.max_subjects).astype(int)
        subjects = [subjects[i] for i in np.unique(idx)]
    print(f"[*] {len(subjects)} eligible unlabeled participants "
          f"({len({s[0] for s in subjects})} datasets)")

    t0 = time.time()
    moments = accumulate(subjects, mask, args.trs_per_subject, args.n_slabs,
                         progress=100)
    print(f"[*] accumulated {moments.n} TRs from {moments.n_subjects} subjects "
          f"in {time.time() - t0:.0f}s")

    bases = {}
    t = time.time()
    bases["corpus-pca"] = fit_pca(moments, args.k, seed=args.seed)
    ev = bases["corpus-pca"]["eigenvalues"]
    share = ev.sum() / float(bases["corpus-pca"]["total_variance"][0])
    print(f"[*] corpus-pca: {ev.shape[0]} components, "
          f"top-{args.k} variance share {share:.3f} ({time.time() - t:.0f}s)")

    t = time.time()
    bases["lr-cca"] = fit_lr_cca(moments, mask, args.k, args.cca_reduce, seed=args.seed)
    cc = bases["lr-cca"]["canonical_correlations"]
    print(f"[*] lr-cca: canonical correlations "
          f"1st {cc[0]:.3f}, 10th {cc[min(9, len(cc) - 1)]:.3f}, "
          f"{args.k}th {cc[-1]:.3f} ({time.time() - t:.0f}s)")

    meta = {
        "n_subjects": moments.n_subjects,
        "n_trs": moments.n,
        "n_voxels": n_vox,
        "k": args.k,
        "trs_per_subject": args.trs_per_subject,
        "datasets": len({s[0] for s in subjects}),
        "include_labeled": bool(args.include_labeled),
        "excluded_datasets": sorted(args.exclude_datasets),
        "n_labeled_subjects": n_labeled,
        "fitted": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = save_basis(args.out, mask, bases, meta)
    print(f"[*] wrote {path} ({path.stat().st_size / 1e6:.0f} MB)")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
