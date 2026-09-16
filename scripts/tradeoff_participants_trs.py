#!/usr/bin/env python3
"""Participants or TRs? The confound the corpus-scaling curve cannot separate.

The scaling result -- `lr-cca` gains +0.154 from 25 to 2000 unlabelled
participants -- grows three things at once: participants, TRs and OpenNeuro
accessions. FINDINGS.md says so and leaves it there, which means the headline
claim ("the unlabelled *participants* are what buy this") is not yet separated
from "more rows help".

This separates them the only way that is clean: **hold the total TR count fixed
and trade participants against TRs per participant.** 2000 x 48, 1000 x 96,
500 x 192, 250 x 384 and 125 x 768 are the same number of rows out of the same
shuffled corpus order, differing only in how many distinct people supplied them.
If the curve is flat, the corpus is buying rows and the participant count is
incidental. If it falls as participants are traded away, the participants are
the resource.

Runs the accumulation once per configuration -- there is no way to reuse a pass
across them, because a 48-TR budget is not a prefix of a 768-TR one under
`_slabs` -- and reports the achieved TR count beside each, since a participant
with a short run cannot supply 768 TRs and the high-TR arms therefore fall
short of the target by an amount worth seeing.

    python scripts/tradeoff_participants_trs.py --voxels <voxdir>
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
from deepmreye.datasource import resolve  # noqa: E402
from deepmreye.unsupervised import (  # noqa: E402
    accumulate, corpus_mask, unlabeled_subjects,
)
from scripts.improve_lrcca import (  # noqa: E402
    Eigs, K_STORE, cca_to_block, fit_cca, project, slicer, summarise,
)

CONFIGS = [(2000, 48), (1000, 96), (500, 192), (250, 384), (125, 768)]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--voxels", required=True)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out", default="results/lrcca_variants/tradeoff.json")
    p.add_argument("--n-slabs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--configs", nargs="+", default=None,
                   help="`subjects x trs` pairs, e.g. 2000x48 2000x192. Default "
                        "is the TR-matched ladder.")
    a = p.parse_args()

    configs = CONFIGS if a.configs is None else [
        tuple(int(v) for v in c.split("x")) for c in a.configs]

    data_dir = resolve(a.data_dir, download=False, quiet=True)
    mask = corpus_mask(data_dir)
    subjects = unlabeled_subjects(data_dir)
    rng = np.random.default_rng(a.seed)
    subjects = [subjects[i] for i in rng.permutation(len(subjects))]
    n_vox = int(mask.sum())
    li = None

    blocks, meta, t0 = [], {}, time.time()
    for n, trs in configs:
        # More slabs when the budget is large, so a 768-TR participant is still
        # sampled across its run rather than in four long blocks.
        slabs = max(a.n_slabs, trs // 12)
        m = accumulate(subjects[:n], mask, trs, slabs, progress=None)
        cov, mu = m.covariance()
        if li is None:
            xs = np.nonzero(mask.reshape(-1))[0] // (mask.shape[1] * mask.shape[2])
            li, ri = np.nonzero(xs < 24)[0], np.nonzero(xs >= 24)[0]
        fit = fit_cca(cov, li, ri, K_STORE, 256, 1e-3, eigs=Eigs(), key=f"{n}x{trs}")
        tag = f"n{n}_t{trs}"
        blocks.append((tag,) + cca_to_block(fit, n_vox, mu) + (K_STORE,))
        meta[tag] = {"n_subjects": m.n_subjects, "n_trs": int(m.n),
                     "rho1": float(fit["rho"][0])}
        print(f"  {tag:<14}{m.n_subjects:>5} subjects {m.n:>7} TRs  "
              f"rho1 {fit['rho'][0]:.3f}  [{time.time() - t0:5.0f}s]", flush=True)
        del m, cov

    recs, index = project(a.voxels, blocks)
    got = {}
    print(f"\n{'config':<24}{'subjects':>9}{'TRs':>8}{'sub-TR':>9}{'1-TR':>9}")
    for n, trs in configs:
        tag = f"n{n}_t{trs}"
        name = f"{n} participants x {trs} TRs"
        got[name] = summarise(probe.lodo(recs, slicer(index, tag, k=a.k, lags=1)),
                              meta[tag])
        print(f"{name:<24}{meta[tag]['n_subjects']:>9}{meta[tag]['n_trs']:>8}"
              f"{got[name]['subtr']:>9.4f}{got[name]['1tr']:>9.4f}", flush=True)
        Path(a.out).write_text(json.dumps(got, indent=1))
    print(f"\n[+] {a.out}")


if __name__ == "__main__":
    main()
