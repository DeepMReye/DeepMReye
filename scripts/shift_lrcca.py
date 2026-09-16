#!/usr/bin/env python3
"""Does a frozen linear basis lose accuracy to residual misregistration?

Every arm in this project reads gaze through a **fixed** spatial filter over a
template crop. That is the one thing a convolutional decoder gets for free and a
linear projection does not: tolerance to a participant whose orbit sits a voxel
off where the corpus average put it. Coregistration is nonlinear and good, but
the eyeball is the worst-behaved structure in the head to register -- it is the
one that moves -- so "a voxel off" is not a hypothetical.

The test needs no labels, and that is the point. For each integer shift `s` the
basis is re-read at `s` (weights moved, not data, which is the same map and
costs one gather), and the shift chosen for a participant is the one that
maximises **the canonical correlation the basis was built on**, measured on that
participant's own timeseries:

    s*(participant) = argmax_s  mean_j  corr( z_L,j(s), z_R,j(s) )

The two orbits are two measurements of one conjugate gaze, so their agreement is
a label-free readout of whether the filter is aimed correctly. `s = 0` is in the
set, so the arm contains its own control, and the distribution of chosen shifts
is reported whatever the score does: if every participant picks zero, the
registration is fine and the idea is closed for one projection pass.

    python scripts/shift_lrcca.py --voxels <voxdir>
"""
import argparse
import itertools
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
    Corpus, Eigs, K_STORE, fit_cca, project, summarise,
)


def shift_gather(mask, shift):
    """Masked-voxel indices to read for each masked voxel, or -1 if unavailable.

    Moving the weights by ``s`` computes what moving the data by ``-s`` would,
    and the gather is only defined where the *source* voxel is also inside the
    mask -- there is no data outside it. Undefined entries become a zero weight,
    which is the honest edge behaviour: a wrapped or extrapolated value would
    invent signal at exactly the orbit boundary the shift is probing.
    """
    flat = mask.reshape(-1)
    order = np.full(flat.shape, -1, dtype=np.int64)
    order[flat] = np.arange(int(flat.sum()))
    grid = np.argwhere(mask)
    tgt = grid + np.asarray(shift)
    ok = np.all((tgt >= 0) & (tgt < np.asarray(mask.shape)), axis=1)
    out = np.full(len(grid), -1, dtype=np.int64)
    idx = np.ravel_multi_index(tuple(tgt[ok].T), mask.shape)
    out[ok] = order[idx]
    return out


def shifted(w, gather):
    out = np.zeros_like(w)
    ok = gather >= 0
    out[ok] = w[gather[ok]]
    return out


def report_shifts(chosen, n):
    counts = {}
    for s in chosen.values():
        counts[str(s)] = counts.get(str(s), 0) + 1
    print(f"\n[*] chosen shifts: {counts.get(str((0, 0, 0)), 0)}/{n} "
          f"participants keep (0,0,0)")
    for s, c in sorted(counts.items(), key=lambda kv: -kv[1])[:8]:
        print(f"      {s:<14}{c:>4}")


def select_shifts(recs, index, shifts, k_select):
    """Per participant, the shift its own two orbits agree best under."""
    chosen, agree = {}, {}
    for rec in recs:
        best = None
        for s in shifts:
            tag = "_".join(str(v) for v in s)
            al, _ = index[f"L{tag}"]
            ar, _ = index[f"R{tag}"]
            zl = rec["z"][:, al:al + k_select].astype(np.float64)
            zr = rec["z"][:, ar:ar + k_select].astype(np.float64)
            zl = zl - zl.mean(0)
            zr = zr - zr.mean(0)
            num = (zl * zr).sum(0)
            den = np.sqrt((zl ** 2).sum(0) * (zr ** 2).sum(0))
            score = float(np.mean(np.abs(num / np.maximum(den, 1e-12))))
            if best is None or score > best[0]:
                best = (score, s, tag)
        chosen[rec["subject"]] = best[1]
        agree[rec["subject"]] = best[0]
        rec["shift"] = best[2]
    return chosen, agree


def oracle_shift(recs, shifts, fixed):
    """The label-chosen upper bound on any per-participant shift rule.

    Each shift gets its own full LODO fit, so train and test are read at the
    same alignment, and each participant then keeps its best row -- chosen with
    the labels, which is why this is a bound and not a method. If it comes back
    at the no-shift number, residual misregistration is not costing this basis
    anything and the question is closed rather than under-searched.
    """
    per_sub = {}
    for s in shifts:
        tag = "_".join(str(v) for v in s)
        res = probe.lodo(recs, fixed(tag))
        for row in res["participants"]:
            key = (row["dataset"], row["subject"])
            cur = per_sub.get(key)
            if cur is None or row["subtr"]["r"] > cur["subtr"]["r"]:
                per_sub[key] = row
        print(f"    shift {tag:<10} {res['median_subtr']:.4f}", flush=True)
    return probe._summarise(list(per_sub.values()), sorted({k[0] for k in per_sub}))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--voxels", required=True)
    p.add_argument("--moments", default="results/lrcca_variants/moments_n2000.npz")
    p.add_argument("--out", default="results/lrcca_variants/shift.json")
    p.add_argument("--radius", type=int, default=1)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--oracle", action="store_true",
                   help="Also compute the label-chosen upper bound on shift "
                        "selection. 27 extra LODO fits.")
    p.add_argument("--k-select", type=int, default=8,
                   help="Canonical directions the agreement criterion averages "
                        "over. The leading ones are the best estimated.")
    a = p.parse_args()

    corpus = Corpus(a.moments)
    li, ri = corpus.orbits()
    n_vox = int(corpus.mask.sum())
    fit = fit_cca(corpus.cov("total"), li, ri, K_STORE, 256, 1e-3, eigs=Eigs(),
                  key="total")

    wl = np.zeros((n_vox, K_STORE), dtype=np.float32)
    wr = np.zeros((n_vox, K_STORE), dtype=np.float32)
    wl[li] = fit["left_weights"][:, :K_STORE]
    wr[ri] = fit["right_weights"][:, :K_STORE]
    off_l = corpus.mu @ wl.astype(np.float64)
    off_r = corpus.mu @ wr.astype(np.float64)

    r = a.radius
    shifts = [s for s in itertools.product(range(-r, r + 1), repeat=3)]
    print(f"[*] {len(shifts)} shifts, radius {r}", flush=True)

    blocks, t0 = [], time.time()
    for s in shifts:
        g = shift_gather(corpus.mask, s)
        tag = "_".join(str(v) for v in s)
        blocks.append((f"L{tag}", shifted(wl, g), off_l, K_STORE))
        blocks.append((f"R{tag}", shifted(wr, g), off_r, K_STORE))
    print(f"[*] weights gathered in {time.time() - t0:.0f}s", flush=True)

    recs, index = project(a.voxels, blocks)

    # Per participant: pick the shift its own two orbits agree best under.
    chosen, agree = select_shifts(recs, index, shifts, a.k_select)

    def fixed(tag):
        al, _ = index[f"L{tag}"]
        ar, _ = index[f"R{tag}"]

        def fn(rec):
            zl = rec["z"][:, al:al + a.k].astype(np.float64)
            zr = rec["z"][:, ar:ar + a.k].astype(np.float64)
            return probe.make_lags(0.5 * (zl + zr), 1)
        return fn

    report_shifts(chosen, len(recs))

    def per_participant(rec):
        al, _ = index[f"L{rec['shift']}"]
        ar, _ = index[f"R{rec['shift']}"]
        zl = rec["z"][:, al:al + a.k].astype(np.float64)
        zr = rec["z"][:, ar:ar + a.k].astype(np.float64)
        return probe.make_lags(0.5 * (zl + zr), 1)

    got = {}
    print(f"\n{'arm':<34}{'sub-TR':>9}{'1-TR':>9}")
    if a.oracle:
        got["oracle per-participant shift"] = summarise(
            oracle_shift(recs, shifts, fixed))
        o = got["oracle per-participant shift"]
        print(f"{'oracle per-participant shift':<34}{o['subtr']:>9.4f}{o['1tr']:>9.4f}",
              flush=True)
    for name, fn in (("no shift  (0,0,0)", fixed("0_0_0")),
                     ("per-participant argmax agreement", per_participant)):
        got[name] = summarise(probe.lodo(recs, fn))
        print(f"{name:<34}{got[name]['subtr']:>9.4f}{got[name]['1tr']:>9.4f}", flush=True)
    got["_chosen"] = {k: list(v) for k, v in chosen.items()}
    got["_agreement"] = agree
    Path(a.out).write_text(json.dumps(got, indent=1))
    print(f"\n[+] {a.out}")


if __name__ == "__main__":
    main()
