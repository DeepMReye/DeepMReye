"""What is the unlabeled corpus worth?

Three measurements, written down once, all on the shipped protocol
(:func:`deepmreye.probe.lodo` -- nine folds, 337 gaze-labeled participants,
scored at sub-TR and 1-TR):

1. ``corpus_scaling.json``  -- every basis refitted at eleven unlabeled corpus
   sizes (25 to 2000 participants), decoded leave-one-dataset-out. Includes
   ``gev-slow``, the negative control: an equally-sized basis out of the same
   accumulators, selected for slow drift instead of bilateral agreement. It
   *degrades* over the same data, which is what distinguishes a real data axis
   from a fitting artefact.
2. ``foldpca_full.json`` / ``foldpca_budget.json`` -- the honest no-corpus
   reference. Without an unlabeled corpus the voxel basis has to come out of the
   gaze-labeled scans themselves, so ``fold-pca`` is refitted per fold *and* per
   labeled budget, never seeing the held-out dataset. Given 200 TRs per
   participant against the corpus bases' 48, so a corpus win cannot be a TR
   budget artefact.
3. ``label_budget.json`` -- held-out accuracy against the number of labeled
   participants the readout may train on. The budget restricts the **training**
   side only (``probe.lodo(train_filter=...)``); every fold is still scored on
   its whole held-out dataset, so the metric's denominator does not move with
   the budget.

``basis_stats.json`` is the label-free companion: the canonical correlation
spectrum at each corpus size. Fitted on 25 participants, cross-orbit CCA reports
sixty-odd directions shared between the orbits above rho = 0.8. They are not
there -- the spectrum contracts as the corpus grows and stops moving by about a
thousand participants. Note the alignment column is to the *largest* basis, and
the sweep is incremental, so those numbers are nested and are a convergence
diagnostic rather than an independent one.

Each stage writes its own JSON and is skipped if that file already exists, so the
script is resumable; ``--force`` reruns everything.

    python scripts/analysis_unlabeled.py --out results/unlabeled_value
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

from deepmreye import probe  # noqa: E402
from deepmreye.datasource import resolve  # noqa: E402
from deepmreye.unsupervised import (  # noqa: E402
    Moments, _slabs, _top_eigenvectors, corpus_mask, load_basis,
)

SIZES = [25, 50, 100, 200, 400, 800, 1039, 1200, 1500, 1800, 2000]
K = 64                      # every basis is carried at 64 and sliced down
FOLDPCA_TRS = 200           # deliberately more than the corpus bases' 48
BUDGETS = [1, 2, 4, 8, 16, 32, None]
SEEDS = [0, 1, 2]


# --------------------------------------------------------------------------- #
# Stage 0: the labeled participants, projected onto every basis in one pass
# --------------------------------------------------------------------------- #

def _labeled_files(root):
    for d in sorted(Path(root).glob("dsL*")):
        for p in sorted(d.glob("*.h5")):
            yield d.name, p


def _read(path):
    """``(masked rows, labels)`` for a labeled participant, or ``None``."""
    with h5py.File(path, "r") as f:
        if "labels" not in f:
            return None
        return f["eye_block"][:], f["labels"][:]


def _stack(blocks):
    """Stack ``[(name, W, mu, shape)]`` into one matmul plus its constant row."""
    cols, offs, index, cursor = [], [], {}, 0
    for name, W, mu, shape in blocks:
        W = np.asarray(W, dtype=np.float32)
        cols.append(W)
        offs.append(np.asarray(mu, dtype=np.float64) @ W.astype(np.float64))
        index[name] = (cursor, cursor + W.shape[1], list(shape))
        cursor += W.shape[1]
    return np.concatenate(cols, axis=1), np.concatenate(offs), index


def _project(root, mask, blocks, verbose=True):
    """Every labeled participant's coordinates under every registered basis.

    One matmul per participant rather than one per basis: they are all linear
    maps of the same masked voxels, and ``(x - mu) @ W = x @ W - (mu @ W)``, so
    the per-basis means become a constant row and the matmul is shared.
    """
    flat = mask.reshape(-1)
    P, off, index = _stack(blocks)
    if verbose:
        print(f"[*] projection matrix {P.shape} ({P.nbytes / 1e9:.2f} GB)", flush=True)
    recs, t0 = [], time.time()
    for ds, path in _labeled_files(root):
        got = _read(path)
        if got is None:
            continue
        block, labels = got
        t = block.shape[-1]
        if t < probe.MIN_TRS or not np.isfinite(labels).any():
            continue
        x = block.reshape(-1, t).T[:, flat].astype(np.float32)
        z = (x @ P).astype(np.float64) - off
        n = min(len(z), len(labels))
        recs.append({"dataset": ds, "subject": path.stem,
                     "z": z[:n].astype(np.float32),
                     "labels": labels[:n].astype(np.float32)})
    if verbose:
        print(f"[+] {len(recs)} participants projected in {time.time() - t0:.0f}s", flush=True)
    return recs, index


def _slicer(index, name, k=None, lags=0):
    """Feature function reading one registered column block out of ``rec['z']``."""
    a, b, shape = index[name]

    def fn(rec):
        z = rec["z"][:, a:b].astype(np.float64)
        if len(shape) == 2:                       # lr-cca is stored per orbit
            z = z.reshape(len(z), shape[0], shape[1])
            kk = k or shape[1]
            z = 0.5 * (z[:, 0, :kk] + z[:, 1, :kk])
        elif k:
            z = z[:, :k]
        return probe.make_lags(z, lags)
    return fn


def _summary(res, extra):
    """One arm's result, flattened. `folds` keeps both resolutions side by side.

    Reporting one resolution and implying the other is the failure
    :mod:`deepmreye.probe` exists to prevent, so nothing downstream gets to see
    a summary that carries only one.
    """
    return {**extra,
            "subtr": res["summary"]["subtr"]["median_r"],
            "1tr": res["summary"]["1tr"]["median_r"],
            "subtr_mean": res["summary"]["subtr"]["mean_r"],
            "1tr_mean": res["summary"]["1tr"]["mean_r"],
            "folds": {d: {"n": f["n"], "subtr": f["subtr"]["r"], "1tr": f["1tr"]["r"]}
                      for d, f in res["folds"].items()}}


# --------------------------------------------------------------------------- #
# Stage 1: basis statistics (no labels involved)
# --------------------------------------------------------------------------- #

def stage_basis_stats(basis_dir, out):
    spec, align = {}, {}
    ref = {side: None for side in ("left", "right")}
    big = np.load(Path(basis_dir) / f"basis_n{SIZES[-1]}.npz", allow_pickle=False)
    for side in ref:
        W = big[f"lr-cca/{side}_weights"][:, :32].astype(np.float64)
        ref[side] = np.linalg.qr(W)[0]
    for n in SIZES:
        z = np.load(Path(basis_dir) / f"basis_n{n}.npz", allow_pickle=False)
        spec[n] = z["lr-cca/canonical_correlations"][:K].tolist()
        vals = []
        for side in ref:
            Q = np.linalg.qr(z[f"lr-cca/{side}_weights"][:, :32].astype(np.float64))[0]
            vals.append(float((np.linalg.svd(Q.T @ ref[side], compute_uv=False) ** 2).mean()))
        align[n] = float(np.mean(vals))
    json.dump({"spectrum": spec, "alignment_to_largest": align},
              open(out, "w"), indent=1)
    print(f"[+] {out}")


# --------------------------------------------------------------------------- #
# Stage 2: corpus scaling
# --------------------------------------------------------------------------- #

def stage_scaling(root, mask, basis_dir, out):
    blocks = []
    for n in SIZES:
        _m, bases, _meta = load_basis(Path(basis_dir) / f"basis_n{n}.npz")
        lr = bases["lr-cca"]
        li, ri = lr["left_index"], lr["right_index"]
        W = np.zeros((int(mask.sum()), 2 * K), dtype=np.float32)
        W[li, :K] = lr["left_weights"][:, :K]
        W[ri, K:] = lr["right_weights"][:, :K]
        blocks.append((f"lrcca_n{n}", W, lr["mean"], (2, K)))
        for kind, tag in (("corpus-pca", "cpca"), ("gev-slow", "gevslow")):
            a = bases[kind]
            blocks.append((f"{tag}_n{n}", a["components"][:, :K], a["mean"], (K,)))
        del bases

    # `raw`: stride-4 in each spatial dimension, 12 x 8 x 5 = 480 voxels. The
    # published DeepMReye 1.0 feature budget, and the arm that needs no corpus.
    sel = np.zeros(mask.shape, dtype=bool)
    sel[::4, ::4, ::4] = True
    idx = np.nonzero(sel.reshape(-1)[mask.reshape(-1)])[0]
    W = np.zeros((int(mask.sum()), len(idx)), dtype=np.float32)
    W[idx, np.arange(len(idx))] = 1.0
    blocks.append(("raw", W, np.zeros(int(mask.sum())), (len(idx),)))

    recs, index = _project(root, mask, blocks)
    arms = {}
    for n in SIZES:
        arms[f"lr-cca:32+lags1@{n}"] = _slicer(index, f"lrcca_n{n}", k=32, lags=1)
        arms[f"lr-cca:32@{n}"] = _slicer(index, f"lrcca_n{n}", k=32)
        arms[f"lr-cca:64@{n}"] = _slicer(index, f"lrcca_n{n}", k=64)
        arms[f"corpus-pca:64@{n}"] = _slicer(index, f"cpca_n{n}", k=64)
        arms[f"gev-slow:64@{n}"] = _slicer(index, f"gevslow_n{n}", k=64)
    arms["raw"] = _slicer(index, "raw")
    arms["raw+lags1"] = _slicer(index, "raw", lags=1)

    got, t0 = {}, time.time()
    for name, fn in arms.items():
        got[name] = _summary(probe.lodo(recs, fn), {})
        print(f"  {name:<28} sub-TR {got[name]['subtr']:.4f}  1-TR {got[name]['1tr']:.4f}"
              f"  [{time.time() - t0:6.0f}s]", flush=True)
        json.dump(got, open(out, "w"), indent=1)
    return recs, index


# --------------------------------------------------------------------------- #
# Stage 3: labeled budget, on the arms that need no refitting
# --------------------------------------------------------------------------- #

def budget_keep(by_ds, budget, seed):
    """Subject ids the readout may train on, per dataset. ``None`` = all."""
    if budget is None:
        return None
    rng = np.random.default_rng(seed)
    keep = {}
    for ds, subs in by_ds.items():
        order = sorted(subs)
        rng.shuffle(order)
        keep[ds] = set(order[:budget])
    return keep


def stage_budget(recs, index, out):
    by_ds = {}
    for r in recs:
        by_ds.setdefault(r["dataset"], []).append(r["subject"])
    arms = {
        "raw": _slicer(index, "raw"),
        "lr-cca:32@25": _slicer(index, "lrcca_n25", k=32),
        "lr-cca:32+lags1@25": _slicer(index, "lrcca_n25", k=32, lags=1),
        "lr-cca:32@2000": _slicer(index, "lrcca_n2000", k=32),
        "lr-cca:32+lags1@2000": _slicer(index, "lrcca_n2000", k=32, lags=1),
        "corpus-pca:64@2000": _slicer(index, "cpca_n2000", k=64),
    }
    rows, t0 = [], time.time()
    for budget in BUDGETS:
        for seed in (SEEDS if budget is not None else [0]):
            keep = budget_keep(by_ds, budget, seed)
            tf = None if keep is None else (
                lambda r, held, k=keep: r["subject"] in k[r["dataset"]])
            for name, fn in arms.items():
                row = _summary(probe.lodo(recs, fn, train_filter=tf),
                               {"budget": budget, "seed": seed, "arm": name})
                rows.append(row)
                print(f"  B={str(budget):<5} s{seed} {name:<24} sub-TR {row['subtr']:.4f}"
                      f"  [{time.time() - t0:6.0f}s]", flush=True)
            json.dump(rows, open(out, "w"), indent=1)


# --------------------------------------------------------------------------- #
# Stage 4: the no-corpus reference, refitted per fold and per budget
# --------------------------------------------------------------------------- #

def fit_fold_bases(root, mask, keep, k=K):
    """One PCA per held-out dataset, over the labeled participants the budget allows."""
    flat = mask.reshape(-1)
    per_ds = {}
    for d in sorted(Path(root).glob("dsL*")):
        m = Moments(int(flat.sum()))
        for p in sorted(d.glob("*.h5")):
            if keep is not None and p.stem not in keep.get(d.name, ()):
                continue
            with h5py.File(p, "r") as f:
                if "labels" not in f:
                    continue
                blk = f["eye_block"]
                for a, b in _slabs(blk.shape[-1], FOLDPCA_TRS, 4):
                    slab = blk[..., a:b]
                    m.add(slab.reshape(-1, slab.shape[-1])[flat].T)
            m.n_subjects += 1
        m.symmetrise()
        per_ds[d.name] = m
    datasets = sorted(per_ds)
    bases = {}
    for held in datasets:
        acc = Moments(int(flat.sum()))
        for ds in datasets:
            if ds == held:
                continue
            acc.c += per_ds[ds].c
            acc.s += per_ds[ds].s
            acc.n += per_ds[ds].n
        if acc.n < k + 2:
            continue
        cov, mu = acc.covariance()
        bases[held] = (_top_eigenvectors(cov, k)[0].astype(np.float32),
                       mu.astype(np.float32), acc.n)
    return bases


def score_foldlocal(recs, folds, train_filter=None, k=K, lags=0):
    """Each fold under its own basis, then one pass through the shipped reducer.

    ``lodo`` applies a single feature function to every record, and a fold-local
    basis is a different matrix per held-out dataset -- so each fold is run with
    its own basis and only that fold's participant rows are kept. The reduction
    from participants to fold medians to a summary is still ``probe._summarise``,
    the same one every other arm goes through.
    """
    rows = []
    for i, held in enumerate(folds):
        def fn(rec, i=i):
            return probe.make_lags(rec["z"][:, i * k:(i + 1) * k].astype(np.float64), lags)
        res = probe.lodo(recs, fn, train_filter=train_filter)
        rows += [p for p in res["participants"] if p["dataset"] == held]
    return probe._summarise(rows, folds)


def stage_foldpca(root, mask, out, budgets, seeds):
    by_ds = {d.name: sorted(p.stem for p in d.glob("*.h5"))
             for d in sorted(Path(root).glob("dsL*"))}
    configs = [(b, s) for b in budgets for s in (seeds if b is not None else [0])]
    rows, t0 = [], time.time()
    for budget, seed in configs:
        keep = budget_keep(by_ds, budget, seed)
        bases = fit_fold_bases(root, mask, keep)
        folds = sorted(bases)
        blocks = [(f, bases[f][0], bases[f][1], (K,)) for f in folds]
        recs, _index = _project(root, mask, blocks, verbose=False)
        tf = None if keep is None else (
            lambda r, held, k=keep: r["subject"] in k[r["dataset"]])
        for lags in (0, 1):
            row = _summary(score_foldlocal(recs, folds, train_filter=tf, lags=lags),
                           {"budget": budget, "seed": seed, "arm": f"fold-pca:64+lags{lags}"})
            rows.append(row)
            print(f"  B={str(budget):<5} s{seed} lags{lags}  sub-TR {row['subtr']:.4f}"
                  f"  1-TR {row['1tr']:.4f}  [{time.time() - t0:6.0f}s]", flush=True)
        json.dump(rows, open(out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--basis-dir", default="results/scaling")
    ap.add_argument("--out", default="results/unlabeled_value")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stages", default="stats,scaling,budget,foldpca,foldpca-budget")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(a.data_dir) if a.data_dir else resolve(None, download=False, quiet=True)
    stages = set(a.stages.split(","))
    mask = corpus_mask(root)
    recs = index = None

    if "stats" in stages and (a.force or not (out / "basis_stats.json").exists()):
        stage_basis_stats(a.basis_dir, out / "basis_stats.json")

    if "scaling" in stages and (a.force or not (out / "corpus_scaling.json").exists()):
        recs, index = stage_scaling(root, mask, a.basis_dir, out / "corpus_scaling.json")

    if "budget" in stages and (a.force or not (out / "label_budget.json").exists()):
        if recs is None:
            recs, index = stage_scaling(root, mask, a.basis_dir, out / "corpus_scaling.json")
        stage_budget(recs, index, out / "label_budget.json")

    if "foldpca" in stages and (a.force or not (out / "foldpca_full.json").exists()):
        stage_foldpca(root, mask, out / "foldpca_full.json", [None], [0])

    if "foldpca-budget" in stages and (a.force or not (out / "foldpca_budget.json").exists()):
        stage_foldpca(root, mask, out / "foldpca_budget.json",
                      [1, 2, 4, 8, 16, None], [0, 1])


if __name__ == "__main__":
    main()
