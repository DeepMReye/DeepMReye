"""How much labelled data the decoder needs: labelled participants, labelled
datasets, a new study's own few, and every decoding scheme -- each against the
references with no corpus behind them."""
import json
import time
from pathlib import Path

import numpy as np

from deepmreye import probe
from unlabeled.common import (
    CURVE_DRAWS, DATASET_BUDGETS, DATASET_DRAWS, K, SEEDS, Moments, _top_eigenvectors, budget_keep,
    dataset_moments, dataset_picks, fit_fold_bases, fit_own_bases, labeled_pca_readout, log,
    mlp_readout, own_slicer, per_participant, pls_readout, project, resume, ridge_auto,
    ridge_by_covariance, score_foldlocal, slicer, summary, voxel_block,
)

BUDGETS = [1, 2, 4, 8, 16, 32, None]


def budget(ctx, out):
    """Held-out accuracy against the number of labelled *participants* per
    training dataset. The budget restricts the training side only
    (``train_filter``); every fold is still scored on its whole held-out
    dataset, so the metric's denominator does not move with it."""
    recs, index = ctx.records()
    by_ds = {}
    for r in recs:
        by_ds.setdefault(r["dataset"], []).append(r["subject"])
    small, n = ctx.sizes[0], ctx.n
    arms = {"raw": slicer(index, "raw"),
            f"lr-cca:32@{small}": slicer(index, f"lrcca_n{small}", k=32),
            f"lr-cca:32+lags1@{small}": slicer(index, f"lrcca_n{small}", k=32, lags=1),
            f"lr-cca:32@{n}": slicer(index, f"lrcca_n{n}", k=32),
            f"lr-cca:32+lags1@{n}": slicer(index, f"lrcca_n{n}", k=32, lags=1),
            f"corpus-pca:64@{n}": slicer(index, f"cpca_n{n}", k=64)}
    rows, t0 = [], time.time()
    for b in BUDGETS:
        for seed in (SEEDS if b is not None else [0]):
            keep = budget_keep(by_ds, b, seed)
            tf = None if keep is None else (lambda r, held, k=keep: r["subject"] in k[r["dataset"]])
            for name, fn in arms.items():
                rows.append(summary(probe.lodo(recs, fn, train_filter=tf),
                                    {"budget": b, "seed": seed, "arm": name}))
                log(f"B={str(b):<5} s{seed} {name:<24} sub-TR {rows[-1]['subtr']:.4f}", t0)
            json.dump(rows, open(out, "w"), indent=1)


def _foldpca(ctx, out, budgets, seeds):
    """The no-corpus reference: a PCA of the labelled scans, refitted per fold
    and per labelled budget, never seeing the held-out dataset. 200 TRs per
    participant against the corpus bases' 48, so a corpus win cannot be a TR
    budget artefact."""
    by_ds = {d.name: sorted(p.stem for p in d.glob("*.h5")) for d in sorted(Path(ctx.root).glob("dsL*"))}
    rows, t0 = [], time.time()
    for b in budgets:
        for seed in (seeds if b is not None else [0]):
            keep = budget_keep(by_ds, b, seed)
            bases = fit_fold_bases(ctx.root, ctx.mask, keep)
            folds = sorted(bases)
            recs, _index = project(ctx.root, ctx.mask, [(f, bases[f][0], bases[f][1], (K,)) for f in folds],
                                   verbose=False)
            tf = None if keep is None else (lambda r, held, k=keep: r["subject"] in k[r["dataset"]])
            for lags in (0, 1):
                rows.append(summary(score_foldlocal(recs, folds, train_filter=tf, lags=lags),
                                    {"budget": b, "seed": seed, "arm": f"fold-pca:64+lags{lags}"}))
                log(f"B={str(b):<5} s{seed} lags{lags}  sub-TR {rows[-1]['subtr']:.4f}"
                    f"  1-TR {rows[-1]['1tr']:.4f}", t0)
            json.dump(rows, open(out, "w"), indent=1)


def foldpca(ctx, out):
    _foldpca(ctx, out, [None], [0])


def foldpca_budget(ctx, out):
    _foldpca(ctx, out, [1, 2, 4, 8, 16, None], [0, 1])


def datasets(ctx, out):
    """Held-out accuracy against how many labelled *datasets* the readout sees.
    Independent acquisitions, not participants, are what a fold is, so this is
    the labelled axis the participant budget cannot measure. A random subset of
    the other datasets per held-out fold; every fold still scored whole. The
    frozen basis gets :data:`CURVE_DRAWS` draws, corpus PCA the first
    :data:`DATASET_DRAWS` of them. Resumable."""
    recs, index = ctx.records([ctx.n])
    names = sorted({r["dataset"] for r in recs})
    ours = f"lr-cca:32+lags1@{ctx.n}"
    arms = {ours: slicer(index, f"lrcca_n{ctx.n}", k=32, lags=1),
            f"corpus-pca:64+lags1@{ctx.n}": slicer(index, f"cpca_n{ctx.n}", k=64, lags=1)}
    rows, done = resume(out, lambda r: (r["arm"], r["datasets"], r["seed"]))
    t0 = time.time()
    for m in DATASET_BUDGETS:
        full = m >= len(names) - 1
        for seed in ([0] if full else range(CURVE_DRAWS)):
            picks = dataset_picks(names, m, seed)
            tf = None if full else (lambda r, held, p=picks: r["dataset"] in p[held])
            for name, fn in arms.items():
                if (name, m, seed) in done or (name != ours and seed >= DATASET_DRAWS):
                    continue
                rows.append(summary(probe.lodo(recs, fn, train_filter=tf),
                                    {"datasets": m, "seed": seed, "arm": name}))
                log(f"m={m} s{seed} {name:<26} sub-TR {rows[-1]['subtr']:.4f}  mean "
                    f"{rows[-1]['subtr_mean']:.4f}", t0)
            json.dump(rows, open(out, "w"), indent=1)


def datasets_supervised(ctx, out):
    """The labelled-dataset budget with no corpus behind it: for every held-out
    dataset and draw, the basis is a PCA of the scans of the datasets that draw
    may train on, so with one labelled dataset the basis and the readout both
    come out of it. Same draws as :func:`datasets`."""
    per_ds = dataset_moments(ctx.root, ctx.mask)
    names = sorted(per_ds)
    configs, blocks = [], []
    for m in DATASET_BUDGETS:
        full = m >= len(names) - 1
        for seed in ([0] if full else range(DATASET_DRAWS)):
            picks = dataset_picks(names, m, seed)
            for held in names:
                acc = Moments(int(ctx.mask.sum()))
                for ds in picks[held]:
                    acc.c += per_ds[ds].c
                    acc.s += per_ds[ds].s
                    acc.n += per_ds[ds].n
                cov, mu = acc.covariance()
                blocks.append((f"m{m}s{seed}_{held}", _top_eigenvectors(cov, K)[0].astype(np.float32),
                               mu.astype(np.float32), (K,)))
                del acc, cov
            configs.append((m, seed, picks))
    del per_ds
    recs, index = project(ctx.root, ctx.mask, blocks)
    rows, t0 = [], time.time()
    for m, seed, picks in configs:
        parts = []
        for held in names:
            res = probe.lodo(recs, slicer(index, f"m{m}s{seed}_{held}", k=K, lags=1),
                             train_filter=lambda r, h, held=held, p=picks: h == held and r["dataset"] in p[held])
            parts += [q for q in res["participants"] if q["dataset"] == held]
        rows.append(summary(probe._summarise(parts, names),
                            {"datasets": m, "seed": seed, "arm": "fold-pca:64+lags1"}))
        log(f"m={m} s{seed} fold-pca:64+lags1  sub-TR {rows[-1]['subtr']:.4f}  mean "
            f"{rows[-1]['subtr_mean']:.4f}", t0)
        json.dump(rows, open(out, "w"), indent=1)


def datasets_voxels(ctx, out, only=None):
    """The labelled-dataset budget for decoders with no unsupervised step at
    all: voxel block means into the same ridge as ours at every budget, and with
    all eight datasets also PLS, an MLP and no temporal context. Does not
    depend on the corpus. The same-ridge arm gets :data:`CURVE_DRAWS` draws
    (the same draws as :func:`datasets`), every other arm the first
    :data:`DATASET_DRAWS` of them. Resumable, because PLS and the MLP are
    slow; cheap arms run first."""
    arms = {"voxels:3+lags1": ("vox3", 1, ridge_by_covariance),
            "voxels:2+lags1": ("vox2", 1, ridge_by_covariance),
            "voxels:3 pls:64+lags1": ("vox3", 1, pls_readout(64)),
            "voxels:3 mlp+lags1": ("vox3", 1, mlp_readout),
            "voxels:3+lags0": ("vox3", 0, ridge_by_covariance),
            "voxels:2+lags0": ("vox2", 0, ridge_by_covariance)}
    names = sorted(d.name for d in Path(ctx.root).glob("dsL*") if d.is_dir())
    rows, done = resume(out, lambda r: (r["arm"], r["datasets"], r["seed"]))
    def draws(m, name):
        if m >= len(names) - 1:
            return [0]
        if name == "voxels:2+lags1":
            # The same ridge on finer blocks: ties 3x3x3 with all eight (0.686 / 0.687)
            # at ~10x the cost, so it is run there only.
            return []
        if name in ("voxels:3 pls:64+lags1", "voxels:3 mlp+lags1"):
            return range(2)      # ~10 min a fold set; context lines in Fig. b, not a comparison
        return range(CURVE_DRAWS if name == "voxels:3+lags1" else DATASET_DRAWS)

    cost = {"voxels:3+lags1": 0, "voxels:3+lags0": 1, "voxels:2+lags0": 2, "voxels:2+lags1": 3,
            "voxels:3 pls:64+lags1": 4, "voxels:3 mlp+lags1": 5}
    todo = [(m, seed, name) for name in sorted(arms, key=cost.get) for m in reversed(DATASET_BUDGETS)
            for seed in draws(m, name) if (name, m, seed) not in done and (only is None or name in only)]
    if not todo:
        return
    recs, index = project(ctx.root, ctx.mask, [voxel_block(ctx.mask, 3), voxel_block(ctx.mask, 2)])
    t0 = time.time()
    for m, seed, name in todo:
        full = m >= len(names) - 1
        picks = dataset_picks(names, m, seed)
        tf = None if full else (lambda r, held, p=picks: r["dataset"] in p[held])
        block, lags, readout = arms[name]
        res = probe.lodo(recs, slicer(index, block, lags=lags), train_filter=tf, readout=readout)
        rows.append(summary(res, {"datasets": m, "seed": seed, "arm": name}))
        log(f"m={m} s{seed} {name:<24} sub-TR {rows[-1]['subtr']:.4f}  1-TR {rows[-1]['1tr']:.4f}  "
            f"mean {rows[-1]['subtr_mean']:.4f}", t0)
        json.dump(rows, open(out, "w"), indent=1)



SPREAD_TOTALS = (8, 16, 32)
SPREAD_DATASETS = (1, 2, 4, 8)
SPREAD_DRAWS = 10


def spread_picks(by_ds, held, total, m, rng):
    """``total`` training participants, ``total // m`` from each of ``m``
    datasets other than ``held``; only datasets with that many can be drawn."""
    per = total // m
    pool = sorted(d for d in by_ds if d != held and len(by_ds[d]) >= per)
    if len(pool) < m:
        raise ValueError(f"{total} participants from {m} datasets: only {len(pool)} have {per}")
    return {(str(d), str(s)) for d in rng.choice(pool, m, replace=False)
            for s in rng.choice(sorted(by_ds[d]), per, replace=False)}


def spread(ctx, out):
    """Datasets or participants, at a matched total: the readout trains on
    ``total`` labelled participants drawn from ``m`` datasets (``total / m``
    from each), for the frozen basis and for the same ridge on voxels. Every
    fold is still scored on its whole held-out dataset. ``m = 8`` with one
    participant each is the participant budget's B = 1. The volumes the
    readout trains on are not matched (runs differ in length) and are
    recorded. Resumable."""
    n = ctx.n
    recs, index = ctx.records([n], extra=[voxel_block(ctx.mask, 3)])
    by_ds, volumes = {}, {}
    for r in recs:
        by_ds.setdefault(r["dataset"], []).append(r["subject"])
        volumes[(r["dataset"], r["subject"])] = len(r["labels"])
    arms = {f"lr-cca:32+lags1@{n}": (slicer(index, f"lrcca_n{n}", k=32, lags=1), None),
            "voxels:3+lags1": (slicer(index, "vox3", lags=1), ridge_by_covariance)}
    rows, done = resume(out, lambda r: (r["arm"], r["total"], r["datasets"], r["draw"]))
    t0 = time.time()
    for total in SPREAD_TOTALS:
        for m in SPREAD_DATASETS:
            for draw in range(SPREAD_DRAWS):
                rng = np.random.default_rng(10000 * total + 100 * m + draw)
                picks = {held: spread_picks(by_ds, held, total, m, rng) for held in sorted(by_ds)}
                tf = lambda r, held, p=picks: (r["dataset"], r["subject"]) in p[held]  # noqa: E731
                vols = float(np.median([sum(volumes[k] for k in p) for p in picks.values()]))
                for name, (fn, readout) in arms.items():
                    if (name, total, m, draw) in done:
                        continue
                    res = probe.lodo(recs, fn, train_filter=tf, readout=readout)
                    rows.append(summary(res, {"total": total, "datasets": m, "draw": draw,
                                              "arm": name, "training_volumes": vols}))
                    log(f"T={total:<3} m={m} d{draw} {name:<24} sub-TR {rows[-1]['subtr']:.4f}  "
                        f"mean {rows[-1]['subtr_mean']:.4f}", t0)
                    json.dump(rows, open(out, "w"), indent=1)

SHOTS = [1, 2, 4, 8, 16, None]
SHOT_DRAWS = 6


def fewshot(ctx, out):
    """A new study that has labelled a few of its own participants
    (``split="within", shots=...``): the readout is fitted on those alone, under
    each corpus basis and under what a study can do with no corpus (stride-4
    voxels, a PCA of its own scans). Shots = 0 is the frozen decoder. The
    ``frozen+calib`` arms refit a ridge from the frozen decoder's output to the
    new study's gaze -- a readout trained elsewhere, calibrated on a few."""
    own = [(f"ownpca_{ds}", W, mu, (K,)) for ds, (W, mu, _n) in fit_own_bases(ctx.root, ctx.mask).items()]
    recs, index = ctx.records(ctx.ladder(), extra=own)
    n = ctx.n
    arms = {}
    for size in ctx.ladder():
        arms[f"lr-cca:32+lags1@{size}"] = slicer(index, f"lrcca_n{size}", k=32, lags=1)
        arms[f"corpus-pca:64+lags1@{size}"] = slicer(index, f"cpca_n{size}", k=64, lags=1)
    arms[f"corpus-pca:32+lags1@{n}"] = slicer(index, f"cpca_n{n}", k=32, lags=1)
    arms["own-pca:64+lags1"] = own_slicer(index, 64, 1, "ownpca")
    arms["own-pca:32+lags1"] = own_slicer(index, 32, 1, "ownpca")
    arms["raw+lags1"] = slicer(index, "raw", lags=1)
    rows, t0 = [], time.time()

    def add(res, arm, shots, seed):
        rows.append(summary(res, {"arm": arm, "shots": shots, "seed": seed}))
        log(f"K={str(shots):<5} s{seed} {arm:<30} sub-TR {rows[-1]['subtr']:.4f}  "
            f"mean {rows[-1]['subtr_mean']:.4f}", t0)

    for name in (f"lr-cca:32+lags1@{n}", f"corpus-pca:64+lags1@{n}", "raw+lags1"):
        res = probe.lodo(recs, arms[name], keep_predictions=name.startswith("lr-cca"))
        add(res, name, 0, 0)
        if "predictions" in res:
            trace = res["predictions"]
            arms[f"frozen+calib|{name}"] = lambda r, p=trace: probe.make_lags(p[(r["dataset"], r["subject"])], 1)
            arms[f"frozen+calib+feat|{name}"] = lambda r, p=trace, f=arms[name]: np.concatenate(
                [probe.make_lags(p[(r["dataset"], r["subject"])], 1), f(r)], axis=1)
    for shots in SHOTS:
        for seed in ([0] if shots is None else range(SHOT_DRAWS)):
            for name, fn in arms.items():
                add(probe.lodo(recs, fn, split="within", shots=shots, seed=seed), name, shots, seed)
            json.dump(rows, open(out, "w"), indent=1)


def fewshot_supervised(ctx, out):
    """The few-shot table for decoders that have seen no unlabelled scan: voxel
    blocks into a ridge, a PCA of the labelled participants' own scans fitted
    inside the readout, and the PCA of the other eight datasets. Same draws as
    :func:`fewshot`; the ridge on every voxel is run for one and two shots only.
    Does not depend on the corpus."""
    folds = fit_fold_bases(ctx.root, ctx.mask, None)
    blocks = [(f"foldpca_{ds}", W, mu, (K,)) for ds, (W, mu, _n) in folds.items()]
    blocks.append(voxel_block(ctx.mask, 2))
    recs, index = project(ctx.root, ctx.mask, blocks)
    voxels = slicer(index, "vox2", lags=1)
    arms = {"fold-pca:64+lags1": (own_slicer(index, 64, 1, "foldpca"), None, SHOTS),
            "labeled-pca:64+lags1": (voxels, labeled_pca_readout(64), SHOTS),
            "labeled-pca:256+lags1": (voxels, labeled_pca_readout(256), SHOTS),
            "voxels+lags1": (voxels, None, [1, 2])}
    rows, t0 = [], time.time()
    for shots in SHOTS:
        for seed in ([0] if shots is None else range(SHOT_DRAWS)):
            for name, (fn, readout, which) in arms.items():
                if shots not in which or (name == "voxels+lags1" and seed >= len(SEEDS)):
                    continue
                res = probe.lodo(recs, fn, split="within", shots=shots, seed=seed, readout=readout)
                rows.append(summary(res, {"arm": name, "shots": shots, "seed": seed}))
                log(f"K={str(shots):<5} s{seed} {name:<24} sub-TR {rows[-1]['subtr']:.4f}", t0)
            json.dump(rows, open(out, "w"), indent=1)


SCHEMES = {
    "own run, first quarter": {"split": "halves", "train_fraction": 0.25},
    "own run, first half": {"split": "halves"},
    "other participants, one": {"split": "within", "shots": 1},
    "other participants, all": {"split": "within"},
    "other datasets, one": {"datasets": 1},
    "other datasets, all eight": {"split": "lodo"},
}


def schemes(ctx, out):
    """Every participant's r under six decoding schemes -- its own run, its
    dataset's other participants, other datasets, each with scarce and with
    full labels -- for both corpus bases, a PCA of the labelled scans, and the
    voxel ridge at two resolutions. Draws are averaged per participant. Cheap
    arms first; resumable, each (scheme, arm) one block."""
    n = ctx.n
    arms = {f"lr-cca:32+lags1@{n}": (f"lrcca_n{n}", 32, None),
            f"corpus-pca:64+lags1@{n}": (f"cpca_n{n}", 64, None),
            "voxels:3+lags1": ("vox3", None, ridge_auto),
            "labeled-pca:64+lags1": ("vox2", None, labeled_pca_readout(64)),
            "voxels:2+lags1": ("vox2", None, ridge_auto)}
    rows, done = resume(out, lambda r: (r["scheme"], r["arm"]))
    todo = [(sc, arm) for arm in arms for sc in SCHEMES if (sc, arm) not in done]
    if not todo:
        return
    recs, index = ctx.records([n], extra=[voxel_block(ctx.mask, 2), voxel_block(ctx.mask, 3)])
    names = sorted({r["dataset"] for r in recs})
    t0 = time.time()
    for scheme, arm in todo:
        block, k, readout = arms[arm]
        fn = slicer(index, block, k=k, lags=1)
        spec = dict(SCHEMES[scheme])
        m = spec.pop("datasets", None)
        draws = DATASET_DRAWS if m is not None else SHOT_DRAWS if "shots" in spec else 1
        got = {}
        for seed in range(draws):
            kw = dict(spec)
            if m is not None:
                picks = dataset_picks(names, m, seed)
                kw["train_filter"] = lambda r, held, p=picks: r["dataset"] in p[held]
            elif "shots" in kw:
                kw["seed"] = seed
            for q in probe.lodo(recs, fn, readout=readout, **kw)["participants"]:
                got.setdefault((q["dataset"], q["subject"]), []).append((q["subtr"]["r"], q["1tr"]["r"]))
        for (ds, sub), v in got.items():
            v = np.array(v, dtype=np.float64)
            rows.append({"scheme": scheme, "arm": arm, "dataset": ds, "subject": sub,
                         "subtr": float(np.nanmean(v[:, 0])), "1tr": float(np.nanmean(v[:, 1]))})
        json.dump(rows, open(out, "w"))
        log(f"{scheme:<28} {arm:<26} {len(got)} participants", t0)


def participants(ctx, out):
    """Every participant's leave-one-dataset-out r for both corpus bases along
    the corpus ladder, the voxel ridge and the PCA of the labelled scans, so a
    figure can bootstrap over participants instead of quoting bare medians."""
    recs, index = ctx.records(ctx.ladder(), extra=[voxel_block(ctx.mask, 3)])
    rows, t0 = [], time.time()
    for n in ctx.ladder():
        for arm, block, k in ((f"lr-cca:32+lags1@{n}", f"lrcca_n{n}", 32),
                              (f"corpus-pca:64+lags1@{n}", f"cpca_n{n}", 64)):
            rows += per_participant(probe.lodo(recs, slicer(index, block, k=k, lags=1)), arm=arm)
            log(arm, t0)
    rows += per_participant(probe.lodo(recs, slicer(index, "vox3", lags=1), readout=ridge_auto),
                            arm="voxels:3+lags1")
    del recs
    folds = fit_fold_bases(ctx.root, ctx.mask, None)
    names = sorted(folds)
    fold_recs, _ = project(ctx.root, ctx.mask, [(f, folds[f][0], folds[f][1], (K,)) for f in names],
                           verbose=False)
    rows += per_participant(score_foldlocal(fold_recs, names, lags=1), arm="fold-pca:64+lags1")
    json.dump(rows, open(out, "w"))
