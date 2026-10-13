"""The corpus bases themselves: their spectrum, how decoding scales with the
corpus, what of it transfers, and every readout-side knob."""
import json
import time
from pathlib import Path

import numpy as np

from deepmreye import probe
from deepmreye.unsupervised import load_basis
from unlabeled.common import (
    K, SEEDS, corpus_blocks, log, mlp_readout, orbit_views, per_participant, project, resume, slicer,
    summary,
)


def basis_stats(ctx, out):
    """Label-free: the canonical-correlation spectrum at each corpus size, and
    how far the leading 32 directions of each are from the largest basis's.
    The sweep is nested, so the alignment is a convergence diagnostic rather
    than an independent replication."""
    spec, align, ref = {}, {}, {}
    _m, big, _meta = ctx.basis()
    for side in ("left", "right"):
        ref[side] = np.linalg.qr(big["lr-cca"][f"{side}_weights"][:, :32].astype(np.float64))[0]
    for n in ctx.sizes:
        _m, bases, _meta = ctx.basis(n)
        lr = bases["lr-cca"]
        spec[n] = lr["canonical_correlations"][:K].tolist()
        vals = []
        for side in ref:
            Q = np.linalg.qr(lr[f"{side}_weights"][:, :32].astype(np.float64))[0]
            vals.append(float((np.linalg.svd(Q.T @ ref[side], compute_uv=False) ** 2).mean()))
        align[n] = float(np.mean(vals))
    json.dump({"spectrum": spec, "alignment_to_largest": align}, open(out, "w"), indent=1)


def scaling(ctx, out):
    """Both bases refitted at every corpus size, decoded leave-one-dataset-out,
    with and without temporal context, and the stride-4 voxels that need no
    corpus."""
    recs, index = ctx.records()
    arms = {}
    for n in ctx.sizes:
        arms[f"lr-cca:32+lags1@{n}"] = slicer(index, f"lrcca_n{n}", k=32, lags=1)
        arms[f"lr-cca:32@{n}"] = slicer(index, f"lrcca_n{n}", k=32)
        arms[f"lr-cca:64@{n}"] = slicer(index, f"lrcca_n{n}", k=64)
        arms[f"corpus-pca:64@{n}"] = slicer(index, f"cpca_n{n}", k=64)
    arms["raw"] = slicer(index, "raw")
    arms["raw+lags1"] = slicer(index, "raw", lags=1)
    got, t0 = {}, time.time()
    for name, fn in arms.items():
        got[name] = summary(probe.lodo(recs, fn), {})
        log(f"{name:<28} sub-TR {got[name]['subtr']:.4f}  1-TR {got[name]['1tr']:.4f}", t0)
        json.dump(got, open(out, "w"), indent=1)


def draw_bases(ctx):
    """``{seed: {n: basis file}}`` for every corpus size below the full one.
    Seed 0 is the main sweep; every further seed is one directory
    ``<basis-dir>/draws/s{seed}`` written by ``fit_corpus_basis.py --seed``."""
    draws = {0: {n: Path(ctx.basis_dir) / f"basis_n{n}.npz" for n in ctx.sizes if n != ctx.n}}
    for d in sorted(Path(ctx.basis_dir).glob("draws/s*")):
        if (d / "sweep.json").exists():            # written last: the draw is complete
            draws[int(d.name[1:])] = {int(p.stem.split("_n")[1]): p for p in d.glob("basis_n*.npz")}
    return draws


def scaling_draws(ctx, out, group=5):
    """The corpus-size curve with its sampling spread: below the full corpus
    there are many ways to draw N participants, so every size is refitted on
    independent random draws (one seed each) and decoded leave-one-dataset-out.
    Per draw and size: both bases' summaries, and the frozen basis's
    per-participant r and orbit agreement, so the most-reliable fraction gets
    an interval too. ``group`` seeds share one projection pass; resumable per
    seed."""
    from deepmreye.metrics import orbit_agreement

    rows = json.load(open(out)) if Path(out).exists() else {}
    todo = {s: b for s, b in draw_bases(ctx).items() if str(s) not in rows and b}
    seeds, t0 = sorted(todo), time.time()
    for g in range(0, len(seeds), group):
        chunk, blocks = seeds[g:g + group], []
        for s in chunk:
            for n, path in sorted(todo[s].items()):
                _m, bases, _meta = load_basis(path)
                blocks += corpus_blocks(ctx.mask, bases, f"s{s}_n{n}", k=64)
                del bases
        recs, index = project(ctx.root, ctx.mask, blocks)
        for s in chunk:
            row = {}
            for n in sorted(todo[s]):
                name = f"lrcca_s{s}_n{n}"
                res = probe.lodo(recs, slicer(index, name, k=32, lags=1), keep_predictions=True,
                                 views=orbit_views(index, name))
                pca = probe.lodo(recs, slicer(index, f"cpca_s{s}_n{n}", k=64, lags=1))
                row[str(n)] = {
                    "lr-cca:32+lags1": summary(res, {}), "corpus-pca:64+lags1": summary(pca, {}),
                    "participants": {f"{q['dataset']}/{q['subject']}": q["subtr"]
                                     for q in per_participant(res)},
                    "agreement": {f"{d}/{p}": orbit_agreement(res["views"]["left"][(d, p)],
                                                              res["views"]["right"][(d, p)])
                                  for (d, p) in res["predictions"]}}
                log(f"seed {s:<3} N={n:<5} lr-cca {res['median_subtr']:.4f}  "
                    f"pca {pca['median_subtr']:.4f}", t0)
            rows[str(s)] = row
            json.dump(rows, open(out, "w"))
        del recs


def transfer(ctx, out):
    """Both corpus bases at every corpus size, the readout trained three ways
    (``probe.lodo(split=...)``): on the other eight datasets, on all nine, inside
    the held-out one. What a basis *contains* against what *transfers*. Both arms
    carry +-1 TR here, so this is also the matched-context comparison."""
    recs, index = ctx.records()
    rows, t0 = {}, time.time()
    for n in ctx.sizes:
        for name, fn in ((f"lr-cca:32+lags1@{n}", slicer(index, f"lrcca_n{n}", k=32, lags=1)),
                         (f"corpus-pca:64+lags1@{n}", slicer(index, f"cpca_n{n}", k=64, lags=1))):
            rows[name] = {}
            for split in probe.SPLITS:
                rows[name][split] = summary(probe.lodo(recs, fn, split=split), {})
                log(f"{name:<28}{split:<8} sub-TR {rows[name][split]['subtr']:.4f}"
                    f"  1-TR {rows[name][split]['1tr']:.4f}", t0)
            json.dump(rows, open(out, "w"), indent=1)


def gap(ctx, out):
    """The shipped arm under every split, with a ridge and with an MLP: the
    supervised non-linear ceiling, measured where transfer cannot confound it."""
    recs, index = ctx.records([ctx.n])
    fn = slicer(index, f"lrcca_n{ctx.n}", k=32, lags=1)
    rows, t0 = {}, time.time()
    for name, readout in (("ridge", None), ("mlp", mlp_readout)):
        rows[name] = {}
        for split in probe.SPLITS:
            rows[name][split] = summary(probe.lodo(recs, fn, split=split, readout=readout), {})
            log(f"{name:<6}{split:<8} sub-TR {rows[name][split]['subtr']:.4f}"
                f"  1-TR {rows[name][split]['1tr']:.4f}", t0)
            json.dump(rows, open(out, "w"), indent=1)


RANKS = [8, 16, 24, 32, 48, 64]


def _agreement(recs, pa, pb):
    """Median over participants of the correlation between two arms' predictions,
    and between their errors (what is left of gaze after each one's best
    per-column affine fit). Errors that correlate are a shared limit, not a
    basis-specific one."""
    cp, ce = [], []
    for r in recs:
        key = (r["dataset"], r["subject"])
        y = r["labels"].reshape(len(r["labels"]), 20)
        ok = np.isfinite(y).all(axis=1)
        if key not in pa or key not in pb or ok.sum() < 30:
            continue
        y = y[ok]
        preds = [p[key][ok] for p in (pa, pb)]
        errs = [np.stack([y[:, j] - np.polyval(np.polyfit(p[:, j], y[:, j], 1), p[:, j])
                          for j in range(20)], axis=1) for p in preds]
        cols = lambda u, v: float(np.mean(  # noqa: E731
            [np.corrcoef(u[:, j], v[:, j])[0, 1] for j in range(20)]))
        cp.append(cols(*preds))
        ce.append(cols(*errs))
    return {"predictions": float(np.median(cp)), "errors": float(np.median(ce))}


def _one_participants_worth(x, y, groups):
    """The protocol's ridge, on as many rows as one participant has, taken
    evenly from every training participant."""
    from sklearn.linear_model import RidgeCV

    ids = np.unique(groups)
    each = max(2, len(x) // len(ids) // len(ids))
    keep = np.concatenate([np.nonzero(groups == g)[0][:each] for g in ids])
    return RidgeCV(alphas=probe.ALPHAS).fit(x[keep], y[keep]).predict


def overlap(ctx, out):
    """Why the two corpus bases tie wherever the readout is fitted inside the
    study: the same subspace? the same signal? short of rows or of people?
    where in each ordering is the gaze?"""
    recs, index = ctx.records([ctx.n])
    base = {"lr-cca": f"lrcca_n{ctx.n}", "corpus-pca": f"cpca_n{ctx.n}"}
    ranked = lambda arm, k, lags=1: slicer(index, base[arm], k=k, lags=lags)  # noqa: E731
    shipped = {"lr-cca": ranked("lr-cca", 32), "corpus-pca": ranked("corpus-pca", 64)}
    got, t0 = {}, time.time()

    def regimes(fn):
        """Within the study, one shot of it, and across studies."""
        one = [summary(probe.lodo(recs, fn, split="within", shots=1, seed=s), {}) for s in SEEDS]
        return {"within": summary(probe.lodo(recs, fn, split="within"), {}),
                "lodo": summary(probe.lodo(recs, fn), {}),
                "shot1": {k: float(np.median([o[k] for o in one])) for k in ("subtr", "1tr", "subtr_mean")}}

    def note(tag, r):
        log(f"{tag:<28} within {r['within']['subtr']:.3f}  one shot {r['shot1']['subtr']:.3f}"
            f"  LODO {r['lodo']['subtr']:.3f} / {r['lodo']['1tr']:.3f}  mean {r['lodo']['subtr_mean']:.3f}", t0)

    # 1. The same subspace? Features centred per participant, lag 0.
    stack = {arm: np.concatenate([f(r) - f(r).mean(axis=0) for r in recs])
             for arm, f in (("lr-cca", ranked("lr-cca", 32, 0)), ("corpus-pca", ranked("corpus-pca", 64, 0)))}
    q = [np.linalg.qr(v)[0] for v in stack.values()]
    got["canonical_correlations"] = np.linalg.svd(q[0].T @ q[1], compute_uv=False).tolist()

    # 2. The same signal? Predictions and errors of the two shipped-rank arms.
    got["agreement"] = {}
    for tag, kw in (("within", {"split": "within"}), ("lodo", {}),
                    ("shot1", {"split": "within", "shots": 1})):
        preds = [probe.lodo(recs, f, keep_predictions=True, **kw)["predictions"] for f in shipped.values()]
        got["agreement"][tag] = _agreement(recs, *preds)
        log(f"agreement {tag:<8} {got['agreement'][tag]}", t0)
    both = lambda r: np.concatenate([f(r) for f in shipped.values()], axis=1)  # noqa: E731
    got["both"] = regimes(both)
    note("both bases concatenated", got["both"])

    # 3. Short of rows, or of people?
    got["rows_or_people"] = {
        arm: summary(probe.lodo(recs, f, split="within", readout=_one_participants_worth), {})
        for arm, f in shipped.items()}
    json.dump(got, open(out, "w"), indent=1)

    # 4. Where in each ordering is the gaze?
    got["ranks"] = {}
    for k in RANKS:
        for arm in base:
            got["ranks"][f"{arm}:{k}+lags1"] = regimes(ranked(arm, k))
            note(f"{arm}:{k}+lags1", got["ranks"][f"{arm}:{k}+lags1"])
        json.dump(got, open(out, "w"), indent=1)


TUNE_RANKS = {"corpus-pca": [16, 32, 48, 64, 96, 128, 192, 256],
              "lr-cca": [16, 24, 32, 48, 64, 96, 128]}
TUNE_LAGS = [0, 1, 2]
WIDE_ALPHAS = np.logspace(-2, 7, 19)


def tune(ctx, out):
    """Every readout-side knob for both corpus bases at the full corpus: rank,
    lag context, whitening of the PCA scores by the corpus eigenvalues, and a
    penalty grid three decades wider than the protocol's (`corpus-pca` scores
    carry the corpus variance as their scale, SD ~6 against ~0.9, so the
    protocol grid might under-regularise it). Best-of-grid for both arms, so
    equally optimistic for both. Resumable."""
    _m, bases, _meta = ctx.basis()
    pca, lr = bases["corpus-pca"], bases["lr-cca"]
    kp, kl = max(TUNE_RANKS["corpus-pca"]), max(TUNE_RANKS["lr-cca"])
    W = np.zeros((int(ctx.mask.sum()), 2 * kl), dtype=np.float32)
    W[lr["left_index"], :kl] = lr["left_weights"][:, :kl]
    W[lr["right_index"], kl:] = lr["right_weights"][:, :kl]
    scale = np.sqrt(np.maximum(pca["eigenvalues"][:kp].astype(np.float64), 1e-12))
    configs = [("corpus-pca", k, lags, whiten, grid)
               for k in TUNE_RANKS["corpus-pca"] for lags in TUNE_LAGS
               for whiten in (False, True) for grid in ("protocol", "wide")]
    configs += [("lr-cca", k, lags, False, grid)
                for k in TUNE_RANKS["lr-cca"] for lags in TUNE_LAGS for grid in ("protocol", "wide")]
    rows, done = resume(out, lambda r: (r["arm"], r["k"], r["lags"], r["whiten"], r["grid"]))
    todo = [c for c in configs if c not in done]
    if not todo:
        return
    recs, index = project(ctx.root, ctx.mask, [("pca", pca["components"][:, :kp], pca["mean"], (kp,)),
                                               ("lr", W, lr["mean"], (2, kl))])
    del bases
    t0 = time.time()
    for arm, k, lags, whiten, grid in todo:
        if arm == "corpus-pca":
            a = index["pca"][0]

            def fn(rec, a=a, k=k, lags=lags, whiten=whiten):
                z = rec["z"][:, a:a + k].astype(np.float64)
                return probe.make_lags(z / scale[:k] if whiten else z, lags)
        else:
            fn = slicer(index, "lr", k=k, lags=lags)
        res = probe.lodo(recs, fn, alphas=probe.ALPHAS if grid == "protocol" else WIDE_ALPHAS)
        rows.append(summary(res, {"arm": arm, "k": k, "lags": lags, "whiten": whiten, "grid": grid}))
        log(f"{arm:<11} k={k:<4} lags={lags} whiten={whiten!s:<5} {grid:<9} sub-TR "
            f"{rows[-1]['subtr']:.4f}  1-TR {rows[-1]['1tr']:.4f}  mean {rows[-1]['subtr_mean']:.4f}", t0)
        json.dump(rows, open(out, "w"), indent=1)
