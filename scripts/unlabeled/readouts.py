"""Does the comparison depend on the readout? Every base -- the frozen basis,
the corpus PCA, voxels with no unsupervised step -- read out every way."""
import json
import time

from deepmreye import probe
from unlabeled.common import (
    log, mlp_readout, ols_readout, pls_readout, resume, ridge_auto, slicer, summary, voxel_block,
)

# (name, lags, readout). PLS keeps 64 components, which is the whole space for
# the 32- and 64-dimensional bases: there it is the ridge again, and says so.
# 2x2x2 voxels with +-2 TR, PLS or the MLP need a second ~20 GB copy of the
# design matrix and are not run.
GRID_READOUTS = [("ridge, +-1 TR", 1, "ridge"), ("ridge, no context", 0, "ridge"),
                 ("ridge, +-2 TR", 2, "ridge"), ("least squares, +-1 TR", 1, "ols"),
                 ("PLS-64 then ridge, +-1 TR", 1, "pls"), ("MLP, +-1 TR", 1, "mlp")]
GRID_SKIP = {("voxels 2x2x2", "ridge, +-2 TR"), ("voxels 2x2x2", "PLS-64 then ridge, +-1 TR"),
             ("voxels 2x2x2", "MLP, +-1 TR")}
MAKE = {"ridge": lambda lags: ridge_auto, "ols": lambda lags: ols_readout,
        "pls": lambda lags: pls_readout(64, lags), "mlp": lambda lags: mlp_readout}


def readout_grid(ctx, out):
    """Each base read out six ways, leave-one-dataset-out on all eight other
    datasets: which base wins does not depend on the readout. The voxel rows do
    not depend on the corpus, so a rerun after refitting the bases keeps them.
    Resumable, keyed by (base, readout)."""
    n = ctx.n
    bases = {f"lr-cca:32 @ N={n}": (f"lrcca_n{n}", 32), f"corpus PCA:64 @ N={n}": (f"cpca_n{n}", 64),
             "voxels 3x3x3": ("vox3", None), "voxels 2x2x2": ("vox2", None)}
    rows, done = resume(out, lambda r: (r["base"], r["readout"]))
    rows = [r for r in rows if r["base"] in bases]            # rows of an older corpus size go
    done = {(r["base"], r["readout"]) for r in rows}
    todo = [(b, r) for b in bases for r in GRID_READOUTS if (b, r[0]) not in done and (b, r[0]) not in GRID_SKIP]
    if not todo:
        return
    recs, index = ctx.records([n], extra=[voxel_block(ctx.mask, 3), voxel_block(ctx.mask, 2)])
    t0 = time.time()
    for base, (name, lags, kind) in todo:
        block, k = bases[base]
        res = probe.lodo(recs, slicer(index, block, k=k, lags=lags), readout=MAKE[kind](lags))
        rows.append(summary(res, {"base": base, "readout": name}))
        log(f"{base:<22} {name:<28} sub-TR {rows[-1]['subtr']:.4f}  1-TR {rows[-1]['1tr']:.4f}  "
            f"mean {rows[-1]['subtr_mean']:.4f}", t0)
        json.dump(rows, open(out, "w"), indent=1)


# The three labelled datasets acquired after DeepMReye 1.0's released CNN was
# trained (datasets_1to6.h5): the only folds it has not seen.
DME1_HELD_OUT = ("dsL07_deepmreye_calib", "dsL08_studyforrest_movie", "dsL11_backtothefuture")
TABLE_METRICS = ("r", "r_x", "r_y", "r2", "euclid_median")
# (row, block, k, lags, readout): ours, and the supervised decoders worth a row --
# the same ridge, the best at sub-TR (PLS), the best at 1-TR (no context), the MLP.
TABLE_ARMS = [("ours: frozen bilateral basis", "lrcca", 32, 1, "ridge"),
              ("ridge on voxels, +-1 TR", "vox3", None, 1, "ridge"),
              ("ridge on voxels, no context", "vox3", None, 0, "ridge"),
              ("PLS-64 on voxels, +-1 TR", "vox3", None, 1, "pls"),
              ("MLP on voxels, +-1 TR", "vox3", None, 1, "mlp")]


# The datasets the released CNN (datasets_1to6.h5) was trained on.
DME1_TRAINED_ON = ("dsL01_guided_fixations", "dsL02_pursuit", "dsL03_pursuit", "dsL04_pursuit",
                   "dsL05_free_viewing", "dsL06_sequences")


def against_dme1(ctx, out, dme1="results/dme1/score_1to6.json", train_on=None):
    """The published CNN against ours and selected supervised decoders, with
    every metric the probe reports (r, its two axes, calibrated R2 and median
    Euclidean error), per fold. The CNN is scored from its saved predictions
    (``scripts/eval_dme1.py score``) through the same ``probe._summarise``.
    Ours and the supervised decoders train on the other eight labelled
    datasets, or only on ``train_on`` (the CNN's own six, for a matched
    comparison). Resumable, keyed by row."""
    rows, done = resume(out, lambda r: r["arm"])
    train_filter = None if train_on is None else (lambda rec, held: rec["dataset"] in train_on)

    def folds(res):
        return {d: {"n": f["n"], **{k: {m: float(f[k][m]) for m in TABLE_METRICS} for k in ("subtr", "1tr")}}
                for d, f in res["folds"].items()}

    if "DeepMReye 1.0" not in done:
        d = json.load(open(dme1))
        parts = d["participants"]
        rows.append({"arm": "DeepMReye 1.0", "weights": d["weights"],
                     "folds": folds(probe._summarise(parts, sorted({p["dataset"] for p in parts})))})
        json.dump(rows, open(out, "w"), indent=1)
    todo = [a for a in TABLE_ARMS if a[0] not in done]
    if not todo:
        return
    n = ctx.n
    recs, index = ctx.records([n], extra=[voxel_block(ctx.mask, 3)])
    t0 = time.time()
    for name, block, k, lags, kind in todo:
        res = probe.lodo(recs, slicer(index, f"lrcca_n{n}" if block == "lrcca" else block, k=k, lags=lags),
                         readout=MAKE[kind](lags), train_filter=train_filter)
        rows.append({"arm": name, "folds": folds(res)})
        held = [res["folds"][d]["subtr"]["r"] for d in DME1_HELD_OUT]
        log(f"{name:<30} sub-TR mean over the three {sum(held) / 3:.4f}", t0)
        json.dump(rows, open(out, "w"), indent=1)


def against_dme1_matched(ctx, out):
    """:func:`against_dme1` with ours and the supervised decoders trained only on
    the six datasets the released CNN was trained on, so every row of the table
    has seen the same labelled data. Resumable, keyed by row."""
    against_dme1(ctx, out, train_on=DME1_TRAINED_ON)
