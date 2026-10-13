"""What every stage of the analysis shares: the run context, the projection of
the labelled participants onto every basis, feature slicers, and the readouts.

Everything here is a parameter or a feature function for
:func:`deepmreye.probe.lodo`; nothing scores. The score has one implementation
and it is the probe's.
"""
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmreye import probe  # noqa: E402
from deepmreye.unsupervised import Moments, _slabs, _top_eigenvectors, load_basis  # noqa: E402

K = 64                      # every corpus basis is carried at 64 and sliced down
FOLDPCA_TRS = 200           # per participant, deliberately more than the corpus bases' 48
SEEDS = [0, 1, 2]
LADDER = (10, 100, 400, 800)   # the corpus sizes the costlier stages are run at, plus the full one


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #

def corpus_sizes(basis_dir):
    """The checkpoints of the basis sweep, smallest first."""
    sweep = Path(basis_dir) / "sweep.json"
    if sweep.exists():
        return list(json.loads(sweep.read_text())["sizes"])
    return sorted(int(p.stem.split("_n")[1]) for p in Path(basis_dir).glob("basis_n*.npz"))


@dataclass
class Context:
    """One run of the analysis: where the data and bases are, and a cache of
    the projected labelled participants, which every stage starts from."""
    root: Path
    mask: np.ndarray
    basis_dir: Path
    sizes: list
    _records: dict = field(default_factory=dict)

    @property
    def n(self):
        """The full corpus: the size every headline number is at."""
        return self.sizes[-1]

    def ladder(self):
        return [s for s in LADDER if s in self.sizes] + [self.n]

    def basis(self, n=None):
        return load_basis(Path(self.basis_dir) / f"basis_n{n or self.n}.npz")

    def records(self, sizes=None, extra=()):
        """The labelled participants under the corpus bases at ``sizes`` (all
        of them by default) and any ``extra`` blocks. Cached without extras."""
        sizes = tuple(sizes or self.sizes)
        if extra:
            return scaling_records(self, sizes, extra)
        if sizes not in self._records:
            full = tuple(self.sizes)
            if full in self._records:          # every smaller set is a subset of it
                return self._records[full]
            self._records.clear()
            self._records[sizes] = scaling_records(self, sizes)
        return self._records[sizes]

    def clear(self):
        self._records.clear()


# --------------------------------------------------------------------------- #
# Projection: the labelled participants onto every basis in one pass
# --------------------------------------------------------------------------- #

def labelled_files(root):
    for d in sorted(Path(root).glob("dsL*")):
        for p in sorted(d.glob("*.h5")):
            yield d.name, p


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


def project(root, mask, blocks, verbose=True):
    """Every labelled participant's coordinates under every registered block.

    One matmul per participant rather than one per basis: they are all linear
    maps of the same masked voxels, and ``(x - mu) @ W = x @ W - (mu @ W)``, so
    the per-basis means become a constant row and the matmul is shared.
    Returns ``(records, index)``; ``index[name]`` is that block's column range
    and shape in ``rec["z"]``.
    """
    flat = mask.reshape(-1)
    P, off, index = _stack(blocks)
    if verbose:
        print(f"[*] projection matrix {P.shape} ({P.nbytes / 1e9:.2f} GB)", flush=True)
    recs, t0 = [], time.time()
    for ds, path in labelled_files(root):
        with h5py.File(path, "r") as f:
            if "labels" not in f:
                continue
            block, labels = f["eye_block"][:], f["labels"][:]
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


def scaling_records(ctx, sizes, extra=()):
    """The labelled participants under both corpus bases at every size in
    ``sizes`` (blocks ``lrcca_n{N}``, per orbit, and ``cpca_n{N}``), the
    stride-4 voxels (``raw``), and any ``extra`` ``(name, W, mu, shape)`` blocks."""
    blocks = list(extra)
    for n in sizes:
        _m, bases, _meta = ctx.basis(n)
        blocks += corpus_blocks(ctx.mask, bases, f"n{n}")
        del bases
    blocks.append(stride4_block(ctx.mask))
    return project(ctx.root, ctx.mask, blocks)


def corpus_blocks(mask, bases, tag, k=K):
    """One fitted basis file as two projection blocks: ``lrcca_{tag}`` (the
    leading ``k`` canonical directions of each orbit) and ``cpca_{tag}``."""
    lr = bases["lr-cca"]
    W = np.zeros((int(mask.sum()), 2 * k), dtype=np.float32)
    W[lr["left_index"], :k] = lr["left_weights"][:, :k]
    W[lr["right_index"], k:] = lr["right_weights"][:, :k]
    pca = bases["corpus-pca"]
    return [(f"lrcca_{tag}", W, lr["mean"], (2, k)),
            (f"cpca_{tag}", pca["components"][:, :k], pca["mean"], (k,))]


def stride4_block(mask):
    """``raw``: stride-4 in each spatial dimension, 12 x 8 x 5 = 480 voxels. The
    published DeepMReye 1.0 feature budget, and an arm that needs no corpus --
    but not a fair supervised baseline, since it keeps 1 voxel in 30."""
    sel = np.zeros(mask.shape, dtype=bool)
    sel[::4, ::4, ::4] = True
    idx = np.nonzero(sel.reshape(-1)[mask.reshape(-1)])[0]
    W = np.zeros((int(mask.sum()), len(idx)), dtype=np.float32)
    W[idx, np.arange(len(idx))] = 1.0
    return ("raw", W, np.zeros(int(mask.sum())), (len(idx),))


def voxel_blocks(mask, size=2):
    """Every masked voxel, averaged in ``size``^3 blocks -- a reduction that
    looks at no data, so an arm built on it has seen no unlabelled scan."""
    cells = np.argwhere(mask) // size
    inv = np.unique(cells, axis=0, return_inverse=True)[1].reshape(-1)
    W = np.zeros((len(cells), int(inv.max()) + 1), dtype=np.float32)
    W[np.arange(len(cells)), inv] = 1.0
    return W / W.sum(axis=0)


def voxel_block(mask, size):
    """:func:`voxel_blocks` as a projection block named ``vox{size}``."""
    W = voxel_blocks(mask, size)
    return (f"vox{size}", W, np.zeros(len(W)), (W.shape[1],))


# --------------------------------------------------------------------------- #
# Feature functions
# --------------------------------------------------------------------------- #

def slicer(index, name, k=None, lags=0):
    """Feature function reading one registered block out of ``rec['z']``: the
    leading ``k`` directions (the orbit average, for a per-orbit block), with
    ``lags`` TRs of context either side."""
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


def orbit_slicer(index, name, side, k=32, lags=1):
    """One orbit's canonical coordinates out of a per-orbit block, lagged: the
    ``views`` that :func:`deepmreye.metrics.orbit_agreement` compares."""
    a, b, shape = index[name]

    def fn(rec):
        z = rec["z"][:, a:b].astype(np.float64).reshape(len(rec["z"]), *shape)
        return probe.make_lags(z[:, side, :k], lags)
    return fn


def orbit_views(index, name, k=32, lags=1):
    return {"left": orbit_slicer(index, name, 0, k, lags),
            "right": orbit_slicer(index, name, 1, k, lags)}


def own_slicer(index, k, lags, prefix):
    """Each participant under the block registered for its *own* dataset,
    ``{prefix}_{dataset}`` -- a fold-local or a study-local basis."""
    def fn(rec):
        a = index[f"{prefix}_{rec['dataset']}"][0]
        return probe.make_lags(rec["z"][:, a:a + k].astype(np.float64), lags)
    return fn


def summary(res, extra):
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


def per_participant(res, **extra):
    """One row per scored participant: ``{**extra, dataset, subject, subtr, 1tr}``."""
    return [{**extra, "dataset": q["dataset"], "subject": q["subject"],
             "subtr": q["subtr"]["r"], "1tr": q["1tr"]["r"]} for q in res["participants"]]


def log(msg, t0):
    print(f"  {msg}  [{time.time() - t0:6.0f}s]", flush=True)


def resume(out, key):
    """Rows already in ``out`` and the set of their ``key(row)`` -- for the
    stages that are slow enough to be resumable."""
    rows = json.load(open(out)) if Path(out).exists() else []
    return rows, {key(r) for r in rows}


# --------------------------------------------------------------------------- #
# Who trains: budgets and dataset draws
# --------------------------------------------------------------------------- #

DATASET_BUDGETS = [1, 2, 4, 6, 8]
DATASET_DRAWS = 4           # draws of the training datasets, for every arm
CURVE_DRAWS = 20            # ... for the two arms of the labelled-dataset curve (Fig. b)


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


def dataset_picks(datasets, m, seed):
    """Which ``m`` datasets each held-out fold may train on, for one draw."""
    rng = np.random.default_rng(1000 * m + seed)
    return {held: set(rng.permutation([d for d in datasets if d != held])[:m].tolist())
            for held in datasets}


# --------------------------------------------------------------------------- #
# The no-corpus references: PCA of the labelled scans themselves
# --------------------------------------------------------------------------- #

def dataset_moments(root, mask, keep=None):
    """Second moments of every labelled dataset's own scans, one accumulator each."""
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
    return per_ds


def pooled_pca(moments, k=K):
    """Top-``k`` principal directions of a sum of accumulators."""
    acc = Moments(moments[0].d)
    for m in moments:
        acc.c += m.c
        acc.s += m.s
        acc.n += m.n
    if acc.n < k + 2:
        return None
    cov, mu = acc.covariance()
    return _top_eigenvectors(cov, k)[0].astype(np.float32), mu.astype(np.float32), acc.n


def fit_fold_bases(root, mask, keep, k=K):
    """One PCA per held-out dataset, over the labelled participants of the
    *other* datasets that the budget allows."""
    per_ds = dataset_moments(root, mask, keep)
    bases = {}
    for held in sorted(per_ds):
        got = pooled_pca([m for ds, m in per_ds.items() if ds != held], k)
        if got is not None:
            bases[held] = got
    return bases


def fit_own_bases(root, mask, k=K):
    """One PCA per dataset, from that dataset's own scans and nothing else.

    What a new study can do with no corpus: its own volumes are unlabelled data
    too. Every participant of the dataset is in the fit, test ones included --
    no labels are, and that is the deployment case, not a leak.
    """
    bases = {}
    for ds, m in dataset_moments(root, mask).items():
        got = pooled_pca([m], k)
        if got is not None:
            bases[ds] = got
    return bases


def score_foldlocal(recs, folds, train_filter=None, k=K, lags=0):
    """Each fold under its own basis, then one pass through the shipped reducer.

    ``lodo`` applies a single feature function to every record, and a fold-local
    basis is a different matrix per held-out dataset -- so each fold is run with
    its own basis and only that fold's participant rows are kept. The reduction
    from participants to fold medians to a summary is still ``probe._summarise``.
    """
    rows = []
    for i, held in enumerate(folds):
        def fn(rec, i=i):
            return probe.make_lags(rec["z"][:, i * k:(i + 1) * k].astype(np.float64), lags)
        res = probe.lodo(recs, fn, train_filter=train_filter)
        rows += [p for p in res["participants"] if p["dataset"] == held]
    return probe._summarise(rows, folds)


# --------------------------------------------------------------------------- #
# Readouts: ``readout(x, y, groups) -> predict`` for ``probe.lodo``
# --------------------------------------------------------------------------- #

def ridge_by_covariance(x, y, groups=None, alphas=probe.ALPHAS, chunk=20000):
    """The protocol's ``RidgeCV``, solved from ``X'X`` instead of an SVD of ``X``.

    Same leave-one-out choice of alpha and the same coefficients (they agree to
    1e-13), at the cost of one p x p eigendecomposition and two passes over the
    rows -- which is what makes a ridge on thousands of voxel columns and
    270,000 rows a two-minute fit rather than an hour's.
    """
    alphas = np.asarray(alphas, dtype=np.float64)
    n, mu, ym = len(x), x.mean(axis=0), y.mean(axis=0)
    evals, V = np.linalg.eigh(x.T @ x - n * np.outer(mu, mu))
    evals = np.maximum(evals, 0.0)
    Zy = V.T @ (x.T @ (y - ym))
    D = 1.0 / (evals[:, None] + alphas[None, :])
    err = np.zeros(len(alphas))
    for a in range(0, n, chunk):
        Z = (x[a:a + chunk] - mu) @ V
        h = (Z ** 2) @ D + 1.0 / n
        for j in range(len(alphas)):
            resid = (y[a:a + chunk] - ym) - Z @ (D[:, j:j + 1] * Zy)
            err[j] += float(((resid / (1.0 - h[:, j:j + 1])) ** 2).sum())
    coef = V @ (Zy / (evals + alphas[int(np.argmin(err))])[:, None])
    return lambda q: (q - mu) @ coef + ym


def ridge_auto(x, y, groups=None):
    """The protocol's ridge by whichever route is cheaper; both give its answer."""
    if len(x) > x.shape[1]:
        return ridge_by_covariance(x, y, groups)
    from sklearn.linear_model import RidgeCV

    return RidgeCV(alphas=probe.ALPHAS).fit(x, y).predict


def ols_readout(x, y, groups=None, rcond=1e-10):
    """Ordinary least squares, no penalty: the minimum-norm solution, from the
    same ``X'X`` eigendecomposition as :func:`ridge_by_covariance`."""
    n, mu, ym = len(x), x.mean(axis=0), y.mean(axis=0)
    evals, V = np.linalg.eigh(x.T @ x - n * np.outer(mu, mu))
    keep = evals > rcond * evals.max()
    coef = V[:, keep] @ ((V[:, keep].T @ (x.T @ (y - ym))) / evals[keep][:, None])
    return lambda q: (q - mu) @ coef + ym


def _centre_basis_readout(k, lags, fit_basis):
    """A basis fitted on the centre lag of the training rows, applied to every
    lag, then the protocol's ridge. ``fit_basis(mid, y, k) -> (mu, R)``."""
    def readout(x, y, groups):
        from sklearn.linear_model import RidgeCV

        p = x.shape[1] // (2 * lags + 1)
        mid = x[:, lags * p:(lags + 1) * p]
        mu, R = fit_basis(mid, y, min(k, len(mid) - 1, p))
        proj = lambda a: np.concatenate(  # noqa: E731
            [(a[:, i * p:(i + 1) * p] - mu) @ R for i in range(2 * lags + 1)], axis=1)
        model = RidgeCV(alphas=probe.ALPHAS).fit(proj(x), y)
        return lambda a: model.predict(proj(a))
    return readout


def pls_readout(k, lags=1):
    """A basis chosen *with* the labels -- the ``k`` voxel directions of partial
    least squares, which covary most with gaze -- then the protocol's ridge."""
    def fit(mid, y, k):
        from sklearn.cross_decomposition import PLSRegression

        return mid.mean(axis=0), PLSRegression(n_components=k, scale=False).fit(mid, y).x_rotations_
    return _centre_basis_readout(k, lags, fit)


def labeled_pca_readout(k, lags=1):
    """A PCA of the training participants' own scans, then the protocol's
    ridge: refitted inside every readout, from exactly the participants whose
    labels it may use, so nothing outside them enters. The same construction as
    :func:`pls_readout`, except that the basis has not seen the gaze."""
    def fit(mid, y, k):
        from sklearn.utils.extmath import randomized_svd

        mu = mid.mean(axis=0)
        return mu, randomized_svd(mid - mu, n_components=k, n_iter=4, random_state=0)[2].T
    return _centre_basis_readout(k, lags, fit)


def mlp_readout(x, y, groups, hidden=(256, 256), epochs=25, seed=0):
    """A small MLP, early-stopped on held-out *participants*.

    Rows of one run are nearly the same sample twice, so a row-wise validation
    split measures memorisation; every eighth participant is held out instead.
    sklearn's, not torch's: nothing in the feature path may load a second
    OpenMP runtime.
    """
    from sklearn.neural_network import MLPRegressor

    ids = np.unique(groups)
    val = np.isin(groups, ids[::8]) if len(ids) > 2 else (np.arange(len(x)) % 8 == 0)
    mu, sd = x[~val].mean(0), x[~val].std(0) + 1e-6
    xt, xv = (x[~val] - mu) / sd, (x[val] - mu) / sd
    net = MLPRegressor(hidden_layer_sizes=hidden, alpha=1e-2, batch_size=512, learning_rate_init=1e-3,
                       random_state=seed)
    best, state = np.inf, None
    for _ in range(epochs):
        net.partial_fit(xt, y[~val])
        err = float(np.mean((net.predict(xv) - y[val]) ** 2))
        if err < best:
            best = err
            state = ([c.copy() for c in net.coefs_], [b.copy() for b in net.intercepts_])
    net.coefs_, net.intercepts_ = state
    return lambda a: net.predict((a - mu) / sd)


# --------------------------------------------------------------------------- #
# After the readout: viewing priors, and keeping the confident participants
# --------------------------------------------------------------------------- #

# Datasets whose participants all saw the same stimulus on the same clock, from
# the paradigm, not from the labels: two feature films, and a guided-fixation
# task whose labels are its target sequence (identical for all 170 participants,
# 25 positions, no within-TR variation). Pursuit, free viewing, sequences and
# the calibration task randomise order per participant: their label ISC at
# matched TRs is 0.00-0.11, against 0.70 and 0.74 for the films.
SHARED_STIMULUS = {"dsL01_guided_fixations", "dsL08_studyforrest_movie", "dsL11_backtothefuture"}


def consensus_prior(recs, predictions, own_weight, other_weight):
    """Each participant's decoded gaze mixed with the mean of the others'.

    Label-free: every trace is z-scored per column before it is averaged, and
    only participants of a dataset in ``SHARED_STIMULUS`` are touched, over the
    first T volumes all of them share.
    """
    out = dict(predictions)
    by = {}
    for r in recs:
        if r["dataset"] in SHARED_STIMULUS and (r["dataset"], r["subject"]) in predictions:
            by.setdefault(r["dataset"], []).append((r["dataset"], r["subject"]))
    for keys in by.values():
        T = min(len(predictions[k]) for k in keys)
        z = {k: (predictions[k][:T] - predictions[k][:T].mean(0)) / (predictions[k][:T].std(0) + 1e-9)
             for k in keys}
        total = sum(z.values())
        for k in keys:
            mixed = predictions[k].copy()
            mixed[:T] = own_weight * z[k] + other_weight * (total - z[k]) / (len(keys) - 1)
            out[k] = mixed
    return out


def keep_most_confident(rows, score, frac, sign=1):
    """Median and mean over datasets of each dataset's most confident ``frac``'s
    median r. ``rows`` are ``{dataset, r, <score>}``."""
    meds = []
    for d in sorted({q["dataset"] for q in rows}):
        m = sorted((q for q in rows if q["dataset"] == d), key=lambda q: -sign * q[score])
        meds.append(np.median([q["r"] for q in m[:max(1, int(round(frac * len(m))))]]))
    return [float(np.median(meds)), float(np.mean(meds))]
