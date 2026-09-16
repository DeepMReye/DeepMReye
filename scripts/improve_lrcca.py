#!/usr/bin/env python3
"""Can the cross-orbit basis be estimated better than `fit_lr_cca` estimates it?

`lr-cca` is closed in `k`, in corpus size, in readout structure and against
nuisance projection (FINDINGS.md). What was never swept is the *estimator*: two
constants inside `deepmreye.unsupervised.fit_lr_cca` -- `n_reduce=256` and
`shrinkage=1e-3` -- and the covariance it is fitted from, which has always been
the **total** one.

Four families, all still label-free, all still a frozen linear map:

`shrink`   the whitening ridge. CCA whitens each orbit by `C^{-1/2}` in a
           256-dimensional reduced space; the smallest of those 256 eigenvalues
           are the worst-estimated, and `1e-3 * lambda_max` barely touches them.
           If "lr-cca needs 400 participants" is an estimation-variance
           statement, this is the knob that says so.

`within`   the covariance. Pooling every TR of every participant around one
           global mean makes `C_total` mostly *between-participant* -- orbit
           size, registration, susceptibility -- which is bilaterally coherent
           and therefore exactly what a cross-orbit CCA will happily lock onto.
           Gaze is a within-participant signal and the probe scores it inside a
           run, so between-participant directions are wasted budget.
           `C_within = C_total - C_between` is exact and free from the moments.

`mirror`   the parameter count. The two orbits are approximate mirror images, so
           a direction in the left orbit has an anatomical twin in the right.
           Constraining `w_R = M w_L` halves the free parameters of the
           cross-covariance and turns the SVD into a symmetric generalized
           eigenproblem whose sign says whether a mode is conjugate-symmetric
           (vertical gaze) or antisymmetric (horizontal). This is the version of
           the bilateral prior that uses the anatomy, not just the correlation.

`combine`  what the readout is handed. `avg` is measured optimal against
           `avg+diff`, but concatenating a *different* basis (`corpus-pca`) has
           not been tried, and the two are selected by different criteria.

Everything is scored through `deepmreye.probe.lodo`, and the incumbent is refit
here rather than quoted, so every row of the table is the same number.

    python scripts/build_labeled_voxels.py --out <voxdir>
    python scripts/fit_lrcca_variants.py --n 2000
    python scripts/improve_lrcca.py --voxels <voxdir> --arms shrink,within
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
from deepmreye.unsupervised import _top_eigenvectors  # noqa: E402

K_STORE = 64          # every candidate is carried at 64 and sliced down
DEFAULT_K = 32
DEFAULT_LAGS = 1


# --------------------------------------------------------------------------- #
# Moments -> covariances
# --------------------------------------------------------------------------- #

class Corpus:
    """The saved moments, and the three covariances they decompose into."""

    def __init__(self, path, lag_path=None):
        d = np.load(path, allow_pickle=False)
        self.c = d["c"].astype(np.float64)
        self.s = d["s"]
        self.n = int(d["n"][0])
        self.mask = d["mask"].astype(bool)
        self.subject_mean = d["subject_mean"]
        self.subject_n = d["subject_n"]
        self.slab_mean = d["slab_mean"]
        self.slab_n = d["slab_n"]
        self.mu = self.s / self.n
        self.lag = None
        if lag_path is not None and Path(lag_path).exists():
            lz = np.load(lag_path, allow_pickle=False)
            sym = lz["lag"].astype(np.float64)
            self.lag = 0.5 * (sym + sym.T) / int(lz["n_pairs"][0])
        self._cache = {}

    def _between(self, mean, counts):
        """`sum_i n_i (mu_i - mu)(mu_i - mu)^T / n`, the group-mean scatter."""
        d = (mean.astype(np.float64) - self.mu) * np.sqrt(counts / self.n)[:, None]
        return d.T @ d

    def cov(self, kind):
        if kind in self._cache:
            return self._cache[kind]
        total = self.c / self.n - np.outer(self.mu, self.mu)
        if kind == "total":
            out = total
        elif kind == "within":
            out = total - self._between(self.subject_mean, self.subject_n)
        elif kind == "within-slab":
            out = total - self._between(self.slab_mean, self.slab_n)
        elif kind == "between":
            out = self._between(self.subject_mean, self.subject_n)
        elif kind in ("fast", "slow"):
            # cov of x_{t+1} -+ x_t, up to the edge terms a 12-TR slab makes
            # negligible. `fast` is a high-pass: it keeps what changes between
            # TRs. Differencing removes any mean, so the *uncentred* second
            # moment is the right one here.
            if self.lag is None:
                raise SystemExit("[!] fast/slow need --lag-moments")
            second = self.c / self.n
            out = 2.0 * (second - self.lag) if kind == "fast" \
                else 2.0 * (second + self.lag)
        else:
            raise ValueError(kind)
        self._cache[kind] = out
        return out

    def orbits(self, split_x=24):
        xs = np.nonzero(self.mask.reshape(-1))[0] // (self.mask.shape[1] * self.mask.shape[2])
        left = xs < split_x
        return np.nonzero(left)[0], np.nonzero(~left)[0]


# --------------------------------------------------------------------------- #
# The mirror map
# --------------------------------------------------------------------------- #

def mirror_pairs(mask, split_x=24):
    """Left-orbit and right-orbit row indices that are anatomical twins.

    The crop is not exactly symmetric, so the reflection plane is *measured*
    (the `x0` maximising mask overlap with its own reflection) and only voxels
    whose reflection is itself in the mask are paired. Everything unpaired is
    dropped, which is the honest version of a symmetry constraint: it applies
    where the symmetry exists.
    """
    nx, ny, nz = mask.shape
    flat = mask.reshape(-1)
    order = np.full(flat.shape, -1, dtype=np.int64)
    order[flat] = np.arange(int(flat.sum()))
    grid = np.argwhere(mask)

    best = None
    for twice_x0 in range(nx, 3 * nx // 2 + nx // 2):   # x0 in [nx/2, nx)
        xm = twice_x0 - grid[:, 0]
        ok = (xm >= 0) & (xm < nx)
        hit = int(mask[xm[ok], grid[ok, 1], grid[ok, 2]].sum())
        if best is None or hit > best[1]:
            best = (twice_x0, hit)
    twice_x0 = best[0]

    xm = twice_x0 - grid[:, 0]
    ok = (xm >= 0) & (xm < nx)
    src = order[np.ravel_multi_index((grid[ok, 0], grid[ok, 1], grid[ok, 2]), mask.shape)]
    dst = np.full(len(src), -1, dtype=np.int64)
    inside = mask[xm[ok], grid[ok, 1], grid[ok, 2]]
    dst[inside] = order[np.ravel_multi_index(
        (xm[ok][inside], grid[ok, 1][inside], grid[ok, 2][inside]), mask.shape)]
    keep = dst >= 0
    src, dst = src[keep], dst[keep]

    # Orient the pair so the first member is always the left orbit.
    xs = np.nonzero(flat)[0] // (ny * nz)
    lo = xs[src] < split_x
    li = np.where(lo, src, dst)
    ri = np.where(lo, dst, src)
    _u, first = np.unique(li, return_index=True)
    return li[np.sort(first)], ri[np.sort(first)], twice_x0 / 2.0, best[1]


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #

class Eigs:
    """Per-(covariance, index set) eigendecompositions, computed once.

    The shrinkage sweep is the point of this class: shrinkage enters *after* the
    decomposition, so refitting at six strengths must not pay for six
    randomized SVDs of a 7156x7156 block.
    """

    def __init__(self, n_reduce=512):
        self.n_reduce = n_reduce
        self._c = {}

    def get(self, cov, key, idx, seed=0):
        if key not in self._c:
            sub = cov[np.ix_(idx, idx)]
            self._c[key] = _top_eigenvectors(sub, min(self.n_reduce, len(idx) - 1), seed)
        return self._c[key]


def _whiten(vecs, vals, n_reduce, shrinkage, alpha=1.0):
    """Whitening raised to a power, so PLS and CCA are two ends of one family.

    CCA whitens each orbit fully (`alpha=1`) before taking the SVD of the cross
    block; partial least squares does not whiten at all (`alpha=0`) and reads
    the raw cross-covariance, which orders directions by shared *magnitude*
    rather than shared *correlation*. The two answer different questions and
    nothing in this project had asked the second one. Anything in between is a
    legitimate estimator too: whitening is what makes CCA able to promote a
    low-variance direction to the top of the spectrum, and that is both its
    whole value and the way it overfits.
    """
    vecs, vals = vecs[:, :n_reduce], vals[:n_reduce]
    return vecs / (vals + shrinkage * float(vals.max())) ** (alpha / 2.0)


def fit_cca(cov, li, ri, k, n_reduce=256, shrinkage=1e-3, seed=0, eigs=None,
            key="", alpha=1.0):
    """`fit_lr_cca`, re-expressed against an arbitrary covariance.

    ``alpha`` is the whitening exponent: 1.0 is CCA (the incumbent), 0.0 is
    cross-orbit PLS.
    """
    eigs = eigs or Eigs()
    wl = _whiten(*eigs.get(cov, (key, "l"), li, seed), n_reduce, shrinkage, alpha)
    wr = _whiten(*eigs.get(cov, (key, "r"), ri, seed), n_reduce, shrinkage, alpha)
    u, s, vt = np.linalg.svd(wl.T @ cov[np.ix_(li, ri)] @ wr, full_matrices=False)
    k = int(min(k, u.shape[1], vt.shape[0]))
    return {"left_index": li, "right_index": ri, "signs": np.ones(k),
            "left_weights": wl @ u[:, :k], "right_weights": wr @ vt[:k].T,
            "rho": s[:k]}


def fit_mirror_cca(cov, li, ri, k, n_reduce=256, shrinkage=1e-3, seed=0, eigs=None,
                   key="", select="abs"):
    """One weight vector for both orbits, applied through the mirror map.

    Maximises `w' A w / w' B w` with `A` the symmetrised cross-covariance of the
    mirrored pair and `B` the averaged within-orbit covariance -- the same
    objective as CCA with `w_R` tied to `w_L`. Eigenvalues are signed: positive
    means the two orbits move together under the reflection (vertical gaze),
    negative means they move oppositely (horizontal). Ranking by `|lambda|`
    keeps both.
    """
    cll, crr = cov[np.ix_(li, li)], cov[np.ix_(ri, ri)]
    clr = cov[np.ix_(li, ri)]
    b = 0.5 * (cll + crr)
    a = 0.5 * (clr + clr.T)
    eigs = eigs or Eigs()
    if (key, "m") not in eigs._c:
        eigs._c[(key, "m")] = _top_eigenvectors(b, min(eigs.n_reduce, len(li) - 1), seed)
    w = _whiten(*eigs._c[(key, "m")], n_reduce, shrinkage)
    m = w.T @ a @ w
    m = 0.5 * (m + m.T)
    lam, q = np.linalg.eigh(m)
    # The sign of an eigenvalue is a label-free statement about what the mode
    # is. Under the reflection, conjugate *vertical* gaze moves the two orbits
    # together (lambda > 0) while *horizontal* gaze moves them oppositely
    # (lambda < 0) -- and the bilaterally coherent nuisance this corpus is full
    # of, global signal and physiology, is symmetric. So the antisymmetric half
    # of the spectrum is a gaze-enriched subset that no correlation magnitude
    # can name.
    pool = {"abs": np.arange(len(lam)),
            "sym": np.nonzero(lam > 0)[0],
            "anti": np.nonzero(lam < 0)[0]}[select]
    take = pool[np.argsort(-np.abs(lam[pool]))[:k]]
    weights = w @ q[:, take]
    return {"left_index": li, "right_index": ri, "signs": np.sign(lam[take]),
            "left_weights": weights, "right_weights": weights,
            "rho": np.abs(lam[take])}


def cca_to_block(fit, n_vox, mu, k=K_STORE):
    """A fitted cross-orbit basis as one dense `[n_vox, k]` map of the average.

    The readout only ever sees `0.5 * (z_L + s * z_R)`, so the two orbit maps
    are folded into a single matrix here and the sign is absorbed into the right
    one. That keeps the projection a plain matmul and makes every arm, mirrored
    or not, the same shape downstream.
    """
    k = int(min(k, fit["left_weights"].shape[1]))
    w = np.zeros((n_vox, k), dtype=np.float32)
    w[fit["left_index"]] = 0.5 * fit["left_weights"][:, :k]
    w[fit["right_index"]] += 0.5 * fit["right_weights"][:, :k] * fit["signs"][:k]
    return w, mu


# --------------------------------------------------------------------------- #
# Projection over the labelled memmap
# --------------------------------------------------------------------------- #

def project(voxdir, blocks, chunk=20000, verbose=True):
    """`[(name, W, mu, k)]` -> per-participant records, in one pass."""
    voxdir = Path(voxdir)
    plan = json.loads((voxdir / "index.json").read_text())
    labels = np.load(voxdir / "labels.npz")
    mm = np.load(voxdir / "voxels.npy", mmap_mode="r")

    cols, offs, index, cursor = [], [], {}, 0
    for name, w, mu, k in blocks:
        w = np.asarray(w, dtype=np.float32)
        cols.append(w)
        # `mu` is the voxel-space mean to centre by, except where a caller has
        # already reduced it: a shifted basis must keep the *unshifted* offset,
        # because the mean belongs to the template and only the filter moved.
        mu = np.asarray(mu, dtype=np.float64)
        offs.append(mu if mu.shape == (w.shape[1],) else mu @ w.astype(np.float64))
        index[name] = (cursor, cursor + k)
        cursor += k
    p = np.concatenate(cols, axis=1)
    off = np.concatenate(offs)
    if verbose:
        print(f"[*] projection {p.shape} over {mm.shape[0]} TRs", flush=True)

    # Centre inside the chunk loop rather than promoting the whole result to
    # float64 afterwards: the stacked projection can be thousands of columns
    # wide and the temporary is otherwise larger than the output.
    z = np.empty((mm.shape[0], p.shape[1]), dtype=np.float32)
    off32 = off.astype(np.float32)
    t0 = time.time()
    for a in range(0, mm.shape[0], chunk):
        b = min(a + chunk, mm.shape[0])
        z[a:b] = np.asarray(mm[a:b]) @ p
        z[a:b] -= off32
    if verbose:
        print(f"[+] projected in {time.time() - t0:.0f}s", flush=True)

    return [{"dataset": r["dataset"], "subject": r["subject"],
             "z": z[r["start"]:r["stop"]],
             "labels": labels[f"{r['dataset']}/{r['subject']}"]}
            for r in plan], index


def slicer(index, name, k=None, lags=DEFAULT_LAGS, scale=None, norm=None):
    """One registered column block as a feature function.

    ``norm="participant"`` z-scores each column inside the participant before
    the readout sees it. That is the one standardisation the probe does not
    already do -- targets are z-scored per training *dataset*, features never --
    and the recorded cross-dataset failure mode is a gain mismatch, so it is
    worth a row even though it cannot change a within-participant correlation
    for a single feature. (It can change one for many: it re-weights how the
    ridge mixes columns whose scales differ between participants.)
    """
    a, b = index[name]

    def fn(rec):
        x = rec["z"][:, a:b].astype(np.float64)
        if k:
            x = x[:, :k]
        if scale is not None:
            x = x * scale[:x.shape[1]]
        if norm == "participant":
            sd = x.std(axis=0)
            sd[sd < 1e-9] = 1.0
            x = (x - x.mean(axis=0)) / sd
        return probe.make_lags(x, lags)
    return fn


def concat(fns):
    return lambda rec: np.concatenate([f(rec) for f in fns], axis=1)


def summarise(res, extra=None):
    return {**(extra or {}),
            "subtr": res["summary"]["subtr"]["median_r"],
            "1tr": res["summary"]["1tr"]["median_r"],
            "subtr_mean": res["summary"]["subtr"]["mean_r"],
            "1tr_mean": res["summary"]["1tr"]["mean_r"],
            "folds": {d: {"n": f["n"], "subtr": f["subtr"]["r"], "1tr": f["1tr"]["r"]}
                      for d, f in res["folds"].items()}}


# --------------------------------------------------------------------------- #
# The sweep
# --------------------------------------------------------------------------- #

SHRINK = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1]
WHITEN = [0.0, 0.25, 0.5, 0.75, 1.0]      # 0 = cross-orbit PLS, 1 = CCA
REDUCE = [64, 128, 256, 512]


def _specs(arms):
    """What to fit, as data: `(fitter, kind, n_reduce, shrinkage, alpha/select)`.

    A table rather than a cascade of `if`s, because the sweep is a table --
    every entry here is one row of the reported one, and a reader should be able
    to check the two against each other without following control flow.
    """
    out = []
    if arms & {"shrink", "incumbent"}:
        out += [("cca", "total", 256, sh, 1.0) for sh in SHRINK]
    if "reduce" in arms:
        out += [("cca", "total", nr, 1e-3, 1.0) for nr in REDUCE if nr != 256]
    if "within" in arms:
        out += [("cca", "within", 256, sh, 1.0) for sh in SHRINK]
        out += [("cca", "within", nr, 1e-3, 1.0) for nr in (128, 512)]
        out += [("cca", "within-slab", 256, 1e-3, 1.0)]
    if "mirror" in arms:
        out += [("mirror", kind, 256, sh, "abs")
                for kind in ("total", "within") for sh in (1e-3, 1e-2, 1e-1)]
    if "sign" in arms:
        out += [("mirror", "total", 256, sh, sel)
                for sel in ("sym", "anti") for sh in (1e-3, 1e-2)]
    if "mirrork" in arms:
        out += [("mirror", "total", 256, 1e-2, "abs")]
    if "whiten" in arms:
        out += [("cca", "total", 256, sh, al)
                for al in WHITEN if al != 1.0 for sh in (1e-3, 1e-2)]
    if "fast" in arms:
        for kind in ("fast", "slow"):
            out += [("cca", kind, 256, sh, 1.0) for sh in (1e-3, 1e-2, 1e-1)]
            out += [("mirror", kind, 256, 1e-2, "abs")]
    return out


def spec_name(fitter, kind, nr, sh, extra):
    if fitter == "cca":
        return f"{kind}_r{nr}_s{sh:g}" + ("" if extra == 1.0 else f"_a{extra:g}")
    tag = "mirror" if extra == "abs" else f"mirror{extra}"
    return f"{tag}-{kind}_r{nr}_s{sh:g}"


def build_candidates(corpus, arms):
    """Every basis to be scored, as `[(name, W, mu, k)]`. No labels involved."""
    n_vox = int(corpus.mask.sum())
    li, ri = corpus.orbits()
    mli, mri, plane, overlap = mirror_pairs(corpus.mask)
    print(f"[*] orbits {len(li)}/{len(ri)} voxels; mirror plane x={plane:.1f}, "
          f"{len(mli)} pairs ({overlap} voxels overlap)", flush=True)

    blocks, eigs, t0 = [], Eigs(), time.time()
    for fitter, kind, nr, sh, extra in _specs(arms):
        cov = corpus.cov(kind)
        if fitter == "cca":
            fit = fit_cca(cov, li, ri, K_STORE, nr, sh, eigs=eigs, key=kind,
                          alpha=extra)
        else:
            fit = fit_mirror_cca(cov, mli, mri, K_STORE, nr, sh, eigs=eigs,
                                 key=f"mirror-{kind}", select=extra)
        name = spec_name(fitter, kind, nr, sh, extra)
        blocks.append((name,) + cca_to_block(fit, n_vox, corpus.mu) + (K_STORE,))
        r = fit["rho"]
        print(f"  fit {name:<32} rho1 {r[0]:.3f}  rho32 {r[min(31, len(r) - 1)]:.3f}"
              f"  [{time.time() - t0:5.0f}s]", flush=True)

    if "random" in arms:
        # The control every frozen basis needs: same shape, same rank, no
        # corpus. A random orthogonal projection of the eye mask says what a
        # k=32 linear map is worth before any criterion is applied to it.
        q = np.linalg.qr(np.random.default_rng(0).standard_normal((n_vox, K_STORE)))[0]
        blocks.append(("random", q.astype(np.float32), corpus.mu, K_STORE))
    if "combine" in arms:
        for kind, tag in (("total", "corpus-pca"), ("within", "within-pca"),
                          ("fast", "fast-pca")):
            if kind == "fast" and corpus.lag is None:
                continue
            vecs, _vals = _top_eigenvectors(corpus.cov(kind), K_STORE)
            blocks.append((tag, vecs.astype(np.float32), corpus.mu, K_STORE))
    return blocks


def _rows(arms):
    """What to report, as `(label, block key, kwargs)`. Same table discipline."""
    out = [("lr-cca:32+lags1  [incumbent]", "total_r256_s0.001", {"k": 32})]
    if "shrink" in arms:
        out += [(f"total  shrink={sh:g}", f"total_r256_s{sh:g}", {"k": 32})
                for sh in SHRINK]
    if "reduce" in arms:
        out += [(f"total  n_reduce={nr}", f"total_r{nr}_s0.001", {"k": 32})
                for nr in REDUCE]
    if "within" in arms:
        out += [(f"within  shrink={sh:g}", f"within_r256_s{sh:g}", {"k": 32})
                for sh in SHRINK]
        out += [(f"within  n_reduce={nr}", f"within_r{nr}_s0.001", {"k": 32})
                for nr in (128, 512)]
        out += [("within-slab  shrink=0.001", "within-slab_r256_s0.001", {"k": 32})]
    if "mirror" in arms:
        out += [(f"mirror-{kind}  shrink={sh:g}", f"mirror-{kind}_r256_s{sh:g}",
                 {"k": 32}) for kind in ("total", "within") for sh in (1e-3, 1e-2, 1e-1)]
    if "sign" in arms:
        out += [(f"mirror-{sel}:{k}  shrink={sh:g}", f"mirror{sel}-total_r256_s{sh:g}",
                 {"k": k}) for sel in ("sym", "anti") for sh in (1e-3, 1e-2)
                for k in (16, 32)]
    if "mirrork" in arms:
        out += [(f"mirror-total:{k}  shrink=0.01", "mirror-total_r256_s0.01", {"k": k})
                for k in (16, 24, 32, 48, 64)]
    out += _rows_extra(arms)
    if "readout" in arms:
        out += [("incumbent  per-participant z", "total_r256_s0.001",
                 {"k": 32, "norm": "participant"}),
                ("incumbent  lags2", "total_r256_s0.001", {"k": 32, "lags": 2}),
                ("incumbent  k=48", "total_r256_s0.001", {"k": 48})]
    if "combine" in arms:
        out += [("corpus-pca:64", "corpus-pca", {"k": 64, "lags": 0}),
                ("within-pca:64", "within-pca", {"k": 64, "lags": 0}),
                ("fast-pca:64", "fast-pca", {"k": 64, "lags": 0})]
    return out


def _rows_extra(arms):
    """The second half of `_rows`, split only to keep either side readable."""
    out = []
    if "whiten" in arms:
        for al in WHITEN:
            label = "CCA" if al == 1.0 else ("PLS" if al == 0.0 else f"a={al:g}")
            key = "total_r256_s{0:g}" + ("" if al == 1.0 else f"_a{al:g}")
            out += [(f"{label:<6} whiten={al:g}  shrink={sh:g}", key.format(sh),
                     {"k": 32}) for sh in (1e-3, 1e-2)]
    if "random" in arms:
        out += [(f"random orthogonal:{k}", "random", {"k": k}) for k in (32, 64)]
    if "fast" in arms:
        for kind in ("fast", "slow"):
            out += [(f"{kind}  shrink={sh:g}", f"{kind}_r256_s{sh:g}", {"k": 32})
                    for sh in (1e-3, 1e-2, 1e-1)]
            out += [(f"mirror-{kind}  shrink=0.01", f"mirror-{kind}_r256_s0.01",
                     {"k": 32})]
    return out


# Arms that concatenate two registered blocks rather than slicing one.
CONCAT = {
    "combine": ("lr-cca:32 + corpus-pca:32",
                [("total_r256_s0.001", 32), ("corpus-pca", 32)]),
    "mirrork": ("lr-cca:32 + mirror:32",
                [("total_r256_s0.001", 32), ("mirror-total_r256_s0.01", 32)]),
    "sign": ("mirror sym:16 + anti:16",
             [("mirrorsym-total_r256_s0.01", 16), ("mirroranti-total_r256_s0.01", 16)]),
}


def build_arms(index, arms):
    """Feature functions over the projected records. Named as they are reported."""
    out = {}
    for label, key, kw in _rows(arms):
        if key in index and label not in out:
            out[label] = slicer(index, key, **kw)
    for arm, (label, parts) in CONCAT.items():
        if arm in arms and all(k in index for k, _ in parts):
            out[label] = concat([slicer(index, k, k=kk) for k, kk in parts])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--voxels", required=True, help="build_labeled_voxels.py output")
    p.add_argument("--moments", default="results/lrcca_variants/moments_n2000.npz")
    p.add_argument("--lag-moments", default="results/lrcca_variants/lag1_n2000.npz")
    p.add_argument("--out", default="results/lrcca_variants/sweep.json")
    p.add_argument("--arms", default="shrink,reduce,within,mirror,fast,combine")
    p.add_argument("--k", type=int, default=DEFAULT_K)
    a = p.parse_args()

    arms = set(a.arms.split(","))
    corpus = Corpus(a.moments, a.lag_moments)
    print(f"[*] moments: {corpus.n} TRs, {len(corpus.subject_n)} subjects", flush=True)

    blocks = build_candidates(corpus, arms)
    del corpus._cache
    recs, index = project(a.voxels, blocks)
    del blocks

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    got = json.loads(out.read_text()) if out.exists() else {}
    t0 = time.time()
    print(f"\n{'arm':<34}{'sub-TR':>9}{'d':>9}{'1-TR':>9}{'d':>9}")
    base = None
    for name, fn in build_arms(index, arms).items():
        if name not in got:
            got[name] = summarise(probe.lodo(recs, fn))
            out.write_text(json.dumps(got, indent=1))
        row = got[name]
        if base is None:
            base = row
        print(f"{name:<34}{row['subtr']:>9.4f}{row['subtr'] - base['subtr']:>+9.4f}"
              f"{row['1tr']:>9.4f}{row['1tr'] - base['1tr']:>+9.4f}"
              f"   [{time.time() - t0:5.0f}s]", flush=True)
    print(f"\n[+] {out}")


if __name__ == "__main__":
    main()
