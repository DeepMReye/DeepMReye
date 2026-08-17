"""Feature bases learned from the *unlabeled* corpus.

The baseline reads gaze off a stride-4 subsample of the eye mask: 480 of the
14236 masked voxels, chosen by nothing but their index. That subsample is not a
modelling choice, it is a budget -- a basis over the full mask cannot be
estimated from six labeled datasets without overfitting the fold it is fitted
on. The unlabeled half of the corpus (1773 participants, 912 OpenNeuro
datasets, no eye tracking) is exactly the data that constraint is missing, and
it is the only thing here that uses it.

Everything in this module is linear and label-free, and all of it comes out of
**one streaming pass** over the unlabeled participants. The pass accumulates a
mean and a second-moment matrix over the masked voxels::

    n  += T
    s  += sum_t x_t
    C  += X^T X            (14236 x 14236, ~1.6 GB float64)

plus the same two quantities over temporal differences ``x_{t+1} - x_t``. Every
basis below is a different decomposition of those accumulators, so adding one
costs an eigendecomposition, not another read of the corpus:

- ``corpus-pca``  top-k eigenvectors of the voxel covariance. The direct
                  replacement for stride subsampling: the k directions along
                  which eye-region signal actually varies, estimated across 900+
                  scanners rather than the five datasets a fold gets to see.
- ``diff-pca``    the same, on the covariance of temporal differences.
                  Differencing removes whatever is static within a run --
                  anatomy, coil bias, slow scanner drift -- and keeps what
                  *moves*. Gaze is the thing in an orbit that moves.
- ``lr-cca``      canonical correlation between the left and right orbit.
                  The eyes rotate together, so a direction in left-orbit voxel
                  space that predicts the right orbit is a direction driven by
                  conjugate gaze, while anything local to one orbit (that side's
                  noise, physiology, susceptibility) is suppressed. Fitted here
                  on unlabeled subjects, it is a *fixed* projection: unlike the
                  per-run CCA in ``scripts/analyze_identifiability.py``, it never
                  sees the run it is applied to and never needs labels to pick
                  which variate is x, which is y, or their signs -- the
                  supervised readout downstream settles that.

The honest control for all three is ``fold-pca`` (see
``deepmreye/evaluate/features.py``): the same construction and the same k,
fitted on the labeled training fold alone. The gap between ``fold-pca`` and
``corpus-pca`` is what the unlabeled corpus bought; the gap between ``raw`` and
``fold-pca`` is merely what using the full mask bought.
"""
import json
import logging
from pathlib import Path

import h5py
import numpy as np

# The eye mask is a fixed crop, so a fully-covered participant has exactly this
# many non-zero voxels. Subjects with fewer are missing coverage (the label-3
# "eyes cut off" case), not carrying zeros meaningfully -- and zeros in a
# mean-centred covariance read as large deviations, so partial subjects would
# steer the basis toward their own missingness pattern rather than toward gaze.
FULL_MASK_VOXELS = 14236

# The two orbits sit either side of a trough at x=24 (per-x mask counts fall to
# 52 there against ~390 at each lobe's centre). Splitting on the trough rather
# than the array midpoint keeps each half to one eyeball.
LR_SPLIT_X = 24

BASIS_KINDS = ("corpus-pca", "diff-pca", "lr-cca", "gev-fast", "gev-slow",
               "band-pca", "nuis-pca8", "nuis-pca32")


def _slabs(n_trs, total, n_slabs):
    """Contiguous TR slabs, evenly spaced through a run.

    Contiguous because the difference accumulator needs consecutive volumes;
    several slabs rather than one because eye position and scanner state drift
    over a run, and a single block from the middle would sample one moment of it.
    """
    per = max(2, total // n_slabs)
    if n_trs <= per:
        return [(0, n_trs)]
    starts = np.linspace(0, n_trs - per, n_slabs).astype(int)
    return [(int(s), int(s + per)) for s in sorted(set(starts.tolist()))]


def corpus_mask(data_dir, max_files=40):
    """The canonical eye mask, as a boolean ``[X, Y, Z]`` array.

    Taken as the union over fully-covered participants. It is a property of the
    template crop, not of any subject, so a handful of files settle it.
    """
    data_dir = Path(data_dir)
    acc = None
    seen = 0
    for path in sorted(data_dir.glob("*/*.h5")):
        if seen >= max_files:
            break
        try:
            with h5py.File(path, "r") as f:
                block = f["eye_block"][..., : min(4, f["eye_block"].shape[-1])]
        except Exception:
            continue
        nz = np.abs(block).sum(-1) > 0
        if nz.sum() < FULL_MASK_VOXELS:
            continue
        acc = nz if acc is None else (acc | nz)
        seen += 1
    if acc is None:
        raise RuntimeError(f"no fully-covered participant found under {data_dir}")
    return acc


def unlabeled_subjects(data_dir, probe_prefix="dsL", min_voxels=FULL_MASK_VOXELS,
                       min_trs=32, include_labeled=False, exclude_datasets=()):
    """Participants eligible to fit a basis: fully covered, and label-free *as
    used here*.

    By default the gaze-labeled datasets are excluded outright, which makes the
    basis strictly inductive. ``include_labeled`` folds their **voxels** in --
    never their labels, which this module cannot even read. That is a different
    and weaker claim, so the two must not be confused:

    - excluded (default): the basis has never seen the evaluation domain at all.
    - included, minus the held-out fold (``exclude_datasets``): unsupervised
      *domain adaptation*. The basis sees the training datasets' voxels, which
      the fold is entitled to, and still never sees the test dataset.
    - included, held-out fold and all (``exclude_datasets=()``): **transductive**.
      Defensible for this application -- at deploy time you do have the
      participant's scan, you just have no eye tracker -- but it is no longer a
      leave-one-dataset-out number, and reporting it as one would be wrong.

    ``exclude_datasets`` is what buys the middle case, one basis per fold.
    """
    exclude = set(exclude_datasets)
    out = []
    for path in sorted(Path(data_dir).glob("*/*.h5")):
        dataset = path.parent.name
        if dataset in exclude:
            continue
        if dataset.startswith(probe_prefix) and not include_labeled:
            continue
        try:
            with h5py.File(path, "r") as f:
                if "labels" in f and not include_labeled:
                    continue
                n_trs = f["eye_block"].shape[-1]
                if n_trs < min_trs:
                    continue
                probe = f["eye_block"][..., : min(4, n_trs)]
        except Exception as e:
            logging.debug(f"skipping {path}: {e}")
            continue
        if int((np.abs(probe).sum(-1) > 0).sum()) < min_voxels:
            continue
        out.append((dataset, path.stem, str(path), int(n_trs)))
    return out


class Moments:
    """Streaming mean and second moment over masked voxels, and over their
    temporal differences.

    Two details carry the whole cost of this class.

    **The accumulators must be Fortran-ordered.** ``scipy``'s ``syrk`` wrapper
    only accumulates into ``c`` in place when ``c`` is already in the layout
    BLAS wants; handed a C-ordered array it quietly copies, updates the copy and
    returns it, so ``overwrite_c=1`` becomes a no-op and the accumulator stays
    at zero. Nothing raises. The resulting covariance is ``-mu mu^T``, which is
    rank one and negative definite, and every basis fitted from it is noise --
    with a plausible-looking leading component, which is how it gets missed.

    **Rows are buffered before each update.** A rank-8 update on a 14236^2
    matrix reads and writes 1.6 GB to do 8e8 flops: it is pure memory traffic,
    and doing one per TR slab made this ~40x slower than the arithmetic
    warrants. Buffering to ``batch_rows`` amortises each pass over the matrix.
    """

    def __init__(self, n_voxels, dtype=np.float32, batch_rows=1024):
        self.d = n_voxels
        self.dtype = dtype
        # order="F": see the class docstring. This is load-bearing.
        self.c = np.zeros((n_voxels, n_voxels), dtype=dtype, order="F")
        self.s = np.zeros(n_voxels, dtype=np.float64)
        self.n = 0
        self.dc = np.zeros((n_voxels, n_voxels), dtype=dtype, order="F")
        self.ds = np.zeros(n_voxels, dtype=np.float64)
        self.dn = 0
        self.n_subjects = 0
        self.batch_rows = batch_rows
        self._buf = {"c": [], "dc": []}
        self._buf_n = {"c": 0, "dc": 0}

    def _syrk(self, x, c):
        from scipy.linalg.blas import dsyrk, ssyrk

        fn = dsyrk if self.dtype == np.float64 else ssyrk
        # trans=1 -> A^T A; beta=1 accumulates into c. Only the upper triangle
        # is written; `symmetrise` mirrors it once at the end.
        out = fn(alpha=1.0, a=np.asfortranarray(x, dtype=self.dtype), trans=1,
                 beta=1.0, c=c, overwrite_c=1)
        if out is not c:  # the copy-instead-of-accumulate case, guarded loudly
            raise RuntimeError(
                "syrk did not accumulate in place -- accumulator layout is wrong")

    def _push(self, key, x, target):
        self._buf[key].append(x)
        self._buf_n[key] += len(x)
        if self._buf_n[key] >= self.batch_rows:
            self._flush(key, target)

    def _flush(self, key, target):
        if not self._buf[key]:
            return
        self._syrk(np.concatenate(self._buf[key]), target)
        self._buf[key] = []
        self._buf_n[key] = 0

    def add(self, x):
        """Add one contiguous slab ``x [T, D]`` of masked voxels."""
        x = np.ascontiguousarray(x, dtype=self.dtype)
        if len(x) < 2:
            return
        self._push("c", x, self.c)
        self.s += x.sum(axis=0, dtype=np.float64)
        self.n += len(x)

        d = np.ascontiguousarray(np.diff(x, axis=0))
        self._push("dc", d, self.dc)
        self.ds += d.sum(axis=0, dtype=np.float64)
        self.dn += len(d)

    def symmetrise(self):
        self._flush("c", self.c)
        self._flush("dc", self.dc)
        for m in (self.c, self.dc):
            iu = np.triu_indices_from(m, k=1)
            m[(iu[1], iu[0])] = m[iu]

    def covariance(self, diff=False):
        """Mean-centred covariance, and the mean that centres it.

        Promoted to float64 here rather than accumulated in it: the update is
        memory-bound, so float32 halves the traffic, while the single
        decomposition afterwards wants the headroom.
        """
        c, s, n = (self.dc, self.ds, self.dn) if diff else (self.c, self.s, self.n)
        if n < 2:
            raise RuntimeError("no data accumulated")
        mu = s / n
        cov = c.astype(np.float64) / n
        cov -= np.outer(mu, mu)
        return cov, mu


def accumulate(subjects, mask, trs_per_subject=48, n_slabs=4, progress=None):
    """One pass over ``subjects``, returning the :class:`Moments`.

    A fixed TR budget per participant, not per run length: a 3600-TR subject
    would otherwise contribute as much as fourteen 260-TR ones and the basis
    would describe that one scanner.
    """
    flat = mask.reshape(-1)
    moments = Moments(int(flat.sum()))
    for i, (_ds, _sub, path, n_trs) in enumerate(subjects):
        try:
            with h5py.File(path, "r") as f:
                block = f["eye_block"]
                for start, stop in _slabs(n_trs, trs_per_subject, n_slabs):
                    slab = block[..., start:stop]
                    x = slab.reshape(-1, slab.shape[-1])[flat].T
                    moments.add(x)
        except Exception as e:
            logging.warning(f"skipping {path}: {e}")
            continue
        moments.n_subjects += 1
        if progress and i % progress == 0:
            print(f"  [{i + 1}/{len(subjects)}] {moments.n} TRs from "
                  f"{moments.n_subjects} subjects", flush=True)
    moments.symmetrise()
    return moments


def _top_eigenvectors(cov, k, seed=0):
    """Top-``k`` eigenvectors of a symmetric PSD matrix, largest first.

    Randomized SVD rather than a full ``eigh``: we want 256 directions out of
    14236, and the full tridiagonalisation costs a hundred times more than the
    range-finder for components that are then discarded.
    """
    from sklearn.utils.extmath import randomized_svd

    k = int(min(k, cov.shape[0]))
    u, s, _ = randomized_svd(cov, n_components=k, n_iter=4, random_state=seed)
    return u, np.maximum(s, 0.0)


def fit_pca(moments, k, diff=False, seed=0):
    cov, mu = moments.covariance(diff=diff)
    total = float(np.trace(cov))
    vecs, vals = _top_eigenvectors(cov, k, seed)
    return {"mean": mu, "components": vecs, "eigenvalues": vals,
            "total_variance": np.array([total])}


def fit_shrunk_pca(rows, corpus, k, lam, seed=0):
    """PCA of the fold covariance **shrunk toward the corpus covariance**.

    The third way of combining a corpus basis with a fold-local one, and the only
    one that combines them where the difference actually lives. Concatenating
    features (with or without per-block penalties) and stacking predictions both
    take the two bases as given and argue about the readout; this argues about the
    *covariance estimate* the basis comes from::

        C(lam) = (1 - lam) * C_fold + lam * C_corpus

    at ``lam = 0`` it is exactly ``fold-pca`` and at ``lam = 1`` exactly
    ``corpus-pca``, so the two incumbents are the endpoints of one curve and any
    interior win is unambiguous. The motivation is that the measured problem with
    ``fold-pca`` is *variance*, not bias: it is estimated from a few hundred
    labeled windows, which is why the labeled-budget sweep shows it scoring 0.847
    at 1000 windows and 0.828 with all of them. Shrinking a noisy estimate toward
    a well-estimated target is the textbook fix (Ledoit-Wolf), with the corpus as
    the target instead of the identity -- and unlike identity shrinkage, the
    target here gets *better* as the unlabeled corpus grows, which is the scaling
    axis this project needs.

    Never forms a 14236^2 matrix. ``C_fold`` is applied as ``X'(Xv)`` and
    ``C_corpus`` through its stored eigendecomposition, so the whole thing is a
    ``LinearOperator`` and ``eigsh`` needs a few hundred matvecs.

    The corpus target is completed to full rank by spreading its *unexplained*
    variance isotropically over the orthogonal complement. Without that, the
    stored rank-256 target assigns exactly zero variance outside its own span, so
    any ``lam > 0`` would silently truncate the fold basis into the corpus
    subspace -- a much stronger claim than shrinkage, and not the one being
    tested.

    Both covariances are trace-normalised first. They are estimated over
    different numbers of TRs and (for the corpus) different acquisitions, so
    without it ``lam`` would mix scales rather than shapes and its value would
    mean nothing across folds.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh

    rows = np.asarray(rows, dtype=np.float64)
    mu = rows.mean(axis=0)
    xc = rows - mu
    d = xc.shape[1]
    k = int(min(k, d - 1))

    u = np.asarray(corpus["components"], dtype=np.float64)
    vals = np.asarray(corpus["eigenvalues"], dtype=np.float64)
    total = float(np.asarray(corpus["total_variance"]).ravel()[0])
    # Isotropic completion of the tail. Guarded because a basis stored with k
    # close to d, or a total_variance that already equals the retained sum,
    # would otherwise give a negative variance.
    resid = max(total - float(vals.sum()), 0.0) / max(d - u.shape[1], 1)

    trace_f = float((xc ** 2).sum() / max(len(xc) - 1, 1))
    s_f = 1.0 / trace_f if trace_f > 0 else 0.0
    s_c = 1.0 / total if total > 0 else 0.0

    def matmat(v):
        v = np.asarray(v, dtype=np.float64)
        fold = xc.T @ (xc @ v) / max(len(xc) - 1, 1)
        proj = u.T @ v
        corp = u @ (vals[:, None] * proj) + resid * (v - u @ proj)
        return (1.0 - lam) * s_f * fold + lam * s_c * corp

    op = LinearOperator(shape=(d, d), dtype=np.float64,
                        matvec=lambda v: matmat(np.reshape(v, (d, 1))).ravel(),
                        matmat=matmat)
    rng = np.random.default_rng(seed)
    w, vecs = eigsh(op, k=k, which="LA", v0=rng.normal(size=d))
    order = np.argsort(w)[::-1]
    return {"mean": mu, "components": vecs[:, order],
            "eigenvalues": np.maximum(w[order], 0.0)}


def lag1_autocorrelation(cov, dcov, vecs):
    """Per-direction lag-1 autocorrelation, for free from the two accumulators.

    For centred stationary ``x``, ``E[(x_{t+1}-x_t)(x_{t+1}-x_t)'] = 2C_0 -
    2 sym(C_1)``, so ``sym(C_1) = C_0 - DC/2`` and along a direction ``w``::

        rho(w) = 1 - (w' DC w) / (2 w' C_0 w)

    No extra pass over the corpus, and no lag-1 accumulator: the difference
    moments already there contain it.

    **Why this is the quantity of interest on this corpus.** Two findings on this
    project point at the same axis from opposite ends. The next-TR entry measured
    that the *predictable* part of an eye block is the nuisance -- global signal,
    motion and drift, concentrated in the leading variance components (0-8
    predicted at R^2 0.59 against 0.09 for 128-256). And the temporal-envelope law
    measured that a gaze trace's own lag-1 autocorrelation sits at 0.13-0.85 and
    predicts how well it decodes. So the corpus can say, without any labels,
    *which* directions are too slow to be gaze -- and variance ordering, which is
    all PCA knows, cannot.
    """
    # `(v * (C @ v)).sum(0)` rather than `einsum("ij,jk,ik->i", ...)`. The einsum
    # expresses the same quadratic form but does not dispatch to BLAS for this
    # pattern; measured at 14236 x 512 it is **143x slower**, which is minutes
    # per basis instead of milliseconds and was the whole cost of this sweep.
    var = (vecs * (cov @ vecs)).sum(axis=0)
    dvar = (vecs * (dcov @ vecs)).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = 1.0 - dvar / (2.0 * var)
    return np.where(var > 0, rho, np.nan), var, dvar


def fit_gev(moments, k, mode="fast", n_reduce=512, shrinkage=1e-2, seed=0,
            cached=None):
    """Generalized eigenvectors of (difference covariance, total covariance).

    Maximises ``w' DC w / w' C_0 w``, i.e. picks directions by how *fast* they
    move relative to their own power, rather than by how much power they have.
    ``mode="slow"`` takes the other end -- Slow Feature Analysis' objective --
    and is here as the control that says whether this axis means anything: the
    slow end should be the drift/motion nuisance and should decode gaze
    *badly*. An arm whose opposite extreme scores the same is measuring nothing.

    Be warned about what the fast end actually contains, because the ratio is
    monotone in lag-1 autocorrelation (``ratio = 2(1 - rho)``) and **thermal noise
    is the whitest thing in the volume**. So the extreme fast directions are
    expected to be noise, not gaze, and ``band-pca`` below is the version that
    takes the prediction seriously. ``gev-fast`` is run anyway because "the naive
    version of this idea fails, here is the measurement" is worth more than
    skipping it.

    ``cached`` takes ``(cov, mu, dcov, vecs, vals)`` so the fast and slow arms --
    which differ only in which end of one spectrum they read -- share a whitener
    rather than each rebuilding a 14236^2 decomposition.
    """
    if cached is None:
        cov, mu = moments.covariance(diff=False)
        dcov, _ = moments.covariance(diff=True)
        # Whiten in the total covariance's leading subspace. Raw whitening of a
        # 14236^2 covariance estimated from ~50k rows is rank-deficient, and the
        # generalized problem would return noise directions at ratio 4 -- the
        # same trap `fit_lr_cca` documents for CCA.
        vecs, vals = _top_eigenvectors(cov, n_reduce, seed)
    else:
        cov, mu, dcov, vecs, vals = cached
    vals = vals + shrinkage * float(vals.max())
    w = vecs / np.sqrt(vals)

    m = w.T @ dcov @ w
    m = 0.5 * (m + m.T)
    evals, evecs = np.linalg.eigh(m)
    order = np.argsort(evals)[::-1] if mode == "fast" else np.argsort(evals)
    idx = order[: int(min(k, len(evals)))]

    comp = w @ evecs[:, idx]
    comp /= np.linalg.norm(comp, axis=0, keepdims=True)
    rho, var, _ = lag1_autocorrelation(cov, dcov, comp)
    return {"mean": mu, "components": comp, "eigenvalues": evals[idx],
            "lag1": rho, "direction_variance": var,
            "total_variance": np.array([float(np.trace(cov))])}


def fit_band_pca(moments, k, rho_lo=-1.0, rho_hi=0.95, n_pool=512, seed=0,
                 cached=None):
    """PCA, then keep only directions whose lag-1 autocorrelation is in a band.

    The point of difference from ``corpus-pca``: the unlabeled corpus is used to
    decide *which directions are nuisance*, not merely which are large. A
    variance-ordered basis spends its first components on global signal, motion
    and drift because those are the highest-variance things in an eye block --
    and the readout downstream cannot recover from that, because every feature
    is standardised to unit variance and shares one ridge alpha, so a nuisance
    component costs exactly as much model capacity as a gaze one.

    Dropping the too-slow end is therefore not cosmetic: it hands ridge ``k``
    directions that could plausibly carry gaze instead of ``k`` that include
    several that provably cannot. ``rho_hi`` is the nuisance cut and ``rho_lo``
    the noise cut (off by default until the measured spectrum says where to put
    it).

    This should also *scale* with unlabeled participants in a way variance
    ordering need not: the top few eigenvectors of a covariance are estimable
    from a few hundred participants, but a per-direction temporal statistic on
    512 directions is a much larger thing to estimate, so more subjects buy a
    cleaner selection.
    """
    if cached is None:
        cov, mu = moments.covariance(diff=False)
        dcov, _ = moments.covariance(diff=True)
        vecs, vals = _top_eigenvectors(cov, n_pool, seed)
    else:
        cov, mu, dcov, vecs, vals = cached
        vecs, vals = vecs[:, :n_pool], vals[:n_pool]
    rho, var, _ = lag1_autocorrelation(cov, dcov, vecs)

    keep = np.isfinite(rho) & (rho >= rho_lo) & (rho <= rho_hi)
    idx = np.nonzero(keep)[0][: int(k)]        # variance order is preserved
    if len(idx) == 0:
        raise RuntimeError(
            f"no direction has lag-1 autocorrelation in [{rho_lo}, {rho_hi}]; "
            f"observed range {np.nanmin(rho):.3f}..{np.nanmax(rho):.3f}")
    return {"mean": mu, "components": vecs[:, idx], "eigenvalues": vals[idx],
            "lag1": rho[idx], "lag1_all": rho, "kept_index": idx,
            "n_dropped_slow": int(np.sum(rho > rho_hi)),
            "n_dropped_fast": int(np.sum(rho < rho_lo)),
            "total_variance": np.array([float(np.trace(cov))])}


def fit_nuisance_projected_pca(moments, k, n_nuisance=16, n_pool=512, seed=0,
                               cached=None):
    """PCA after projecting out the *slowest* directions in the corpus.

    This is the suggestion `CLAUDE.md` has carried since the next-TR result --
    "prediction after projecting out the global/motion components" -- done to the
    basis instead of to a predictor, and it is a better-posed knob than
    ``band-pca``'s correlation threshold. The measured lag-1 spectrum is why:
    leading principal directions sit at rho 0.57-0.88 and the bulk at ~0.06, but
    a *gaze* trace on this corpus reaches rho 0.851 (dsL02), so no threshold
    separates nuisance from gaze cleanly. A count does: remove the ``J`` slowest
    directions, where ``J`` is a small integer directly comparable to the
    next-TR finding that components 0-8 carry the predictable nuisance.

    The slow subspace is taken as the ``J`` directions of highest lag-1
    autocorrelation among the leading ``n_pool`` principal directions -- i.e.
    what is both high-variance *and* slow, which is what global signal, motion
    and drift are. Anything slow but tiny is left alone; it costs the readout
    nothing.

    The deflation is implicit at apply time. The returned components are
    eigenvectors of ``P C P`` with ``P = I - UU'``, so they are orthogonal to the
    slow subspace by construction and ``x @ components`` already equals
    ``(Px) @ components``. No change to ``project`` is needed, which is what keeps
    this arm interchangeable with every other basis at the same k.
    """
    if cached is None:
        cov, mu = moments.covariance(diff=False)
        dcov, _ = moments.covariance(diff=True)
        vecs, vals = _top_eigenvectors(cov, n_pool, seed)
    else:
        cov, mu, dcov, vecs, vals = cached
        vecs, vals = vecs[:, :n_pool], vals[:n_pool]

    rho, var, _ = lag1_autocorrelation(cov, dcov, vecs)
    finite = np.where(np.isfinite(rho), rho, -np.inf)
    slow = np.argsort(finite)[::-1][: int(n_nuisance)]
    u = vecs[:, slow]
    # Re-orthonormalise: eigenvectors of one covariance are orthogonal, but this
    # keeps the projector exact if a caller ever passes a non-orthogonal pool.
    u, _ = np.linalg.qr(u)

    # Deflate: C_res = (I - UU') C (I - UU'), formed without materialising the
    # 14236^2 projector.
    cu = cov @ u
    cres = cov - cu @ u.T - u @ cu.T + u @ (u.T @ cu) @ u.T
    cres = 0.5 * (cres + cres.T)

    comp, cvals = _top_eigenvectors(cres, k, seed)
    rho_kept, _, _ = lag1_autocorrelation(cov, dcov, comp)
    return {"mean": mu, "components": comp, "eigenvalues": cvals,
            "lag1": rho_kept, "n_nuisance": np.array([int(n_nuisance)]),
            "removed_lag1": rho[slow],
            "total_variance": np.array([float(np.trace(cov))])}


def fit_lr_cca(moments, mask, k, n_reduce=256, shrinkage=1e-3, seed=0,
               split_x=LR_SPLIT_X, cached=None):
    """Canonical correlation between the left and right orbit.

    Solved in a PCA-reduced, whitened space rather than on the raw halves:
    each half is ~7000 voxels with far fewer effective degrees of freedom, so
    the raw whitening ``C_ll^{-1/2}`` is rank-deficient and CCA would return
    directions that are pure noise at correlation 1.0. Reducing to ``n_reduce``
    principal directions per half and shrinking the eigenvalues bounds that.

    ``cached`` takes ``(cov, mu)`` to skip rebuilding the covariance.
    """
    cov, mu = cached if cached is not None else moments.covariance(diff=False)

    # Which rows of the masked-voxel vector belong to which orbit.
    xs = np.nonzero(mask.reshape(-1))[0] // (mask.shape[1] * mask.shape[2])
    left = xs < split_x
    li, ri = np.nonzero(left)[0], np.nonzero(~left)[0]
    if len(li) == 0 or len(ri) == 0:
        raise ValueError(
            f"split_x={split_x} puts every masked voxel on one side "
            f"(left {len(li)}, right {len(ri)}) -- there is no second orbit to "
            f"correlate against")

    def whitener(idx):
        sub = cov[np.ix_(idx, idx)]
        vecs, vals = _top_eigenvectors(sub, min(n_reduce, len(idx) - 1), seed)
        vals = vals + shrinkage * float(vals.max())
        return vecs / np.sqrt(vals)          # [n_half, m]

    wl, wr = whitener(li), whitener(ri)
    # Cross-covariance expressed in the two whitened bases; its singular values
    # are the canonical correlations.
    m = wl.T @ cov[np.ix_(li, ri)] @ wr
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    k = int(min(k, u.shape[1], vt.shape[0]))

    return {
        "mean": mu,
        "left_index": li,
        "right_index": ri,
        "left_weights": wl @ u[:, :k],       # [n_left,  k]
        "right_weights": wr @ vt[:k].T,      # [n_right, k]
        "canonical_correlations": s[:k],
    }


def save_basis(path, mask, bases, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {"mask": mask, "_meta": np.frombuffer(
        json.dumps(meta).encode("utf8"), dtype=np.uint8)}
    for kind, arrays in bases.items():
        for name, value in arrays.items():
            out[f"{kind}/{name}"] = np.asarray(value, dtype=np.float32) \
                if np.asarray(value).dtype.kind == "f" else np.asarray(value)
    np.savez_compressed(path, **out)
    return path


def load_basis(path):
    z = np.load(path, allow_pickle=False)
    mask = z["mask"].astype(bool)
    meta = json.loads(bytes(z["_meta"]).decode("utf8"))
    bases = {}
    for key in z.files:
        if "/" not in key:
            continue
        kind, name = key.split("/", 1)
        bases.setdefault(kind, {})[name] = z[key]
    return mask, bases, meta


def project(kind, arrays, x, k=None):
    """Apply a fitted basis to masked voxels ``x [N, D]`` -> ``[N, k]``."""
    x = np.asarray(x, dtype=np.float64)
    # Every PCA-shaped basis: a mean plus an orthonormal-ish `components` matrix.
    # Keyed off the stored arrays rather than a hand-maintained name list, which
    # is what let `nuis-pca8`/`nuis-pca32` be fitted, registered in both
    # registries, and then raise here at apply time -- after the expensive part.
    if kind != "lr-cca" and "components" in arrays:
        comp = arrays["components"]
        if k:
            comp = comp[:, :k]
        # diff-pca's mean is the mean *difference* (~0) and centring by it would
        # be meaningless for a position feature, so both centre on the voxel
        # mean of the corpus.
        return (x - arrays["mean"]) @ comp
    if kind == "lr-cca":
        mu = arrays["mean"]
        li, ri = arrays["left_index"], arrays["right_index"]
        wl, wr = arrays["left_weights"], arrays["right_weights"]
        if k:
            wl, wr = wl[:, :k], wr[:, :k]
        xc = x - mu
        # The two orbits give two estimates of the same conjugate gaze; their
        # mean is the denoised one, which is the entire point of fitting across
        # eyes rather than pooling their voxels.
        return 0.5 * (xc[:, li] @ wl + xc[:, ri] @ wr)
    raise ValueError(f"unknown basis kind {kind!r}")
