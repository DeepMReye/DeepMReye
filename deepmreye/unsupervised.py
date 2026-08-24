"""The two linear feature bases learned from the *unlabeled* corpus.

The published baseline reads gaze off a stride-4 subsample of the eye mask: 480
of the 14236 masked voxels, chosen by nothing but their index. That subsample is
not a modelling choice, it is a budget -- a basis over the full mask cannot be
estimated from a handful of labeled datasets without overfitting the fold it is
fitted on. The unlabeled half of the corpus (1878 participants, 915 OpenNeuro
accessions, no eye tracking) is exactly the data that constraint is missing.

Both bases come out of **one streaming pass** over the unlabeled participants,
which accumulates a mean and a second-moment matrix over the masked voxels::

    n += T
    s += sum_t x_t
    C += X^T X            (14236 x 14236, ~810 MB float32)

- ``corpus-pca``  top-k eigenvectors of the voxel covariance. The direct
                  replacement for stride subsampling: the k directions along
                  which eye-region signal actually varies, estimated across 900+
                  scanners rather than the few datasets a fold gets to see.
- ``lr-cca``      canonical correlation between the left and right orbit, and
                  the arm this package exists to ship. The eyes rotate together,
                  so a direction in left-orbit voxel space that predicts the
                  right orbit is a direction driven by conjugate gaze, while
                  anything local to one orbit (that side's noise, physiology,
                  susceptibility) is suppressed. Fitted on unlabeled subjects it
                  is a *fixed* projection: it never sees the run it is applied
                  to and never needs labels to pick which variate is x, which is
                  y, or their signs -- the supervised readout settles that.

Six further bases were fitted and measured here and all of them lost; see
FINDINGS.md. They needed a second accumulator over temporal differences, which
is why this pass is now half the memory and half the cost it used to be.
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

BASIS_KINDS = ("corpus-pca", "lr-cca")


def _slabs(n_trs, total, n_slabs):
    """Contiguous TR slabs, evenly spaced through a run.

    Several slabs rather than one because eye position and scanner state drift
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
    """Streaming mean and second moment over masked voxels.

    Two details carry the whole cost of this class.

    **The accumulator must be Fortran-ordered.** ``scipy``'s ``syrk`` wrapper
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
        self.n_subjects = 0
        self.batch_rows = batch_rows
        self._buf = {"c": []}
        self._buf_n = {"c": 0}

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

    def symmetrise(self):
        self._flush("c", self.c)
        iu = np.triu_indices_from(self.c, k=1)
        self.c[(iu[1], iu[0])] = self.c[iu]

    def covariance(self):
        """Mean-centred covariance, and the mean that centres it.

        Promoted to float64 here rather than accumulated in it: the update is
        memory-bound, so float32 halves the traffic, while the single
        decomposition afterwards wants the headroom.
        """
        c, s, n = self.c, self.s, self.n
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


def fit_pca(moments, k, seed=0):
    cov, mu = moments.covariance()
    total = float(np.trace(cov))
    vecs, vals = _top_eigenvectors(cov, k, seed)
    return {"mean": mu, "components": vecs, "eigenvalues": vals,
            "total_variance": np.array([total])}


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
    cov, mu = cached if cached is not None else moments.covariance()

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
    # Keyed off the stored arrays rather than a hand-maintained name list, so a
    # basis that was fitted and saved can always be applied.
    if kind != "lr-cca" and "components" in arrays:
        comp = arrays["components"]
        if k:
            comp = comp[:, :k]
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


def orbit_projections(rows, arrays, k=None):
    """Masked voxel rows ``[T, 14236]`` -> the two orbits' canonical coords.

    :func:`project` averages the two orbits, which is the feature the shipped
    readout uses. This keeps them apart, because whether to average or
    concatenate them is a question a cache that has already averaged cannot
    answer -- and the measured answer (average) is not self-evident.
    """
    rows = np.asarray(rows, dtype=np.float64)
    li, ri = np.asarray(arrays["left_index"]), np.asarray(arrays["right_index"])
    wl, wr = arrays["left_weights"], arrays["right_weights"]
    k = int(min(k or wl.shape[1], wl.shape[1], wr.shape[1]))
    centred = rows - arrays["mean"]
    return centred[:, li] @ wl[:, :k], centred[:, ri] @ wr[:, :k]
