"""The frozen basis: canonical correlation between the left and right orbit.

Both eyes rotate together, so gaze is part of the signal the two orbits share. A
direction in left-orbit voxel space whose projection correlates with a direction
in right-orbit voxel space is therefore driven by conjugate gaze, while what is
local to one orbit (its own noise, physiology, susceptibility) is suppressed.
The basis is fitted once, on unlabelled fMRI, and never sees gaze.

It comes out of **one streaming pass** over the corpus, which accumulates a mean
and a second-moment matrix over the 14,236 eye-mask voxels::

    n += T
    s += sum_t x_t
    C += X^T X            (14236 x 14236, ~810 MB float32)

and is then solved in closed form (:func:`fit_lr_cca`): each orbit reduced to its
256 leading principal directions, whitened, and the whitened cross-covariance
decomposed by SVD. Its singular vectors are the canonical pairs.

The shipped corpus (:func:`corpus_subjects`) is every eligible participant of
the published eye-block corpus that is not a gaze-labelled participant under any
name: 2,060 participants of 652 OpenNeuro datasets.
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

# The OpenNeuro accessions three gaze-labelled datasets were ingested from. The
# same runs sit under both names, so the corpus must exclude the accession as
# well as the `dsL*` folder, or the basis is fitted on evaluation participants
# (it was, for four of them, until 6 October 2026).
EVALUATION_SOURCES = {
    "dsL07_deepmreye_calib": "ds006833",
    "dsL08_studyforrest_movie": "ds000113",
    "dsL11_backtothefuture": "ds006642",
}


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


def unlabeled_subjects(data_dir, labelled_prefix="dsL", min_voxels=None, min_trs=32,
                       exclude_sources=True):
    """Participants eligible for the corpus: fully covered, long enough, no gaze.

    A participant is eligible when its file carries no ``labels``, its dataset
    is not a gaze-labelled one (``labelled_prefix``) or, with
    ``exclude_sources``, one of the OpenNeuro accessions those were taken from
    (:data:`EVALUATION_SOURCES`, which hold the same runs), it has at least
    ``min_trs`` volumes, and at least ``min_voxels`` (default: every eye-mask
    voxel) carry signal. This is eligibility only; :func:`corpus_subjects`
    makes the draw.
    """
    min_voxels = FULL_MASK_VOXELS if min_voxels is None else min_voxels
    exclude = set(EVALUATION_SOURCES.values()) if exclude_sources else set()
    out = []
    for path in sorted(Path(data_dir).glob("*/*.h5")):
        dataset = path.parent.name
        if dataset in exclude or dataset.startswith(labelled_prefix):
            continue
        try:
            with h5py.File(path, "r") as f:
                if "labels" in f:
                    continue
                n_trs = f["eye_block"].shape[-1]
                if n_trs < min_trs:
                    continue
                first = f["eye_block"][..., : min(4, n_trs)]
        except Exception as e:
            logging.debug(f"skipping {path}: {e}")
            continue
        if int((np.abs(first).sum(-1) > 0).sum()) < min_voxels:
            continue
        out.append((dataset, path.stem, str(path), int(n_trs)))
    return out


def corpus_subjects(data_dir, n=None, seed=0, **eligibility):
    """The unlabeled corpus: every eligible participant
    (:func:`unlabeled_subjects`, which drops the evaluation sources) in a
    seed-``seed`` shuffle, the first ``n`` of them (all if ``None``).

    Shuffled rather than sorted because the sorted order is by accession: a
    prefix would confound "more participants" with "fewer acquisitions". The
    order is nested in ``n``: the first 400 of a seed are the first 400 of
    every larger corpus of that seed; another seed is another random draw.
    """
    subjects = unlabeled_subjects(data_dir, **eligibility)
    order = np.random.default_rng(seed).permutation(len(subjects))
    subjects = [subjects[i] for i in order]
    return subjects if n is None else subjects[:n]


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


def fit_lr_cca(moments, mask, k, n_reduce=256, shrinkage=1e-3, seed=0,
               split_x=None, cached=None):
    """Canonical correlation between the left and right orbit.

    Solved in a PCA-reduced, whitened space rather than on the raw halves:
    each half is ~7000 voxels with far fewer effective degrees of freedom, so
    the raw whitening ``C_ll^{-1/2}`` is rank-deficient and CCA would return
    directions that are pure noise at correlation 1.0. Reducing to ``n_reduce``
    principal directions per half and shrinking the eigenvalues bounds that.

    ``cached`` takes ``(cov, mu)`` to skip rebuilding the covariance.
    """
    cov, mu = cached if cached is not None else moments.covariance()
    split_x = LR_SPLIT_X if split_x is None else split_x

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


def project(arrays, x, k=None):
    """Masked voxels ``x [N, 14236]`` -> the bilateral feature ``[N, k]``.

    The two orbits give two estimates of the same conjugate gaze; their mean is
    the denoised one, which is the entire point of fitting across eyes rather
    than pooling their voxels.
    """
    zl, zr = orbit_projections(x, arrays, k)
    return 0.5 * (zl + zr)


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


def fit_corpus(data_dir, out, k=256, n=None, seed=0, trs_per_subject=48, n_slabs=4,
               n_reduce=256, shrinkage=1e-3, progress=100):
    """Fit the basis on a corpus directory and save it to ``out``.

    ``data_dir`` is laid out as ``<dataset>/<subject>.h5`` (:mod:`deepmreye.data`).
    The corpus is :func:`corpus_subjects` -- every fully covered, unlabelled
    participant with at least 32 volumes, never a gaze-labelled dataset or its
    source accession -- in a seed-``seed`` shuffle, the first ``n`` of them (all
    if ``None``). Each contributes ``trs_per_subject`` volumes, in ``n_slabs``
    blocks spread over its run, so that long runs do not dominate.

    ``k`` canonical pairs are stored (the decoder uses the leading 32). Costs one
    streaming pass: about three minutes for the full corpus on a 12-core laptop,
    and ~1 GB of memory for the accumulator.
    """
    import time

    mask = corpus_mask(data_dir)
    subjects = corpus_subjects(data_dir, n=n, seed=seed)
    if not subjects:
        raise RuntimeError(f"no eligible participants under {data_dir}")
    print(f"[*] {len(subjects)} participants from {len({s[0] for s in subjects})} datasets",
          flush=True)
    moments = accumulate(subjects, mask, trs_per_subject, n_slabs, progress=progress)
    arrays = fit_lr_cca(moments, mask, k, n_reduce=n_reduce, shrinkage=shrinkage, seed=seed)
    meta = {"n_subjects": moments.n_subjects, "n_trs": moments.n, "k": k,
            "trs_per_subject": trs_per_subject, "n_slabs": n_slabs, "seed": seed,
            "n_reduce": n_reduce, "shrinkage": shrinkage,
            "datasets": len({s[0] for s in subjects}),
            "subjects": [f"{d}/{s}" for d, s, *_ in subjects],
            "fitted": time.strftime("%Y-%m-%d %H:%M:%S")}
    path = save_basis(out, mask, {"lr-cca": arrays}, meta)
    cc = arrays["canonical_correlations"]
    print(f"[+] {meta['n_subjects']} participants, {moments.n} volumes; canonical r "
          f"1st {cc[0]:.3f}, 32nd {cc[min(31, len(cc) - 1)]:.3f} -> {path}", flush=True)
    return path
