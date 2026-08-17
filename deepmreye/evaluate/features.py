"""Feature sources for the gaze probe: what the readout is fitted on.

The readout zoo (``baselines.py``) is held fixed and the feature source is
varied, so that a difference between arms is a difference in the
representation and not in how it was fitted. Five sources, in the order they
have to be argued for:

- ``raw``        stride-4 voxels, 480 of 14236. The published baseline. The
                 stride is a budget, not a model: no basis over the full mask
                 can be fitted from six labeled datasets without overfitting.
- ``fold-pca``   PCA over the *full* mask, fitted on the labeled training fold
                 only. The control that matters. It uses every voxel and
                 compresses to the same k as the corpus bases, so the gap
                 between it and ``raw`` measures what full-resolution voxels are
                 worth, and the gap between it and ``corpus-pca`` measures what
                 the *unlabeled corpus* is worth. Without it, any win by
                 ``corpus-pca`` is unattributable.
- ``corpus-pca`` / ``diff-pca`` / ``lr-cca``
                 the unsupervised bases, fitted once on 1773 unlabeled
                 participants and frozen. See ``deepmreye/unsupervised.py``.

All five are linear projections of the same pooled voxels, so they are
interchangeable at the same output dimensionality -- which is what makes the
comparison a comparison.
"""
import numpy as np

from deepmreye.unsupervised import project

CORPUS_KINDS = ("corpus-pca", "diff-pca", "lr-cca", "gev-fast", "gev-slow",
                "band-pca", "nuis-pca8", "nuis-pca32")
# The cross-orbit JEPA (`deepmreye/orbitjepa.py`). Unlike every other trained
# arm here, its untrained control is **not** a random projection: the encoders
# are a linear identity path plus a zero-initialised MLP over the frozen
# canonical pre-projection, so `jepa-random` reproduces `lr-cca:k` exactly (see
# `test_untrained_jepa_reproduces_lr_cca_exactly`). That makes the control the
# 0.825 arm itself and the trained-minus-untrained margin a margin over the best
# linear corpus basis, measured on identical folds and windows.
JEPA_KINDS = ("jepa", "jepa-random")
# The corpus basis and the fold-local one combined at the *covariance* level
# rather than by concatenating their outputs: PCA of
# `(1 - lam) C_fold + lam C_corpus` (`unsupervised.fit_shrunk_pca`). `lam=0` is
# `fold-pca` and `lam=1` is `corpus-pca` exactly, so the two incumbents are the
# endpoints of this arm and it cannot win by accident.
HYBRID_KINDS = ("fold-shrunk-pca",)
FEATURE_KINDS = (("raw", "fold-pca", "fold-srm", "fold-pls")
                 + CORPUS_KINDS + HYBRID_KINDS + JEPA_KINDS)


class JepaExtractor:
    """Orbit-JEPA latents as a feature source. Pure numpy -- no torch import.

    Applied to the **pooled** bins rather than to the window at full TR
    resolution, unlike `OrbitExtractor`. That is deliberate: at initialisation
    the encoders are linear, so pooling-then-encoding and encoding-then-pooling
    coincide, and running on the pooled bins keeps `jepa-random` exactly equal to
    `lr-cca:k` under the identical pooling the linear arm uses. Encoding at TR
    resolution instead would introduce a second difference (where the
    non-linearity sits relative to the temporal average) on top of the one being
    measured, and the attribution would no longer be clean.
    """

    needs_fit = False

    def __init__(self, kind, mask, basis, weights, m, head, regress_motion,
                 n_components=None):
        self.kind = kind
        self.mask_flat = mask.reshape(-1)
        self.basis = basis
        self.weights = weights
        self.m = m
        self.head = head
        self.regress_motion = regress_motion
        self.n_components = n_components

    @property
    def parts(self):
        return (self,)

    def __call__(self, pooled, raw=None, n_t=None, subject_ids=None):
        from deepmreye.orbitjepa import jepa_features

        selected = pooled[..., self.mask_flat]
        out = jepa_features(self.weights, selected, self.basis,
                            m=self.m, head=self.head,
                            regress_motion=self.regress_motion)
        return out[..., : self.n_components] if self.n_components else out


def pool_time(x, n_t):
    """Mean-pool a window ``[B, X, Y, Z, W]`` into ``n_t`` temporal bins.

    Returns ``[B, n_t, V]`` as **numpy**, over the flattened voxel grid. Pooling
    before any projection is free: every basis here is linear, so pooling then
    projecting and projecting then pooling give the same answer, and this way
    the projection runs on 20 rows per window rather than 100.

    Numpy rather than torch, deliberately. LightGBM and PyTorch each bring their
    own OpenMP runtime, and a threaded torch reduction that runs *after*
    LightGBM has fitted in the same process deadlocks outright -- no error, no
    traceback, the process simply stops. ``eval_probe`` hits exactly that
    ordering with ``--readouts lgbm`` on a multi-fold protocol, where fold 2's
    feature extraction follows fold 1's LightGBM fit. Nothing here needs a
    tensor, so the whole class of failure is avoided by not creating one.
    """
    x = np.asarray(x)
    b, _, _, _, w = x.shape
    per_bin = int(np.ceil(w / n_t))
    pad = per_bin * n_t - w
    if pad:
        x = np.pad(x, [(0, 0)] * 4 + [(0, pad)])
    return x.reshape(b, -1, n_t, per_bin).mean(axis=3).transpose(0, 2, 1)


class FeatureExtractor:
    """Turns a pooled window into the feature rows a readout is fitted on."""

    def __init__(self, kind, mask=None, basis=None, n_components=None, stride=4,
                 grid_shape=(47, 29, 18), shrink_lambda=0.5):
        self.kind = kind
        self.stride = stride
        self.n_components = n_components
        self.basis = basis
        self.shrink_lambda = shrink_lambda
        self.mask_flat = None
        self.fold_basis = None
        self.srm = None

        if kind == "raw":
            # Reproduce the baseline's stride selection as an index into the
            # flattened grid, so every arm shares one code path.
            sel = np.zeros(grid_shape, dtype=bool)
            sel[::stride, ::stride, ::stride] = True
            self.mask_flat = sel.reshape(-1)
        else:
            if mask is None:
                raise ValueError(f"{kind} needs the corpus eye mask")
            self.mask_flat = mask.reshape(-1)

    @property
    def needs_fit(self):
        return self.kind in ("fold-pca", "fold-srm", "fold-pls", "fold-shrunk-pca")

    def select(self, pooled):
        """``[B, n_t, V]`` -> ``[B, n_t, D]`` on the selected voxels."""
        return pooled[..., self.mask_flat]

    def fit(self, rows, targets=None, subject_ids=None, seed=0):
        """Fit the fold-local basis on masked voxel rows ``[N, D]``."""
        from sklearn.utils.extmath import randomized_svd
        from sklearn.cross_decomposition import PLSRegression
        from deepmreye.evaluate.srm import SharedResponseModel

        rows = np.asarray(rows, dtype=np.float64)
        if self.kind == "fold-srm":
            k = int(min(self.n_components or 64, min(rows.shape) - 1))
            self.srm = SharedResponseModel(n_components=k)
            if subject_ids is None:
                subject_ids = np.zeros(rows.shape[0], dtype=int)
            else:
                subject_ids = np.asarray(subject_ids)

            subject_data = {}
            for sub_id in np.unique(subject_ids):
                idx = np.where(subject_ids == sub_id)[0]
                subject_data[sub_id] = rows[idx]
            self.srm.fit(subject_data, seed=seed)
        elif self.kind == "fold-shrunk-pca":
            from deepmreye.unsupervised import fit_shrunk_pca

            if self.basis is None:
                raise ValueError("fold-shrunk-pca needs the corpus-pca basis")
            k = int(min(self.n_components or 64, min(rows.shape) - 1))
            self.fold_basis = fit_shrunk_pca(rows, self.basis, k,
                                             self.shrink_lambda, seed=seed)
        elif self.kind == "fold-pls":
            k = int(min(self.n_components or 64, min(rows.shape) - 1))
            if targets is None:
                raise ValueError("fold-pls requires targets for basis fitting")
            pls = PLSRegression(n_components=k, scale=True)
            pls.fit(rows, targets)
            self.fold_basis = pls
        else:
            mu = rows.mean(axis=0)
            k = int(min(self.n_components or 256, min(rows.shape) - 1))
            _, _, vt = randomized_svd(rows - mu, n_components=k, n_iter=4,
                                      random_state=seed)
            self.fold_basis = {"mean": mu, "components": vt.T}

    def transform(self, selected, subject_ids=None):
        """``[B, n_t, D]`` (numpy) -> ``[B, n_t, k]``."""
        if self.kind == "raw":
            return selected
        b_win, n_t, d_vox = selected.shape
        flat = selected.reshape(-1, d_vox)
        if self.kind in ("fold-pca", "fold-shrunk-pca"):
            if self.fold_basis is None:
                raise RuntimeError(f"{self.kind} used before fit()")
            out = (flat - self.fold_basis["mean"]) @ self.fold_basis["components"]
        elif self.kind == "fold-pls":
            if self.fold_basis is None:
                raise RuntimeError("fold-pls used before fit()")
            out = self.fold_basis.transform(flat)
        elif self.kind == "fold-srm":
            if self.srm is None:
                raise RuntimeError("fold-srm used before fit()")
            if subject_ids is None:
                flat_subs = np.zeros(flat.shape[0], dtype=int)
            else:
                flat_subs = np.repeat(subject_ids, n_t)
            out = self.srm.transform(flat, flat_subs)
        else:
            out = project(self.kind, self.basis, flat, k=self.n_components)
        return out.reshape(b_win, n_t, -1)

    @property
    def parts(self):
        return (self,)

    def __call__(self, pooled, raw=None, n_t=None, subject_ids=None):
        return self.transform(self.select(pooled), subject_ids=subject_ids)


class CompositeExtractor:
    """Several feature sources concatenated, e.g. ``fold-pca+lr-cca``.

    The interesting question a composite answers is not "is the corpus basis
    better than a fold-local one" -- that is the ``--features`` comparison -- but
    "does it carry anything the fold-local one does not". A corpus basis that
    loses on its own can still add complementary directions, and concatenation
    is how that shows up. Ridge sorts out the redundancy.
    """

    def __init__(self, name, parts):
        self.kind = name
        self._parts = tuple(parts)
        # Set on the first call, and read by `banded-ridge` / `stack-ridge`,
        # which need to know where one block ends and the next begins. Recorded
        # from the actual output rather than from the requested `:k`, because a
        # budget is clamped against the data (`_n_components`) and a block width
        # that disagreed with the features would silently mis-assign penalties.
        self.block_widths = None

    @property
    def parts(self):
        return self._parts

    @property
    def needs_fit(self):
        return any(p.needs_fit for p in self._parts)

    def __call__(self, pooled, raw=None, n_t=None, subject_ids=None):
        outs = [p(pooled, raw, n_t, subject_ids=subject_ids) for p in self._parts]
        widths = [o.shape[-1] for o in outs]
        if self.block_widths is not None and list(self.block_widths) != widths:
            raise RuntimeError(
                f"{self.kind}: block widths changed mid-fold, "
                f"{self.block_widths} -> {widths}")
        self.block_widths = widths
        return np.concatenate(outs, axis=-1)


def parse_spec(spec):
    """``"fold-pca+lr-cca:32"`` -> ``(("fold-pca", None), ("lr-cca", 32))``.

    The optional ``:k`` is a per-part component budget, and it exists because a
    concatenation without one is not a fair test of whether a corpus basis adds
    anything. The readouts standardise every feature to unit variance, so
    gluing 256 corpus components onto 256 fold-local ones hands ridge 512
    equally-scaled dimensions under a single alpha -- it cannot downweight the
    added block, and the extra noise directions cost more than the signal they
    carry. ``fold-pca+lr-cca:32`` asks the question the fair way.
    """
    parts = []
    for token in (s for s in spec.split("+") if s):
        kind, _, budget = token.partition(":")
        if kind not in FEATURE_KINDS:
            raise ValueError(
                f"unknown feature source {kind!r} in {spec!r}; "
                f"known: {', '.join(FEATURE_KINDS)} (join with '+' to "
                f"concatenate, append ':k' for a per-part budget)")
        if budget and not budget.isdigit():
            raise ValueError(f"component budget must be an integer, got {budget!r}")
        budget_val = int(budget) if budget else None
        parts.append((kind, budget_val))

    if not parts:
        raise ValueError(f"empty feature spec {spec!r}")
    return tuple(parts)
