"""Combining feature blocks properly: one penalty per block, or one weight per
block's predictions.

The concatenation results on this project were all measured with `ridge-cv`, and
`parse_spec`'s own docstring says why that is the wrong instrument: every readout
wraps its features in a `StandardScaler`, so gluing a 32-component corpus basis
onto a 64-component fold-local one hands ridge 96 equally-scaled dimensions under
**a single alpha**. It cannot penalise the two blocks differently, so it cannot
express "trust the fold-local block, shrink the corpus block hard" -- which is
the only combination anyone expected to win. The per-part `:k` budget was a
workaround for exactly this, and it is a blunt one: truncation is a 0/1 prior on
whole components, when what is wanted is a continuous one.

Two estimators, in increasing order of how much they let the blocks disagree:

- ``BandedRidge``   one regularisation strength per block, selected by grouped
                    cross-validation. Standard practice in voxelwise encoding
                    (Nunez-Elizalde et al. 2019 NeuroImage; Dupre la Tour et al.
                    2022 NeuroImage, "Feature-space selection with banded ridge
                    regression"), where joint models over feature spaces of very
                    different dimensionality and predictive power are the norm.
                    Scaling block *j* by ``w_j`` and applying one global alpha is
                    algebraically identical to penalising it by ``alpha / w_j^2``,
                    so the search is over block scalings.
- ``StackedRidge``  a separate ridge per block, then a non-negative weighted
                    average of their *predictions*, with the weights fitted on
                    out-of-fold predictions (stacked generalisation; the fMRI
                    form is Lin et al. 2024 NeuroImage). Weaker than banded ridge
                    in principle -- it cannot form a direction that mixes blocks
                    -- but it is the right shape for *this* corpus, where the two
                    arms win different folds, and its weights are directly
                    readable as "how much did the corpus basis contribute".

**Both select on grouped CV, by participant.** Windows overlap and neighbouring
TRs are correlated, so an ungrouped (or leave-one-out) inner split scores a
model on rows that are near-duplicates of its training rows, which
systematically prefers too little regularisation. Since choosing regularisation
is the entire content of these two estimators, that is not a detail -- an
ungrouped selector would report a combination win that is really a leak. When
no groups are passed the split falls back to `KFold`, and that is a degraded
mode, not an equivalent one.

Deliberately numpy + scipy only. Nothing in the feature/readout path may import
torch (see the OpenMP deadlock note in CLAUDE.md).
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

# Same grid as the rest of the zoo, so a banded fit that lands on a single
# shared scaling is exactly `ridge-cv` and the comparison is nested.
ALPHA_GRID = np.logspace(-2, 6, 17)

# Per-block scalings tried, as multiples of the first block's. Log-spaced over
# four orders of magnitude because the useful answer may well be "shrink the
# second block to nothing" -- that is the outcome the redundancy story predicts,
# and the grid has to be able to express it.
RATIO_GRID = np.logspace(-2, 2, 17)


def dyadic_blocks(widths, first=8):
    """Subdivide each feature block into log-spaced sub-blocks.

    ``[64, 32]`` -> ``[8, 8, 16, 32, 8, 8, 16]``. The point is that every basis
    here is **variance-ordered**, so a per-component penalty that decreases down
    the spectrum is exactly the prior the data suggests -- and ``:k`` truncation
    is the crudest possible version of it, a step from "trust completely" to
    "discard". Banding the spectrum dyadically lets cross-validation *learn* the
    taper instead, at the cost of one penalty per band rather than per component
    (Nunez-Elizalde et al. 2019, non-spherical priors for encoding models).

    Dyadic rather than uniform because eigenvalue spectra fall roughly
    geometrically: the first 8 components deserve their own penalty, components
    129-256 do not need 16 separate ones.
    """
    out = []
    for w in widths:
        w = int(w)
        edges, e = [], int(first)
        while e < w:
            edges.append(e)
            e *= 2
        edges.append(w)
        bands = list(np.diff([0] + edges))
        # Absorb a runt tail rather than giving it its own band: a 4-column band
        # would have its penalty estimated from almost nothing.
        if len(bands) > 1 and bands[-1] < first:
            tail = bands.pop()
            bands[-1] += tail
        out.extend(int(b) for b in bands)
    return out


def _splits(n, groups, n_splits, seed):
    """Grouped folds by participant, falling back to plain KFold."""
    if groups is not None:
        groups = np.asarray(groups)
        n_groups = len(np.unique(groups))
        if n_groups >= 2:
            k = int(min(n_splits, n_groups))
            return list(GroupKFold(n_splits=k).split(np.zeros(n), groups=groups))
    return list(KFold(n_splits=int(min(n_splits, n)), shuffle=True,
                      random_state=seed).split(np.zeros(n)))


def _expand(weights, blocks):
    """Per-block scalars -> one scalar per feature column."""
    return np.repeat(np.asarray(weights, dtype=np.float64), blocks)


def _ridge_path(gram, xty, alphas, scale):
    """Ridge coefficients for every alpha, for one candidate column scaling.

    Takes the *unscaled* ``X'X`` and ``X'y`` because scaling column *j* by
    ``w_j`` turns the Gram into ``D G D`` for diagonal ``D`` -- so the O(N D^2)
    product is computed once per fold by the caller and every candidate scaling
    costs only a D x D eigendecomposition, which one also serves the whole alpha
    grid. That is what makes the nested scaling x alpha search affordable on a
    pooled leave-one-dataset-out fold.
    """
    m = gram * scale[:, None] * scale[None, :]
    b = xty * scale[:, None]
    vals, vecs = np.linalg.eigh(m)
    vals = np.maximum(vals, 0.0)
    vb = vecs.T @ b
    return [(vecs @ (vb / (vals + a)[:, None])) * scale[:, None] for a in alphas]


class BandedRidge(BaseEstimator, RegressorMixin):
    """Ridge with one regularisation strength per feature block.

    ``blocks`` is the width of each block, in the column order they were
    concatenated -- ``[64, 32]`` for ``fold-pca:64+lr-cca:32``. With a single
    block this reduces to ridge with a CV-selected alpha, which is the point:
    the combination arm and its own single-block control run through the same
    code.

    Scaling is done inside rather than by an outer ``StandardScaler`` pipeline,
    because the selection needs participant groups and threading a fit
    parameter through a pipeline for that is more fragile than owning it.
    """

    def __init__(self, blocks=None, alphas=ALPHA_GRID, ratios=RATIO_GRID,
                 n_splits=5, seed=0):
        self.blocks = blocks
        self.alphas = alphas
        self.ratios = ratios
        self.n_splits = n_splits
        self.seed = seed

    def _candidates(self, n_blocks, rng):
        if n_blocks == 1:
            return [np.ones(1)]
        if n_blocks == 2:
            # Exhaustive: the first block is pinned at 1 and only the ratio
            # matters, since a global rescaling is absorbed by alpha.
            return [np.array([1.0, r]) for r in self.ratios]
        # Beyond two blocks the grid is exponential, so sample the simplex --
        # the random search Dupre la Tour et al. use for the same reason.
        n_iter = 10 * len(self.ratios)
        return [np.ones(n_blocks)] + [
            np.exp(rng.uniform(np.log(self.ratios[0]), np.log(self.ratios[-1]),
                               size=n_blocks))
            for _ in range(n_iter)]

    def fit(self, x, y, groups=None):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        blocks = list(self.blocks) if self.blocks else [x.shape[1]]
        if sum(blocks) != x.shape[1]:
            raise ValueError(f"blocks {blocks} sum to {sum(blocks)}, "
                             f"features are {x.shape[1]}")

        self.scaler_ = StandardScaler().fit(x)
        xs = self.scaler_.transform(x)
        rng = np.random.default_rng(self.seed)
        candidates = self._candidates(len(blocks), rng)
        folds = _splits(len(xs), groups, self.n_splits, self.seed)

        # [candidate, alpha] validation error, summed over folds.
        err = np.zeros((len(candidates), len(self.alphas)))
        for tr, va in folds:
            x_tr, x_va = xs[tr], xs[va]
            mu_x, mu_y = x_tr.mean(axis=0), y[tr].mean(axis=0)
            xc, yc = x_tr - mu_x, y[tr] - mu_y
            xv = x_va - mu_x
            gram, xty = xc.T @ xc, xc.T @ yc
            for i, w in enumerate(candidates):
                scale = _expand(w, blocks)
                for j, coef in enumerate(_ridge_path(gram, xty, self.alphas, scale)):
                    resid = (xv @ coef + mu_y) - y[va]
                    err[i, j] += float((resid ** 2).sum())

        i, j = np.unravel_index(int(np.argmin(err)), err.shape)
        self.block_weights_ = np.asarray(candidates[i], dtype=float)
        self.alpha_ = float(self.alphas[j])
        # The penalty each block actually received, which is the interpretable
        # output: alpha / w^2, so a small weight means a hard-shrunk block.
        self.block_alphas_ = self.alpha_ / self.block_weights_ ** 2
        self.blocks_ = blocks
        self.cv_error_ = float(err[i, j] / len(xs))

        mu_x, mu_y = xs.mean(axis=0), y.mean(axis=0)
        scale = _expand(self.block_weights_, blocks)
        xc, yc = xs - mu_x, y - mu_y
        coefs = _ridge_path(xc.T @ xc, xc.T @ yc, [self.alpha_], scale)
        self.coef_ = coefs[0]
        self.intercept_ = mu_y - mu_x @ self.coef_
        return self

    def predict(self, x):
        xs = self.scaler_.transform(np.asarray(x, dtype=np.float64))
        return xs @ self.coef_ + self.intercept_


class StackedRidge(BaseEstimator, RegressorMixin):
    """One ridge per block, combined by non-negative weights on their predictions.

    The weights are fitted on **out-of-fold** predictions, so a block that only
    looks good on its own training rows cannot buy influence. They are
    constrained non-negative and normalised to sum to one, which makes the fit a
    convex combination: it can never do worse than the best single block by more
    than the weight estimation error, and ``stack_weights_`` reads directly as
    each block's contribution -- the number the redundancy question actually
    wants.

    Fitted per output dimension, because horizontal and vertical gaze are not
    equally decodable and need not weight the blocks the same way (``dsL06``
    decodes x at 0.947 and y at 0.343 from identical scans).
    """

    def __init__(self, blocks=None, alphas=ALPHA_GRID, n_splits=5, seed=0):
        self.blocks = blocks
        self.alphas = alphas
        self.n_splits = n_splits
        self.seed = seed

    def _block_slices(self, n_features):
        blocks = list(self.blocks) if self.blocks else [n_features]
        if sum(blocks) != n_features:
            raise ValueError(f"blocks {blocks} sum to {sum(blocks)}, "
                             f"features are {n_features}")
        edges = np.cumsum([0] + blocks)
        return blocks, [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]

    def _fit_one(self, x, y):
        """Ridge on one block, alpha by grouped CV on that block alone."""
        model = BandedRidge(blocks=[x.shape[1]], alphas=self.alphas,
                            n_splits=self.n_splits, seed=self.seed)
        return model.fit(x, y, groups=self._groups)

    def fit(self, x, y, groups=None):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        self.blocks_, slices = self._block_slices(x.shape[1])
        self._groups = groups
        folds = _splits(len(x), groups, self.n_splits, self.seed)

        oof = np.full((len(slices), len(x), y.shape[1]), np.nan)
        for tr, va in folds:
            g_tr = None if groups is None else np.asarray(groups)[tr]
            for b, sl in enumerate(slices):
                m = BandedRidge(blocks=[sl.stop - sl.start], alphas=self.alphas,
                                n_splits=self.n_splits, seed=self.seed)
                m.fit(x[tr][:, sl], y[tr], groups=g_tr)
                oof[b, va] = m.predict(x[va][:, sl])

        ok = np.all(np.isfinite(oof), axis=(0, 2))
        self.stack_weights_ = np.zeros((len(slices), y.shape[1]))
        for d in range(y.shape[1]):
            from scipy.optimize import nnls

            a = oof[:, ok, d].T
            w, _ = nnls(a, y[ok, d])
            self.stack_weights_[:, d] = w / w.sum() if w.sum() > 0 else 1.0 / len(slices)

        self.models_ = [self._fit_one(x[:, sl], y) for sl in slices]
        self.slices_ = slices
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        preds = np.stack([m.predict(x[:, sl])
                          for m, sl in zip(self.models_, self.slices_)])
        return np.einsum("bnd,bd->nd", preds, self.stack_weights_)
