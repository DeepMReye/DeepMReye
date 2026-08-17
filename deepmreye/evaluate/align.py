"""Unsupervised feature alignment: fixing the transfer, not the representation.

Five experiments have now moved the *representation* barely at all (0.759-0.779
across every basis and objective tried), while ``dsL03_pursuit`` sits at
r ~ 0.20 under every one of them -- and decodes at ~0.88 *within* run. The gap
is therefore not missing signal, it is covariate shift between acquisitions,
and it is where the headroom is: mean across folds 0.687 against a median of
0.779, dragged down almost entirely by that one fold.

The methods here are the standard answer to exactly that problem in neural
decoding, and none of them looks at a gaze label. They only use the target
participant's own unlabeled fMRI -- which you always have at deploy time, since
the thing you are missing is the eye tracker, not the scan.

- ``center``  subtract each group's feature mean. The cheapest possible fix.
- ``zscore``  per-feature mean and standard deviation within each group. This is
              the *diagonal* case, and it is what ``analyze_calibration.py``
              already tried as ``feat-std`` (it scored 0.003, i.e. nothing). It
              is kept here as the reference the full-covariance methods have to
              beat, because if they do not, the extra machinery is unjustified.
- ``ea``      **Euclidean Alignment**: whiten each group by the inverse square
              root of its own covariance, so every group arrives with identity
              covariance. Proposed for cross-subject EEG transfer and validated
              across 13 BCI paradigms; closed-form, label-free. The step up from
              ``zscore`` is that it removes the *correlation structure* between
              components, not merely their individual scales -- and on a shared
              PCA basis different scanners genuinely differ in which directions
              carry variance, which a diagonal correction cannot touch.
- ``coral``   **Correlation Alignment**: instead of whitening everything to
              identity, map each target group onto the pooled *source* second
              moment. Same information, different destination: ``ea`` discards
              the source's covariance too, ``coral`` preserves it and moves the
              target onto it.

``ea`` and ``coral`` differ in what happens to the training side, and that
matters. ``ea`` whitens source groups as well, so the readout is fitted on
already-aligned features; ``coral`` leaves the source alone and moves only the
target. Both are worth having -- which one wins is an empirical question about
whether the source's covariance is worth keeping.
"""
import numpy as np

ALIGN_METHODS = ("none", "center", "zscore", "ea", "coral")

# Groups here are single participants, which on the shorter labeled runs means
# ~80 feature rows against 256 dimensions -- a badly rank-deficient covariance.
# Shrinkage is not a refinement, it is what makes the inverse square root exist
# at all; Ledoit-Wolf picks the coefficient from the data rather than by taste.
DEFAULT_SHRINKAGE = "lw"


def _covariance(x, shrinkage=DEFAULT_SHRINKAGE):
    """Regularised covariance of centred rows ``x [N, D]``."""
    n, d = x.shape
    if shrinkage == "lw":
        from sklearn.covariance import LedoitWolf

        # assume_centered: the caller has already removed the group mean, and
        # letting LedoitWolf re-centre would undo that on tiny groups.
        return LedoitWolf(assume_centered=True).fit(x).covariance_
    cov = (x.T @ x) / max(n - 1, 1)
    a = float(shrinkage)
    return (1 - a) * cov + a * (np.trace(cov) / d) * np.eye(d)


def _inv_sqrt(cov, eps=1e-10):
    """``cov^{-1/2}`` for a symmetric PSD matrix."""
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, eps * max(float(vals.max()), eps))
    return (vecs / np.sqrt(vals)) @ vecs.T


def _sqrt(cov, eps=1e-10):
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, eps * max(float(vals.max()), eps))
    return (vecs * np.sqrt(vals)) @ vecs.T


def fit_reference(x, shrinkage=DEFAULT_SHRINKAGE):
    """Pooled source statistics that ``coral`` maps the target onto."""
    mu = x.mean(axis=0)
    return {"mean": mu, "sqrt_cov": _sqrt(_covariance(x - mu, shrinkage))}


def align(x, groups, method, reference=None, shrinkage=DEFAULT_SHRINKAGE):
    """Align ``x [N, D]`` within each group of ``groups [N]``.

    Every method is fitted on the group's *own* rows and uses no labels, so it
    is applicable to a participant you have never seen and have no eye tracking
    for. ``reference`` is required by ``coral`` and ignored otherwise.
    """
    if method == "none":
        return np.asarray(x, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    groups = np.asarray(groups)
    out = np.empty_like(x)

    for g in np.unique(groups):
        m = groups == g
        xg = x[m]
        mu = xg.mean(axis=0)
        centred = xg - mu

        if method == "center":
            out[m] = centred
        elif method == "zscore":
            out[m] = centred / (xg.std(axis=0) + 1e-8)
        elif method in ("ea", "coral"):
            # A group with fewer rows than dimensions cannot support even a
            # shrunk whitening; fall back rather than emit a near-singular map.
            if len(xg) < 3:
                out[m] = centred
                continue
            w = _inv_sqrt(_covariance(centred, shrinkage))
            if method == "coral":
                if reference is None:
                    raise ValueError("coral needs a source reference")
                out[m] = centred @ w @ reference["sqrt_cov"] + reference["mean"]
            else:
                out[m] = centred @ w
        else:
            raise ValueError(f"unknown alignment {method!r}; "
                             f"known: {', '.join(ALIGN_METHODS)}")
    return out


def apply_pair(x_tr, g_tr, x_te, g_te, method, shrinkage=DEFAULT_SHRINKAGE):
    """Align a train/test pair consistently.

    ``ea`` whitens both sides, so the readout is fitted on aligned features.
    ``coral`` leaves the source untouched and moves only the target onto the
    pooled source statistics -- which is the whole difference between them.
    """
    if method == "none":
        return np.asarray(x_tr, dtype=np.float64), np.asarray(x_te, dtype=np.float64)
    if method == "coral":
        ref = fit_reference(np.asarray(x_tr, dtype=np.float64), shrinkage)
        return (np.asarray(x_tr, dtype=np.float64),
                align(x_te, g_te, "coral", ref, shrinkage))
    return (align(x_tr, g_tr, method, None, shrinkage),
            align(x_te, g_te, method, None, shrinkage))
