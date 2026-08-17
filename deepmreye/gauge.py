"""Zero-label gaze decoding: the corpus basis supplies the gauge.

`scripts/analyze_identifiability.py` established that gaze is recoverable from a
single run with **no labels in the fit** -- CCA between the two orbits reaches
|r| ~ 0.75, against 0.57 for a *supervised* cross-dataset ridge. It was
nonetheless not a method, for one reason: CCA is invariant to permuting and
negating its components, so which variate is horizontal, which is vertical, and
what sign each carries had to be read off labels. That script reports
`median(|r|)`, which hides the problem rather than solving it.

**The frozen corpus basis has no gauge freedom.** It is one fixed set of filters
applied to every participant, so component `j` means the same thing in everyone
and its sign is fixed by construction. Measured over 75 labeled participants in 7
datasets (`scripts/diagnose_gauge.py`):

- component **21** tracks gaze **x** at signed r **+0.563**, sign agreement
  **100%**;
- component **7** tracks gaze **y** at signed r **-0.544**, sign agreement 96%;
- leave-one-dataset-out selects the same two components on **7 of 7** folds;
- the circular-shift null gives **-0.003 / +0.010**.

That yields two decoders, and the second is the point of this module:

``fixed``     apply the two corpus components directly. No fitting whatsoever on
              the target participant -- the entire decoder is two integers and
              two signs.
``adapted``   refit CCA on the target run (unsupervised, as before) and take its
              gauge by **temporal agreement with the corpus components**. The
              corpus acts as a label-free teacher that names the run-specific
              variates; the run-specific fit supplies the precision the frozen
              filters lack.

**What is and is not label-free.** Nothing here touches the target
participant's, or the target *study's*, gaze. The two component indices and
their signs were chosen once on other datasets and are constants of the method
from then on -- about 9 bits, fixed for all future use. The geometric route that
would have removed even those (naming the components from the spatial dipole of
their patterns) was measured and is **not** clean: component 21's dipole is z/y
dominated rather than x, so it cannot name the horizontal axis. Do not claim
"no labels anywhere"; claim "no labels from the target study", which is what
was verified.

numpy and sklearn only -- nothing in the feature path may import torch (see the
OpenMP deadlock note in CLAUDE.md).
"""
import numpy as np

from deepmreye.unsupervised import project

# Selected by `scripts/diagnose_gauge.py` on the labeled corpus and stable under
# leave-one-dataset-out on 7/7 folds. Stored as the method's constants; pass an
# explicit gauge to `decode` to override, which is what the LODO arm does.
DEFAULT_GAUGE = {"x": (21, +1.0), "y": (7, -1.0)}

# Components of the per-run CCA. 8 is inherited from
# `analyze_identifiability.py`, where it was measured that 3 is too few (the
# horizontal variate is simply absent for dsL02/dsL05 at 3). That number was
# tuned with labels, which is a mild design-time leak; it is checked at 6 and 12
# in the harness rather than left implicit.
N_CC = 8
N_PCA = 20
MIN_TR = 120


def corpus_variates(block, mask, basis, k=32):
    """``[X, Y, Z, T]`` eye block -> corpus lr-cca variates ``[T, k]``."""
    x = as_rows(block, mask)
    return project("lr-cca", basis, x, k=k)


def as_rows(block, mask):
    """``[X, Y, Z, T]`` -> ``[T, n_masked]`` float64, in corpus-mask order."""
    block = np.asarray(block)
    t = block.shape[-1]
    return block.reshape(-1, t).T[:, mask.reshape(-1)].astype(np.float64)


def orbit_views(rows, basis):
    """Split masked rows into the two orbits, using the basis's own indices.

    The split has to come from the basis rather than from a local constant:
    `fit_lr_cca` defines the halves by `x < LR_SPLIT_X` over the masked voxels
    and stores the resulting indices, and a run split any other way would put
    the two estimators in different coordinate systems. Note this is *not*
    `crossorbit.split_orbits`, which additionally drops the midline and mirrors
    the right orbit -- applying that mirroring here would flip the sign of
    horizontal gaze relative to the corpus filters.
    """
    li = np.asarray(basis["left_index"])
    ri = np.asarray(basis["right_index"])
    return rows[:, li], rows[:, ri]


def motion_proxy(rows):
    """Stand-in for realignment parameters, which the corpus does not store.

    Rigid head motion translates the whole crop coherently, so the mean signal
    over the orbit voxels and its temporal derivative capture some of it. This
    is markedly weaker than true 6-DOF parameters and is a stated limitation,
    not a fix: a canonical variate could still be partly motion. Kept identical
    to `analyze_identifiability.motion_proxy` so the arms stay comparable.
    """
    live = rows.std(axis=0) > 1e-6
    g = rows[:, live].mean(axis=1)
    return np.column_stack([g, np.gradient(g)])


def regress_out(x, confounds):
    """Project the confound subspace out of every column of ``x``."""
    c = np.column_stack([np.ones(len(x)), confounds])
    beta, *_ = np.linalg.lstsq(c, x, rcond=None)
    return x - c @ beta


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or np.std(a[ok]) < 1e-9 or np.std(b[ok]) < 1e-9:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def run_cca(left, right, fit, n_cc=N_CC, n_pca=N_PCA):
    """Per-run CCA between the orbits, fitted on ``fit`` and applied to all TRs.

    Returns the averaged canonical variates ``[T, n_cc]``. Averaging the two
    views halves the independent per-view noise: both are estimates of the same
    shared latent, which is the whole reason CCA is the identifying estimator
    here.
    """
    from sklearn.cross_decomposition import CCA
    from sklearn.decomposition import PCA

    n_pca = int(min(n_pca, len(np.atleast_1d(np.arange(left.shape[0])[fit])) - 1,
                    left.shape[1], right.shape[1]))
    n_cc = int(min(n_cc, n_pca))
    pl, pr = PCA(n_pca).fit(left[fit]), PCA(n_pca).fit(right[fit])
    cca = CCA(n_components=n_cc, max_iter=1000).fit(pl.transform(left[fit]),
                                                    pr.transform(right[fit]))
    u, v = cca.transform(pl.transform(left), pr.transform(right))
    return (u + v) / 2.0


def gauge_by_teacher(variates, teacher):
    """Name a run's variates by temporal agreement with a corpus component.

    ``teacher`` is one corpus variate ``[T]`` -- a filter with a fixed, known
    meaning. The run-specific variate that agrees with it most is the same
    latent, and the sign of that agreement resolves the sign. This is the whole
    gauge mechanism, and it uses no labels: both series are unsupervised
    projections of the same voxels.

    Returns ``(index, sign, quality)``. ``quality`` is the |correlation| to the
    teacher, which is worth carrying because a run where nothing matches the
    teacher is a run whose gauge should not be trusted.
    """
    scores = np.array([corr(variates[:, k], teacher)
                       for k in range(variates.shape[1])])
    if not np.any(np.isfinite(scores)):
        return 0, 1.0, float("nan")
    k = int(np.nanargmax(np.abs(scores)))
    return k, float(np.sign(scores[k]) or 1.0), float(abs(scores[k]))


def decode(block, mask, basis, gauge=None, k=32, mode="adapted",
           n_cc=N_CC, n_pca=N_PCA, fit_frac=0.5):
    """Predict gaze for one run without using any of its labels.

    ``mode="fixed"`` returns the two gauged corpus components directly -- a
    decoder with no free parameters at all. ``mode="adapted"`` refits CCA on the
    run and uses the corpus components only to name the result.

    ``fit_frac`` splits the run for the adapted fit. Fitting on all of it would
    not be circular (nothing here sees gaze), but half-and-half keeps this
    directly comparable to `analyze_identifiability.py`, which is the number
    being improved on. Returns ``(pred [T, 2], info)``.
    """
    gauge = gauge or DEFAULT_GAUGE
    rows = as_rows(block, mask)
    corpus = project("lr-cca", basis, rows, k=k)
    teachers = {ax: gauge[ax][1] * corpus[:, gauge[ax][0]] for ax in ("x", "y")}

    if mode == "fixed":
        pred = np.column_stack([teachers["x"], teachers["y"]])
        return pred, {"mode": "fixed", "quality_x": 1.0, "quality_y": 1.0}

    left, right = orbit_views(rows, basis)
    conf = motion_proxy(rows)
    left, right = regress_out(left, conf), regress_out(right, conf)

    t = len(rows)
    fit = slice(0, max(int(t * fit_frac), n_pca + 1))
    variates = run_cca(left, right, fit, n_cc, n_pca)

    cols, info = [], {"mode": mode}
    for ax in ("x", "y"):
        idx, sign, quality = gauge_by_teacher(variates, teachers[ax])
        cols.append(sign * variates[:, idx])
        info[f"variate_{ax}"], info[f"quality_{ax}"] = idx, quality
    return np.column_stack(cols), info


def oracle_gauge(variates, gaze, fit):
    """The gauge a *label-using* method would have chosen. Upper bound only.

    This is what `analyze_identifiability.py` does implicitly, and it exists
    here so the harness can measure how much the label-free gauge gives up --
    which is the only honest way to report the method.
    """
    out = {}
    for ax, axis in (("x", 0), ("y", 1)):
        scores = [corr(variates[fit, k], gaze[fit, axis])
                  for k in range(variates.shape[1])]
        if not np.any(np.isfinite(scores)):
            out[ax] = (0, 1.0)
            continue
        k = int(np.nanargmax(np.abs(scores)))
        out[ax] = (k, float(np.sign(scores[k]) or 1.0))
    return out


def select_gauge(datasets_runs, k=32):
    """Choose (component, sign) per axis from labeled runs -- route (b).

    ``datasets_runs`` can be a list/dict of datasets (each containing
    ``(corpus_variates [T, k], gaze [T, 2])`` runs), or a flat list of runs.
    Averages per dataset first so single large datasets (e.g. dsL01 with 170
    participants) do not overwhelm the gauge selection.

    Used leave-one-dataset-out by the harness, so the gauge applied to a fold is
    never chosen on that fold.
    """
    if len(datasets_runs) == 0:
        raise ValueError("no runs to select a gauge from")

    if isinstance(datasets_runs, dict):
        datasets_list = list(datasets_runs.values())
    elif isinstance(datasets_runs, (list, tuple)):
        # Check if element is a run (tuple of 2 arrays) vs a list of runs
        if len(datasets_runs) > 0 and isinstance(datasets_runs[0], (tuple, list)) and len(datasets_runs[0]) == 2 and isinstance(datasets_runs[0][0], np.ndarray):
            datasets_list = [datasets_runs]
        else:
            datasets_list = list(datasets_runs)
    else:
        datasets_list = [datasets_runs]

    ds_means = []
    for ds in datasets_list:
        per_run = []
        for item in ds:
            variates, gaze = item[0], item[1]
            per_run.append(np.array([[corr(variates[:, j], gaze[:, ax])
                                      for ax in (0, 1)]
                                     for j in range(variates.shape[1])]))
        if per_run:
            ds_means.append(np.nanmean(np.stack(per_run), axis=0))

    if not ds_means:
        raise ValueError("no valid runs in datasets")

    stacked = np.stack(ds_means)                       # [n_datasets, k, 2]
    out = {}
    for ax, name in ((0, "x"), (1, "y")):
        m = np.nanmean(stacked[:, :, ax], axis=0)
        j = int(np.nanargmax(np.abs(m)))
        out[name] = (j, float(np.sign(m[j]) or 1.0))
    return out
