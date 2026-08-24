"""Gaze decoding metrics, and the calibration that makes two of them meaningful.

**Pearson r needs nothing.** It is invariant to gain and offset, so it can be
computed straight from the readout's output and is the metric to compare methods
on.

**R-squared and Euclidean error do not survive that invariance**, and the
protocol they are computed under makes the problem unavoidable: `probe.lodo`
z-scores gaze per *training* dataset before pooling, because the per-dataset
Euclidean scale spans 21 to 595 and one pooled ridge would otherwise follow
whichever dataset has the largest target variance. Its predictions therefore
come out in z-units against test labels in degrees, and an R-squared computed
across that is not small, it is meaningless.

The gap is a real one and not an artifact of pooling. Cross-dataset predictions
are mis-calibrated in *gain* (measured 0.11 to 2.27 against the training scale)
with offsets near zero, and the required gain is about
``test_gaze_SD / train_gaze_SD`` -- the target's marginal spread, which is
exactly what differs between a fixation task and a free-viewing task. Degrees of
visual angle depend on screen size and viewing distance, and neither is in the
BOLD. Every unsupervised correction tried failed (z-match -0.921, quantile
-0.973, feature standardisation 0.003, mean shift 0.071).

So this module fixes it with the smallest honest amount of supervision:
:func:`fit_affine` estimates one gain and one offset per axis on the **other
participants of the same held-out dataset**. No participant ever sees its own
labels, and the scenario is the realistic one -- calibrate a new study on a few
subjects with an eye tracker, then decode the rest without one. R-squared and
Euclidean error are reported under that calibration and must be quoted as
calibrated numbers; Pearson r is reported without it.
"""
import numpy as np


def pearson(pred, true):
    """Pearson r over finite pairs; NaN when either side is degenerate."""
    ok = np.isfinite(pred) & np.isfinite(true)
    if ok.sum() < 10 or np.std(true[ok]) < 1e-6 or np.std(pred[ok]) < 1e-6:
        return np.nan
    return float(np.corrcoef(pred[ok], true[ok])[0, 1])


def r_squared(pred, true):
    """Coefficient of determination against the test mean.

    Not `pearson ** 2`: this one punishes gain and offset error, which is the
    whole reason it is reported next to r. It goes negative when the prediction
    is worse than predicting the mean, and that is informative rather than a bug.
    """
    ok = np.isfinite(pred) & np.isfinite(true)
    if ok.sum() < 10:
        return np.nan
    resid = np.sum((true[ok] - pred[ok]) ** 2)
    total = np.sum((true[ok] - true[ok].mean()) ** 2)
    return float(1.0 - resid / total) if total > 1e-12 else np.nan


def euclidean(pred, true):
    """Per-sample gaze error ``sqrt(dx^2 + dy^2)`` -> (median, mean).

    Median first because the distribution is heavy-tailed: track loss and
    blinks put a handful of samples many degrees out, and a mean over those
    describes the artifacts rather than the decoding.
    """
    ok = np.isfinite(pred).all(axis=1) & np.isfinite(true).all(axis=1)
    if ok.sum() < 10:
        return np.nan, np.nan
    d = np.linalg.norm(pred[ok] - true[ok], axis=1)
    return float(np.median(d)), float(d.mean())


def fit_affine(pred, true):
    """Per-axis gain and offset mapping predictions onto label units.

    Two parameters per axis by ordinary least squares. Deliberately not a full
    2x2 linear map: the axes are not confusable here (the (pred, true) 2x2
    correlation matrix is diagonal and positive on every dataset), so off-
    diagonal terms would only fit noise.
    """
    gain, offset = np.ones(2), np.zeros(2)
    for a in range(2):
        ok = np.isfinite(pred[:, a]) & np.isfinite(true[:, a])
        if ok.sum() < 10 or np.std(pred[ok, a]) < 1e-9:
            continue
        g, b = np.polyfit(pred[ok, a], true[ok, a], 1)
        gain[a], offset[a] = float(g), float(b)
    return gain, offset


def apply_affine(pred, gain, offset):
    return pred * gain + offset


def score(pred, true, gain=None, offset=None):
    """Every metric for one participant at one resolution.

    ``pred`` and ``true`` are ``[N, 2]`` in that participant's own rows. Pass a
    calibration to get R-squared and Euclidean error in label units; without one
    they are computed on the raw prediction and will be meaningless if it is in
    z-units, which is why :func:`probe.lodo` always supplies one.
    """
    cal = pred if gain is None else apply_affine(pred, gain, offset)
    r_x, r_y = pearson(pred[:, 0], true[:, 0]), pearson(pred[:, 1], true[:, 1])
    q_x, q_y = r_squared(cal[:, 0], true[:, 0]), r_squared(cal[:, 1], true[:, 1])
    med, mean = euclidean(cal, true)
    return {
        "r_x": r_x, "r_y": r_y, "r": float(np.mean([r_x, r_y])),
        "r2_x": q_x, "r2_y": q_y, "r2": float(np.mean([q_x, q_y])),
        "euclid_median": med, "euclid_mean": mean,
        "gain_x": None if gain is None else float(gain[0]),
        "gain_y": None if gain is None else float(gain[1]),
    }


def nanmedian(values):
    """Median over the finite entries; NaN when there are none."""
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")
