"""Fitting and scoring the linear gaze probe.

The probe is the control for the whole method: it asks whether the
self-supervised representation carries gaze information, by fitting a linear map
from frozen encoder features to gaze coordinates.

Two things here are deliberate, because getting either wrong makes the numbers
meaningless rather than merely worse:

**Time is not pooled away.** The encoder produces one token per (spatial patch,
temporal patch). Pooling over *both* axes gives a single vector per window and
forces the target to be the mean gaze over that window -- 80 to 250 seconds
depending on the dataset's TR. Measured on the labeled corpus, that discards 84
to 96% of the gaze variance (within-window SD 2.4-7.1 degrees, SD of the window
means 0.12-1.11), so the probe is asked to predict a nearly constant target.
Pooling over space alone keeps one prediction per temporal patch.

**NaNs are averaged around, not propagated.** Missing gaze samples are marked
NaN and are common: windows containing at least one NaN are 100% of
``dsL03_pursuit`` and ``dsL06_sequences``, 61% of ``dsL05_free_viewing``. A
plain ``mean`` turns those into NaN targets that get dropped, which silently
removed two of six labeled datasets from the evaluation entirely. ``nanmean``
keeps a temporal bin as long as *any* of its samples is valid.
"""
import warnings

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import r2_score


def temporal_targets(labels, n_t):
    """Reduce gaze labels to one coordinate per temporal token.

    ``labels`` is ``[B, W, 10, 2]`` -- W TRs, 10 sub-TR gaze samples, x and y.
    Returns ``[B, n_t, 2]``, each entry the nanmean over the TRs (and their
    sub-samples) covered by that temporal patch. A bin with no valid sample at
    all stays NaN and is masked downstream.
    """
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    labels = np.asarray(labels, dtype=np.float64)

    b, w = labels.shape[0], labels.shape[1]
    per_bin = int(np.ceil(w / n_t))

    # Pad the time axis so it divides evenly, matching the patcher, which pads
    # T up to a multiple of temp_patch_size before binning.
    pad = per_bin * n_t - w
    if pad > 0:
        labels = np.concatenate(
            [labels, np.full((b, pad) + labels.shape[2:], np.nan)], axis=1)

    binned = labels.reshape(b, n_t, per_bin, *labels.shape[2:])
    axes = tuple(range(2, binned.ndim - 1))  # everything but batch, bin, and x/y
    with warnings.catch_warnings():
        # An all-NaN bin is expected, not exceptional -- it means no gaze was
        # recorded there, and it is masked rather than dropped.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(binned, axis=axes)


def pool_spatial(context_reps, n_s, n_t):
    """Mean-pool encoder tokens over space, keeping the temporal axis.

    ``context_reps`` is ``[B, n_s * n_t, D]`` flattened as ``s * n_t + t`` by
    the patcher. Returns ``[B, n_t, D]``.
    """
    b, total, d = context_reps.shape
    if total != n_s * n_t:
        raise ValueError(f"expected {n_s * n_t} tokens, got {total}")
    return context_reps.view(b, n_s, n_t, d).mean(dim=1)


def flatten_valid(features, targets):
    """Stack ``[B, n_t, ...]`` into rows and drop rows with a NaN target."""
    x = np.asarray(features).reshape(-1, np.asarray(features).shape[-1])
    y = np.asarray(targets).reshape(-1, np.asarray(targets).shape[-1])
    keep = ~np.isnan(y).any(axis=1)
    return x[keep], y[keep]


def flatten_valid_groups(features, targets, *groups):
    """:func:`flatten_valid`, also expanding per-window group labels to rows.

    ``features``/``targets`` are ``[N, n_t, ...]`` -- one row per *temporal bin*
    -- while ``groups`` (dataset name, subject id) carry one entry per *window*.
    Expanding them here, against the same NaN mask that drops target rows, is
    the only place the correspondence can be got right; doing it at the call
    site has silently misaligned predictions from their dataset before.
    """
    features = np.asarray(features)
    targets = np.asarray(targets)
    n_t = targets.shape[1]

    y = targets.reshape(-1, targets.shape[-1])
    keep = ~np.isnan(y).any(axis=1)
    x = features.reshape(-1, features.shape[-1])[keep]

    expanded = [np.repeat(np.asarray(g), n_t)[keep] for g in groups]
    return (x, y[keep], *expanded)


def aggregate_by_subject(targets, predictions, subjects, baseline=None, min_rows=20):
    """Score each participant separately, then take the median across them.

    This is the right unit of analysis and it is not the same number as pooling
    every row together. Pooled across participants, a model that predicts only
    *which participant this is* scores well: if one subject's gaze sits left of
    another's, the between-subject offset alone produces a correlation, with no
    within-subject decoding at all. Per-subject correlation cannot be gamed that
    way -- each is computed within one participant, where the between-subject
    variance is constant by construction.

    The median rather than the mean because per-subject r is bounded and skewed,
    and a single failed registration should not move the headline number.

    Returns the median of each metric plus ``n_subjects``, and keeps the whole
    per-subject distribution under ``"per_subject"`` so spread can be reported.
    """
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    subjects = np.asarray(subjects)

    per_subject = {}
    for sub in np.unique(subjects):
        sel = subjects == sub
        if sel.sum() < min_rows:
            # Too few valid timepoints for a correlation to mean anything.
            continue
        per_subject[str(sub)] = compute_probe_metrics(
            targets[sel], predictions[sel], baseline)

    if not per_subject:
        return {"n_subjects": 0, "per_subject": {}}

    out = {"n_subjects": len(per_subject), "per_subject": per_subject}
    for key in ("euclidean_error", "pearson_r_x", "pearson_r_y",
                "r2_x", "r2_y", "r2_vs_baseline"):
        vals = [m[key] for m in per_subject.values()
                if key in m and np.isfinite(m[key])]
        out[key] = float(np.median(vals)) if vals else np.nan
    return out


def compute_probe_metrics(targets, predictions, baseline=None):
    """Euclidean error, Pearson r and R^2 for predicted vs true gaze.

    ``targets``/``predictions`` are ``[N, 2]``. ``baseline``, when given, is the
    constant prediction (typically the training-set mean) that R^2 is measured
    against -- without it, R^2 is computed against the *test* mean, which
    flatters a model that has only learned the test set's centre.
    """
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()

    tgt = np.asarray(targets, dtype=np.float64)
    pred = np.asarray(predictions, dtype=np.float64)

    valid = ~np.isnan(tgt).any(axis=1)
    tgt, pred = tgt[valid], pred[valid]

    empty = {"n": 0, "euclidean_error": np.nan, "pearson_r_x": np.nan,
             "pearson_r_y": np.nan, "r2_x": np.nan, "r2_y": np.nan}
    if len(tgt) < 2:
        return empty

    out = {
        "n": int(len(tgt)),
        "euclidean_error": float(np.mean(np.sqrt(np.sum((tgt - pred) ** 2, axis=1)))),
    }

    for i, axis in enumerate("xy"):
        # A constant column makes Pearson undefined; report NaN rather than a
        # warning-generated nan that looks like a computation failure.
        if np.std(tgt[:, i]) < 1e-9 or np.std(pred[:, i]) < 1e-9:
            out[f"pearson_r_{axis}"] = np.nan
        else:
            out[f"pearson_r_{axis}"] = float(pearsonr(tgt[:, i], pred[:, i])[0])
        out[f"r2_{axis}"] = float(r2_score(tgt[:, i], pred[:, i]))

    if baseline is not None:
        const = np.broadcast_to(np.asarray(baseline, dtype=np.float64), tgt.shape)
        ss_res = np.sum((tgt - pred) ** 2)
        ss_tot = np.sum((tgt - const) ** 2)
        out["r2_vs_baseline"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        out["baseline_euclidean"] = float(
            np.mean(np.sqrt(np.sum((tgt - const) ** 2, axis=1))))

    return out
