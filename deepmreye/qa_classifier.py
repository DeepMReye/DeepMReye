"""Triage model over extracted eye blocks: which subjects probably show eyes.

This does **not** approve datasets. Approval is manual, and stays manual --
see the "Classifier removed" note in ``CLAUDE.md``. What this buys is ordering
and reach:

- **Ordering.** With ~2400 datasets to get through, labeling the subjects the
  model is least sure about first is worth far more than labeling in accession
  order.
- **Reach.** Full extraction pulls in every subject of an approved dataset, and
  nobody ever looks at those. A dataset can pass QA on its two sampled subjects
  and still contain individuals with bad coverage. This flags them for review.

It differs from the deleted ``transform_probability`` gate in what it reads: the
old one scored the registration's affine statistics, this one measures the
extracted eyeball voxels -- the same thing you judge from the QA report.

Features are deliberately cheap and interpretable. Occupancy of the eye mask
separates the classes strongly on its own (~0.58 for clean subjects against
~0.10 for a badly covered one), so a small model on a few hundred labels is a
reasonable fit; a deep one is not.
"""
import logging

import h5py
import numpy as np

# Label semantics from the QA UI: 1 = eyes, 3 = eyes but cut off, 4 = eyes but faint,
# 0 = no eyes / bad transform, 2 = no eyes / good transform.
#
# The model predicts the *exact* label rather than a binary eyes/no-eyes, so the
# UI can pre-select the right radio button -- including "cut off" or "faint".
EYES = 1
LABEL_NAMES = {0: "no eyes (bad transform)", 1: "eyes", 2: "no eyes (good transform)",
               3: "eyes but cut off", 4: "eyes but faint"}

import os
import re

FEATURE_NAMES = [
    # Inner eye block features (10)
    "nonzero_frac",
    "mean_abs",
    "std",
    "temporal_std_mean",
    "temporal_std_std",
    "spatial_kurtosis",
    "center_edge_ratio",
    "left_right_balance",
    "active_voxel_frac",
    "temporal_snr",
    # Registration & Surrounding Mask features (8)
    "step0_affine_sum",
    "step1_bigmask_affine_sum",
    "step2_smallmask_affine_sum",
    "step0_trans_mag",
    "step1_trans_mag",
    "step2_trans_mag",
    "step1_vs_step2_affine_diff",
    "step1_vs_step2_trans_diff",
    # Temporal & Scan Metadata features (3)
    "repetition_time",
    "n_trs",
    "scan_duration_sec",
]


def extract_report_features(report_path):
    """Extract registration transform features from the HTML report header or file."""
    feats = np.zeros(8, dtype=np.float32)
    if not report_path or not os.path.exists(report_path):
        return feats
    try:
        with open(report_path, "r", encoding="utf-8", errors="replace") as f:
            html = f.read()

        stats_match = re.search(r"Transform Stats:\s*(\[\[\[.*?\]\]\])", html, re.DOTALL)
        if stats_match:
            nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+e[-+]?\d+|[-+]?\d+\.\d+|[-+]?\d+", stats_match.group(1))]
            if len(nums) >= 36:
                s0 = np.array(nums[0:12], dtype=np.float32)
                s1 = np.array(nums[12:24], dtype=np.float32)
                s2 = np.array(nums[24:36], dtype=np.float32)

                feats[0] = float(np.sum(s0))
                feats[1] = float(np.sum(s1))
                feats[2] = float(np.sum(s2))
                feats[3] = float(np.linalg.norm(s0[9:12]))
                feats[4] = float(np.linalg.norm(s1[9:12]))
                feats[5] = float(np.linalg.norm(s2[9:12]))
                feats[6] = float(np.linalg.norm(s1 - s2))
                feats[7] = float(np.linalg.norm(s1[9:12] - s2[9:12]))
    except Exception as e:
        logging.debug(f"Failed to extract report features from {report_path}: {e}")
    return feats


def extract_features(eye_block, report_path=None, tr=2.0, n_trs=100.0):
    """Summarise one ``[X, Y, Z, T]`` block, report, and TR/volume stats into the 21-feature vector."""
    block = np.asarray(eye_block, dtype=np.float32)
    mask = np.any(block != 0, axis=-1)  # voxels with any signal across time
    n_vox = mask.sum()

    if n_vox == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    vals = block[mask]                      # [n_vox, T]
    temporal_std = vals.std(axis=1)         # per-voxel variability over time
    spatial = vals.mean(axis=1)             # per-voxel mean signal

    # Eyes sit at the centre of the crop; a failed registration tends to smear
    # signal toward the edges, so the centre/edge contrast is informative.
    x, y, z = mask.shape
    cx, cy, cz = x // 2, y // 2, z // 2
    center = mask[cx - x // 6:cx + x // 6, cy - y // 6:cy + y // 6, cz - z // 6:cz + z // 6]
    center_frac = center.mean() if center.size else 0.0
    edge_frac = mask.mean()

    # The crop holds both eyes side by side; a lopsided result usually means the
    # registration drifted off one of them.
    left, right = mask[: x // 2], mask[x // 2:]
    lr_sum = left.mean() + right.mean()
    lr_balance = min(left.mean(), right.mean()) / (lr_sum / 2) if lr_sum > 0 else 0.0

    denom = temporal_std.mean()
    f_inner = np.array([
        float(mask.mean()),
        float(np.abs(vals).mean()),
        float(vals.std()),
        float(temporal_std.mean()),
        float(temporal_std.std()),
        float(_kurtosis(spatial)),
        float(center_frac / edge_frac) if edge_frac > 0 else 0.0,
        float(lr_balance),
        float((temporal_std > temporal_std.mean()).mean()),
        float(np.abs(spatial).mean() / denom) if denom > 0 else 0.0,
    ], dtype=np.float32)

    f_report = extract_report_features(report_path)

    # TR and time volume metadata
    actual_tr = float(tr) if (tr is not None and tr > 0) else 2.0
    actual_ntr = float(n_trs) if (n_trs is not None and n_trs > 0) else float(block.shape[-1])
    f_meta = np.array([actual_tr, actual_ntr, actual_tr * actual_ntr], dtype=np.float32)

    return np.concatenate([f_inner, f_report, f_meta])


def _kurtosis(a):
    a = np.asarray(a, dtype=np.float64)
    if a.size < 2:
        return 0.0
    sd = a.std()
    return float(((a - a.mean()) ** 4).mean() / sd**4 - 3.0) if sd > 0 else 0.0


def open_h5_file(path, mode="r"):
    """Open HDF5 file safely without file locking issues."""
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    try:
        return h5py.File(path, mode, locking=False)
    except (TypeError, ValueError):
        return h5py.File(path, mode)


def features_from_file(path, report_path=None, tr=None, n_trs=None):
    """Feature vector for one participant file, or None if unreadable."""
    try:
        with open_h5_file(path, "r") as f:
            block = f["eye_block"][:]
            file_tr = tr or f.attrs.get("repetition_time", 2.0)
            file_ntr = n_trs or f.attrs.get("n_trs", block.shape[-1])
            return extract_features(block, report_path=report_path, tr=file_tr, n_trs=file_ntr)
    except Exception as e:
        logging.warning(f"Failed to featurise {path}: {e}")
        return None


def build_training_set(data_dir, registry_path=None, manual_only=False):
    """Collect ``(X, y, keys)`` from every QA-labeled participant on disk."""
    from pathlib import Path

    from deepmreye.storage import subject_path

    data_dir = Path(data_dir)
    registry_path = Path(registry_path or data_dir / "datasets.h5")

    labeled = []
    with open_h5_file(registry_path, "r") as f:
        for ds_name in f.keys():
            for sub_id in f[ds_name].keys():
                sub_grp = f[ds_name][sub_id]
                approved = sub_grp.attrs.get("approved", -1)
                is_manual = sub_grp.attrs.get("is_manual", False)
                report_path = sub_grp.attrs.get("report_html_path", "")
                tr = sub_grp.attrs.get("repetition_time", None)
                ntr = sub_grp.attrs.get("n_trs", None)
                if approved in (0, 1, 2, 3, 4):
                    if manual_only:
                        if is_manual:
                            labeled.append((ds_name, sub_id, int(approved), report_path, tr, ntr))
                    else:
                        labeled.append((ds_name, sub_id, int(approved), report_path, tr, ntr))

    X, y, keys = [], [], []
    for ds_name, sub_id, approved, report_path, tr, ntr in labeled:
        path = subject_path(data_dir, ds_name, sub_id)
        if not path.exists():
            continue
        feats = features_from_file(path, report_path=report_path, tr=tr, n_trs=ntr)
        if feats is None:
            continue
        X.append(feats)
        y.append(approved)  # exact label, so the UI can pre-select it
        keys.append((ds_name, sub_id))

    if not X:
        return np.empty((0, len(FEATURE_NAMES)), np.float32), np.empty(0, int), []
    return np.vstack(X), np.asarray(y), keys


def train(X, y, groups=None, seed=0):
    """Fit the triage model and cross-validate it.

    ``groups`` should be the dataset each subject came from. Subjects of one
    dataset share a scanner, sequence and often a failure mode, so a split that
    puts them on both sides reports an accuracy the model will not reproduce on
    a new dataset -- which is exactly how it gets used.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_val_score

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",  # no-eyes subjects are the minority
        random_state=seed,
        n_jobs=-1,
    )

    scores = None
    counts = np.bincount(y, minlength=5)
    n_min = int(min(c for c in counts if c > 0)) if counts.any() else 0
    if n_min >= 3 and len(set(y)) >= 2:
        # Accuracy, not ROC-AUC: this is multiclass now, and what matters for
        # pre-selection is simply how often the pre-checked button is right.
        if groups is not None and len(set(groups)) >= 3:
            folds = min(5, len(set(groups)), n_min)
            cv = GroupKFold(n_splits=folds).split(X, y, groups)
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        else:
            scores = cross_val_score(model, X, y, cv=min(5, n_min), scoring="accuracy")

    model.fit(X, y)
    return model, scores
