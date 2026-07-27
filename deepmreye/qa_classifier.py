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

# Label semantics from the QA UI: 1 = eyes, 3 = eyes but cut off,
# 0 = no eyes / bad transform, 2 = no eyes / good transform.
#
# The model predicts the *exact* label rather than a binary eyes/no-eyes, so the
# UI can pre-select the right radio button -- including "cut off", which is the
# distinction that is tedious to make by eye and worth automating.
EYES = 1
LABEL_NAMES = {0: "no eyes (bad transform)", 1: "eyes", 2: "no eyes (good transform)",
               3: "eyes but cut off"}

FEATURE_NAMES = [
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
]


def extract_features(eye_block):
    """Summarise one ``[X, Y, Z, T]`` block into the feature vector.

    Everything here is computed over the masked-in voxels only. Voxels outside
    the eye mask are exactly 0 by construction, so including them would mostly
    measure how much empty bounding box a subject has.
    """
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
    return np.array([
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


def _kurtosis(a):
    a = np.asarray(a, dtype=np.float64)
    if a.size < 2:
        return 0.0
    sd = a.std()
    return float(((a - a.mean()) ** 4).mean() / sd**4 - 3.0) if sd > 0 else 0.0


def features_from_file(path):
    """Feature vector for one participant file, or None if unreadable."""
    try:
        with h5py.File(path, "r") as f:
            return extract_features(f["eye_block"][:])
    except Exception as e:
        logging.warning(f"Failed to featurise {path}: {e}")
        return None


def build_training_set(data_dir, registry_path=None):
    """Collect ``(X, y, keys)`` from every QA-labeled participant on disk."""
    from pathlib import Path

    from deepmreye.storage import subject_path

    data_dir = Path(data_dir)
    registry_path = Path(registry_path or data_dir / "datasets.h5")

    labeled = []
    with h5py.File(registry_path, "r") as f:
        for ds_name in f.keys():
            for sub_id in f[ds_name].keys():
                approved = f[ds_name][sub_id].attrs.get("approved", -1)
                if approved in (0, 1, 2, 3):
                    labeled.append((ds_name, sub_id, int(approved)))

    X, y, keys = [], [], []
    for ds_name, sub_id, approved in labeled:
        path = subject_path(data_dir, ds_name, sub_id)
        if not path.exists():
            continue
        feats = features_from_file(path)
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
    counts = np.bincount(y, minlength=4)
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
