"""QA labels: the vocabulary, which datasets pass, and a versionable backup.

The authoritative labels live as ``approved`` attributes inside
``data/datasets.h5`` (a mutable binary file). This module mirrors every
labeling action into an append-only ``data/labels.csv`` so the effort is
recoverable and git-trackable even if the HDF5 registry is deleted, corrupted,
or rebuilt from scratch.

CSV schema (one row per labeling event, newest rows win on restore):

    timestamp,dataset,scope,subject,label

- ``scope="dataset"`` with an empty ``subject``: a whole-dataset action, i.e.
  the skip flag (label ``-99``).
- ``scope="subject"``: a per-subject label (the ``LBL_*`` vocabulary below).
"""
import csv
from datetime import datetime, timezone
from pathlib import Path

import h5py

CSV_FIELDS = ["timestamp", "dataset", "scope", "subject", "label"]

# QA label vocabulary, stored on each subject's ``approved`` attribute.
LBL_UNLABELED = -1
LBL_NO_EYES_BAD_TRANSFORM = 0
LBL_EYES = 1
LBL_NO_EYES_GOOD_TRANSFORM = 2
LBL_EYES_CUT = 3  # eyeballs visible but clipped by the bounding box
LBL_EYES_FAINT = 4  # eyeballs visible but faint
LBL_DATASET_SKIPPED = -99

# Labels that count as usable eye signal. `LBL_EYES_CUT` and `LBL_EYES_FAINT` are
# included: clipped or faint eyeballs still carry gaze information, and dropping
# those would cost whole datasets under the all-or-nothing rule. They are kept as
# distinct labels (rather than folded into `LBL_EYES`) so the corpus can be
# filtered on them later.
APPROVED_LABELS = (LBL_EYES, LBL_EYES_CUT, LBL_EYES_FAINT)


def labels_pass(labels):
    """The all-or-nothing rule: some labelled participant, and every one shows eyes.

    A single 'no eyes' participant drops the whole dataset, since the same
    scanner and protocol tend to fail the same way across participants.
    """
    labels = [int(x) for x in labels if int(x) != LBL_UNLABELED]
    return len(labels) > 0 and all(x in APPROVED_LABELS for x in labels)


def qa_status(h5_path):
    """``{dataset: "pass" | "fail" | "not labelled" | "skipped"}`` for every
    dataset in the registry, by :func:`labels_pass` on its subjects' labels."""
    out = {}
    with h5py.File(h5_path, "r") as f:
        for ds in f.keys():
            labels = [int(f[ds][s].attrs.get("approved", LBL_UNLABELED)) for s in f[ds].keys()]
            if f[ds].attrs.get("approved", LBL_UNLABELED) == LBL_DATASET_SKIPPED:
                out[ds] = "skipped"
            elif all(x == LBL_UNLABELED for x in labels):
                out[ds] = "not labelled"
            else:
                out[ds] = "pass" if labels_pass(labels) else "fail"
    return out


def _now():
    return datetime.now(timezone.utc).isoformat()


def append_label_events(csv_path, events):
    """Append labeling events to the CSV history, creating it with a header if new.

    ``events`` is an iterable of ``(dataset, scope, subject, label)`` tuples.
    ``subject`` may be None/"" for dataset-scope events.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    ts = _now()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_FIELDS)
        for dataset, scope, subject, label in events:
            writer.writerow([ts, dataset, scope, subject or "", int(label)])


def export_labels(h5_path, csv_path):
    """Write a full snapshot of the current labels in the registry to the CSV.

    Appends one event batch capturing every dataset-skip flag and every labeled
    subject currently in the HDF5. Returns the number of events written.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"Registry not found at {h5_path}")

    events = []
    with h5py.File(h5_path, "r") as f:
        for ds in f.keys():
            ds_grp = f[ds]
            if ds_grp.attrs.get("approved", -1) == -99:
                events.append((ds, "dataset", "", -99))
            for sub in ds_grp.keys():
                lbl = ds_grp[sub].attrs.get("approved", -1)
                if lbl != -1:
                    events.append((ds, "subject", sub, int(lbl)))

    if events:
        append_label_events(csv_path, events)
    return len(events)


def _latest_labels(csv_path):
    """Collapse the append-only history to the latest label per (dataset, scope, subject)."""
    latest = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["dataset"], row["scope"], row["subject"])
            # Rows are appended in chronological order, so the last one wins.
            latest[key] = row
    return latest


def restore_labels(h5_path, csv_path):
    """Replay the latest label per key from the CSV back into an existing registry.

    Only touches datasets/subjects that already exist in the HDF5 (the CSV is a
    label backup, not a data backup) and only sets the ``approved`` attribute.
    Returns ``(applied, skipped_missing)`` counts.
    """
    h5_path = Path(h5_path)
    csv_path = Path(csv_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"Registry not found at {h5_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Label backup not found at {csv_path}")

    applied = 0
    skipped_missing = 0
    with h5py.File(h5_path, "a") as f:
        for (dataset, scope, subject), row in _latest_labels(csv_path).items():
            label = int(row["label"])
            if dataset not in f:
                skipped_missing += 1
                continue
            if scope == "dataset":
                f[dataset].attrs["approved"] = label
                applied += 1
            elif scope == "subject":
                if subject in f[dataset]:
                    f[dataset][subject].attrs["approved"] = label
                    applied += 1
                else:
                    skipped_missing += 1
    return applied, skipped_missing
