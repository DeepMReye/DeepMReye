"""Versionable, human-readable backup of manual QA labels.

The authoritative labels live as ``approved`` attributes inside
``data/datasets.h5`` (a mutable binary file). This module mirrors every
labeling action into an append-only ``data/labels.csv`` so the effort is
recoverable and git-trackable even if the HDF5 registry is deleted, corrupted,
or rebuilt from scratch.

CSV schema (one row per labeling event, newest rows win on restore):

    timestamp,dataset,scope,subject,label

- ``scope="dataset"`` with an empty ``subject``: a whole-dataset action, i.e.
  the skip flag (label ``-99``).
- ``scope="subject"``: a per-subject label (1 eyes, 0 no eyes/bad transform,
  2 no eyes/good transform).
"""
import csv
from datetime import datetime, timezone
from pathlib import Path

import h5py

CSV_FIELDS = ["timestamp", "dataset", "scope", "subject", "label"]


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
