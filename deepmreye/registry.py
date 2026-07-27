"""Registry writes that survive parallel workers and a live labeling UI.

``datasets.h5`` holds the QA labels, and the labeling UI keeps it open in
append mode while you work. HDF5 permits exactly one writer per file, so an
extraction worker that also wrote the registry would either corrupt it or lose
the labels -- and the labels are the expensive, irreplaceable part.

So workers never touch ``datasets.h5``. Each one appends a JSON line to its own
sidecar under ``_pending/``, and :func:`merge_pending` folds those into the
registry later, when no UI is running. Writes are one-line-per-subject appends,
which are atomic enough at this size that a killed worker truncates at most its
last line.
"""
import json
import os
from pathlib import Path

import h5py
import numpy as np

PENDING_DIRNAME = "_pending"


def pending_dir(data_dir):
    return Path(data_dir) / PENDING_DIRNAME


def record(data_dir, dataset, subject, meta, worker_id=None):
    """Append one subject's extraction metadata to this worker's sidecar."""
    d = pending_dir(data_dir)
    d.mkdir(parents=True, exist_ok=True)

    if worker_id is None:
        worker_id = os.environ.get("SLURM_ARRAY_TASK_ID") or os.getpid()

    payload = {"dataset": dataset, "subject": subject}
    for key, value in meta.items():
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            value = value.item()
        payload[key] = value

    with open(d / f"worker_{worker_id}.jsonl", "a") as f:
        f.write(json.dumps(payload) + "\n")
        f.flush()


def read_pending(data_dir):
    """Every pending record on disk, tolerating a truncated final line."""
    d = pending_dir(data_dir)
    if not d.exists():
        return []

    records = []
    for path in sorted(d.glob("worker_*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # A worker killed mid-write leaves one partial line; the
                    # subject is simply re-extracted on the next pass.
                    continue
    return records


def merge_pending(data_dir, registry_path=None, clear=True):
    """Fold pending worker records into ``datasets.h5``.

    Never touches an ``approved`` attribute, so merging can't clobber QA work.
    Returns the number of subject records applied.
    """
    data_dir = Path(data_dir)
    registry_path = Path(registry_path or data_dir / "datasets.h5")
    records = read_pending(data_dir)
    if not records:
        return 0

    applied = 0
    with h5py.File(registry_path, "a") as f:
        for rec in records:
            ds_name = rec.pop("dataset")
            sub_id = rec.pop("subject")

            ds_grp = f[ds_name] if ds_name in f else f.create_group(ds_name)
            if "approved" not in ds_grp.attrs:
                ds_grp.attrs["approved"] = -1

            sub_grp = ds_grp[sub_id] if sub_id in ds_grp else ds_grp.create_group(sub_id)
            for key, value in rec.items():
                if isinstance(value, list):
                    value = np.asarray(value, dtype=np.float32)
                sub_grp.attrs[key] = value
            # Leave `approved` alone -- QA labels outrank anything a worker knows.
            if "approved" not in sub_grp.attrs:
                sub_grp.attrs["approved"] = -1
            applied += 1

    if clear:
        for path in pending_dir(data_dir).glob("worker_*.jsonl"):
            path.unlink()

    return applied


def ensure_dataset_entry(registry_path, ds_name, description=None, graphql=None):
    """Create a dataset group with its metadata if it isn't registered yet."""
    with h5py.File(registry_path, "a") as f:
        if ds_name in f:
            return False
        grp = f.create_group(ds_name)
        grp.attrs["approved"] = -1
        if description is not None:
            grp.attrs["dataset_description"] = description
        if graphql is not None:
            grp.attrs["graphql_metadata"] = graphql
        return True
