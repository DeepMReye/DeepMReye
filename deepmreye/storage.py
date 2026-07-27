"""On-disk layout for extracted eye blocks.

One HDF5 file per participant, foldered by dataset::

    <data_dir>/<dataset>/<subject>.h5
        eye_block  [X, Y, Z, T] float32   normalized, non-eye voxels zeroed
        labels     [T, 10, 2]   float32   gaze coordinates, only when known

This mirrors how BIDS/OpenNeuro lay out their source data, and it is what makes
the extraction parallelisable: every worker owns exactly one output file, so
there is no shared handle to serialise on and a crashed worker can only damage
its own subject. The older layout wrote every subject of a dataset into one
``<dataset>.h5``, which forced append-mode writes and made concurrent
extraction unsafe.

Labeled and unlabeled participants use the *same* container. ``labels`` is
simply absent when gaze is unknown, so the JEPA and probe loaders read one
format and the published artifact is internally consistent.
"""
import h5py
import numpy as np

# Windowed loaders read `window_size` TRs at a time (default 100). Chunking the
# full spatial extent against a slab of time means one window touches a handful
# of chunks instead of striding across the whole run.
TIME_CHUNK = 50

# Written into every file so a consumer can tell which extraction produced it
# without consulting the repo. Bump when the on-disk semantics change.
FORMAT_VERSION = 2


def subject_path(data_dir, dataset, subject):
    """Where a given participant's file lives."""
    from pathlib import Path
    return Path(data_dir) / dataset / f"{subject}.h5"


def _chunks_for(shape):
    """Chunk shape favouring contiguous reads of a time window."""
    x, y, z, t = shape
    return (x, y, z, min(TIME_CHUNK, t))


def write_subject(path, eye_block, labels=None, attrs=None, compression_opts=4):
    """Write one participant's eye block (and gaze labels, if any).

    ``eye_block`` is stored as float32 ``[X, Y, Z, T]``. Writes go to a
    temporary file that is renamed into place, so an interrupted job leaves
    either the previous file or nothing -- never a half-written file that
    later reads as truncated.
    """
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    eye_block = np.asarray(eye_block, dtype=np.float32)
    if eye_block.ndim != 4:
        raise ValueError(f"eye_block must be 4D [X, Y, Z, T], got shape {eye_block.shape}")

    if labels is not None:
        labels = np.asarray(labels, dtype=np.float32)
        if labels.shape[0] != eye_block.shape[-1]:
            raise ValueError(
                f"labels/eye_block length mismatch: {labels.shape[0]} labels "
                f"vs {eye_block.shape[-1]} TRs"
            )

    tmp = path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset(
            "eye_block",
            data=eye_block,
            chunks=_chunks_for(eye_block.shape),
            compression="gzip",
            compression_opts=compression_opts,
        )
        if labels is not None:
            f.create_dataset(
                "labels",
                data=labels,
                chunks=(min(TIME_CHUNK, labels.shape[0]),) + labels.shape[1:],
                compression="gzip",
                compression_opts=compression_opts,
            )
        f.attrs["format_version"] = FORMAT_VERSION
        f.attrs["n_trs"] = int(eye_block.shape[-1])
        f.attrs["has_labels"] = labels is not None
        for key, value in (attrs or {}).items():
            if value is not None:
                f.attrs[key] = value

    tmp.replace(path)
    return path


def read_subject(path, start=None, end=None, with_labels=True):
    """Read a participant, optionally only the TR window ``[start, end)``."""
    sl = slice(start, end)
    with h5py.File(path, "r") as f:
        eye_block = f["eye_block"][..., sl]
        labels = None
        if with_labels and "labels" in f:
            labels = f["labels"][sl]
        attrs = dict(f.attrs)
    return eye_block, labels, attrs


def iter_subjects(data_dir):
    """Yield ``(dataset, subject, path)`` for every participant file on disk."""
    from pathlib import Path

    root = Path(data_dir)
    if not root.exists():
        return
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for sub_file in sorted(ds_dir.glob("*.h5")):
            yield ds_dir.name, sub_file.stem, sub_file


def is_intact(path):
    """Whether a participant file is readable and carries an eye block.

    Uploads and interrupted writes leave truncated HDF5 files that only fail on
    open, so the published artifact gets checked with this rather than by
    trusting the directory listing.
    """
    try:
        with h5py.File(path, "r") as f:
            if "eye_block" not in f:
                return False
            f["eye_block"].shape
            return True
    except Exception:
        return False
