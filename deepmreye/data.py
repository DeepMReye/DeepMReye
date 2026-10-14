"""The eye-block corpus: where it is, and how a participant is stored.

**Layout.** One HDF5 file per participant (or run), foldered by dataset::

    <data_dir>/<dataset>/<subject>.h5
        eye_block  [47, 29, 18, T] float32   normalised, non-eye voxels zeroed
        labels     [T, 10, 2]      float32   gaze x/y, only when known

``labels`` holds ten gaze samples per volume: sample ``j`` of volume ``t`` is the
mean gaze over ``[(t + j/10) * TR, (t + (j+1)/10) * TR)``, NaN where the eye
tracker recorded nothing. Gaze-labelled datasets are the ``dsL*`` folders.

**The published corpus** is the HuggingFace dataset ``DeepMReye/eyeballs``:
eye blocks of 3,684 participants in 921 dataset folders (77 GB), drawn from
OpenNeuro. Nine of the folders, ``dsL*`` (337 participants, 20 GB), carry gaze.
Resolution order for a command that needs data:

1. an explicit ``data_dir`` (``--data-dir``);
2. else ``$DEEPMREYE_DATA``;
3. else ``./data`` if it holds a corpus;
4. else the local cache (``$DEEPMREYE_CACHE``, default ``~/.cache/deepmreye``),
   downloading what the command needs on first use.
"""
import os
from pathlib import Path

import h5py
import numpy as np

# h5py takes an exclusive lock by default, which fails on the network
# filesystems a corpus often lives on.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

DEFAULT_REPO = os.environ.get("DEEPMREYE_HF_REPO", "DeepMReye/eyeballs")

# What each kind of command downloads.
LABELLED = ["index.parquet", "dsL*/*.h5"]
CORPUS = ["index.parquet", "*/*.h5"]

# Chunk the full spatial extent against a slab of time, so a window of volumes
# touches a handful of chunks.
TIME_CHUNK = 50

# Written into every file, so a consumer can tell which layout produced it.
FORMAT_VERSION = 2


# --------------------------------------------------------------------------- #
# Finding the corpus
# --------------------------------------------------------------------------- #

def _looks_like_corpus(path):
    path = Path(path)
    return path.is_dir() and ((path / "index.parquet").exists() or any(path.glob("*/*.h5")))


def cache_dir():
    """Where a downloaded corpus is kept."""
    base = os.environ.get("DEEPMREYE_CACHE")
    if base:
        return Path(base).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg).expanduser() if xdg else Path.home() / ".cache") / "deepmreye"


def resolve(data_dir=None, patterns=None, download=True, repo_id=None, quiet=False):
    """A local directory holding the corpus, downloading ``patterns`` if needed.

    An explicit ``data_dir`` is used as given and never downloaded into. Raises
    ``FileNotFoundError`` when nothing is found and ``download`` is off.
    """
    if data_dir is not None:
        path = Path(data_dir).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"no such directory: {path}")
        return path

    env = os.environ.get("DEEPMREYE_DATA")
    for candidate in ([Path(env).expanduser()] if env else []) + [Path.cwd() / "data"]:
        if _looks_like_corpus(candidate):
            if not quiet:
                print(f"[data] using {candidate}")
            return candidate

    target = cache_dir()
    if _looks_like_corpus(target) and all(any(target.glob(p)) for p in patterns or ["*/*.h5"]):
        return target
    if download:
        return fetch(patterns, target=target, repo_id=repo_id, quiet=quiet)
    if _looks_like_corpus(target):
        return target
    raise FileNotFoundError(
        f"no corpus found (looked in $DEEPMREYE_DATA, ./data and {target}); pass "
        f"--data-dir, set $DEEPMREYE_DATA, or allow downloading")


def fetch(patterns=None, target=None, repo_id=None, quiet=False):
    """Download ``patterns`` of the published corpus into ``target``.

    Files already present are not downloaded again, so this is also how a cache
    is topped up. ``patterns=None`` takes everything.
    """
    from huggingface_hub import snapshot_download

    repo_id = repo_id or DEFAULT_REPO
    target = Path(target or cache_dir())
    target.mkdir(parents=True, exist_ok=True)
    if not quiet:
        what = "everything" if not patterns else ", ".join(patterns)
        print(f"[data] {repo_id}: {what} -> {target}", flush=True)
    try:
        snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(target),
                          allow_patterns=patterns)
    except Exception as e:
        if type(e).__name__ in ("GatedRepoError", "RepositoryNotFoundError"):
            # The published corpus is open; this is a fork, a private copy, or a gate.
            raise RuntimeError(
                f"no access to {repo_id}: if it is gated, accept its terms at "
                f"https://huggingface.co/datasets/{repo_id} and log in with `hf auth login`") from e
        raise
    return target


# --------------------------------------------------------------------------- #
# One participant on disk
# --------------------------------------------------------------------------- #

def subject_path(data_dir, dataset, subject):
    """Where a given participant's file lives."""
    return Path(data_dir) / dataset / f"{subject}.h5"


def _chunks_for(shape):
    x, y, z, t = shape
    return (x, y, z, min(TIME_CHUNK, t))


def write_subject(path, eye_block, labels=None, attrs=None, compression_opts=4):
    """Write one participant's eye block (and gaze labels, if any).

    Writes go to a temporary file that is renamed into place, so an interrupted
    job leaves either the previous file or nothing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    eye_block = np.asarray(eye_block, dtype=np.float32)
    if eye_block.ndim != 4:
        raise ValueError(f"eye_block must be 4D [X, Y, Z, T], got shape {eye_block.shape}")
    if labels is not None:
        labels = np.asarray(labels, dtype=np.float32)
        if labels.shape[0] != eye_block.shape[-1]:
            raise ValueError(f"labels/eye_block length mismatch: {labels.shape[0]} labels "
                             f"vs {eye_block.shape[-1]} volumes")

    tmp = path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("eye_block", data=eye_block, chunks=_chunks_for(eye_block.shape),
                         compression="gzip", compression_opts=compression_opts)
        if labels is not None:
            f.create_dataset("labels", data=labels,
                             chunks=(min(TIME_CHUNK, labels.shape[0]),) + labels.shape[1:],
                             compression="gzip", compression_opts=compression_opts)
        f.attrs["format_version"] = FORMAT_VERSION
        f.attrs["n_trs"] = int(eye_block.shape[-1])
        f.attrs["has_labels"] = labels is not None
        for key, value in (attrs or {}).items():
            if value is not None:
                f.attrs[key] = value
    tmp.replace(path)
    return path


def read_subject(path, start=None, end=None, with_labels=True):
    """``(eye_block, labels or None, attrs)``, optionally only volumes ``[start, end)``."""
    sl = slice(start, end)
    with h5py.File(path, "r") as f:
        eye_block = f["eye_block"][..., sl]
        labels = f["labels"][sl] if with_labels and "labels" in f else None
        attrs = dict(f.attrs)
    return eye_block, labels, attrs


def iter_subjects(data_dir, pattern="*"):
    """Yield ``(dataset, subject, path)`` for every participant file under ``data_dir``."""
    root = Path(data_dir)
    if not root.exists():
        return
    for ds_dir in sorted(p for p in root.glob(pattern) if p.is_dir()):
        for sub_file in sorted(ds_dir.glob("*.h5")):
            yield ds_dir.name, sub_file.stem, sub_file


def is_intact(path):
    """Whether a participant file is readable and carries an eye block."""
    try:
        with h5py.File(path, "r") as f:
            return "eye_block" in f and len(f["eye_block"].shape) == 4
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Gaze labels from an eye tracker
# --------------------------------------------------------------------------- #

def bin_gaze(times, x, y, n_volumes, tr, n_sub=10, min_samples=1):
    """A raw gaze stream -> labels ``[n_volumes, n_sub, 2]``.

    ``times`` are seconds from the onset of the first volume (subtract the
    scanner trigger's time first). Sub-bin ``j`` of volume ``t`` covers
    ``[(t + j/n_sub) * tr, (t + (j+1)/n_sub) * tr)`` and holds the mean of the
    samples in it, so the mean over a volume's ``n_sub`` bins is its mean gaze.
    Bins with fewer than ``min_samples`` finite samples are NaN; pass blinks and
    track loss as NaN and they are skipped. Units are whatever ``x`` and ``y``
    are in (degrees of visual angle, for calibration).
    """
    times = np.asarray(times, dtype=np.float64)
    xy = np.stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)], axis=1)
    n_bins = int(n_volumes) * int(n_sub)
    out = np.full((n_bins, 2), np.nan)
    # floor, so a sample exactly on a bin edge belongs to the bin it opens.
    idx = np.floor(times / (float(tr) / n_sub)).astype(np.int64)
    keep = np.isfinite(times) & (idx >= 0) & (idx < n_bins)
    for col in range(2):
        ok = keep & np.isfinite(xy[:, col])
        counts = np.bincount(idx[ok], minlength=n_bins)
        sums = np.bincount(idx[ok], weights=xy[ok, col], minlength=n_bins)
        enough = counts >= min_samples
        out[enough, col] = sums[enough] / counts[enough]
    return out.reshape(int(n_volumes), int(n_sub), 2).astype(np.float32)


def read_gaze_tsv(path):
    """``(times, x, y)`` from a tab- or comma-separated file with a header row
    naming ``time``, ``x`` and ``y`` columns (``onset`` is accepted for ``time``)."""
    text = Path(path).read_text().splitlines()
    sep = "\t" if "\t" in text[0] else ","
    header = [h.strip().lower() for h in text[0].split(sep)]
    col = {name: header.index(name) for name in header}
    t_key = "time" if "time" in col else "onset"
    if t_key not in col or "x" not in col or "y" not in col:
        raise ValueError(f"{path}: need columns time (or onset), x and y; found {header}")
    rows = np.genfromtxt(text[1:], delimiter=sep, dtype=np.float64, missing_values=("", "n/a", "NaN"),
                         filling_values=np.nan)
    rows = np.atleast_2d(rows)
    return rows[:, col[t_key]], rows[:, col["x"]], rows[:, col["y"]]
