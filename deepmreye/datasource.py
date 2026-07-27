"""Find the corpus, wherever this machine happens to keep it.

The same commands run on the cluster, where the data sits on scratch and is
produced locally, and on a laptop, where it has to come down from HuggingFace.
Rather than making every command take a path, resolution goes through here:

1. an explicit ``--data-dir`` (or ``data_dir=`` argument) always wins;
2. else ``$DEEPMREYE_DATA``;
3. else ``./data`` if it already looks like a corpus;
4. else the local HuggingFace cache, downloading the repo on first use.

So on the cluster you set ``DEEPMREYE_DATA`` once and nothing ever touches the
network; on a laptop you set nothing and the first command pulls what it needs.
"""
import os
from pathlib import Path

# Where the published corpus lives. Override with $DEEPMREYE_HF_REPO to work
# against a fork or a private staging copy.
DEFAULT_REPO = os.environ.get("DEEPMREYE_HF_REPO", "DeepMReye/eyeballs")

# A directory is "a corpus" if it has the registry; that is what every stage
# reads first, and it is small enough to be a cheap existence check.
REGISTRY_NAME = "datasets.h5"

# The small files every stage reads: labels, their backup, and the index.
# A few MB, against ~29 GB of blocks and ~8 GB of reports.
REGISTRY_FILES = ["datasets.h5", "labels.csv", "index.parquet"]

# What each stage actually needs, so a laptop is not made to download the whole
# corpus before it can do anything. Labeling reads reports, and those arrive one
# dataset at a time via `ensure_reports` as you reach them; training reads
# blocks and no reports at all. Stages absent from here get everything.
STAGE_PATTERNS = {
    "qa": REGISTRY_FILES,
    "train": REGISTRY_FILES + ["*/*.h5"],
}


def _looks_like_corpus(path):
    path = Path(path)
    return path.is_dir() and (path / REGISTRY_NAME).exists()


def local_candidates():
    """Paths searched before falling back to a download, in priority order."""
    out = []
    env = os.environ.get("DEEPMREYE_DATA")
    if env:
        out.append(Path(env).expanduser())
    out.append(Path.cwd() / "data")
    return out


def cache_dir():
    """Where a downloaded corpus is kept."""
    base = os.environ.get("DEEPMREYE_CACHE")
    if base:
        return Path(base).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "deepmreye"


def resolve(data_dir=None, repo_id=None, download=True, patterns=None, quiet=False):
    """Return a local directory holding the corpus, fetching it if needed.

    ``patterns`` restricts what is downloaded (``huggingface_hub`` glob syntax),
    which matters because the QA reports are larger than the data itself --
    labeling wants them, training does not.

    Raises ``FileNotFoundError`` if nothing is found and ``download`` is off.
    """
    if data_dir is not None:
        path = Path(data_dir).expanduser()
        # An explicit path is a statement of intent: create it rather than
        # silently downloading somewhere else.
        path.mkdir(parents=True, exist_ok=True)
        return path

    for candidate in local_candidates():
        if _looks_like_corpus(candidate):
            if not quiet:
                print(f"[data] using {candidate}")
            return candidate

    target = cache_dir()
    if _looks_like_corpus(target):
        if not quiet:
            print(f"[data] using cached corpus at {target}")
        # The cache is ours to keep complete, and stages ask for different
        # slices of it: labeling pulls only the registry, so a later `train`
        # would otherwise find a "corpus" with no blocks in it. Topping up is
        # cheap -- already-present files are skipped.
        if download and patterns:
            fetch(repo_id=repo_id, target=target, patterns=patterns, quiet=True)
        return target

    if not download:
        searched = ", ".join(str(p) for p in local_candidates())
        raise FileNotFoundError(
            f"No corpus found (looked in: {searched}, {target}). "
            f"Set $DEEPMREYE_DATA, pass --data-dir, or allow downloading."
        )

    return fetch(repo_id=repo_id, target=target, patterns=patterns, quiet=quiet)


def fetch(repo_id=None, target=None, patterns=None, quiet=False):
    """Download the corpus from HuggingFace into ``target``."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is needed to download the corpus. "
            "Install it with `uv pip install huggingface_hub`, or point "
            "$DEEPMREYE_DATA at an existing copy."
        ) from e

    repo_id = repo_id or DEFAULT_REPO
    target = Path(target or cache_dir())
    target.mkdir(parents=True, exist_ok=True)

    if not quiet:
        what = "everything" if not patterns else ", ".join(patterns)
        print(f"[data] downloading {what} from {repo_id} -> {target}")
        print("[data] first run only; later runs reuse this copy.")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target),
        allow_patterns=patterns,
    )
    return target


def ensure_reports(data_dir, datasets, repo_id=None, quiet=False):
    """Make sure the QA reports for ``datasets`` are present locally.

    Reports are ~5 MB per subject and the full set is larger than the eye
    blocks, so they are fetched per dataset as labeling reaches them instead of
    all at once. Datasets already on disk cost nothing.
    """
    data_dir = Path(data_dir)
    missing = [d for d in datasets if not any((data_dir / d).glob("*/*.html"))]
    if not missing:
        return []

    fetch(
        repo_id=repo_id,
        target=data_dir,
        patterns=[f"{d}/*/*.html" for d in missing],
        quiet=quiet,
    )
    return missing
