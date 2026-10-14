"""A small synthetic world in which every README recipe can run end to end.

The eye block is shrunk to ``[8, 3, 2]`` (48 voxels, the orbits split at
x = 4), and gaze drives both orbits through a fixed "anatomy" plus noise local
to each orbit -- the generative model the method assumes. Everything else is
the real code path: the CLI, the basis fit, the readout, the files on disk. Only
ANTs registration is replaced (by a lookup of the run's block), because it needs
a real head; ``test_integration.py`` runs it on one.

Tests marked ``network`` reach the HuggingFace Hub and run only with
``DEEPMREYE_NETWORK_TESTS=1``.
"""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SHAPE = (8, 3, 2)
N_VOX = int(np.prod(SHAPE))
SPLIT_X = 4
TR = 2.0


def pytest_collection_modifyitems(config, items):
    if os.environ.get("DEEPMREYE_NETWORK_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="reaches the Hub; set DEEPMREYE_NETWORK_TESTS=1")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


def _gaze(rng, t):
    """Smooth gaze, ten samples per volume: ``[t, 10, 2]``, unit variance."""
    walk = np.cumsum(rng.normal(size=(t * 10, 2)), axis=0)
    kernel = np.ones(25) / 25
    g = np.stack([np.convolve(walk[:, a], kernel, mode="same") for a in range(2)], axis=1)
    g = (g - g.mean(axis=0)) / g.std(axis=0)
    return g.reshape(t, 10, 2)


def _block(rng, gaze, anatomy):
    """Both orbits driven by the volume's mean gaze, plus noise local to each."""
    per_volume = gaze.mean(axis=1)
    t = len(per_volume)
    half = N_VOX // 2
    left = per_volume @ anatomy["left"] + 0.3 * rng.normal(size=(t, half))
    right = per_volume @ anatomy["right"] + 0.3 * rng.normal(size=(t, half))
    # Flat C order over (x, y, z): the first half of the voxels is x < SPLIT_X.
    return np.concatenate([left, right], axis=1).T.reshape(SHAPE + (t,)).astype(np.float32)


@pytest.fixture
def small_world(monkeypatch):
    """The real module constants, resized to the synthetic eye block."""
    from deepmreye import basis

    monkeypatch.setattr(basis, "FULL_MASK_VOXELS", N_VOX)
    monkeypatch.setattr(basis, "LR_SPLIT_X", SPLIT_X)


@pytest.fixture(scope="session")
def world(tmp_path_factory):
    """Files on disk, the way a user would have them:

    - ``corpus/``: 12 unlabelled datasets of 3 participants each, and the three
      "published" labelled datasets ``dsL01..03`` (4 participants, each dataset
      in its own units);
    - ``mystudy/``: 6 participants of a new study, gaze in "degrees" (x2.5 +1);
    - ``other/dsX_lab/``: a labelled dataset from another setup;
    - ``raw/``: stand-ins for NIfTI runs, whose blocks ``extract_eye_block``
      returns;
    - ``release/``: the "pretrained" basis and decoder, fitted on ``corpus/``.
    """
    from deepmreye import basis
    from deepmreye.data import write_subject
    from deepmreye.model import Decoder

    root = tmp_path_factory.mktemp("world")
    rng = np.random.default_rng(0)
    anatomy = {"left": rng.normal(size=(2, N_VOX // 2)), "right": rng.normal(size=(2, N_VOX // 2))}
    truth, raw = {}, {}

    def run(name, t, gain=1.0, offset=0.0, labelled=True, folder=None):
        g = _gaze(rng, t)
        block = _block(rng, g, anatomy)
        labels = gain * g + offset
        truth[name] = labels
        if folder is not None:
            write_subject(root / folder / f"{name.split('/')[-1]}.h5", block,
                          labels=labels if labelled else None, attrs={"repetition_time": TR})
        return block

    for d in range(12):
        for s in range(3):
            run(f"ds{d:06d}/sub-{s:02d}", 60, labelled=False, folder=f"corpus/ds{d:06d}")
    for d, gain in zip(range(1, 4), (1.0, 5.0, 20.0)):
        for s in range(4):
            run(f"dsL0{d}/sub-{s:02d}", 80, gain=gain, folder=f"corpus/dsL0{d}_task")
    for s in range(1, 7):
        name = f"mystudy/sub-{s:02d}"
        raw[f"sub-{s:02d}_bold"] = run(name, 80, gain=2.5, offset=1.0, folder="mystudy")
    for s in range(3):
        run(f"dsX_lab/sub-{s:02d}", 80, gain=3.0, folder="other/dsX_lab")
    (root / "raw").mkdir()
    for name in raw:
        (root / "raw" / f"{name}.nii.gz").write_bytes(b"")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(basis, "FULL_MASK_VOXELS", N_VOX)
        mp.setattr(basis, "LR_SPLIT_X", SPLIT_X)
        basis_path = basis.fit_corpus(root / "corpus", root / "release" / "basis.npz", k=16,
                                      progress=None)
        from deepmreye.__main__ import _labelled_runs

        model = Decoder.from_basis(basis_path)
        feats, labels, datasets = [], [], []
        for dataset, _s, block, y in _labelled_runs(root / "corpus", "dsL*"):
            feats.append(model.features(block))
            labels.append(y)
            datasets.append(dataset)
        model.fit_features(feats, labels, datasets)
        model.meta["version"] = "test"
        decoder_path = model.save(root / "release" / "decoder.npz")
    return SimpleNamespace(root=root, truth=truth, raw=raw, basis=basis_path,
                           decoder=decoder_path)


@pytest.fixture
def env(world, small_world, monkeypatch, tmp_path):
    """The world as the CLI sees it: published data, basis and decoder found
    through the environment, registration replaced by a lookup, cwd a scratch dir."""
    from deepmreye import preprocess, thumbnail

    monkeypatch.setenv("DEEPMREYE_DATA", str(world.root / "corpus"))
    monkeypatch.setenv("DEEPMREYE_BASIS", str(world.basis))
    monkeypatch.setenv("DEEPMREYE_MODEL", str(world.decoder))
    monkeypatch.chdir(tmp_path)

    def fake_extract(func, masks=None, transforms=None, qa_png=None, tr=None):
        block = world.raw[Path(str(func)).name.replace(".nii.gz", "")]
        if qa_png is not None:
            thumbnail.save(thumbnail.from_block(block), qa_png)
        return block, TR if tr is None else float(tr)

    monkeypatch.setattr(preprocess, "extract_eye_block", fake_extract)
    return world
