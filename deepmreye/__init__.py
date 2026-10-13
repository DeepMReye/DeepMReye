"""Decode eye gaze from fMRI without an eye tracker.

    import deepmreye

    result = deepmreye.decode("sub-01_task-movie_bold.nii.gz")
    result["gaze"]          # [T, 10, 2]: ten gaze samples per volume, x then y
    result["confidence"]    # agreement of the two eyes' decodings, no labels needed

See README.md for training your own readout, calibrating to degrees, and
fitting the basis on your own corpus.
"""
import os

# h5py takes an exclusive lock by default, which fails on the network
# filesystems a corpus often lives on.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

try:
    from ._version import __version__
except ImportError:  # running from a source tree without a build
    __version__ = "unknown"

from deepmreye.model import Decoder  # noqa: E402

__all__ = ["Decoder", "decode", "extract_eye_block", "load_model", "__version__"]


def load_model(path=None):
    """The pretrained decoder, or one saved with :meth:`Decoder.save` at ``path``."""
    return Decoder.pretrained() if path is None else Decoder.load(path)


def extract_eye_block(func, **kwargs):
    """A functional run (NIfTI path or ``ants`` image) -> ``(eye_block, tr)``.

    See :func:`deepmreye.preprocess.extract_eye_block`.
    """
    from deepmreye.preprocess import extract_eye_block as _extract

    return _extract(func, **kwargs)


def decode(func, model=None, qa_png=None, tr=None):
    """Preprocess one run and decode its gaze.

    ``func`` is a 4D NIfTI path, an ``ants`` image, or an eye block already
    extracted (a ``[47, 29, 18, T]`` array or a ``.h5`` written by
    :func:`deepmreye.data.write_subject`). ``model`` is a :class:`Decoder`
    (default: the pretrained one). Returns ``{"gaze", "gaze_left", "gaze_right",
    "confidence", "tr", "units"}``.
    """
    import numpy as np

    model = model or Decoder.pretrained()
    if isinstance(func, np.ndarray):
        block = func
    elif str(func).endswith(".h5"):
        from deepmreye.data import read_subject

        block, _labels, attrs = read_subject(func, with_labels=False)
        tr = tr if tr is not None else attrs.get("repetition_time")
    else:
        block, tr = extract_eye_block(func, qa_png=qa_png, tr=tr)
    left, right = model.predict_orbits(block)
    return {"gaze": model.predict(block), "gaze_left": left, "gaze_right": right,
            "confidence": model.confidence(block),
            "tr": None if tr is None else float(tr), "units": model.units}
