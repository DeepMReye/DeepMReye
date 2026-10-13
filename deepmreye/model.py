"""The decoder: a frozen basis, temporal context, and a linear readout.

A participant's eye block ``[47, 29, 18, T]`` goes through three fixed steps::

    voxels  [T, 14236]   the eye-mask voxels of every volume
    z       [T, 32]      the bilateral canonical coordinates (basis.project)
    x       [T, 96]      z at the volume before, the volume itself and the one after
    gaze    [T, 10, 2]   a ridge regression to ten gaze samples per volume, x and y

Only the last step ever sees gaze labels. The pretrained decoder's readout is
fitted on every gaze-labelled participant of the published corpus (337
participants, nine datasets).

**Units.** Gaze is z-scored per training dataset before the readout is fitted,
because the datasets' gaze scales differ by more than an order of magnitude
(screen size, viewing distance, task). Predictions therefore come out in those
standard units -- x grows to the right, y grows *downwards* (screen coordinates)
-- and say where a participant looks relative to the centre of their own gaze
distribution. Degrees need a gain and offset per axis, which :meth:`Decoder.calibrate`
fits from a few participants with an eye tracker.

**Confidence.** The readout is linear in the average of the two orbits'
coordinates, so applying it to each orbit alone gives two decodings that average
to the real one. How well they agree (:meth:`Decoder.confidence`) ranks a
dataset's participants by how well they decode, without labels.
"""
import json
import os
import warnings
from pathlib import Path

import numpy as np

from deepmreye import metrics
from deepmreye.basis import load_basis, orbit_projections

DEFAULT_K = 32
DEFAULT_LAGS = 1
ALPHAS = np.logspace(-2, 4, 13)
SAMPLES_PER_TR = 10

# Where the pretrained decoder lives. Override with $DEEPMREYE_MODEL_REPO, or
# point $DEEPMREYE_MODEL at a local file to skip the Hub entirely.
PRETRAINED_REPO = os.environ.get("DEEPMREYE_MODEL_REPO", "DeepMReye/deepmreye-2")
PRETRAINED_FILE = "deepmreye2.npz"
# The full basis (256 canonical pairs) beside it, for training and evaluation.
PRETRAINED_BASIS = "basis.npz"

FORMAT = "deepmreye-decoder/1"
_BASIS_KEYS = ("mean", "left_index", "right_index", "left_weights", "right_weights")


def pretrained_basis(repo_id=None, revision=None):
    """Local path to the published basis; ``$DEEPMREYE_BASIS`` takes precedence."""
    local = os.environ.get("DEEPMREYE_BASIS")
    if local:
        return Path(local)
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id or PRETRAINED_REPO, PRETRAINED_BASIS, revision=revision))


def make_lags(z, lags=0):
    """Stack ``z`` at offsets ``-lags..+lags`` along the feature axis.

    Edge padding, not zero padding: a run's first and last TRs have no earlier
    or later neighbour, and padding them with zeros would inject a fake
    excursion to the corpus mean at exactly the rows a readout sees as extreme.
    """
    if lags == 0:
        return z
    parts = []
    for lag in range(-lags, lags + 1):
        if lag < 0:
            parts.append(np.pad(z[:lag], ((-lag, 0), (0, 0)), mode="edge"))
        elif lag > 0:
            parts.append(np.pad(z[lag:], ((0, lag), (0, 0)), mode="edge"))
        else:
            parts.append(z)
    return np.concatenate(parts, axis=1)


def fit_ridge(parts, alphas=ALPHAS, readout=None):
    """The readout on every finite row of ``parts``, gaze z-scored per dataset.

    ``parts`` is a list of ``(dataset, features [T, F], labels [T, 10, 2])``, one
    per participant. This is the only implementation of the readout fit: the
    evaluation protocol and :meth:`Decoder.fit` both call it.

    Three details are not free choices:

    - **Targets are z-scored per training dataset.** Without it a pooled ridge
      follows whichever dataset has the largest gaze variance.
    - **Datasets are concatenated in sorted order**, participants in the order
      given, so the fit does not depend on set iteration order.
    - **Every valid row is used.** A row is valid when all 20 targets and all
      features are finite.

    Returns the fitted ``sklearn.linear_model.RidgeCV`` (penalty chosen over
    ``alphas`` by efficient leave-one-out on the training rows), or, with a
    custom ``readout(x, y, groups) -> predict``, whatever that returns. ``None``
    when no dataset has ten usable rows.
    """
    from sklearn.linear_model import RidgeCV

    rows, ys, gs = [], [], []
    for ds in sorted({p[0] for p in parts}):
        members = [p for p in parts if p[0] == ds]
        y = np.concatenate([np.asarray(lab).reshape(len(lab), 2 * SAMPLES_PER_TR)
                            for _, _, lab in members])
        g = np.concatenate([np.full(len(f), len(gs) * 100000 + i)
                            for i, (_, f, _) in enumerate(members)])
        oks = [np.isfinite(f).all(axis=1) for _, f, _ in members]
        ok = np.isfinite(y).all(axis=1) & np.concatenate(oks)
        if ok.sum() < 10:
            continue
        sd = y[ok].std(axis=0)
        sd[sd < 1e-9] = 1.0
        ys.append((y[ok] - y[ok].mean(axis=0)) / sd)
        gs.append(g[ok])
        bounds = np.cumsum([0] + [len(o) for o in oks])
        rows += [(f, ok[a:b]) for (_, f, _), a, b in zip(members, bounds[:-1], bounds[1:])]
    if not ys:
        return None

    # One preallocated design matrix: a voxel-sized feature set does not fit
    # three copies of itself in memory.
    y_all = np.concatenate(ys)
    x_all = np.empty((len(y_all), rows[0][0].shape[1]))
    pos = 0
    for f, ok in rows:
        n = int(ok.sum())
        x_all[pos:pos + n] = f[ok]
        pos += n
    if readout is None:
        return RidgeCV(alphas=alphas).fit(x_all, y_all)
    return readout(x_all, y_all, np.concatenate(gs))


class Decoder:
    """Gaze from an eye block: ``Decoder.pretrained().predict(block)``.

    Parameters
    ----------
    basis : dict
        The lr-cca arrays (``mean``, ``left_index``, ``right_index``,
        ``left_weights``, ``right_weights``) and the boolean eye ``mask``
        ``[47, 29, 18]``, as :func:`deepmreye.basis.load_basis` returns them.
    k : int
        Canonical pairs used (32).
    lags : int
        Volumes of context either side (1).
    """

    def __init__(self, basis, k=DEFAULT_K, lags=DEFAULT_LAGS, coef=None, intercept=None,
                 gain=None, offset=None, meta=None):
        self.mask = np.asarray(basis["mask"], dtype=bool)
        self.k = int(min(k, basis["left_weights"].shape[1]))
        self.lags = int(lags)
        self.basis = {key: np.asarray(basis[key]) for key in _BASIS_KEYS}
        self.basis["left_weights"] = self.basis["left_weights"][:, :self.k]
        self.basis["right_weights"] = self.basis["right_weights"][:, :self.k]
        self.coef = None if coef is None else np.asarray(coef, dtype=np.float64)
        self.intercept = None if intercept is None else np.asarray(intercept, dtype=np.float64)
        self.gain = None if gain is None else np.asarray(gain, dtype=np.float64)
        self.offset = None if offset is None else np.asarray(offset, dtype=np.float64)
        self.meta = dict(meta or {})

    # ------------------------------------------------------------------ #
    # Construction and persistence
    # ------------------------------------------------------------------ #

    @classmethod
    def from_basis(cls, path, k=DEFAULT_K, lags=DEFAULT_LAGS):
        """An untrained decoder on a basis file written by :func:`deepmreye.basis.fit_corpus`."""
        mask, bases, meta = load_basis(path)
        if "lr-cca" not in bases:
            raise ValueError(f"{path} holds no lr-cca basis (found {sorted(bases)})")
        basis_meta = {key: meta[key] for key in ("n_subjects", "n_trs", "datasets", "fitted")
                      if key in meta}
        return cls({**bases["lr-cca"], "mask": mask}, k=k, lags=lags,
                   meta={"basis": basis_meta})

    @classmethod
    def load(cls, path):
        """A decoder saved by :meth:`save`."""
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} file (format {meta.get('format')!r})")
        basis = {key: z[key] for key in _BASIS_KEYS}
        basis["mask"] = z["mask"]
        get = lambda key: z[key] if key in z.files else None  # noqa: E731
        return cls(basis, k=meta["k"], lags=meta["lags"], coef=get("coef"),
                   intercept=get("intercept"), gain=get("gain"), offset=get("offset"),
                   meta=meta)

    @classmethod
    def pretrained(cls, repo_id=None, revision=None, filename=PRETRAINED_FILE):
        """The released decoder: the basis fitted on 2,060 unlabelled participants
        and a readout fitted on all 337 gaze-labelled ones.

        ``$DEEPMREYE_MODEL`` (a local file) takes precedence over the Hub. The
        download is a few megabytes and is cached by ``huggingface_hub``.
        """
        local = os.environ.get("DEEPMREYE_MODEL")
        if local:
            return cls.load(local)
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id or PRETRAINED_REPO, filename, revision=revision)
        return cls.load(path)

    def save(self, path):
        """Write the decoder (basis, readout and calibration) to one ``.npz``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {**self.meta, "format": FORMAT, "k": self.k, "lags": self.lags,
                "units": self.units}
        arrays = {key: value for key, value in self.basis.items()}
        arrays["mask"] = self.mask
        for key in ("coef", "intercept", "gain", "offset"):
            value = getattr(self, key)
            if value is not None:
                arrays[key] = value
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
        return path

    # ------------------------------------------------------------------ #
    # Features
    # ------------------------------------------------------------------ #

    def voxels(self, block):
        """``[X, Y, Z, T]`` eye block -> ``[T, 14236]`` eye-mask voxels."""
        block = np.asarray(block)
        if block.ndim != 4 or block.shape[:3] != self.mask.shape:
            raise ValueError(f"expected an eye block of shape {self.mask.shape + ('T',)}, "
                             f"got {block.shape}; run deepmreye.preprocess first")
        rows = block.reshape(-1, block.shape[-1]).T[:, self.mask.reshape(-1)]
        covered = int((np.abs(rows[: min(4, len(rows))]).sum(axis=0) > 0).sum())
        if covered < rows.shape[1]:
            warnings.warn(f"only {covered} of {rows.shape[1]} eye-mask voxels carry signal: "
                          "the eyes are not fully inside the field of view or the registration "
                          "failed, and gaze may decode poorly", stacklevel=3)
        return rows.astype(np.float64)

    def orbits(self, block):
        """The two orbits' canonical coordinates, ``([T, k], [T, k])``."""
        return orbit_projections(self.voxels(block), self.basis, k=self.k)

    def features(self, block):
        """The readout's input, ``[T, k * (2 * lags + 1)]``."""
        zl, zr = self.orbits(block)
        return make_lags(0.5 * (zl + zr), self.lags)

    # ------------------------------------------------------------------ #
    # Readout
    # ------------------------------------------------------------------ #

    @property
    def is_fitted(self):
        return self.coef is not None

    @property
    def units(self):
        return self.meta.get("calibrated_units", "deg") if self.gain is not None else "z"

    def fit(self, blocks, labels, datasets=None, alphas=ALPHAS):
        """Fit the readout on labelled eye blocks.

        ``blocks`` and ``labels`` are sequences, one entry per run: ``[47, 29,
        18, T]`` blocks and ``[T, 10, 2]`` gaze (NaN where unknown). ``datasets``
        names each run's dataset; gaze is z-scored within each before pooling,
        so pass it whenever the runs come from more than one setup. Clears any
        calibration.
        """
        return self.fit_features([self.features(b) for b in blocks], labels, datasets, alphas)

    def fit_features(self, features, labels, datasets=None, alphas=ALPHAS):
        """:meth:`fit` on precomputed :meth:`features`."""
        datasets = ["data"] * len(features) if datasets is None else list(datasets)
        if not len(features) == len(labels) == len(datasets):
            raise ValueError("features, labels and datasets must have one entry per run")
        parts = []
        for ds, f, y in zip(datasets, features, labels):
            n = min(len(f), len(y))
            parts.append((str(ds), np.asarray(f, dtype=np.float64)[:n], np.asarray(y)[:n]))
        model = fit_ridge(parts, alphas)
        if model is None:
            raise ValueError("no dataset has ten volumes with finite gaze on all 20 samples")
        self.coef, self.intercept = np.asarray(model.coef_), np.asarray(model.intercept_)
        self.gain = self.offset = None
        self.meta.update({"alpha": float(model.alpha_),
                          "trained_on": sorted(set(map(str, datasets))),
                          "n_runs": len(parts)})
        self.meta.pop("calibrated_units", None)
        return self

    def _readout(self, x):
        if not self.is_fitted:
            raise RuntimeError("the readout is not fitted; use Decoder.pretrained() or fit()")
        return (x @ self.coef.T + self.intercept).reshape(len(x), SAMPLES_PER_TR, 2)

    def _calibrated(self, gaze, calibrated):
        if calibrated and self.gain is not None:
            return gaze * self.gain + self.offset
        return gaze

    def predict(self, block, calibrated=True):
        """Gaze ``[T, 10, 2]``: ten samples per volume, x then y.

        In the training datasets' standard units (``units == "z"``) unless the
        decoder has been calibrated, then in the calibration labels' units.
        """
        return self._calibrated(self._readout(self.features(block)), calibrated)

    def predict_orbits(self, block, calibrated=True):
        """The readout applied to each orbit alone: ``(left, right)``, each ``[T, 10, 2]``."""
        zl, zr = self.orbits(block)
        return tuple(self._calibrated(self._readout(make_lags(z, self.lags)), calibrated)
                     for z in (zl, zr))

    def confidence(self, block):
        """Agreement of the two orbits' decodings (Pearson r, mean over axes).

        A per-run confidence that needs no labels: within a dataset it flags
        the runs that decode worst. It does not compare datasets, and it cannot
        flag single volumes.
        """
        left, right = self.predict_orbits(block, calibrated=False)
        return metrics.orbit_agreement(left, right)

    def calibrate(self, blocks, labels, units="deg"):
        """Fit one gain and one offset per axis mapping predictions onto ``labels``.

        A few runs with an eye tracker from the study being decoded are enough;
        the readout itself is unchanged. ``units`` only labels the result.
        """
        preds, trues = [], []
        for block, y in zip(blocks, labels):
            p = self.predict(block, calibrated=False)
            n = min(len(p), len(y))
            preds.append(p[:n].reshape(-1, 2))
            trues.append(np.asarray(y, dtype=np.float64)[:n].reshape(-1, 2))
        self.gain, self.offset = metrics.fit_affine(np.concatenate(preds), np.concatenate(trues))
        self.meta["calibrated_units"] = units
        return self

    def __repr__(self):
        state = "fitted" if self.is_fitted else "untrained"
        return f"Decoder(k={self.k}, lags={self.lags}, {state}, units={self.units!r})"
