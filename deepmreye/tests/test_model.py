"""The decoder: features, readout, confidence, calibration, persistence.

The property that matters most is that the decoder and the evaluation protocol
compute the same thing -- one feature path, one readout fit -- so that the
published numbers describe the model a user downloads.
"""
import warnings

import numpy as np
import pytest

from deepmreye import evaluate
from deepmreye.__main__ import build_parser, write_gaze_tsv
from deepmreye.model import Decoder, make_lags


def _basis(seed=0, shape=(6, 2, 2), k=4):
    """A toy lr-cca basis on a fully covered 24-voxel mask, orbits split along x."""
    rng = np.random.default_rng(seed)
    mask = np.ones(shape, dtype=bool)
    d = int(mask.sum())
    half = d // 2
    return {"mask": mask, "mean": rng.normal(size=d),
            "left_index": np.arange(half), "right_index": np.arange(half, d),
            "left_weights": rng.normal(size=(half, k)),
            "right_weights": rng.normal(size=(d - half, k))}


def _runs(model, n=6, t=90, datasets=("dsA", "dsB"), seed=1):
    """Blocks whose gaze is a fixed linear map of the decoder's own features."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(model.k * (2 * model.lags + 1), 2))
    blocks, labels, names = [], [], []
    for i in range(n):
        block = rng.normal(size=model.mask.shape + (t,))
        y = model.features(block) @ w
        blocks.append(block)
        labels.append(np.repeat(y[:, None, :], 10, axis=1) * (1 + i % 2))
        names.append(datasets[i % len(datasets)])
    return blocks, labels, names


def test_features_are_the_evaluation_features():
    """The decoder's features of a block equal the protocol's features of its
    cached record: the published numbers are about this model."""
    model = Decoder(_basis(), k=3, lags=1)
    block = np.random.default_rng(2).normal(size=model.mask.shape + (40,))
    zl, zr = model.orbits(block)
    rec = {"z": np.stack([zl, zr], axis=1)}
    assert np.allclose(model.features(block), evaluate.features(3, 1)(rec))
    assert np.allclose(model.features(block), make_lags(0.5 * (zl + zr), 1))


def test_fit_recovers_a_linear_map_and_predicts_ten_samples_per_volume():
    model = Decoder(_basis(), k=4, lags=1)
    blocks, labels, names = _runs(model)
    model.fit(blocks, labels, names)
    pred = model.predict(blocks[0])
    assert pred.shape == (90, 10, 2)
    for axis in range(2):
        assert np.corrcoef(pred[:, 0, axis], labels[0][:, 0, axis])[0, 1] > 0.99
    assert model.units == "z" and model.meta["trained_on"] == ["dsA", "dsB"]


def test_orbit_decodings_average_to_the_prediction():
    model = Decoder(_basis(), k=4, lags=1)
    blocks, labels, names = _runs(model)
    model.fit(blocks, labels, names)
    left, right = model.predict_orbits(blocks[1])
    assert np.allclose(0.5 * (left + right), model.predict(blocks[1]))
    assert -1.0 <= model.confidence(blocks[1]) <= 1.0


def test_calibration_maps_onto_label_units():
    model = Decoder(_basis(), k=4, lags=1)
    blocks, labels, names = _runs(model)
    model.fit(blocks, labels, names)
    # Two runs of one dataset (dsA), whose gaze shares one scale.
    degrees = [3.0 * labels[i] + 1.5 for i in (0, 2)]
    model.calibrate([blocks[0], blocks[2]], degrees)
    assert model.units == "deg"
    err = np.abs(model.predict(blocks[4]) - (3.0 * labels[4] + 1.5)).mean()
    assert err < 0.05 * np.abs(3.0 * labels[4] + 1.5).mean()
    # Refitting the readout clears a calibration that no longer applies.
    model.fit(blocks, labels, names)
    assert model.units == "z" and model.gain is None


def test_save_load_round_trip(tmp_path):
    model = Decoder(_basis(), k=4, lags=1)
    blocks, labels, names = _runs(model)
    model.fit(blocks, labels, names).calibrate([blocks[0], blocks[2]], [labels[0], labels[2]],
                                               units="px")
    path = model.save(tmp_path / "m.npz")
    back = Decoder.load(path)
    assert (back.k, back.lags, back.units) == (4, 1, "px")
    assert np.allclose(back.predict(blocks[3]), model.predict(blocks[3]))


def test_load_refuses_a_file_that_is_not_a_decoder(tmp_path):
    path = tmp_path / "x.npz"
    np.savez(path, meta=np.array('{"format": "something-else"}'))
    with pytest.raises(ValueError, match="not a"):
        Decoder.load(path)


def test_untrained_decoder_refuses_to_predict():
    model = Decoder(_basis(), k=4)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(np.zeros(model.mask.shape + (5,)))


def test_wrong_block_shape_is_refused():
    model = Decoder(_basis(), k=4)
    with pytest.raises(ValueError, match="eye block"):
        model.features(np.zeros((3, 3, 3, 5)))


def test_uncovered_eyes_are_warned_about():
    model = Decoder(_basis(), k=4)
    block = np.random.default_rng(3).normal(size=model.mask.shape + (10,))
    block[0] = 0.0
    with pytest.warns(UserWarning, match="eye-mask voxels"):
        model.features(block)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.features(np.random.default_rng(4).normal(size=model.mask.shape + (10,)))


def test_k_is_capped_by_the_basis():
    assert Decoder(_basis(k=4), k=32).k == 4


def test_gaze_tsv_has_one_row_per_sample_with_onsets(tmp_path):
    gaze = np.arange(3 * 10 * 2, dtype=float).reshape(3, 10, 2)
    path = write_gaze_tsv(tmp_path / "g.tsv", gaze, tr=2.0)
    rows = path.read_text().strip().split("\n")
    assert rows[0] == "volume\tsample\tonset\tx\ty"
    assert len(rows) == 31
    vol, sample, onset, x, y = rows[12].split("\t")       # volume 1, sample 1
    assert (vol, sample, float(onset)) == ("1", "1", 2.2)
    assert (float(x), float(y)) == (22.0, 23.0)


def test_cli_parses_every_command():
    parser = build_parser()
    for argv in (["decode", "run.nii.gz"], ["preprocess", "run.nii.gz", "-o", "s.h5"],
                 ["train", "-o", "m.npz"], ["calibrate", "--data-dir", "d", "-o", "m.npz"],
                 ["evaluate"], ["fit-basis", "-o", "b.npz"], ["download"]):
        assert callable(parser.parse_args(argv).func)
