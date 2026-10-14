"""Every recipe in the README, end to end through the CLI.

Each test is one README recipe run the way the README says, on the synthetic
world of ``conftest.py``, and checks what a user would check: the files exist,
the units are what the README promises, and the gaze is right. If a recipe in
the README changes, its test here changes with it.
"""
import json

import numpy as np
import pytest

import deepmreye
from deepmreye.__main__ import main
from deepmreye.data import CORPUS, LABELLED, read_subject
from deepmreye.model import Decoder

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def cli(*argv):
    main([str(a) for a in argv])


def _tsv(path):
    rows = np.genfromtxt(path, delimiter="\t", names=True)
    return rows, np.stack([rows["x"], rows["y"]], axis=1).reshape(-1, 10, 2)


def _r(pred, true):
    return np.mean([np.corrcoef(pred[..., a].ravel(), true[..., a].ravel())[0, 1] for a in range(2)])


# --- Quick start / recipe 1: decode with the pretrained decoder -----------------------


def test_recipe_1_decode_a_nifti_run(env, tmp_path):
    cli("decode", env.root / "raw" / "sub-05_bold.nii.gz", "-o", "gaze", "--qa", "--save-block")
    rows, gaze = _tsv("gaze/sub-05_bold_gaze.tsv")
    assert gaze.shape == (80, 10, 2)
    # volume, sample, onset: sample j of volume t starts at (t + j/10) TR.
    assert (rows["volume"][12], rows["sample"][12], rows["onset"][12]) == (1, 2, 2.4)
    assert _r(gaze, env.truth["mystudy/sub-05"]) > 0.8
    summary = json.loads(open("gaze/sub-05_bold_decode.json").read())
    assert summary["units"] == "z" and summary["volumes"] == 80 and summary["tr"] == 2.0
    assert summary["confidence"] > 0.5
    assert (tmp_path / "gaze" / "sub-05_bold_qa.png").stat().st_size > 0
    # --save-block keeps the eye block, and decoding it again gives the same gaze.
    block, _labels, attrs = read_subject("gaze/sub-05_bold.h5")
    assert block.shape[-1] == 80 and attrs["repetition_time"] == 2.0
    cli("decode", "gaze/sub-05_bold.h5", "-o", "again")
    assert np.allclose(_tsv("again/sub-05_bold_gaze.tsv")[1], gaze, atol=1e-4)


def test_quick_start_python_api(env):
    path = env.root / "mystudy" / "sub-05.h5"
    block = read_subject(path)[0]
    from_array = deepmreye.decode(block)
    from_file = deepmreye.decode(path)
    assert np.allclose(from_array["gaze"], from_file["gaze"])
    assert from_file["tr"] == 2.0 and from_file["units"] == "z"
    assert np.allclose(0.5 * (from_file["gaze_left"] + from_file["gaze_right"]), from_file["gaze"])
    assert np.allclose(deepmreye.load_model().predict(block), from_array["gaze"])
    assert np.allclose(deepmreye.load_model(env.decoder).predict(block), from_array["gaze"])


# --- preprocessing: the first step of every recipe that uses labels --------------------


def test_preprocess_bins_raw_eye_tracker_samples(env, tmp_path):
    t = np.arange(0, 80 * 2.0, 0.004)
    x, y = np.sin(t / 3), np.cos(t / 5)
    x[(t > 10) & (t < 10.5)] = np.nan                          # a blink
    np.savetxt("tracker.tsv", np.c_[t + 3.0, x, y], delimiter="\t", header="time\tx\ty",
               comments="")
    cli("preprocess", env.root / "raw" / "sub-01_bold.nii.gz", "-o", "mine/sub-01.h5",
        "--gaze", "tracker.tsv", "--gaze-offset", 3.0, "--qa")
    _block, labels, attrs = read_subject("mine/sub-01.h5")
    assert labels.shape == (80, 10, 2) and attrs["repetition_time"] == 2.0
    assert labels[0, 0, 0] == pytest.approx(np.sin(t[t < 0.2] / 3).mean(), abs=1e-3)
    assert np.isnan(labels[5, 1, 0]) and np.isfinite(labels[5, 1, 1])  # 10.2-10.4 s, x only
    assert (tmp_path / "mine" / "sub-01.png").exists()


def test_preprocess_takes_binned_labels_and_refuses_the_wrong_length(env):
    np.save("per_volume.npy", np.ones((80, 2), np.float32))
    cli("preprocess", env.root / "raw" / "sub-01_bold.nii.gz", "-o", "a.h5", "--labels", "per_volume.npy")
    assert read_subject("a.h5")[1].shape == (80, 10, 2)
    np.save("short.npy", np.ones((79, 10, 2), np.float32))
    with pytest.raises(SystemExit):
        cli("preprocess", env.root / "raw" / "sub-01_bold.nii.gz", "-o", "b.h5", "--labels", "short.npy")


# --- recipe 2: calibrate to degrees ---------------------------------------------------


def test_recipe_2_calibrate_to_degrees(env):
    study = env.root / "mystudy"
    cli("calibrate", "--data-dir", study, "--subjects", "sub-01", "sub-02", "-o", "cal.npz")
    cli("decode", study / "sub-05.h5", "--model", "cal.npz", "-o", "gaze")
    summary = json.loads(open("gaze/sub-05_decode.json").read())
    assert summary["units"] == "deg"
    gaze, true = _tsv("gaze/sub-05_gaze.tsv")[1], env.truth["mystudy/sub-05"]
    assert np.abs(gaze - true).mean() < 0.5 * true.std()       # in degrees, not z-units
    # Calibration leaves the readout alone: the correlation is the pretrained one.
    pretrained = Decoder.load(env.decoder).predict(read_subject(study / "sub-05.h5")[0])
    assert _r(gaze, true) == pytest.approx(_r(pretrained, true), abs=1e-6)


# --- recipe 3: train on some participants, decode the rest ----------------------------


def test_recipe_3_train_on_some_participants(env):
    study = env.root / "mystudy"
    cli("train", "--data-dir", study, "--subjects", "sub-0[1-4]*", "-o", "mystudy.npz")
    model = Decoder.load("mystudy.npz")
    assert model.meta["trained_on"] == ["mystudy"] and model.meta["n_runs"] == 4
    cli("decode", study / "sub-05.h5", study / "sub-06.h5", "--model", "mystudy.npz", "-o", "gaze")
    for sub in ("sub-05", "sub-06"):
        assert json.loads(open(f"gaze/{sub}_decode.json").read())["units"] == "labels"
        gaze, true = _tsv(f"gaze/{sub}_gaze.tsv")[1], env.truth[f"mystudy/{sub}"]
        assert _r(gaze, true) > 0.8
        assert np.abs(gaze - true).mean() < 0.5 * true.std()   # the study's own units


# --- recipe 4: one readout per participant --------------------------------------------


def test_recipe_4_one_readout_per_participant_from_a_calibration_block(env, tmp_path):
    from deepmreye.data import write_subject

    # Gaze known only for the first half of each run, NaN after.
    for sub in ("sub-01", "sub-02"):
        block, labels, attrs = read_subject(env.root / "mystudy" / f"{sub}.h5")
        labels[40:] = np.nan
        write_subject(tmp_path / "calib" / f"{sub}.h5", block, labels=labels, attrs=attrs)
    for sub in ("sub-01", "sub-02"):
        cli("train", "--data-dir", "calib", "--subjects", sub, "-o", f"decoders/{sub}.npz")
        cli("decode", f"calib/{sub}.h5", "--model", f"decoders/{sub}.npz", "-o", "gaze")
        gaze, true = _tsv(f"gaze/{sub}_gaze.tsv")[1], env.truth[f"mystudy/{sub}"]
        assert Decoder.load(f"decoders/{sub}.npz").meta["n_runs"] == 1
        assert _r(gaze[40:], true[40:]) > 0.7                 # the unlabelled half


# --- recipe 5: add your labelled datasets to the published ones -----------------------


def test_recipe_5_add_labelled_datasets_to_the_published_ones(env):
    cli("train", "--data-dir", env.root / "other", "--include-published", "-o", "general.npz")
    model = Decoder.load("general.npz")
    assert model.meta["trained_on"] == ["dsL01_task", "dsL02_task", "dsL03_task", "dsX_lab"]
    assert model.units == "z"
    # Without --data-dir, train is the pretrained decoder again.
    cli("train", "-o", "published.npz")
    block = read_subject(env.root / "mystudy" / "sub-05.h5")[0]
    assert np.allclose(Decoder.load("published.npz").predict(block),
                       Decoder.load(env.decoder).predict(block))


# --- recipe 6: basis features only ----------------------------------------------------


def test_recipe_6_basis_features(env):
    cli("features", env.root / "raw" / "sub-05_bold.nii.gz", "-o", "f")
    cli("features", env.root / "raw" / "sub-05_bold.nii.gz", "-o", "f_orbits", "--orbits")
    k = Decoder.load(env.decoder).k
    avg, both = np.load("f/sub-05_bold_basis.npy"), np.load("f_orbits/sub-05_bold_basis.npy")
    assert avg.shape == (80, k) and both.shape == (80, 2, k)
    assert np.allclose(avg, both.mean(axis=1), atol=1e-5)
    # The leading pair tracks gaze: the representation, before any readout.
    true = env.truth["mystudy/sub-05"].mean(axis=1)
    assert max(abs(np.corrcoef(avg[:, 0], true[:, a])[0, 1]) for a in range(2)) > 0.5


# --- recipe 7: fit the basis on your own corpus ---------------------------------------


def test_recipe_7_fit_the_basis_and_train_on_it(env):
    cli("fit-basis", "-o", "my_basis.npz", "--k", 16)
    from deepmreye.basis import load_basis

    _mask, bases, meta = load_basis("my_basis.npz")
    assert meta["n_subjects"] == 36                 # the unlabelled participants only
    assert not any(s.startswith("dsL") for s in meta["subjects"])
    cc = bases["lr-cca"]["canonical_correlations"]
    assert cc[0] > 0.9 and cc[0] >= cc[-1]
    cli("train", "--basis", "my_basis.npz", "-o", "mine.npz")
    cli("decode", env.root / "mystudy" / "sub-05.h5", "--model", "mine.npz", "-o", "gaze")
    assert _r(_tsv("gaze/sub-05_gaze.tsv")[1], env.truth["mystudy/sub-05"]) > 0.8


# --- reproduce the evaluation, and the data ---------------------------------------------


def test_evaluate_reports_every_labelled_dataset(env):
    cli("evaluate", "--json", "eval.json")
    res = json.loads(open("eval.json").read())
    assert set(res["folds"]) == {"dsL01_task", "dsL02_task", "dsL03_task"}
    assert res["median_subtr"] > 0.8 and res["median_1tr"] > 0.8
    # --check refuses to vouch for anything but the published corpus and basis.
    with pytest.raises(SystemExit):
        cli("evaluate", "--check")


def test_download_fetches_the_labelled_folders_or_everything(env, monkeypatch, tmp_path):
    from deepmreye import data

    calls = []
    monkeypatch.setattr(data, "fetch", lambda patterns, target=None, **kw: calls.append(patterns))
    cli("download", "--data-dir", tmp_path / "d")
    cli("download", "--all", "--data-dir", tmp_path / "d")
    assert calls == [LABELLED, CORPUS]


def test_pretrained_comes_from_the_published_repo(monkeypatch, world):
    """The names a release is published under, as the package looks them up."""
    import huggingface_hub

    from deepmreye import model

    asked = []
    monkeypatch.delenv("DEEPMREYE_MODEL", raising=False)
    monkeypatch.delenv("DEEPMREYE_BASIS", raising=False)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo, filename, revision=None: asked.append((repo, filename))
                        or str(world.decoder if filename == model.PRETRAINED_FILE else world.basis))
    Decoder.pretrained()
    model.pretrained_basis()
    assert asked == [("DeepMReye/deepmreye-unsupervised", "decoder.npz"),
                     ("DeepMReye/deepmreye-unsupervised", "basis.npz")]
