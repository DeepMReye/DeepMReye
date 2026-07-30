"""The in-training gaze probe: folds, and the metrics it hands to wandb.

The monitor this replaced mean-pooled the encoder's spatial tokens and split by
subject only, which put it 0.11-0.24 above ``eval_probe.py`` and ranked configs
in the opposite order. What is pinned here is that the replacement computes the
same thing ``eval_probe.py --protocol dataset`` does:

- the folds are leave-one-dataset-out, and the training side of a fold never
  contains a row of the dataset being held out;
- one embedding of the whole labeled set covers every fold, which is what makes
  running this inside the training loop affordable at all -- so ``split_by="all"``
  has to be exactly the union of a fold's two sides;
- every metric reaches the log dict under a stable key, per dataset and in
  aggregate, for both protocols.
"""
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from deepmreye.data.probe_dataset import ProbeDataset
from deepmreye.storage import subject_path, write_subject

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from train_jepa import (  # noqa: E402
    build_probe,
    evaluate_probe,
    probe_folds,
    run_probe,
)

LABEL_ATTRS = {"repetition_time": 2.0}
DATASETS = ("dsL01_guided_fixations", "dsL02_pursuit", "dsL05_free_viewing")


def _corpus(tmp_path, n_trs=150, subjects=2):
    """A labeled corpus with the real block shape, so the token grid is 6x4x3."""
    rng = np.random.default_rng(0)
    for ds in DATASETS:
        for i in range(subjects):
            write_subject(
                subject_path(tmp_path, ds, f"{ds}-sub-{i}"),
                rng.normal(size=(47, 29, 18, n_trs)).astype(np.float32),
                labels=rng.normal(size=(n_trs, 10, 2)).astype(np.float32),
                attrs=LABEL_ATTRS)
    return tmp_path


def _args(**overrides):
    defaults = dict(window_size=100, probe_windows=0, spatial_pool="6x4x3",
                    probe_readout="ridge-cv", probe_protocols=["dataset", "subject"],
                    batch_size=4, num_workers=0)
    return Namespace(**{**defaults, **overrides})


def test_split_all_is_exactly_the_union_of_a_folds_two_sides(tmp_path):
    """The probe embeds every labeled window once and splits the features, so
    'all' must miss nothing a per-fold ProbeDataset would have loaded."""
    _corpus(tmp_path)
    common = dict(labeled_data_dir=tmp_path, window_size=100)
    holdout = {"dsL02_pursuit"}

    def keys(ds):
        return {(s["dataset"], s["subject"], s["start"]) for s in ds.samples}

    everything = keys(ProbeDataset(split="train", split_by="all", **common))
    fold = (keys(ProbeDataset(split="train", holdout=holdout, **common))
            | keys(ProbeDataset(split="test", holdout=holdout, **common)))

    assert everything == fold
    assert len(everything) == len(ProbeDataset(split="train", split_by="all", **common).samples)


def test_split_all_ignores_the_split_argument(tmp_path):
    _corpus(tmp_path)
    common = dict(labeled_data_dir=tmp_path, split_by="all", window_size=100)
    assert (len(ProbeDataset(split="train", **common))
            == len(ProbeDataset(split="test", **common)) > 0)


def test_dataset_folds_hold_out_each_dataset_exactly_once():
    ds_rows = np.array(["a", "a", "b", "b", "c"])
    sub_rows = np.array(["s1", "s2", "s3", "s4", "s5"])

    folds = probe_folds("dataset", ds_rows, sub_rows, held_out_subjects=set())

    assert [name for name, _ in folds] == ["a", "b", "c"]
    for name, test in folds:
        # The training side is the complement, and it must not contain a single
        # row of the held-out dataset -- that is the whole claim of the protocol.
        assert set(ds_rows[test]) == {name}
        assert name not in set(ds_rows[~test])
    # Every row is held out exactly once, so the union of the folds' per-subject
    # scores is a corpus-wide number with nothing counted twice.
    assert sum(test.sum() for _, test in folds) == len(ds_rows)


def test_subject_fold_holds_out_the_named_participants_across_every_dataset():
    ds_rows = np.array(["a", "a", "b", "b"])
    sub_rows = np.array(["s1", "s2", "s3", "s4"])

    (name, test), = probe_folds("subject", ds_rows, sub_rows, {"s2", "s4"})

    assert name == "held-out subjects"
    assert set(sub_rows[test]) == {"s2", "s4"}
    assert set(ds_rows[test]) == {"a", "b"}


def test_unknown_protocol_is_an_error():
    with pytest.raises(ValueError):
        probe_folds("paradigm", np.array(["a"]), np.array(["s"]), set())


def test_run_probe_recovers_a_linear_signal_and_scores_every_held_out_dataset():
    rng = np.random.default_rng(0)
    n_per_subject = 60
    ds_rows, sub_rows, feats, targets = [], [], [], []
    # One shared linear map, so a readout fitted on two datasets transfers to
    # the third and the per-dataset r is high everywhere.
    w = rng.normal(size=(6, 2))
    for ds in ("a", "b", "c"):
        for s in range(2):
            x = rng.normal(size=(n_per_subject, 6))
            feats.append(x)
            targets.append(x @ w)
            ds_rows += [ds] * n_per_subject
            sub_rows += [f"{ds}{s}"] * n_per_subject

    per_dataset, overall = run_probe(
        "dataset", np.vstack(feats), np.vstack(targets),
        np.array(ds_rows), np.array(sub_rows), set(), "ridge-cv")

    assert set(per_dataset) == {"a", "b", "c"}
    for m in per_dataset.values():
        assert m["n_subjects"] == 2
        assert m["pearson_r_x"] > 0.9 and m["pearson_r_y"] > 0.9
    # Six subjects, each scored in the one fold that held it out.
    assert overall["n_subjects"] == 6
    assert overall["pearson_r_x"] > 0.9


def test_run_probe_scores_a_dataset_that_only_appears_in_the_test_side():
    """The subject protocol tests every dataset at once, so its report has to
    break down by dataset rather than collapsing to one number."""
    rng = np.random.default_rng(1)
    ds_rows, sub_rows, feats, targets = [], [], [], []
    w = rng.normal(size=(6, 2))
    for ds in ("a", "b"):
        for s in range(2):
            x = rng.normal(size=(60, 6))
            feats.append(x)
            targets.append(x @ w)
            ds_rows += [ds] * 60
            sub_rows += [f"{ds}{s}"] * 60

    per_dataset, overall = run_probe(
        "subject", np.vstack(feats), np.vstack(targets),
        np.array(ds_rows), np.array(sub_rows), {"a1", "b1"}, "ridge-cv")

    assert set(per_dataset) == {"a", "b"}
    assert overall["n_subjects"] == 2


@pytest.mark.parametrize("spatial_pool", ["6x4x3", "mean"])
def test_evaluate_probe_logs_every_metric_for_both_protocols(tmp_path, spatial_pool):
    from deepmreye.models.jepa import JEPAModel

    _corpus(tmp_path)
    probe = build_probe(_args(spatial_pool=spatial_pool), tmp_path)
    assert probe is not None

    model = JEPAModel(embed_dim=8, encoder_depth=1, predictor_depth=1, num_heads=2)
    logs = evaluate_probe(model, probe, torch.device("cpu"), epoch=0, epochs=1)

    for protocol in ("dataset", "subject"):
        for ds in DATASETS:
            for metric in ("pearson_r", "pearson_r_x", "pearson_r_y", "r2",
                           "euclidean", "n_subjects"):
                assert f"probe/{protocol}/{ds}/{metric}" in logs
        for key in ("mean_r", "mean_r_x", "mean_r_y", "mean_r2", "mean_euclidean"):
            assert f"probe/{protocol}/{key}" in logs
        assert logs[f"probe/{protocol}/all/n_subjects"] > 0

    # The unprefixed aliases follow the first protocol, so a sweep can sort on
    # one curve without knowing which protocols a run enabled.
    assert logs["probe/mean_r"] == logs["probe/dataset/mean_r"]
    assert np.isfinite(logs["probe/dataset/mean_r"])


def test_evaluate_probe_feature_width_follows_the_spatial_pool(tmp_path, capsys):
    from deepmreye.models.jepa import JEPAModel

    _corpus(tmp_path)
    probe = build_probe(_args(probe_protocols=["dataset"]), tmp_path)
    model = JEPAModel(embed_dim=8, encoder_depth=1, predictor_depth=1, num_heads=2)

    evaluate_probe(model, probe, torch.device("cpu"), epoch=0, epochs=1)

    # 72 spatial tokens at embed_dim 8, not 8 -- the mean-pooled width is what
    # made the old monitor incomparable with the reported numbers.
    assert f"x {72 * 8:,}" in capsys.readouterr().out


def test_probe_is_disabled_rather_than_crashing_without_labeled_data(tmp_path):
    (tmp_path / "empty").mkdir()
    assert build_probe(_args(), tmp_path) is None


def _eval_probe_args(tmp_path, **overrides):
    defaults = dict(protocol="dataset", arms=["random"], readouts=["ridge-cv"],
                    checkpoint=None, window_size=100, temp_patch_size=5,
                    voxel_stride=4, spatial_pool="6x4x3", n_components=32, gap=0,
                    batch_size=4, num_workers=0, seed=0, embed_dim=8,
                    encoder_depth=1, predictor_depth=1, num_heads=2,
                    max_windows=None, feature_cache=None)
    return Namespace(**{**defaults, **overrides})


def test_eval_probe_random_arm_uses_one_encoder_for_both_splits(tmp_path, monkeypatch):
    """The `random` control must be *one* untrained encoder, not two.

    A fresh JEPAModel() per split encodes the fit set and the scoring set in two
    unrelated random bases, so the readout transfers nothing and the arm scores
    ~0 whatever the architecture does. That is not a measurement of an untrained
    encoder, and it is how `random` came to read -0.003 at full spatial
    resolution while the same untrained weights, shared, read ~0.8 on the same
    fold.
    """
    import eval_probe

    _corpus(tmp_path)
    args = _eval_probe_args(tmp_path)

    seen = []
    real = eval_probe.encoder_features

    def spy(model, *a, **kw):
        seen.append(id(model))
        return real(model, *a, **kw)

    monkeypatch.setattr(eval_probe, "encoder_features", spy)
    arm_models = eval_probe.build_arms(args, torch.device("cpu"))
    eval_probe.run_fold("dsL02_pursuit", {"dsL02_pursuit"}, tmp_path, args,
                        torch.device("cpu"), arm_models)

    assert len(seen) == 2, "expected one encoder pass per split"
    assert len(set(seen)) == 1, "train and test were encoded by different models"


def test_eval_probe_random_encoder_is_reproducible(tmp_path):
    """Seeded, so the control is the same encoder from run to run."""
    import eval_probe

    args = _eval_probe_args(tmp_path)
    a = eval_probe.build_model(args, torch.device("cpu"))
    b = eval_probe.build_model(args, torch.device("cpu"))
    for (name, p_a), (_, p_b) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(p_a, p_b), name
