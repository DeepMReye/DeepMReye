"""Tests for bringing the labeled gaze datasets into the corpus layout.

The labeled participants are the control for the whole method, so the two
things that make them usable are worth pinning: they land under their corpus
name (``dsL*``, the glob the probe selects on), and they carry a repetition
time -- which exists nowhere in the source data and comes from the protocol
table in the converter.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from convert_labeled_to_h5 import (
    DATASET_ALIASES,
    DATASET_TR,
    corpus_name,
    run_convert,
    tr_for,
)

from deepmreye.pipeline import LBL_EYES, is_dataset_approved


def _write_npz(path, n_trs=8, seed=0):
    """A source export: one (47, 29, 18) volume and one (10, 2) label per TR."""
    rng = np.random.default_rng(seed)
    arrays = {}
    for i in range(n_trs):
        arrays[f"data_{i}"] = rng.normal(size=(47, 29, 18)).astype(np.float32)
        arrays[f"label_{i}"] = rng.normal(size=(10, 2)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _labeled_source(tmp_path, source_name, subjects):
    for sub in subjects:
        _write_npz(tmp_path / "labeled" / source_name / f"{sub}.npz")
    return tmp_path / "labeled"


def test_source_directories_are_renamed_to_their_corpus_name(tmp_path):
    labeled = _labeled_source(tmp_path, "dataset1_guided_fixations", ["sub-A", "sub-B"])
    out = tmp_path / "data"

    run_convert(labeled, out, skip_registry=True)

    assert (out / "dsL01_guided_fixations" / "sub-A.h5").exists()
    assert not (out / "dataset1_guided_fixations").exists()


def test_every_alias_keeps_the_dsL_prefix_the_probe_globs_on():
    # `dsL*/*.h5` is how the labeled subset is selected without opening a file,
    # so an alias that lost the prefix would silently drop a dataset from it.
    assert all(name.startswith("dsL") for name in DATASET_ALIASES.values())
    assert len(set(DATASET_ALIASES.values())) == len(DATASET_ALIASES)


def test_unknown_source_name_passes_through_unrenamed():
    assert corpus_name("dataset7_something") == "dataset7_something"


def test_tr_comes_from_the_protocol_table_when_the_name_has_none():
    # Only dataset 6 encodes its TR per subject; the rest have it nowhere.
    assert tr_for("dataset1_guided_fixations", "sub-NDARAA948VFH") == 0.800
    assert tr_for("dataset6_sequences", "S4_0004_TR1250_2MM") == 1.25


def test_subject_name_tr_wins_over_the_protocol_table():
    # Dataset 6 is the same participant resampled, so a single dataset-level TR
    # would be wrong for five of its six subjects.
    assert tr_for("dataset6_sequences", "S4_0006_TR2500_2MM") == 2.5


@pytest.mark.parametrize("source_name", sorted(DATASET_ALIASES))
def test_every_labeled_dataset_has_a_resolvable_tr(source_name):
    assert source_name in DATASET_TR or source_name == "dataset6_sequences"


def test_converted_participant_carries_tr_labels_and_shape(tmp_path):
    labeled = _labeled_source(tmp_path, "dataset2_pursuit", ["sub-A"])
    out = tmp_path / "data"

    run_convert(labeled, out, skip_registry=True)

    with h5py.File(out / "dsL02_pursuit" / "sub-A.h5") as f:
        assert f["eye_block"].shape == (47, 29, 18, 8)
        assert f["labels"].shape == (8, 10, 2)
        assert f.attrs["repetition_time"] == pytest.approx(0.870)
        assert f.attrs["has_labels"]
        assert f.attrs["dataset"] == "dsL02_pursuit"


def test_registration_makes_the_labeled_datasets_visible_to_qa(tmp_path):
    # Before this, the labeled sets existed only as folders: `is_dataset_approved`
    # could not see them and they indexed with a null QA label.
    labeled = _labeled_source(tmp_path, "dataset5_free_viewing", ["sub-A", "sub-B"])
    out = tmp_path / "data"
    out.mkdir()
    with h5py.File(out / "datasets.h5", "w"):
        pass

    run_convert(labeled, out)

    with h5py.File(out / "datasets.h5") as f:
        grp = f["dsL05_free_viewing"]
        assert sorted(grp.keys()) == ["sub-A", "sub-B"]
        assert grp["sub-A"].attrs["approved"] == LBL_EYES
        assert is_dataset_approved(grp)

    # The labels are mirrored so a rebuilt registry does not lose them.
    assert "dsL05_free_viewing,subject,sub-A,1" in (out / "labels.csv").read_text()


def test_rerunning_registers_subjects_already_on_disk(tmp_path):
    # The second pass skips the rewrite; it must still see those subjects, or a
    # resumed conversion would leave them out of the registry entirely.
    labeled = _labeled_source(tmp_path, "dataset3_pursuit", ["sub-A"])
    out = tmp_path / "data"
    out.mkdir()
    with h5py.File(out / "datasets.h5", "w"):
        pass

    run_convert(labeled, out, skip_registry=True)
    run_convert(labeled, out)

    with h5py.File(out / "datasets.h5") as f:
        assert f["dsL03_pursuit"]["sub-A"].attrs["approved"] == LBL_EYES
