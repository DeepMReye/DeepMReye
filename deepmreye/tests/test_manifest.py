"""Tests for the manifest sharding that drives the SLURM extraction array."""
import json
import sys
from pathlib import Path

import pytest

# stage_downloads / extract_staged live in slurm/, not scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "slurm"))
from extract_staged import load_manifest


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "manifest.jsonl"
    with open(path, "w") as f:
        for i in range(10):
            f.write(json.dumps({"dataset": "ds1", "subject": f"sub-{i:02d}",
                                "key": f"k{i}", "local": f"/tmp/{i}.nii.gz"}) + "\n")
    return path


def test_shards_partition_the_work_exactly(manifest):
    """Every subject must be claimed by exactly one array task -- a subject in
    two shards is extracted twice, a subject in none is silently dropped."""
    stride = 4
    seen = []
    for task_id in range(stride):
        seen.extend(e["subject"] for e in load_manifest(manifest, task_id, stride))

    assert sorted(seen) == sorted(f"sub-{i:02d}" for i in range(10))
    assert len(seen) == len(set(seen))


def test_shards_are_balanced(manifest):
    stride = 3
    sizes = [len(load_manifest(manifest, t, stride)) for t in range(stride)]
    assert max(sizes) - min(sizes) <= 1


def test_more_tasks_than_subjects_is_safe(manifest):
    """Oversized arrays are common; the extra tasks must just no-op."""
    stride = 25
    total = sum(len(load_manifest(manifest, t, stride)) for t in range(stride))
    assert total == 10
    assert load_manifest(manifest, 20, stride) == []


def test_single_task_takes_everything(manifest):
    assert len(load_manifest(manifest, 0, 1)) == 10


def test_blank_lines_ignored(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text('{"dataset": "d", "subject": "s", "key": "k", "local": "l"}\n\n')
    assert len(load_manifest(path, 0, 1)) == 1


def test_large_datasets_are_trimmed_not_dropped(tmp_path, monkeypatch):
    """Skipping oversized datasets outright discarded 40% of the corpus; they
    must be trimmed to the cap instead."""
    import stage_downloads as sd

    listing = {f"sub-{i:04d}": f"ds/sub-{i:04d}/func/x_bold.nii.gz" for i in range(300)}
    monkeypatch.setattr(sd, "find_bold_by_subject", lambda client, ds: dict(listing))
    monkeypatch.setattr(sd, "make_s3_client", lambda: None)

    entries = sd.build_manifest(
        tmp_path / "data", tmp_path / "staging", tmp_path / "reg.h5",
        max_subjects=200, datasets=["ds_big"], workers=1,
        resolved_path=tmp_path / "resolved.jsonl",
    )
    assert len(entries) == 200
    assert all(e["dataset"] == "ds_big" for e in entries)


def test_sample_takes_precedence_over_cap(tmp_path, monkeypatch):
    """The QA pass takes 2 subjects even from a dataset far above the cap."""
    import stage_downloads as sd

    listing = {f"sub-{i:04d}": f"k{i}" for i in range(500)}
    monkeypatch.setattr(sd, "find_bold_by_subject", lambda client, ds: dict(listing))
    monkeypatch.setattr(sd, "make_s3_client", lambda: None)

    entries = sd.build_manifest(
        tmp_path / "data", tmp_path / "staging", tmp_path / "reg.h5",
        max_subjects=200, sample=2, datasets=["ds_big"], workers=1,
        resolved_path=tmp_path / "resolved.jsonl",
    )
    assert len(entries) == 2


def test_oversized_inputs_are_deferred_not_lost(tmp_path):
    """Deferring an oversized subject must leave a rerunnable record. Without
    one it silently vanishes from the corpus with only a log line to say why."""
    from extract_staged import extract_one

    big = tmp_path / "big.nii.gz"
    big.write_bytes(b"0" * 2_000_000)
    entry = {"dataset": "ds_x", "subject": "sub-1", "key": "k", "local": str(big)}
    deferred = tmp_path / "deferred_0.jsonl"

    status, err = extract_one(entry, tmp_path, (None,) * 6,
                              max_input_gb=0.001, deferred_path=deferred)

    assert status == "too_large"
    rec = json.loads(deferred.read_text().strip())
    assert rec["dataset"] == "ds_x" and rec["subject"] == "sub-1"
    assert rec["local"] == str(big)      # rerunnable as-is
    assert rec["size_gb"] > 0


def test_missing_staged_file_is_reported(tmp_path):
    from extract_staged import extract_one

    entry = {"dataset": "d", "subject": "s", "key": "k", "local": str(tmp_path / "gone.nii.gz")}
    status, err = extract_one(entry, tmp_path, (None,) * 6)
    assert status == "missing"
