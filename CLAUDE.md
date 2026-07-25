# CLAUDE.md

Orientation for an agent picking up this project. Read this first, then
`overview.md` for the method in depth.

## What this is

DeepMReye 2.0: decode eye gaze from fMRI without an eye tracker. The signal
around the eyeballs in a BOLD volume carries gaze position. Version 1 (published,
`frey2021deepmreye`) did this with a supervised regressor. This branch (`pytorch`)
rewrites it as a **self-supervised JEPA**: pretrain a representation on unlabeled
eye-region fMRI from OpenNeuro, then fit a light linear probe to gaze on the few
labeled datasets. The bet: unlabeled fMRI is abundant, gaze labels are scarce.

Status: pipeline runs end-to-end, tests pass, not yet trained at scale. The
immediate goal is to run the full pipeline on a cluster and iterate on the model.

## Pipeline (single entry point)

Everything runs through `python -m deepmreye <command>` (see `deepmreye/__main__.py`).
`run_pipeline.sh` is a thin `.venv` wrapper around the same calls.

1. `compile`    — sample ~2 subjects/dataset from OpenNeuro into `data/datasets.h5`.
2. `qa`         — Flask browser UI to label each subject eyes / no-eyes.
3. `preprocess` — download + extract all subjects of approved datasets.
4. `train`      — train the JEPA + linear probe.

Plus `export-labels` / `restore-labels` for the label backup (see below).

```bash
python -m deepmreye compile --data-dir data --limit 5
python -m deepmreye qa --data-dir data
python -m deepmreye preprocess --data-dir data
python -m deepmreye train --data-dir data -- --epochs 50   # args after `--` go to train_jepa.py
```

`--data-dir` goes AFTER the command (subparser-only, by design).

## Layout

- `deepmreye/__main__.py` — the CLI. Every stage dispatches from here.
- `deepmreye/pipeline.py` — shared S3 download / coregister / extract / write.
  `compile` and `preprocess` both call `process_subject` here. Single source of
  truth for the ingestion logic and for `is_dataset_approved`.
- `deepmreye/preprocess.py` — coregistration and eye-mask extraction (ANTs).
- `deepmreye/labels.py` — CSV label backup (export / restore).
- `deepmreye/validation.py` — TR extraction/validation from NIfTI headers.
- `deepmreye/data/{jepa,probe}_dataset.py` — windowed HDF5 dataloaders.
- `deepmreye/models/{jepa,patcher}.py` — ViT encoders/predictor, patchify + masking.
- `deepmreye/evaluate/probe.py` — gaze probe metrics (R^2, Pearson r).
- `scripts/` — the stage implementations (`run_compile`, `run_preprocess`,
  `run_labeler`), imported by the CLI. `train_jepa.py` is the training loop.
- `overview.md` — detailed method reference. `paper/` — ICLR 2026 draft.

## Data model

- `data/datasets.h5` — central registry. One group per dataset, one subgroup per
  subject. Manual QA labels live as the `approved` attribute:
  `1` eyes, `0` no eyes / bad transform, `2` no eyes / good transform,
  `-1` unlabeled, `-99` whole dataset skipped.
- `data/<ds>/<ds>.h5` — per-dataset extracted `eye_block` arrays `[X, Y, Z, T]`
  plus `transform_stats` (diagnostic only).
- `data/labels.csv` — append-only backup of every QA label (see below).

## Key decisions (context you won't get from the code)

- **Classifier removed.** An earlier design trained a decision tree to auto-QA
  coregistration quality (`transform_probability`). We deleted it. Approval is
  now purely manual labels. Don't reintroduce a probability gate.
- **Dataset-level all-or-nothing approval.** A dataset is used for training only
  if EVERY labeled subject shows eyes; one "no eyes" subject drops the whole
  dataset. This is intentional: the same scanner/experiment tends to fail the
  same way across subjects, and OpenNeuro has more datasets than we need. Logic
  lives in `is_dataset_approved` (`deepmreye/pipeline.py`), used by both
  preprocess and the training dataset.
- **Labels are precious and backed up.** The QA UI mirrors every save into
  `data/labels.csv`. Re-running `compile`/`preprocess` never deletes labels. If
  the registry is rebuilt or corrupted, `python -m deepmreye restore-labels`
  replays the CSV. Commit `labels.csv` to version the QA effort.
- **TR is validated but not yet used to resample.** Windows are a fixed number
  of TRs, not fixed duration, so datasets with different TRs give windows of
  different real length. Known limitation (see `overview.md` §Discussion). If you
  work on temporal handling, this is the open thread.

## Dev

- Env: `uv` + `pyproject.toml` / `uv.lock` (no `requirements.txt`). The wrapper
  script expects a `.venv`. Use `.venv/bin/python` on this machine.
- Tests: `pytest deepmreye/tests/ -q` (jepa forward/masking, label round-trip,
  TR validation). Run before pushing.
- Device: training auto-selects cuda / mps / cpu in `train_jepa.py`.

## Paper sync

`paper/main.tex` Method section mirrors `overview.md`. When the pipeline or model
changes substantially, update `overview.md` AND the paper's Method, and note it
here if it changes a key decision above.
