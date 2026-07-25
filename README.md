# DeepMReye 2.0: JEPA for fMRI Eye Tracking

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)
![Architecture: JEPA](https://img.shields.io/badge/Architecture-JEPA-blue.svg)

Decode eye gaze from fMRI without an eye tracker. The BOLD signal around the
eyeballs carries gaze position; DeepMReye 1.0 decoded it with a supervised model.
DeepMReye 2.0 pretrains a representation on **unlabeled** eye-region fMRI from
OpenNeuro using a Joint Embedding Predictive Architecture (JEPA), then fits a
lightweight linear probe to gaze coordinates. The motivation: unlabeled fMRI is
abundant, simultaneous eye-tracking labels are scarce.

![Logo](media/deepmreye_logo.png)

## Installation

Requires Python 3.9–3.11. Dependencies are managed with `uv` (`pyproject.toml` +
`uv.lock`).

```bash
python -m venv .venv
source .venv/bin/activate
uv pip install -e .
```

Training uses CUDA (NVIDIA) or MPS (Apple Silicon) if available, otherwise CPU.

## Running the pipeline

Everything runs through a single CLI. `--data-dir` goes after the command.

```bash
python -m deepmreye compile --data-dir data --limit 5     # 1. sample subjects from OpenNeuro
python -m deepmreye qa --data-dir data                    # 2. label datasets in the browser
python -m deepmreye preprocess --data-dir data            # 3. extract all subjects of approved datasets
python -m deepmreye train --data-dir data -- --epochs 50  # 4. train JEPA + probe
```

Extra training arguments after `--` are forwarded to `scripts/train_jepa.py`.
`run_pipeline.sh <command>` is a thin `.venv` wrapper around these same calls.

### The four stages

1. **compile** — Samples a few subjects per OpenNeuro dataset, coregisters them
   to the eye template, extracts the eye bounding box, and builds the
   `data/datasets.h5` registry with an HTML QA report per subject.
2. **qa** — A local Flask web app for manual quality control. For each subject
   you mark eyes / no-eyes. A dataset is used for training only if all of its
   labeled subjects show eyes (one bad subject drops the dataset, since scanner
   or experiment failures tend to be shared across subjects). Labels are stored
   in `data/datasets.h5` and mirrored to `data/labels.csv`.
3. **preprocess** — Downloads and extracts every subject of the approved
   datasets into per-dataset HDF5 files.
4. **train** — Trains the JEPA model and evaluates a linear gaze probe.

## Label backup

Manual QA labels are the expensive part. Every save in the QA UI is appended to
`data/labels.csv`, and re-running `compile` or `preprocess` never deletes labels.
If the registry is ever rebuilt or corrupted:

```bash
python -m deepmreye export-labels --data-dir data    # snapshot current labels to CSV
python -m deepmreye restore-labels --data-dir data    # replay CSV back into datasets.h5
```

Commit `labels.csv` to version your labeling effort.

## Repository layout

- `deepmreye/__main__.py` — the CLI entry point.
- `deepmreye/pipeline.py` — shared OpenNeuro download / coregistration / extraction.
- `deepmreye/models/` — the JEPA ViT (`jepa.py`) and patchification + masking (`patcher.py`).
- `deepmreye/data/` — HDF5 dataloaders for pretraining (`jepa_dataset.py`) and probing (`probe_dataset.py`).
- `deepmreye/evaluate/probe.py` — gaze probe metrics.
- `deepmreye/labels.py` — CSV label backup.
- `scripts/` — stage implementations and `train_jepa.py`.
- `overview.md` — detailed method reference.
- `paper/` — ICLR 2026 manuscript draft.

## Data formats

Per-dataset HDF5 files hold the extracted blocks:

```text
<dataset>.h5
 └── sub-01
     ├── eye_block        # [X, Y, Z, T] float, gzip-compressed
     └── transform_stats  # affine QA statistics (diagnostic)
```

Manual QA labels live as `approved` attributes in `data/datasets.h5`:
`1` eyes, `0` / `2` no eyes, `-1` unlabeled, `-99` dataset skipped.

## Testing

```bash
pytest deepmreye/tests/ -q
```

Covers patchification and masking geometry, EMA target updates, the label
backup round-trip, and TR validation.

## Correspondence

Questions about the implementation: contact the primary developers.
