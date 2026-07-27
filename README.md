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
   datasets into per-participant HDF5 files.
4. **train** — Trains the JEPA model and evaluates a linear gaze probe.

Note the ordering constraint: QA labeling needs the HTML reports, and those are
produced by coregistration. So `compile` registers a sample of subjects to give
you something to look at, you label, and only then does `preprocess` fetch the
remaining subjects of the datasets that passed.

### Where the data lives

No stage needs a path. The corpus is resolved in this order, so the same
commands work on a laptop and on a cluster:

1. `--data-dir` if you pass one;
2. `$DEEPMREYE_DATA`;
3. `./data`, if it holds a registry;
4. otherwise it is downloaded from HuggingFace into `~/.cache/deepmreye` on
   first use.

```bash
python -m deepmreye qa                    # laptop: pulls what it needs, then labels
DEEPMREYE_DATA=/scratch/.../data \
  python -m deepmreye qa                  # cluster: uses the local copy, no network
python -m deepmreye qa --no-download      # never reach for HuggingFace
```

Each stage pulls only what it reads. `qa` starts from the registry alone (a few
MB) and fetches QA reports one dataset at a time as you reach them — they are
~5 MB per subject, more in total than the eye blocks themselves, so downloading
them up front would mean waiting hours to label the first dataset. `train` pulls
the blocks and no reports. A directory you point at is never topped up from the
network; only the cache is.

To download up front instead — before a flight, or to work offline:

```bash
python -m deepmreye fetch                 # blocks + registry (~29 GB)
python -m deepmreye fetch --reports       # everything, labeling included (~37 GB)
python -m deepmreye fetch --labels-only   # just the registry and index (MB)
```

### On a cluster

If compute nodes have no outbound network, or login sessions are memory-capped,
`compile` and `preprocess` cannot run as single commands — they need network and
memory in the same process. Everything SLURM-specific lives in [`slurm/`](slurm/)
with its own README; the rest of the repo is portable.

## QA triage

Labeling every sampled subject across ~2400 OpenNeuro datasets is the slow part.
A small scikit-learn model over the extracted eyeball voxels (occupancy,
centre/edge contrast, temporal SNR) helps in two places:

```bash
python scripts/train_qa_classifier.py --data-dir data --rank   # label uncertain subjects first
python scripts/train_qa_classifier.py --data-dir data --flag   # screen the full download
```

`--rank` orders unlabeled subjects by model uncertainty so labeling effort goes
where it changes the model most. `--flag` lists likely no-eyes participants among
the subjects that full extraction pulls in — QA only ever samples 2 subjects per
dataset, so those are otherwise never inspected.

**The model never approves anything.** It ranks and flags for your review;
dataset approval stays manual. Accuracy is reported as ROC-AUC cross-validated
with datasets held out, since subjects within a dataset share a scanner and
failure mode.

## Label backup

Manual QA labels are the expensive part. Every save in the QA UI is appended to
`data/labels.csv`, and re-running `compile` or `preprocess` never deletes labels.
If the registry is ever rebuilt or corrupted:

```bash
python -m deepmreye export-labels --data-dir data     # snapshot current labels to CSV
python -m deepmreye restore-labels --data-dir data    # replay CSV back into datasets.h5
```

Labels live with the corpus, not in git — the data directory is on scratch or in
the HuggingFace cache. They are versioned by pushing them to the Hub; see
*Working across machines*.

## Repository layout

- `deepmreye/__main__.py` — the CLI entry point.
- `deepmreye/pipeline.py` — shared OpenNeuro download / coregistration / extraction.
- `deepmreye/models/` — the JEPA ViT (`jepa.py`) and patchification + masking (`patcher.py`).
- `deepmreye/data/` — HDF5 dataloaders for pretraining (`jepa_dataset.py`) and probing (`probe_dataset.py`).
- `deepmreye/evaluate/probe.py` — gaze probe metrics.
- `deepmreye/labels.py` — CSV label backup.
- `deepmreye/storage.py` — the per-participant HDF5 layout (all block I/O).
- `deepmreye/registry.py` — worker sidecar records and their merge into the registry.
- `deepmreye/datasource.py` — finds the corpus (local dir, `$DEEPMREYE_DATA`, or
  HuggingFace) so no command needs a path.
- `deepmreye/qa_classifier.py` — eye-detection triage features and model.
- `scripts/` — portable stages: labeling UI, `train_jepa.py`, `build_index.py`,
  `train_qa_classifier.py`, `upload_to_hf.py`, `sync_labels.py`,
  `convert_labeled_to_h5.py`.
- `slurm/` — everything cluster-specific (staging, extraction array). Has its
  own README. Nothing outside this folder needs SLURM.
- `overview.md` — detailed method reference.
- `paper/` — ICLR 2026 manuscript draft.

## Data formats

One HDF5 file per participant, foldered by dataset:

```text
data/
├── ds000001/
│   ├── sub-01.h5
│   │   ├── eye_block   # [47, 29, 18, T] float32, gzip, chunked over time
│   │   └── labels      # [T, 10, 2] float32 — only when gaze is known
│   └── sub-02.h5
├── datasets.h5         # QA registry
└── index.parquet       # one row per participant
```

Labeled and unlabeled participants use the identical container; `labels` is
simply absent when gaze is unknown. Blocks are normalized at extraction
(z-scored per voxel and per volume, clipped at 5 SD), so every file in the
corpus is directly comparable. Per-participant files are what make extraction
parallelisable and keep one corrupt write from costing a whole dataset.

Build the index (and validate every file) with:

```bash
python scripts/build_index.py --data-dir data --deep
```

`--deep` reads every voxel, catching interior corruption, all-zero blocks, and
values outside the claimed normalization range. Anything it flags is excluded
from the index rather than shipped.

## Working across machines

Extraction needs a cluster; labeling needs your eyes and a browser. The Hub is
what joins them. The corpus goes up once, then only labels travel — a few MB
against ~29 GB of blocks.

Authenticate once per machine (`hf auth login`, or set `$HF_TOKEN`), and set
`DEEPMREYE_HF_REPO` so no command needs `--repo-id`:

```bash
export DEEPMREYE_HF_REPO=DeepMReye/eyeballs
```

**1. Cluster — push the working copy** (blocks + QA reports + registry, ~37 GB;
runs for hours, so detach it):

```bash
export DATA=/leonardo_scratch/fast/AIFAC_S07_154/mfrey/dme/data
python scripts/upload_to_hf.py --data-dir $DATA --repo-id $DEEPMREYE_HF_REPO \
    --reports --private --dry-run          # check the tally first
python scripts/upload_to_hf.py --data-dir $DATA --repo-id $DEEPMREYE_HF_REPO \
    --reports --private
```

This is a *working copy*: every subject, including ones already labeled
no-eyes. That is deliberate — you cannot revise a label on a subject that was
filtered out of the copy you are labeling from. `--publish` applies the QA
filter, and belongs to the final artifact only.

**2. Laptop — label.** Nothing to configure; the registry comes down in
seconds and reports stream in per dataset:

```bash
python -m deepmreye qa                     # UI on http://localhost:5050
```

**3. Laptop — push the labels back.** Small and quick, so do it often:

```bash
python scripts/sync_labels.py push
```

**4. Cluster — collect them**, then stage and extract the approved datasets in
full (see [`slurm/`](slurm/)):

```bash
python scripts/sync_labels.py pull --data-dir $DATA
```

`pull` **merges**: it fills unlabeled slots, leaves anything already labeled on
this machine alone, and reports conflicts instead of resolving them silently. So
labeling in two places cannot lose work, and pulling twice changes nothing the
second time.

## Publishing

The final artifact, once labeling and full extraction are done:

```bash
python scripts/build_index.py --data-dir $DATA --deep
python scripts/upload_to_hf.py --data-dir $DATA --repo-id $DEEPMREYE_HF_REPO \
    --publish --dry-run
```

The dry run reports exactly what would be uploaded and why anything was left
out. Two independent gates apply: technical validation (above), and — with
`--publish` — QA status: subjects labeled no-eyes are dropped even though their
files are perfectly valid, since the corpus exists for eye-region signal.
Subjects not yet labeled are kept, but called out in the summary.
`--labeled-only` publishes just the gaze-labeled subset.

Manual QA labels live as `approved` attributes in `data/datasets.h5`:
`1` eyes, `3` eyes but clipped by the bounding box (also approved), `0` no eyes
or bad transform, `2` no eyes with a good transform, `-1` unlabeled, `-99` whole
dataset skipped.

## Testing

```bash
pytest deepmreye/tests/ -q
```

Covers patchification and masking geometry, EMA target updates, the label
backup round-trip, TR validation, the on-disk storage format (atomic writes,
truncation detection, label alignment), registry merging under parallel
workers, the windowed dataloaders and their train/test splits, and the manifest
sharding that partitions work across the extraction array.

## Correspondence

Questions about the implementation: contact the primary developers.
