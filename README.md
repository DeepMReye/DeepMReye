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
   `data/datasets.h5` registry with a small QA thumbnail per subject.
2. **qa** — A local Flask web app for manual quality control. For each subject
   you mark eyes (`1` clean, `4` faint, `3` cut off) vs no-eyes (`0` bad transform, `2` good transform). A dataset is used for training only if all of its
   labeled subjects show eyes (one bad subject drops the dataset, since scanner
   or experiment failures tend to be shared across subjects). Labels are stored
   in `data/datasets.h5` and mirrored to `data/labels.csv`.
3. **preprocess** — Downloads and extracts every subject of the approved
   datasets into per-participant HDF5 files.
4. **train** — Trains the JEPA model and evaluates a linear gaze probe. Writes
   checkpoints to `runs/jepa/` (`last.pt` plus one every `--save-every` epochs).

Note the ordering constraint: QA labeling needs the thumbnails, and those are
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

Each stage pulls only what it reads. `qa` takes the registry plus every QA
thumbnail — ~20 KB per subject, ~30 MB in total, so it arrives in one go and you
can start labeling immediately. `train` pulls the blocks and no images; `probe`
pulls only `dsL*/*.h5`, the gaze-labeled subset. A directory you point at is
never topped up from the network; only the cache is.

To download up front instead — before a flight, or to work offline:

```bash
python -m deepmreye fetch                 # blocks + registry (~29 GB)
python -m deepmreye fetch --reports       # add the full HTML reports too (~37 GB)
python -m deepmreye fetch --labels-only   # just the registry and index (MB)
```

### On a cluster

If compute nodes have no outbound network, or login sessions are memory-capped,
`compile` and `preprocess` cannot run as single commands — they need network and
memory in the same process. Everything SLURM-specific lives in [`slurm/`](slurm/)
with its own README; the rest of the repo is portable.

## Evaluation

The claim of the method is that a self-supervised representation decodes gaze
better than what you can read straight off the voxels. `scripts/eval_probe.py`
is the table that has to show it, crossing two axes.

**Generalization level** (`--protocol`), in increasing strictness:

| | train / test |
|---|---|
| `within` | same participant, early timepoints vs late |
| `subject` | held-out participants, same scanner and paradigm |
| `dataset` | leave one dataset out, each in turn |
| `paradigm` | leave one paradigm out (`dsL02/03/04` are all pursuit) |

**Feature source** (`--arms`): `voxels` (downsampled raw voxels, no learning),
`random` (the same architecture untrained), `trained` (a JEPA checkpoint).

**Readout** (`--readouts`): `mean`, `linear`, `ridge`, `ridge-cv`, `pca-ridge`,
`pls`, `rf`, `gbt` — see `deepmreye/evaluate/baselines.py`. Every readout runs
on every arm's features, so nothing wins by being tuned better than what it is
compared against.

```bash
python scripts/eval_probe.py --protocol dataset --feature-cache .cache/features
python scripts/eval_probe.py --protocol dataset --checkpoint runs/jepa/last.pt
```

Feature extraction is the expensive half and the readouts take seconds, so
`--feature-cache` makes it cheap to add a readout to a table you already ran.

Two things about how the numbers are reported. Metrics are aggregated **per
participant, then median across participants** — pooling every row together lets
a model score well by predicting only which subject it is looking at (`--pooled`
prints that number for comparison). And **Pearson r is the headline rather than
R²**, because cross-dataset predictions are mis-calibrated in gain, which
destroys R² while leaving the correlation intact; `scripts/analyze_calibration.py`
measures exactly that and shows why no unsupervised correction fixes it.

`scripts/analyze_identifiability.py` is an **analysis, not a baseline**: it fits
CCA between the left and right orbit of a single run and recovers gaze without
labels. It cannot be deployed (it is fitted on the run it scores), but it
separates "the representation cannot carry gaze" from "the readout does not
transfer".

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

## QA thumbnails

Every participant gets a `<subject>.png` beside its HDF5: the z=-30 brain slice
with the eye mask in red, then the extracted eye block from two sides. That is
enough to answer the only question QA asks — are the eyeballs in there.

It replaced the 5 MB Plotly report as the default artifact. Measured over the QA
sample, 1773 reports came to 9.1 GB and the same 1773 thumbnails to 29 MB, and a
full extraction would have put the reports over 100 GB. Pass `--report html` (or
`both`) to `slurm/extract_staged.py` when you want the histogram and timecourses
for a specific subject.

For a corpus extracted before thumbnails existed:

```bash
python scripts/backfill_thumbnails.py --data-dir data --workers 8
```

This reads each subject's report where one exists, and falls back to the stored
block otherwise — so the gaze-labeled participants, which never had a report,
get one too. Do it before deleting any reports: they are the only surviving
record of the pre-normalization volumes.

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
- `deepmreye/evaluate/probe.py` — gaze probe metrics and per-subject aggregation.
- `deepmreye/evaluate/baselines.py` — the readout models every arm is scored with.
- `deepmreye/labels.py` — CSV label backup.
- `deepmreye/storage.py` — the per-participant HDF5 layout (all block I/O).
- `deepmreye/registry.py` — worker sidecar records and their merge into the registry.
- `deepmreye/datasource.py` — finds the corpus (local dir, `$DEEPMREYE_DATA`, or
  HuggingFace) so no command needs a path.
- `deepmreye/qa_classifier.py` — eye-detection triage features and model.
- `deepmreye/thumbnail.py` — the QA thumbnail every participant gets.
- `scripts/` — portable stages: labeling UI, `train_jepa.py`, `eval_probe.py`,
  `analyze_identifiability.py`, `analyze_calibration.py`, `build_index.py`,
  `train_qa_classifier.py`, `upload_to_hf.py`, `sync_labels.py`,
  `convert_labeled_to_h5.py`, `backfill_thumbnails.py`.
- `docs/ssl_design_brief.md` — the open questions on the SSL objective.
- `results/` — evaluation output.
- `slurm/` — everything cluster-specific (staging, extraction array). Has its
  own README. Nothing outside this folder needs SLURM.
- `overview.md` — detailed method reference.
- `paper/` — ICLR 2026 manuscript draft.

## Data formats

One HDF5 file per participant, foldered by dataset:

```text
data/
├── ds000001/                    # an OpenNeuro accession — unlabeled
│   ├── sub-01.h5
│   │   ├── eye_block   # [47, 29, 18, T] float32, gzip, chunked over time
│   │   └── labels      # [T, 10, 2] float32 — only when gaze is known
│   ├── sub-01.png      # ~20 KB QA thumbnail
│   └── sub-02.h5
├── dsL01_guided_fixations/      # a gaze-labeled dataset
│   └── sub-NDARAA948VFH.h5
├── datasets.h5         # QA registry
└── index.parquet       # one row per participant
```

Labeled and unlabeled participants use the identical container; `labels` is
simply absent when gaze is unknown. The `dsL` prefix is the only thing that
distinguishes them by path, which makes `dsL*/*.h5` the glob for the probe set.
Blocks are normalized at extraction (z-scored per voxel and per volume, clipped
at 5 SD), so every file in the corpus is directly comparable. Per-participant
files are what make extraction parallelisable and keep one corrupt write from
costing a whole dataset.

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

**1. Cluster — push the working copy** (blocks + thumbnails + registry; runs
for hours, so detach it. `--reports` adds the ~5 MB HTML reports, which is only
worth it if you want to inspect subjects in depth off-cluster):

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

**2. Laptop — label.** Nothing to configure; the registry and every thumbnail
come down in seconds:

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
