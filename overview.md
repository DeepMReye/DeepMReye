# DeepMReye 2.0 Architectural Overview

This document is the deep method reference for this branch (`pytorch`): data
extraction, tensor shapes, windowed data loading, and the classic-regressor
gaze probe. It explains the mechanics without relying on reading source code.

For a short orientation (project state, key decisions, layout) see `CLAUDE.md`.
For install and usage see `README.md`.

A self-supervised JEPA pretraining approach (patchification, masking, a ViT
encoder/predictor) was built and evaluated on this codebase and set aside —
see `CLAUDE.md`'s "What this is" for why. That architecture is documented on
the **`pytorch-jepa`** branch, not here.

## 0. Running the Pipeline

There is a single entry point. Every stage runs through it:

```bash
python -m deepmreye compile --data-dir data --limit 5   # 1. sample subjects for QA
python -m deepmreye qa --data-dir data                  # 2. browser labeling UI (eyes / no eyes)
python -m deepmreye preprocess --data-dir data           # 3. extract all subjects of approved datasets
python -m deepmreye all --data-dir data                  # run 1-3, pausing for QA
python scripts/eval_probe.py --protocol dataset --readouts ridge-cv svr lgbm mlp
```

`--data-dir` goes *after* the command. `run_pipeline.sh <command>` is a thin
`.venv` wrapper around these same calls. Manual QA labels are stored as
`approved` attributes in `data/datasets.h5` and are preserved across reruns of
any stage; re-running `compile` or `preprocess` never deletes them.

On a cluster where compute nodes are offline and login sessions are
memory-capped, steps 1 and 3 split into a download stage and an extraction
stage — see "Running on Leonardo" in `CLAUDE.md`. The per-subject work is
identical either way.

## 1. Data Ingestion & Preprocessing

DeepMReye trains on raw fMRI data sourced from OpenNeuro. The process is broken into metadata compilation and BOLD sequence extraction.

### Metadata Compilation (`scripts/compile_openneuro.py`)
- The pipeline queries the OpenNeuro GraphQL API for a list of all available datasets.
- A centralized HDF5 registry (`data/datasets.h5`) is created. Each dataset becomes a root group (e.g., `/ds000001`), and each subject becomes a subgroup.
- **Manual QA & Rapid Audit**: Through a Flask app (`scripts/label_datasets.py`), researchers manually review the sampled subjects using detailed report views or the high-speed **Rapid Visual Audit tab (`/rapid`)** displaying side-by-side $z=-30$ axial slice images of Subject 1 & Subject 2. Those images are the per-subject QA thumbnails (`deepmreye/thumbnail.py`), served as files rather than re-derived per request. Labels are written as `approved` attributes on the subject subgroups (`1` eyes, `4` eyes faint, `3` eyes cut off, `0`/`2` no eyes, `-1` unlabeled) and mirrored to `data/labels.csv`. A 21-feature Random Forest triage classifier (`deepmreye/qa_classifier.py`, evaluating inner block features, 3-stage ANTs registration transform statistics, and TR/n_trs metadata with 78.5% CV accuracy) pre-selects predictions and ranks subjects by uncertainty. A dataset is approved for training only if all of its labeled subjects show eyes (`1`, `3`, `4`); a single "no eyes" subject drops the whole dataset (`is_dataset_approved` in `deepmreye/pipeline.py`).

### Extraction & Coregistration (`scripts/download_and_preprocess.py`)
- The script iterates through the approved datasets and downloads the raw `_bold.nii.gz` sequences natively bypassing heavy local storage by unpacking on-the-fly.
- **Coregistration**: For every subject, the sequence is registered to a standard space (MTI) using `ANTsPy` (specifically `Affine` and `SyNAggro` transforms).
- **Metadata Validation**: Prior to registration, the raw sequence is strictly validated using `deepmreye.validation`. The Repetition Time (TR) is extracted from the raw NIfTI header. If the TR is missing or invalid (<= 0), the dataset is skipped entirely to preserve temporal dynamics. Valid TRs are written to the `datasets.h5` registry.
- **Voxel Extraction**: A binary eye-mask is applied to the registered BOLD sequence. The pipeline crops out the bounding box containing the eyes. All voxels outside the precise eyeball mask are explicitly zeroed out (`replace_with=0`). The crop is fixed by the mask, so every eye block is `[47, 29, 18, T]`.
- **Normalization**: The cropped block is passed through `normalize_img`: each voxel is z-scored across time, each volume is then z-scored across space, and values are clipped at 5 SD. Masked-out voxels stay exactly 0.
- **QA artifact**: Each subject gets a ~20 KB thumbnail, `data/<dataset>/<subject>.png`: the registered brain at $z=-30$ with the eye mask overlaid in red, then the extracted eye block collapsed along two axes. It is rendered from the *raw* volumes, before normalization — per-voxel z-scoring across time flattens the temporal mean, so anatomy is only visible beforehand. The full Plotly report (~5 MB) is opt-in via `--report html`; over a full extraction it would cost more than 100 GB against roughly 0.5 GB of thumbnails.
- **Transform statistics**: During registration, an affine transformation matrix is produced. Its flattened statistics are recorded in the registry and shown in the subject's HTML QA report. These are diagnostic only; approval is decided by manual labels, not an automatic quality score.
- **Serialization**: Each participant is written to its own gzip-compressed HDF5 file, `data/<dataset>/<subject>.h5`, holding `eye_block` `[X, Y, Z, T]` float32 and — when gaze is known — `labels` `[T, 10, 2]`. OpenNeuro datasets keep their accession as the folder name (`ds000001`); the six gaze-labeled datasets are `dsL01_guided_fixations` … `dsL06_sequences`, so `dsL*/*.h5` selects the probe set by path alone. Chunks span the full spatial extent against a 50-TR slab, so reading one training window touches a few chunks instead of striding the whole run. One file per participant (rather than one per dataset) is what allows parallel extraction: each worker owns its output file outright.

## 2. Windowed Data Loading (`ProbeDataset`)

The PyTorch `Dataset` handles the 4D per-participant arrays by leveraging HDF5's native chunking.

- Reads the per-participant layout, restricted to files that carry a `labels` dataset — the 270 gaze-labeled participants across `dsL01`-`dsL06`.
- **Windowing**: fMRI sequences vary wildly in temporal length ($T$); the dataset samples continuous windows of `window_size` TRs (default 100, stride 50). A sequence shorter than the window cannot be used.
- Splitting is one of four, in increasing strictness: temporally **within each subject** (`split_by="time"`), subject-wise **within each dataset** (`split_by="subject"`, the default), whole datasets held out (`split_by="dataset"`), or a named `holdout={...}` fold, which is what leave-one-dataset-out and leave-one-paradigm-out use. Splitting a pooled shuffle across datasets would leak subjects of the same dataset into both sides and inflate the metrics. The `time` split cuts the *timeline*, not the window index — windows overlap by half a window, so an index-wise split would put the same TRs on both sides.
- Yields the dataset name and the **subject id** alongside the block, because the subject is the unit metrics are aggregated over (§3).
- Outputs `[B, X, Y, Z, window_size]` BOLD blocks alongside `[B, window_size, 10, 2]` label arrays (10 sub-TR sampling points, X & Y coordinates in degrees of visual angle). NaNs are preserved — they mark TRs with no valid gaze sample, and the evaluation masks them; dropping them here would misalign block and gaze in time.
- Windows are a fixed **TR count**, not a fixed duration — datasets differ in TR (0.80-1.25 s), so a window covers a different real span across them. Known limitation; see §4.

## 3. Evaluation: the Gaze Probe and its Baselines

The probe asks how well gaze can be read out of a feature source with a
simple, non-learned readout. The harness is `scripts/eval_probe.py`; the
metrics live in `deepmreye/evaluate/probe.py` and the readouts in
`deepmreye/evaluate/baselines.py`.

### Feature extraction

A batch of `[B, X, Y, Z, window_size]` labeled blocks is downsampled
(stride-4 subsample, then mean-pooled per temporal patch of 5 TRs) into
`[B, N_t, D]` features, keeping the temporal axis rather than collapsing the
whole window to one vector — a window spans 80-250 seconds depending on the
dataset's TR, and pooling it away would force the target to be the mean gaze
over that span. `temporal_targets` reduces the `[B, window_size, 10, 2]`
labels to `[B, N_t, 2]` with the matching binning.

`nanmean`, not `mean`: missing gaze samples are marked NaN and are common
(windows containing at least one NaN are 100% of `dsL03_pursuit` and
`dsL06_sequences`, 61% of `dsL05_free_viewing`). A plain mean turns those into
NaN targets that get dropped, which silently removed two of six labeled datasets
from the evaluation entirely.

### Generalization levels (`--protocol`)

| | train / test |
|---|---|
| `within` | same participant, early timepoints vs late. Train and test share no timepoint; we store no run boundaries, so they are temporally adjacent. |
| `subject` | held-out participants, same scanner and paradigm |
| `dataset` | leave one dataset out, each in turn |
| `paradigm` | leave one paradigm out — `dsL02/03/04` are all smooth pursuit, so holding out one alone still trains on the same task |

### Readouts

`mean` (constant), `linear` (OLS), `ridge`, `ridge-cv` (alpha by inner
LOO-GCV), `pca-ridge`, `pls`, `rf`, `gbt`, `svr`, `lgbm`, `mlp` — the last
three reproducing the comparison in `media/deepmreye_benchmarks.ipynb`
(`svm.SVR`, `lgb.LGBMRegressor`, `neural_network.MLPRegressor`) against that
notebook's own DeepMReye 1.0 CNN, which is not reproduced here.

**`pca-ridge` is the readout to compare everything else against**: unsupervised
compression, then a linear map, with no peek at gaze. If a non-linear readout
(`svr`/`lgbm`/`mlp`/`rf`/`gbt`) does not clear it, the extra complexity is not
earning its keep on this feature source. Ridge alpha is chosen by inner CV
(`ridge-cv`) rather than pinned at `alpha=1.0`, since an under-tuned ridge
baseline is the first thing a reviewer attacks.

### Metrics

Euclidean error in degrees, Pearson $r$ per axis, and $R^2$ against the
**training-set** mean gaze (against the test mean it would flatter a model that
has only learned where this dataset's gaze sits on average).

Aggregation is **per participant, then median across participants**. Pooling
every row together is gameable: if one subject's gaze sits left of another's, a
model that predicts only *which subject this is* scores a high pooled $r$ with
zero within-subject decoding.

**Pearson $r$ is the headline rather than $R^2$.** Cross-dataset predictions are
mis-calibrated in gain (measured 0.11–2.27 against the training scale) with
offsets near zero, which destroys $R^2$ while leaving the correlation intact.
See §4 and `scripts/analyze_calibration.py`.

## 4. Discussion / open limitations

- **Fixed-TR-count windows, not fixed duration.** See §2. Not yet addressed —
  resampling to a common real-time window is the natural fix but is unmeasured.
- **Cross-dataset gain miscalibration.** See §3's metrics note.
  `scripts/analyze_calibration.py` measures it; no unsupervised correction
  tested so far recovers it (z-match, quantile matching, feature
  standardisation, mean-shift all fail — see `CLAUDE.md`).
- **`dsL03_pursuit` is a standing anomaly**: decodes fine within-run/within-paradigm but fails under leave-one-dataset-out — a transfer/calibration failure, not a missing-signal one (consistent with the CCA analysis in `CLAUDE.md`).
