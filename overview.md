# DeepMReye 2.0 Explicit Architectural Overview

This document provides a comprehensive, highly-detailed breakdown of the end-to-end DeepMReye 2.0 Joint Embedding Predictive Architecture (JEPA) pipeline. It explains the mechanics of data extraction, tensor shapes, data loading, architectural patchification, 2D continuous masking, and supervised probing without relying on reading source code.

For a short orientation (project state, key decisions, layout) see `CLAUDE.md`.
For install and usage see `README.md`. This file is the deep method reference.

> **Paper sync:** the Method section of the paper (`paper/main.tex`) mirrors the
> method described here. When the pipeline or model changes substantially,
> update both. See `paper/README.md` for the build command and the section-to-code
> mapping.

## 0. Running the Pipeline

There is a single entry point. Every stage runs through it:

```bash
python -m deepmreye compile --data-dir data --limit 5   # 1. sample subjects for QA
python -m deepmreye qa --data-dir data                  # 2. browser labeling UI (eyes / no eyes)
python -m deepmreye preprocess --data-dir data          # 3. extract all subjects of approved datasets
python -m deepmreye train --data-dir data -- --epochs 50 # 4. train JEPA (extra args after `--` go to train_jepa)
python -m deepmreye all --data-dir data                 # run 1-4, pausing for QA
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
- **Normalization**: The cropped block is passed through `normalize_img`: each voxel is z-scored across time, each volume is then z-scored across space, and values are clipped at 5 SD. Masked-out voxels stay exactly 0. This runs at extraction so stored data matches the labeled gaze datasets, which were normalized the same way — pretraining on raw BOLD and probing on z-scored data would put two different input distributions through one encoder.
- **QA artifact**: Each subject gets a ~20 KB thumbnail, `data/<dataset>/<subject>.png`: the registered brain at $z=-30$ with the eye mask overlaid in red, then the extracted eye block collapsed along two axes. It is rendered from the *raw* volumes, before normalization — per-voxel z-scoring across time flattens the temporal mean, so anatomy is only visible beforehand. The full Plotly report (~5 MB) is opt-in via `--report html`; over a full extraction it would cost more than 100 GB against roughly 0.5 GB of thumbnails.
- **Transform statistics**: During registration, an affine transformation matrix is produced. Its flattened statistics are recorded in the registry and shown in the subject's HTML QA report. These are diagnostic only; approval is decided by manual labels, not an automatic quality score.
- **Serialization**: Each participant is written to its own gzip-compressed HDF5 file, `data/<dataset>/<subject>.h5`, holding `eye_block` `[X, Y, Z, T]` float32 and — when gaze is known — `labels` `[T, 10, 2]`. OpenNeuro datasets keep their accession as the folder name (`ds000001`); the six gaze-labeled datasets are `dsL01_guided_fixations` … `dsL06_sequences`, so `dsL*/*.h5` selects the probe set by path alone. Chunks span the full spatial extent against a 50-TR slab, so reading one training window touches a few chunks instead of striding the whole run. One file per participant (rather than one per dataset) is what allows parallel extraction: each worker owns its output file outright.

## 2. Pytorch Data Loading & Batching

The PyTorch `Dataset` instances handle the massive 4D arrays securely by leveraging HDF5's native chunking.

### `JEPADataset` (Unsupervised Training)
- **Initialization**: Scans the registry and includes every subject that belongs to a manually approved dataset (`is_dataset_approved`) and has an extracted `eye_block` on disk. There is no automatic quality gate; QA is the manual eyes / no-eyes labeling.
- **Windowing**: fMRI sequences vary wildly in temporal length ($T$). To standardize batches, the dataset dynamically samples random continuous "windows" of `100` TRs (`window_size=100`). If a sequence is less than 100 TRs, it cannot be used.
- **Output Shape**: Every item yielded by the dataset is a 4D tensor of shape `[X, Y, Z, 100]` where `X, Y, Z` are the spatial bounding box dimensions covering the eyes. A DataLoader batches these into `[B, X, Y, Z, 100]`.

### `ProbeDataset` (Supervised Evaluation)
- Reads the same per-participant layout as `JEPADataset`, restricted to files that carry a `labels` dataset. Labeled and unlabeled participants are byte-format identical, so one code path serves both.
- Splitting is one of four, in increasing strictness: temporally **within each subject** (`split_by="time"`), subject-wise **within each dataset** (`split_by="subject"`, the default), whole datasets held out (`split_by="dataset"`), or a named `holdout={...}` fold, which is what leave-one-dataset-out and leave-one-paradigm-out use. Splitting a pooled shuffle across datasets would leak subjects of the same dataset into both sides and inflate the metrics. The `time` split cuts the *timeline*, not the window index — windows overlap by half a window, so an index-wise split would put the same TRs on both sides.
- Yields the dataset name and the **subject id** alongside the block, because the subject is the unit metrics are aggregated over (§6).
- Outputs `[B, X, Y, Z, 100]` BOLD blocks alongside `[B, 100, 10, 2]` label arrays (100 TRs, 10 sub-TR sampling points, X & Y coordinates) in degrees of visual angle. NaNs are preserved — they mark TRs with no valid gaze sample, and the evaluation masks them; dropping them here would misalign block and gaze in time.

## 3. Patchification (`fMRIPatcher`)

The initial layer of the `JEPAModel` translates the continuous 5D batches `[B, X, Y, Z, T]` into a sequence of flat transformer tokens.

- **Spatial Grouping**: The 3D volume is chopped into small cubes, by default `spat_patch_size = 8`. This means $8 \times 8 \times 8$ voxel blocks.
- **Temporal Grouping**: The temporal dimension (100) is chopped into chunks of `temp_patch_size = 5` TRs.
- **Mask-Aware Extraction**: Because the spatial bounding box contains a lot of empty space (corners outside the spherical eyes), the patcher only creates tokens for spatial cubes that contain actual brain/eye data (non-zero variance). Let's say out of the grid, $N_s$ spatial blocks are valid (e.g., ~30 blocks). 
- **Token Grid**: The temporal dimension produces $N_t = 100 / 5 = 20$ temporal chunks. The grid size is exactly $N_s \times N_t$ tokens per batch. 
- **Linear Projection**: Each extracted patch (a flat array of $8 \times 8 \times 8 \times 5 = 2560$ float values) is passed through an `nn.Linear` layer projecting it into the transformer embedding dimension `embed_dim=256`.

### Positional Embeddings
Because transformers lack inherent geometric knowledge, the model adds two distinct sets of learned embeddings to the tokens:
1. **Spatial Embeddings (`pos_s`)**: Learned for each of the $N_s$ spatial locations.
2. **Temporal Embeddings (`pos_t`)**: Learned for each of the $N_t$ temporal bins.
They are broadcasted and added: `Token[s, t] = Projection + pos_s[s] + pos_t[t]`.

## 4. Continuous 2D Masking Curriculum

Unlike traditional BERT-style random dropout, DeepMReye 2.0 uses "Double-Cross" contiguous masking controlled by two hyperparameters: `spatial_ratio` and `temporal_ratio` (both $\in [0, 1]$).

1. Determine drop counts: `num_drop_s = int(N_s * spatial_ratio)` and `num_drop_t = int(N_t * temporal_ratio)`.
2. Randomly sample `num_drop_s` unique spatial indices to drop.
3. Randomly sample `num_drop_t` unique temporal indices to drop.
4. **Token Resolution**: A token at coordinate `(s, t)` is categorized as a **Target** (masked out) if its spatial index `s` is dropped OR its temporal index `t` is dropped. Otherwise, it is a **Context** token (visible).
5. **Curriculum learning**: During epoch progression, training starts "easy". 
   - Epoch 1: `s_ratio=0.1, t_ratio=0.1` (model sees almost the whole volume).
   - Epoch N: `s_ratio=0.5, t_ratio=0.5` (model is starved, forcing deep feature interpolations).

## 5. Core ViT Architecture (`JEPAModel`)

The network is composed of three Vision Transformers:

1. **Target Encoder (EMA)**
   - Inputs: Only the **Target** tokens (masked parts).
   - Operation: Computes the latent representations of what the model *should* predict.
   - Weights: Does not receive gradients. Its weights are mathematically updated every step as an Exponential Moving Average (EMA) of the Context Encoder. 
2. **Context Encoder**
   - Inputs: Only the **Context** tokens (visible parts).
   - Operation: Learns the observable geometry. Backpropagates actively.
3. **Predictor**
   - Inputs: The output representations of the Context Encoder, PLUS a learnable `[MASK]` token appended with the original positional embeddings of the **Target** coordinates.
   - Operation: Attempts to guess the exact output vectors produced by the Target Encoder for the masked regions.
   - Loss: Computes the `SmoothL1Loss` or Euclidean distance between the Predictor's output vectors and the Target Encoder's output vectors.

## 6. Evaluation: the Gaze Probe and its Baselines

The probe is a *measurement of the representation*, not a model, so it is kept
deliberately weak: a linear map, fitted in closed form, on frozen features. The
harness is `scripts/eval_probe.py`; the metrics live in
`deepmreye/evaluate/probe.py` and the readouts in
`deepmreye/evaluate/baselines.py`.

### Feature extraction

A batch of `[B, X, Y, Z, 100]` labeled blocks runs through the frozen **Context
Encoder** with zero masking, producing `[B, N_s * N_t, D]` tokens.

**Pooling is over space only.** Mean-pooling over both axes gives one vector per
window and forces the target to be the mean gaze over that whole window — 80 to
250 seconds depending on the dataset's TR. Measured on the labeled corpus, that
discards **84–96% of the gaze variance** (within-window SD 2.4–7.1°, SD of the
window means 0.12–1.11°), so the probe would be asked to predict a nearly
constant target. `pool_spatial` therefore returns `[B, N_t, D]`, one embedding
per temporal patch, and `temporal_targets` reduces the `[B, 100, 10, 2]` labels
to `[B, N_t, 2]` with the same binning and padding the patcher uses.

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

### Arms and readouts

Three feature sources — `voxels` (stride-4 downsampled raw voxels, no learning),
`random` (the same architecture untrained), `trained` (a JEPA checkpoint) —
crossed with the same readouts: `mean`, `linear` (OLS), `ridge`, `ridge-cv`
(alpha by inner LOO-GCV), `pca-ridge`, `pls`, `rf`, `gbt`.

The bar the method must clear is **`pca-ridge` on raw voxels**, not the random
encoder. A random ViT is a non-linear random projection that scores *below* raw
voxels everywhere, so beating it demonstrates nothing; `pca-ridge` is the honest
competitor because it is the same shape of method — compress without seeing
gaze, then fit a linear map.

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
See §Discussion and `scripts/analyze_calibration.py`.

## 7. Crucial Hyperparameters & Optimization

- `embed_dim`, `encoder_depth`, `predictor_depth`: Defines the transformer capacities globally. If memory allows, increasing `embed_dim` to 512 and `encoder_depth` to 12 heavily pushes parameter horizons.
- `batch_size`: Defaults around 32, highly sensitive given native 4D voxel block buffers. Set to maximum available GPU VRAM.
- `max_n_s` and `max_n_t`: Set locally in `Patchify` limits to pad positional embeddings for sequences exceeding normal matrix shapes (e.g. padding to 500 spatial tokens absolute bounds).
- `EMA Momentum`: Linearly anneals from `0.996` towards `1.0`. A high EMA prevents the Target Encoder from collapsing into providing zero-variance trivial outputs.
- `window_size`: Number of TRs per training window (default `100`). Sequences shorter than this are skipped.
