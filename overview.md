# DeepMReye 2.0 Explicit Architectural Overview

This document provides a comprehensive, highly-detailed breakdown of the end-to-end DeepMReye 2.0 Joint Embedding Predictive Architecture (JEPA) pipeline. It explains the mechanics of data extraction, tensor shapes, data loading, architectural patchification, 2D continuous masking, and supervised probing without relying on reading source code.

## 1. Data Ingestion & Preprocessing

DeepMReye trains on raw fMRI data sourced from OpenNeuro. The process is broken into metadata compilation and BOLD sequence extraction.

### Metadata Compilation (`compile_openneuro.py`)
- The pipeline queries the OpenNeuro GraphQL API for a list of all available datasets.
- A centralized HDF5 registry (`data/datasets.h5`) is created. Each dataset becomes a root group (e.g., `/ds000001`), and each subject becomes a subgroup.
- **Manual QA**: Through a Streamlit app (`scripts/label_datasets.py`), researchers manually review dataset samples and flag them by adding an `approved=1` attribute to the dataset group in the `.h5` registry.

### Extraction & Coregistration (`download_and_preprocess.py`)
- The script iterates through the `approved=1` datasets and downloads the raw `_bold.nii.gz` sequences natively bypassing heavy local storage by unpacking on-the-fly.
- **Coregistration**: For every subject, the sequence is registered to a standard space (MTI) using `ANTsPy` (specifically `Affine` and `SyNAggro` transforms).
- **Voxel Extraction**: A binary eye-mask is applied to the registered BOLD sequence. The pipeline crops out the bounding box containing the eyes. All voxels outside the precise eyeball mask are explicitly zeroed out (`replace_with=0`).
- **Quality Assurance via ML**: During registration, an affine transformation matrix is produced. A pre-trained machine learning model (`DecisionTreeClassifier` or Random Forest) evaluates the flattened affine statistics to predict how successful the spatial alignment was. This outputs a float between 0.0 and 1.0 and is saved as the `transform_probability` attribute.
- **Serialization**: The extracted 4D bounding box is saved tightly as a contiguous matrix of shape `[X, Y, Z, T]` using `gzip` compression directly inside dataset-specific `.h5` files (e.g., `data/ds000001/ds000001.h5`).

## 2. Pytorch Data Loading & Batching

The PyTorch `Dataset` instances handle the massive 4D arrays securely by leveraging HDF5's native chunking.

### `JEPADataset` (Unsupervised Training)
- **Initialization**: Scans all `[ds_name].h5` files. It evaluates the `transform_probability` for each subject. If a subject's coregistration quality score is below the `prob_threshold` (default `0.7`), the subject is dropped from the training pool entirely.
- **Windowing**: fMRI sequences vary wildly in temporal length ($T$). To standardize batches, the dataset dynamically samples random continuous "windows" of `100` TRs (`window_size=100`). If a sequence is less than 100 TRs, it cannot be used.
- **Output Shape**: Every item yielded by the dataset is a 4D tensor of shape `[X, Y, Z, 100]` where `X, Y, Z` are the spatial bounding box dimensions covering the eyes. A DataLoader batches these into `[B, X, Y, Z, 100]`.

### `ProbeDataset` (Supervised Evaluation)
- Loads from separate, pre-converted labeled HDF5 databases (e.g., `dataset1_guided_fixations.h5`).
- It applies a strict dataset-wise or subject-wise `train/test` split geometry to ensure validation generalization.
- Outputs `[B, X, Y, Z, 100]` BOLD blocks alongside `[B, 100, 10, 2]` label arrays (100 TRs, 10 sub-TR sampling points, X & Y coordinates). 

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

## 6. Evaluation Loop & Linear Probing

Because the unsupervised training learns representations, we validate the semantic meaning of these embeddings by mapping them to explicit gaze tracking coordinates.

- **Linear Probe**: A single shallow `nn.Linear(256, 2)` layer.
- **Inference**: A batch of `[B, X, Y, Z, 100]` labeled data runs through the frozen **Context Encoder** (with zero masking, meaning it processes all tokens).
- **Pooling**: The resulting token sequences are mean-pooled across the $N_s \times N_t$ dimension, resulting in a single `[B, 256]` latent semantic vector per subject sequence.
- **Decoding**: The probe multiplies this by weights to predict a scalar `[X, Y]` coordinate.
- **Metrics**: 
   - A PyTorch Euclidean Distance loss (`MSELoss` / `SmoothL1`) trains the probe.
   - The validation loop cleanly filters out `NaN` ground-truth tracking elements safely. 
   - Performance is quantized into $R^2$ Variance Explained and Pearson Correlation ($r$).

## 7. Crucial Hyperparameters & Optimization

- `embed_dim`, `encoder_depth`, `predictor_depth`: Defines the transformer capacities globally. If memory allows, increasing `embed_dim` to 512 and `encoder_depth` to 12 heavily pushes parameter horizons.
- `batch_size`: Defaults around 32, highly sensitive given native 4D voxel block buffers. Set to maximum available GPU VRAM.
- `max_n_s` and `max_n_t`: Set locally in `Patchify` limits to pad positional embeddings for sequences exceeding normal matrix shapes (e.g. padding to 500 spatial tokens absolute bounds).
- `EMA Momentum`: Linearly anneals from `0.996` towards `1.0`. A high EMA prevents the Target Encoder from collapsing into providing zero-variance trivial outputs.
- `prob_threshold`: The minimum strictness metric required before considering a scanned OpenNeuro anatomy structurally sound for sequence inclusion.
