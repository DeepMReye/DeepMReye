# DeepMReye 2.0: Joint Embedding Predictive Architecture for fMRI Eye Tracking

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)
![DeepMReye JEPA](https://img.shields.io/badge/Architecture-JEPA-blue.svg)

DeepMReye 2.0 upgrades the original supervised eye-tracking regression models into a **Joint Embedding Predictive Architecture (JEPA)**. By training exclusively on unsupervised fMRI datasets extracted from OpenNeuro, the network maps spatial/temporal neuro-imaging matrices into scalable representational embeddings securely before deploying a lightweight Linear Probe for gaze coordinate decoding.

![Logo](media/deepmreye_logo.png)

---

## 🚀 Installation & Environment

DeepMReye 2.0 requires extensive dependencies for managing heavy 4D Matrix transformations (`torch`, `h5py`, `numpy`, `wandb`). We highly recommend using `uv` or standard Python `venv` environments to securely sandbox the dependencies.

### Environment Setup
1. **Initialize the Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   # OR using UV for lightning-fast resolution
   uv pip install -r requirements.txt
   ```
3. **Verify PyTorch Backend**: 
   The matrices require deep accelerated computation. Ensure your environment has CUDA (NVIDIA) or MPS (Apple Silicon) enabled properly via `torch.backends`.

---

## 🧠 Architecture Overview

The repository is modularized cleanly into scalable blocks separating Data Extraction, PyTorch Dataloading, and ViT Model topographies. 

### `/scripts` (Execution Pipelines)
- **`compile_openneuro.py`**: Queries the OpenNeuro GraphQL API, mining available fMRI modalities and building the `data/datasets.h5` core registry.
- **`download_and_preprocess.py`**: Executes massive multi-threaded parallelization downloading S3 buckets. Utilizes `deepmreye/preprocess.py` voxel coregistration dynamically bounding 4D spatial grids, and writes the output `.h5` objects natively.
- **`convert_labeled_to_h5.py`**: Synthesizes the provided supervised ground-truth `.npz` records directly into the standardized HDF5 architecture format `[X, Y, Z, T]`.
- **`train_jepa.py`**: The main execution loop processing the heavy Continuous Spatial-Temporal Masking Curriculums.

### `/deepmreye` (PyTorch Core Blocks)
- **`models/jepa.py`**: The Vision Transformer (ViT) implementation spanning Context Encoders, EMA Target Encoders, dual $1D/3D$ positional embeddings, and the Mask Predictor topologies.
- **`models/patcher.py`**: Spatially unfolds `[B, X, Y, Z, T]` datasets into grid tokens (e.g., $8\times8\times8$ cubes) and computes the exact `spatial_ratio` and `temporal_ratio` missing masks seamlessly.
- **`data/jepa_dataset.py` & `probe_dataset.py`**: Dynamic arrays reading native subsets of sequence windows straight from chunked `.h5` matrices off disk to prevent RAM cascading.
- **`evaluate/probe.py`**: The validation modules stripping arbitrary invalid `NaN` label coordinates natively matching the original 1.0 specifications exactly.

---

## 🏃 Execution Instructions & BIDS Datasets

The entire pipeline natively downloads directly from the OpenNeuro AWS S3 buckets. Instead of hard-coding datasets, the process dynamically pulls any resting-state fMRI dataset mapped over GraphQL.

1. **Mine OpenNeuro Metadata**: 
   ```bash
   python scripts/compile_openneuro.py
   ```
2. **Select & QA BIDS Datasets**:
   Run the Streamlit Labeling GUI to manually approve datasets (this ensures raw fMRI acquisitions contain valid eye boundaries). The GUI permanently tags records in `data/datasets.h5` with `approved=1`.
   ```bash
   streamlit run scripts/label_datasets.py
   ```
3. **Extract 4D Sequences**:
   Downloads all `approved=1` BIDS bold records, corespatially masks the eyeballs, and serializes the 4D geometries natively.
   ```bash
   python scripts/download_and_preprocess.py
   ```
4. **Train Unsupervised Masking Representations**:
   ```bash
   HDF5_USE_FILE_LOCKING=FALSE python scripts/train_jepa.py \
       --epochs 100 \
       --batch-size 32 \
       --lr 1e-4 \
       --wandb-project "deepmreye-jepa"
   ```

---

## ⚙️ Hyperparameter Configuration

DeepMReye 2.0 splits configurations dynamically into two tiers:
1. **Global Paths & Augmentations**: The `deepmreye/config.py` Python class controls universal states like base `data_dir` mappings, $train/test$ split proportions, and volumetric data augmentation limits ($rotation/shift/zoom$).
2. **JEPA CLI Parameters**: Local hyperparameters unique to the deep learning architecture are passed explicitly into `scripts/train_jepa.py` preventing hardcoded structural bounds. Key parameters include:
   - `--batch-size`: Scales the heavily intensive 4D tensors per GPU mapping.
   - `--s-ratio-start` & `--t-ratio-start`: The Curriculum spatial/temporal masking drop probability rates linearly annealing into the `--*-end` arrays limits dynamically forcing ViT predictions.

---

## 🧪 Testing and Validation

Comprehensive unit tests cover the integrity of the patchification, target EMA gradients, dimensional grid projections, and Euclidean metrics.

To validate your machine topologies locally before launching large batching workloads:
```bash
pytest deepmreye/tests/test_jepa.py -v
```
This executes isolated geometric `[B, X, Y, Z, T]` continuous blocks mocking real neuro-images sequentially verifying the `fMRIPatcher` scaling architectures and ViT output vectors exactly.

---

## 📁 Data Formats

Unlike the earlier version, DeepMReye 2.0 drops Pickles in favor of native PyTorch-optimized HDF5 datalakes (`.h5`). 
Each dataset structure internally possesses the following sequence:
```
dataset_name.h5
 └── /sub-01
     ├── /eye_block (Matrix: [X, Y, Z, T], Int16, gzip)
     └── /attrs (ML Classification Probabilities)
```

## 🚀 Future Improvements Roadmap

DeepMReye 2.0 provides an enormous architectural leap, but several components can be scaled further:
- **K-Fold Linear Ensembles**: The current supervised linear probe (`evaluate/probe.py`) utilizes a rigid test-split. Moving evaluations toward $k$-fold cross-dataset validations will tighten convergence precision.

## Correspondence
If you have questions regarding the implementation algorithms, mathematical continuous mapping drops, or PyTorch cluster optimizations, contact the primary developers: [EMAIL_ADDRESS] & [EMAIL_ADDRESS]
