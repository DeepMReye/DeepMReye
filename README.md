# DeepMReye 2.0

Decode eye gaze from fMRI without an eye tracker.

The signal in the voxels around the eyeballs in a BOLD volume carries gaze
position. Version 1 read it out with a supervised 3-D CNN trained on gaze
labels. This version reads it out with a **frozen linear basis learned from
unlabeled fMRI** plus a ridge readout, and matches the supervised reference
while needing **no data from the target study**.

The method, in full:

1. Fit a linear basis on 2,000 unlabeled OpenNeuro participants by canonical
   correlation between the **left and right orbit** (`lr-cca`). Both eyes rotate
   together, so a direction in left-orbit voxel space that predicts the right
   orbit is a direction driven by conjugate gaze; anything local to one eye is
   suppressed. No gaze labels are involved.
2. Project a new participant's eye-mask voxels onto the leading 32 directions
   and stack them at lags -1, 0, +1.
3. Fit a ridge on the gaze-labeled datasets and read gaze out.

That is the whole model. It is linear end to end and takes seconds to fit.

## Results

Leave-one-dataset-out over 337 gaze-labeled participants in 9 datasets: the
readout is fitted on 8 datasets and scored on the 9th, so every number answers
"does this transfer to a study it has never seen".

| | Pearson r | r_x | r_y | R² * | error * |
|---|---|---|---|---|---|
| **sub-TR** (10 samples per TR) | **0.770** | 0.814 | 0.749 | 0.543 | 2.87° |
| **1-TR** (mean gaze per TR) | **0.838** | 0.862 | 0.817 | 0.616 | 2.34° |

Median over the 9 folds, of the median over participants within a fold. Mean
over folds is **0.721** (sub-TR) and **0.777** (1-TR) -- report it beside the
median, because two folds are genuinely hard and a median alone hides them.

`*` R² and error are **calibrated** -- see "About the metrics" below. Error is
in degrees of visual angle.

The noise floor on a 9-fold median here is about **0.02**, so differences below
that are ties regardless of their direction.

Per fold, sub-TR / 1-TR Pearson r:

| dataset | n | sub-TR | 1-TR | |
|---|---|---|---|---|
| `dsL02_pursuit` | 9 | 0.920 | 0.929 | |
| `dsL04_pursuit` | 34 | 0.851 | 0.864 | |
| `dsL05_free_viewing` | 27 | 0.803 | 0.839 | |
| `dsL03_pursuit` | 24 | 0.776 | 0.808 | resolution-limited, see `FINDINGS.md` |
| `dsL01_guided_fixations` | 170 | 0.770 | 0.786 | |
| `dsL07_deepmreye_calib` | 15 | 0.744 | 0.837 | |
| `dsL06_sequences` | 6 | 0.673 | 0.734 | vertical axis barely sampled |
| `dsL11_backtothefuture` | 37 | 0.671 | 0.848 | |
| `dsL08_studyforrest_movie` | 14 | 0.283 | 0.348 | 7T, worst registration in the corpus |

**Read `FINDINGS.md` before proposing an improvement.** It records what was
tried, what the controls said, and which directions are closed. The short
version: gaze is *linearly* accessible from these features, so a non-linear
encoder in front of a linear readout has nothing to add, and eight separate
attempts confirmed it.

### About the metrics

Pearson r is the headline and needs no calibration -- it is invariant to gain
and offset. R² and Euclidean error are **not**, and the protocol makes that
unavoidable: it z-scores gaze per training dataset before pooling (the
per-dataset scale spans 21 to 595, and without it the pooled ridge follows
whichever dataset has the largest target variance), so predictions come out in
z-units. `deepmreye/metrics.py` fixes this with the smallest honest amount of
supervision: one gain and one offset per axis, fitted on the **other
participants of the same held-out dataset**. No participant sees its own labels,
and the scenario is the realistic one -- calibrate a new study on a few subjects
with an eye tracker, decode the rest without one. Quote R² and error as
*calibrated*; quote r as it stands.

## Install

```bash
uv sync
```

`ants` is needed for ingestion only; evaluation is pure numpy/sklearn.

## Use

Everything runs through one entry point.

```bash
# Fit the basis on the unlabeled corpus (one streaming pass, ~1.3 GB of
# accumulators). Excludes every gaze-labeled dataset by construction.
python -m deepmreye fit-basis --out results/basis.npz --k 256

# Leave-one-dataset-out decoding: r, R2 and error at both resolutions.
python -m deepmreye evaluate --basis results/basis.npz --build-cache

# Refuse to report unless the protocol reproduces its known headline numbers.
python -m deepmreye evaluate --basis results/basis.npz --calibrate
```

To extend the corpus (see "Extending the data" below):

```bash
python -m deepmreye compile --limit 5      # sample subjects for QA
python -m deepmreye qa                     # browser UI: mark eyes / no eyes
python -m deepmreye preprocess             # extract every subject of approved datasets
```

`--data-dir` goes **after** the command. Unset, it resolves `$DEEPMREYE_DATA`,
then `./data`, then downloads from HuggingFace.

## What is in the package

```
deepmreye/
  unsupervised.py   The two linear bases and the streaming accumulator they
                    come from. `lr-cca` is what ships; `corpus-pca` is the
                    variance-ordered reference it is compared against.
  probe.py          The evaluation protocol, written down once: leave-one-
                    dataset-out, scored at sub-TR and 1-TR, with a cache whose
                    guard includes a corpus fingerprint.
  metrics.py        Pearson r, R2, Euclidean error, and the per-dataset affine
                    calibration that makes the last two mean something.

  pipeline.py       Ingestion: S3 download, coregistration, extraction, write.
  preprocess.py     Coregistration and eye-mask extraction (ANTs).
  eyetracking.py    Gaze ingest for OpenNeuro datasets that ship eye tracking.
                    The time origin is *recovered*, never assumed.
  storage.py        The per-participant HDF5 layout. Every read and write of an
                    eye block goes through here.
  registry.py       Worker sidecars, so parallel extraction never writes the
                    registry directly.
  datasource.py     Finds the corpus and decides what each stage downloads.
  labels.py         CSV backup of the QA labels.
  validation.py     TR extraction and validation from NIfTI headers.
  thumbnail.py      The ~20 KB QA image every participant gets.
  qa_classifier.py  Triage model: ranks unlabeled subjects so the uncertain ones
                    get labeled first, and pre-selects a label in the UI. It
                    never approves anything.

scripts/            Portable stages: basis fitting, the QA UI, gaze ingest and
                    its verification, corpus indexing, HuggingFace sync.
slurm/              Cluster-specific staging and extraction, plus run.sbatch.
```

`paper/` and `docs/`, if you have them, are gitignored: the COSYNE abstract, the
figure and its input snapshot, and a note on the claims it makes. Nothing in the
method depends on them.

### The analysis scripts, and what each one answers

These produce the tables in `FINDINGS.md`. Their outputs land in `results/`,
which is gitignored -- **`FINDINGS.md` is the record, not the JSON** -- and the
first four are caches that make the rest cheap.

```
fit_lrcca_variants.py      One corpus pass -> second moment, per-participant and
                           per-slab means, and (--lag1) the lag-1 cross moment.
                           Every covariance in the family is then a subtraction.
build_labeled_voxels.py    All 337 labelled participants' masked voxels as one
                           22 GB memmap, so applying a new basis is a matmul.
corpus_pc_timeseries.py    The corpus as 96,000 x 512 orbital PCs (212 MB), for
                           criteria that need more than a covariance (ICA).
analysis_unlabeled.py      The corpus-scaling, labelled-budget and fold-local
                           references behind the figure.

improve_lrcca.py           Estimator, covariance, symmetry and criterion sweeps.
beyond_cca.py              ICA / dictionary learning as subspace selection.
hybrid_lrcca.py            Corpus vs labelled voxels vs both, one basis per fold.
tradeoff_participants_trs.py  Participants against TRs at a fixed row count.
shift_lrcca.py             Per-participant registration shift, plus its oracle.
anatomical_basis.py        The analytic eyeball-rotation basis (no fitting).
temporal_contrast.py       Lag-1 hard negatives; cross-orbit CCA over a window.
slice_time.py              z-band features and the within-TR sample profile.
subtr_align.py             Is the corpus aligned at sub-TR, not just at TR?
report_lrcca.py            Ranks any of those JSONs by fold, not by median.
eval_dme1.py               The published DeepMReye 1.0 CNN, from OSF weights.
figure_unlabeled.py        The figure; panels selected by letter.
```

## Data

`data/<dataset>/<subject>.h5`, one file per participant: `eye_block [47, 29, 18, T]`
float32, plus `labels [T, 10, 2]` when gaze was recorded. Ten gaze samples per
TR is what makes the sub-TR resolution possible. `data/datasets.h5` is the
registry that carries the manual QA label per subject.

Dataset names carry the subset: `ds######` is an OpenNeuro accession (keep the
real accession -- it is the provenance), `dsL##_<name>` is gaze-labeled. So
`dsL*/*.h5` selects the evaluation set without opening a file.

Current corpus:

- **337 gaze-labeled participants across 9 datasets** -- `dsL01` 170, `dsL11` 37,
  `dsL04` 34, `dsL05` 27, `dsL03` 24, `dsL07` 15, `dsL08` 15, `dsL02` 9,
  `dsL06` 6. All in degrees of visual angle.
- **2,064 unlabeled participants eligible across 654 OpenNeuro accessions**,
  almost all contributing exactly 2 participants each. The shipped basis is fit
  on the first **2,000** of a seed-0 shuffle of those, spanning 649 accessions
  and 96,000 TRs (48 per participant). Count them with
  `unsupervised.unlabeled_subjects`; a participant is eligible only if its eye
  mask is fully covered (14,236 voxels) and its run is at least 32 TRs, which is
  why this is smaller than the raw download.

### Extending the data

Two directions, and they are worth very different amounts.

**More unlabeled participants: probably not worth it.** The corpus-size curve
saturates. `lr-cca` gains +0.15 going from 25 to 800 participants and then
flattens; going from 1039 to 2000 buys nothing measurable. **More TRs per
participant is saturated too** -- eight times the TR budget at N=2000 is worth
+0.004 -- and at a fixed total row count, trading participants for TRs is a tie
from 2000 down to 250 (see *Participants or TRs?* in `FINDINGS.md`). The
mechanism is that a 64-dimensional linear subspace of a 14236-voxel eye mask is
simply easy to estimate, so more data approaches a ceiling that is set by the
target being easy.
If you want to try anyway, `slurm/stage_downloads.py --sample 5` stages five
subjects per dataset instead of two, and `slurm/submit_extraction.sh` extracts
them; the split exists because Leonardo's compute nodes have no network and its
login nodes have a 32 GB cap. See `slurm/README.md`.

**Two fields to add at ingest, next time anything is re-extracted.** Both cost
one line in `pipeline.py` and both unlock an experiment that is otherwise a
460 GB re-download: the per-participant `masked_eye.mean(-1)` and `.std(-1)`
*before* `normalize_img` (the stored block is z-scored per voxel, so the
anatomy is gone), and the `SliceTiming` vector (a volume is a sweep, not an
instant, and under multiband the map from z to acquisition time is periodic).
See the analytic-basis and slice-timing entries in `FINDINGS.md`.

**More labeled datasets: this is the scarce resource.** Every claim here rests
on nine leave-one-dataset-out folds, and independent *acquisitions* -- not
participants -- are what a fold is. `scripts/scan_eyetracking_datasets.py` finds
OpenNeuro accessions that already ship gaze; `scripts/fetch_eyetracking.py`
ingests one; `scripts/verify_gaze_sync.py` proves the alignment before it counts.
Do not skip the verification: three datasets passed a lag sweep with healthy
margins while their vertical axis was inverted, and two were retired only after
a cross-dataset readout caught it.

## Development

```bash
pytest deepmreye/tests/ -q
```

The tests worth knowing about are the ones guarding failures that are otherwise
*silent*: the covariance accumulator that returns zeros without raising
(`test_unsupervised.py`), the cache that loads a retired corpus without
complaint (`test_probe.py`), and the R² that produces a plausible number from
mismatched units (`test_metrics.py`).
