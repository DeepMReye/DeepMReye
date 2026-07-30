# Corpus status and provenance

What exists right now, how it was made, and what is next. For the method see
`overview.md`; for design decisions and cluster constraints see `CLAUDE.md`;
for how to run anything see `README.md` and `slurm/README.md`.

Last updated **2026-07-29**.

## The corpus

Eye-region blocks extracted from OpenNeuro, on scratch and pushed to
`DeepMReye/eyeballs` (private) on HuggingFace.

| | |
|---|---|
| participants | **2043** |
| source datasets | **918** |
| TRs | **1,007,592** |
| size | **46.7 GB** blocks, +32 MB QA thumbnails |
| with gaze labels | **270** participants (6 datasets, `dsL01`–`dsL06`) |
| unlabeled (QA sample) | 1773 participants, 912 datasets |
| usable for 100-TR windows | 97.8% |

Every block is `[47, 29, 18, T]` float32, normalized identically (per-voxel
z-score across time, per-volume z-score across space, clipped at 5 SD),
`format_version` 2. Median 270 TRs per participant. Every participant also has a
~20 KB QA thumbnail beside its HDF5 (`deepmreye/thumbnail.py`); the 1773
OpenNeuro subjects additionally still have their 5 MB HTML reports, 9.1 GB in
total, which are no longer the default artifact.

The unlabeled half is the **QA sample** — 2 subjects per dataset, used to decide
dataset eyeball visibility. Manual QA and Rapid Visual Audit (`/rapid`) are
**COMPLETE**: **697 of 912 datasets approved**, all 1772 sampled subjects
labeled (1420 eyes / 352 no-eyes). Labels are pushed to HuggingFace.

The labeled half is the probe control, and is now complete: all 270 gaze-labeled
participants are converted, registered and carry their acquisition TR (0.80 /
0.87 / 1.02 / 1.00 / 1.00 s for datasets 1-5, per-subject for dataset 6). This
replaces the earlier state where only 6 of them existed in the corpus.

Full extraction of all subjects across the 697 approved datasets is next on the
critical path.

### Paths

```
/leonardo_work/EUHPC_D21_101/mfrey/dme/DeepMReye     repo (+ .venv, Python 3.11)
/leonardo_scratch/fast/AIFAC_S07_154/mfrey/dme/
    data/          <dataset>/<subject>.h5, datasets.h5, index.parquet, labels.csv
    staging/       downloaded .nii.gz + manifest.jsonl + resolved.jsonl
    labeled_data/  source labeled gaze datasets (nested <dataset>.h5)
```

`/leonardo_work` is 96% full — keep data on scratch.

## How it was made

OpenNeuro has **2394** datasets; ~1206 contain BOLD at all. Of those, 2287
subjects resolved to a downloadable functional run and **1801** downloaded
(the rest mostly HTTP 403 on restricted datasets — accepted, not a gap worth
chasing). Those 1801 went through ANTs coregistration to the DeepMReye template
(`Affine`, `Affine`, `SyNAggro`), eye-mask extraction, and normalization, in a
46-task SLURM array.

Losses at extraction, ~2% in total, all recorded in
`staging/deferred_*.jsonl` rather than dropped: 16 unreadable NIfTI, 11
contained ANTs memory blowups, 9 missing or invalid TR headers. Retrying them
is possible and not worth it.

All 270 gaze-labeled participants were converted into the same container from
`labeled_data/` (`scripts/convert_labeled_to_h5.py`), renamed to `dsL01`-`dsL06`,
and entered in the registry as approved. Shapes, label alignment and TRs verified
for every one. Labeled and unlabeled participants are indistinguishable in
format; `labels` is simply present or absent, and the `dsL` prefix is the only
thing that separates them by path.

## Where it stands

**QA labeling is complete and synced to HuggingFace**, and the labeled control
set is now complete too. The ground-truth labels across the 912 OpenNeuro
datasets have been verified via detailed QA and the **Rapid Visual Audit tab
(`/rapid`)**, and pushed to `DeepMReye/eyeballs`.

Since then:
- **QA thumbnails replaced the HTML reports** as the default artifact
  (`deepmreye/thumbnail.py`). 1773 reports = 9.1 GB; the same 1773 thumbnails =
  29 MB, 310x smaller. Extraction writes PNG by default (`--report png|html|both`),
  the `qa` stage now downloads every thumbnail up front instead of streaming
  reports per dataset, and `/zview` serves a file rather than parsing 5 MB of
  embedded base64 per request. Backfill with `scripts/backfill_thumbnails.py`.
- **The 264 remaining gaze-labeled participants** were converted, named `dsL*`,
  given their acquisition TRs, and registered.

Key improvements implemented earlier:
- **21-Feature Triage Classifier**: Evaluates 10 inner-mask features, 8 3-stage ANTs registration transform statistics (including `step1_vs_step2_affine_diff` and `step1_vs_step2_trans_diff`), and 3 sequence metadata metrics (`repetition_time`, `n_trs`, `scan_duration_sec`). Achieves **78.5% ($\pm 4.9\%$) GroupKFold CV Accuracy**.
- **Rapid Visual Audit UI (`/rapid`)**: Interactive high-density grid displaying on-the-fly $z=-30$ axial brain slice + red eye mask overlay PNG images side-by-side for Subject 1 & Subject 2 across all 739 qualifying eye-present datasets. Real-time click-to-remove toggling synced instantly to `datasets.h5` and `labels.csv`.

---

## Current phase: replicate the classic-regressor benchmark on the current corpus

JEPA self-supervised pretraining was tried on this codebase (see the
`pytorch-jepa` branch) and set aside: after correcting a broken random-encoder
control, an *untrained* encoder scored the same as every trained configuration
tested (widths 8-256, 7 learning rates, 4 mask schedules) — nothing showed
training helps, so there is nothing to build further on right now. This branch
drops JEPA and goes back to the question DeepMReye 1.0 originally answered with
a supervised CNN: how well can gaze be read straight off fMRI voxels with
classic regressors? `media/deepmreye_benchmarks.ipynb` (an old branch's
notebook) compared Ridge, SVR, LightGBM (`lgb.LGBMRegressor`) and an MLP
against DeepMReye 1.0's CNN, per dataset. `deepmreye/evaluate/baselines.py` now
has all three non-CNN regressors (`svr`, `lgbm`, `mlp`, alongside the existing
`ridge-cv`/`pca-ridge`/`pls`/`rf`/`gbt`), reproducing that comparison is the
current goal.

Full extraction (20-28k more subjects) is **not** the next step regardless: the
unlabeled corpus does not matter to this comparison at all, only the 270
gaze-labeled participants (`dsL01`-`dsL06`) do.

1. **Baselines, `ridge-cv` only — done, as a table, not a number.**
   `scripts/eval_probe.py`, four generalization levels (`within` / `subject` /
   `dataset` / `paradigm`). Rerun with:
   ```
   python scripts/eval_probe.py --protocol dataset --readouts mean linear ridge-cv pca-ridge pls
   ```
   **Headline numbers** (per-subject median Pearson r, `ridge-cv` readout on
   raw stride-4 voxels):

   | protocol | best case | worst case |
   |---|---|---|
   | `within` (258 subj) | r 0.84/0.84, R² 0.58 | — |
   | `subject` (54 held out) | r 0.83/0.81, R² 0.54 | — |
   | `dataset` (leave-one-out) | dsL02 r 0.89/0.83, R² 0.59 | **dsL03 r 0.14/0.22, R² −0.78** |
   | `paradigm` (leave-task-out) | fixation r 0.84/0.77 | pursuit r 0.63/0.60, R² −0.22 |

   `dsL03_pursuit` is a standing anomaly: decodes fine within-run/within-paradigm
   but fails under leave-one-dataset-out — a transfer/calibration failure, not a
   missing-signal one (consistent with the CCA analysis, see `CLAUDE.md`). GBT
   vs `ridge-cv` on raw voxels is a coin flip on every fold (±0.05 R²) — no
   nonlinear gain to be had on this feature source with tree models; whether
   SVR/LightGBM/MLP do better is the open question below.

2. **Next**: run with the new readouts and record the result here.
   ```
   python scripts/eval_probe.py --protocol dataset --readouts ridge-cv svr lgbm mlp
   ```
   SVR is O(n²)-O(n³) in training rows and `--protocol dataset` trains each
   fold on five pooled datasets (tens of thousands of rows after flattening) —
   watch for it being impractically slow; `--max-windows` subsamples if so.

### Full extraction, when it is time

0. **Size the job first.** `stage_downloads.py --resolve-only` gives the exact
   subject count over the approved datasets in ~15 min. Extrapolating from
   48.7k subjects across ~1206 BOLD datasets puts it at roughly 20-28k, i.e.
   320-450 GB of blocks — over 10x the current corpus.
1. Stage on a login node, extract on compute: `slurm/submit_extraction.sh`.
   `python -m deepmreye preprocess` does both in one process and is unusable at
   this scale (see `CLAUDE.md`, Running on Leonardo). Staging is the
   constraint, not the output: raw NIfTI averages 155 MB, so `--cleanup` is not
   optional. Do **not** pass `--report html`; that is the >100 GB path.
2. **QA at scale**: 703 contact sheets of <=200 thumbnails each, rather than
   25k individual subjects. Sort by triage-classifier confidence so a cutoff can
   be picked by scrolling, then include/exclude by hand at the margin. Plus
   `python scripts/auto_label_datasets.py` (or `qa_classifier --flag`) to flag
   outlier no-eyes subjects that the 2-subject QA sample never saw.
3. **Publish corpus**: `python scripts/build_index.py --deep`, then
   `python scripts/upload_to_hf.py --publish`.

## Open questions

- `dsL06_sequences`'s 6 subjects are the *same participant* (S4_0004–S4_0009) at
  different TRs, so a subject-wise probe split there is not independent. The
  other five labeled datasets are now in the corpus, so the probe no longer
  rests on that dataset alone — split with `split_by="dataset"` to check
  transfer across scanner and paradigm.
- `dsL02_pursuit` has 9 subjects, converted from `.npz`. Its nested
  `dataset2_pursuit.h5` was truncated mid-upload; the `.npz` exports were intact
  and are what the corpus was built from, so this is no longer blocking.
- `ds006190/sub-24630` is extracted on disk but has no registry record — a
  worker sidecar that was never merged. `python -m deepmreye merge-registry` on
  the cluster fixes it.
- `MAX_SUBJECTS_PER_DATASET` is 200 (trim, not drop). At full extraction that is
  ~36k subjects; revisit if that is more than needed.
