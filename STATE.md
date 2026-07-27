# Corpus status and provenance

What exists right now, how it was made, and what is next. For the method see
`overview.md`; for design decisions and cluster constraints see `CLAUDE.md`;
for how to run anything see `README.md` and `slurm/README.md`.

Last updated **2026-07-27**.

## The corpus

Eye-region blocks extracted from OpenNeuro, on scratch and pushed to
`DeepMReye/eyeballs` (private) on HuggingFace.

| | |
|---|---|
| participants | **1779** |
| source datasets | **913** |
| TRs | **662,833** |
| size | **29.0 GB** blocks, +8 GB QA reports |
| with gaze labels | 6 participants (1 dataset) |
| usable for 100-TR windows | 97.5% |

Every block is `[47, 29, 18, T]` float32, normalized identically (per-voxel
z-score across time, per-volume z-score across space, clipped at 5 SD),
`format_version` 2. TRs per participant: median 264, p90 722, max 3600.

This is the **QA sample** — 2 subjects per dataset, enough to decide whether a
dataset has visible eyeballs. Full extraction of the approved datasets comes
after labeling.

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

The 6 gaze-labeled participants were converted into the same container from
`labeled_data/` and verified bit-exact. Labeled and unlabeled participants are
indistinguishable in format; `labels` is simply present or absent.

## Where it stands

**Labeling is the only thing on the critical path.** Everything upstream is
done; everything downstream waits on QA labels. 14 of 1779 subjects are labeled
so far (11 eyes, 3 no-eyes); the registry covers 1772 of them, the remaining 7
being the gaze-labeled participants, which need no QA.

1. **Label**, on a laptop — `python -m deepmreye qa`. See *Working across
   machines* in `README.md` for the Hub round trip.
2. **Triage model** at ~200 labels — `scripts/train_qa_classifier.py --rank`
   orders the rest by uncertainty. It ranks and flags only, never approves.
3. **Full extraction** of the approved datasets: stage without `--sample`,
   rerun the array (`slurm/README.md`). Consider `--cleanup`, since staging the
   full corpus is far larger than the 405 GB QA sample. Then `--flag` the
   classifier over the newly extracted subjects — QA never sees them by hand.
4. **Publish** — `build_index.py --deep`, then `upload_to_hf.py --publish`.

## Open questions

- `dataset2_pursuit.h5` was truncated mid-upload and disappeared from
  `labeled_data/`. Re-upload it and rerun `convert_labeled_to_h5.py`.
- All 6 gaze-labeled subjects are the *same participant* (S4_0004–S4_0009) at
  different TRs, so a subject-wise probe split there is not independent. This
  is the control for the whole method, so it needs the other labeled datasets
  before the probe numbers mean anything.
- `MAX_SUBJECTS_PER_DATASET` is 200 (trim, not drop). At full extraction that is
  ~36k subjects; revisit if that is more than needed.
