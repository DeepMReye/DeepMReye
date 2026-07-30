# Corpus status and provenance

What exists right now, how it was made, and what is next. For the method see
`overview.md`; for design decisions and cluster constraints see `CLAUDE.md`;
for how to run anything see `README.md` and `slurm/README.md`.

Last updated **2026-07-29**.

**Corpus note:** the raw staged NIfTI have been deleted — 428 GB, every one of
them verified present on HuggingFace at a byte-identical size first. The 29 that
never produced a block are kept and listed in `staging/retained_raw.jsonl`, as
are `manifest.jsonl` / `resolved.jsonl` / `deferred_*.jsonl`. `staging/` is now
6.5 GB. The raw data is re-downloadable from OpenNeuro; the labels and blocks
are not, and those are on the Hub.

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

## Current phase: test the model on the corpus we already have

Full extraction is deliberately **not** the next step. The unlabeled half is
still the 2-subjects-per-dataset QA sample (1332 usable subjects across 703
approved datasets, `dsL*` excluded), and that is enough to answer the question
that gates everything else: *does the self-supervised representation beat
reading gaze straight off the voxels?* If it does not, extracting 20k more
subjects does not help.

So the order is: baselines → JEPA → compare → only then scale up. The first
three are done locally on a laptop (CPU/MPS); **JEPA training itself needs
Leonardo GPUs next** — a local epoch takes minutes and 30 epochs was not
finished before moving over.

1. **Baselines** — done, as a table, not a number. `scripts/eval_probe.py`,
   four generalization levels (`within` / `subject` / `dataset` / `paradigm`)
   crossed with the readout zoo (`mean`, `linear`, `ridge-cv`, `pca-ridge`,
   `pls`, `gbt`). Rerun locally with:
   ```
   python scripts/eval_probe.py --protocol dataset --arms voxels random \
       --feature-cache .cache/features --readouts mean linear ridge-cv pca-ridge pls
   ```
   **Headline numbers** (per-subject median Pearson r, `ridge-cv` readout on
   raw stride-4 voxels — the bar to beat, *not* the random encoder, which
   scores near zero everywhere post-fix):

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
   nonlinear gain to be had on this feature source, so if JEPA beats the table
   it will not be because a nonlinear map became available.

2. **JEPA pretraining bugs found and fixed while wiring up the eval** (both
   have regression tests):
   - The `split_by="time"` (within-subject) split silently dropped `dsL01` —
     170 of 270 labeled subjects — from the test side, because window starts
     sit on a stride-50 grid and `start >= cut` had no solution for 270-TR
     runs. Fixed in `deepmreye/data/probe_dataset.py`.
   - **Positional embeddings were drowning the patch signal.**
     `nn.Embedding` defaulted to `N(0,1)` while patch tokens come out at
     std ~0.36 — the input was 6% of token variance before the transformer
     even started, diluted to 0.05% after 6 layers. Measured effect: an
     untrained random encoder fit training subjects at r≈0.45 and transferred
     at r≈0.00 (only the position pattern, identical for every window,
     survived pooling). Fixed with `std=0.02` init
     (`deepmreye/models/jepa.py`) — signal share 6% → 99.4%. **This landed
     after one epoch of what was otherwise a real training run; any
     checkpoint from before this fix is invalid and was discarded.**
   - `train_jepa.py` was separately broken before this session (pointed at a
     `labeled_data/` dir that no longer exists, dead `ds_key` reference, no
     checkpoint saving at all) and was rewritten. It now writes `last.pt`
     every epoch plus `epoch###.pt` every `--save-every`, each carrying its
     own architecture dict so `eval_probe.py` cannot load weights into a
     mismatched model shape.
   - A 3-epoch smoke run (batch 16, embed 256, depth 6) confirmed the full
     loop end-to-end: loss decreasing, and the `trained` arm on the `subject`
     protocol went from random-encoder r≈0.03 to r≈0.56 — real learning, just
     not yet near the voxel baseline (r≈0.83) at 3 epochs, which is expected.
     That checkpoint was for wiring verification only and was not kept as a
     result.

3. **JEPA on Leonardo GPUs — done. The answer is yes, once the probe stopped
   averaging away the signal it was supposed to measure.** `slurm/` now has
   `train_jepa.sbatch`, `sweep_jepa.sbatch`, `eval_probe.sbatch` and
   `eval_checkpoints.sbatch`; see `CLAUDE.md` §GPU training for the environment
   traps each cost a job. The baseline table above **reproduces exactly** on the
   cluster (dsL02 r 0.887/0.825 R² 0.589; dsL03 r 0.136/0.224 R² −0.784;
   `subject` r 0.834/0.805 R² 0.535), so the corpus is identical and the arms
   are comparable.

   **First pass was wrong, and here is why.** `encoder_features` fed
   `pool_spatial`, which mean-pools the encoder's 72 spatial tokens (a 6×4×3
   grid, patch size 8 over the 47×29×18 box) into one 256-vector per temporal
   bin. Gaze direction is encoded as *contrast across the orbit* — averaging it
   away is not a small loss. Measured on one fold (dsL01, same checkpoint,
   same everything else): pooled r 0.454/0.392, unpooled r 0.855/0.768. The
   `--protocol dataset` headline computed under pooling was **0.37 mean r
   against voxels' 0.647** and looked like a hard plateau (13-config sweep,
   21-epoch curves, all within 0.03 of each other, `random` encoder wrongly
   scoring 0.009). None of that was false — it correctly measured a broken
   readout. `deepmreye/evaluate/probe.py`'s `pool_spatial` is unchanged (it is
   also what the in-training monitor and, notably, `analyze_identifiability.py`
   use); `scripts/eval_probe.py` grew `--spatial-pool {mean,none,GXxGYxGZ}` to
   read the same checkpoints without pooling.

   **Second pass, full spatial resolution (`--spatial-pool 6x4x3`, 18,432
   features/bin), `ridge-cv` only (it beat `pca-ridge` at every grid tested —
   the latter's global 32-component compression costs ~0.1 r, so do not use
   it here), mean over 6 leave-one-dataset-out folds:**

   | arm | mean r |
   |---|---|
   | `voxels` (ridge-cv) | **0.623** |
   | `trained` JEPA (`base`/`mask-temporal`/`lr-low`, epochs 3–21) | **0.61 – 0.64** |
   | `random` encoder, same resolution | ~~**−0.003**~~ — **wrong, see below** |

   Every trained config, at every epoch from 3 onward, sits within noise of the
   voxel baseline. Plot: epoch curves for the three long-run configs against
   both controls — see `results/curve_controls.json` and
   `results/ckpt_*_dataset_epoch*.json` (`wide`, embed_dim 384 → 27,648
   features, timed out at 3h10m/6-fold and was dropped rather than requeued at
   this priority).

   **The `random` row was an artifact, and correcting it removes the evidence
   that training does anything.** `eval_probe.py` built a *fresh*
   `JEPAModel()` for the train split and another for the test split
   (`get_features` called `build_model` per split). For `trained` that is
   harmless — both loads come from the same checkpoint — but for `random` it
   means the readout was fitted in one random basis and asked to predict in an
   unrelated one. It scored ≈0 by construction, whatever the architecture does.
   Fixed 2026-07-30: `build_arms` constructs one seeded encoder per arm for the
   whole evaluation, with a regression test
   (`test_eval_probe_random_arm_uses_one_encoder_for_both_splits`).

   Same fold, same weights, same everything else — one shared random encoder
   instead of two (`embed_dim=32`, `--spatial-pool 6x4x3`, `ridge-cv`):

   | dsL01 fold | r_x | r_y | R² |
   |---|---|---|---|
   | before (two random inits) | −0.277 | 0.019 | −0.446 |
   | after (one random init) | **0.848** | **0.758** | **0.373** |

   Over all six folds the corrected `random` control is **mean r 0.610**
   (`results/random_d32_sharedweights.json`: 0.803 / 0.796 / 0.141 / 0.639 /
   0.693 / 0.586, reproducing even the `dsL03` anomaly) — level with `voxels`
   at 0.623 and inside the trained band of 0.60–0.66. The in-training probe
   agrees independently: at **epoch 0, before a single gradient step**, it reads
   mean r 0.635 across the same six folds.

   So the honest reading of this table is now: at `embed_dim=32` and full
   spatial resolution, an **untrained** encoder already matches raw voxels, and
   no trained configuration has been shown to beat it. The patcher's first layer
   is a random linear map of 8³×5 raw voxels into 32 dims, i.e. a random
   projection, and a random projection preserves linear decodability — which is
   what the corrected control measures. Unmeasured: the same control at
   `embed_dim=256`, where the earlier `base` runs sit; the bug applied there
   too, so that `random` row is equally void.

   **How wide should a patch's embedding be — measured, not guessed.**
   `pool_spatial` was one axis (positions collapsed to 1); the other is
   `embed_dim` (dims per position). Four 3-epoch runs at `mask-temporal`'s mask
   schedule, evaluated the same way:

   | embed_dim | features (72×d) | r @ epoch 1 | r @ epoch 3 |
   |---|---|---|---|
   | 8 | 576 | 0.572 | 0.524 |
   | 16 | 1,152 | 0.608 | 0.617 |
   | **32** | **2,304** | 0.634 | **0.636** |
   | 64 | 4,608 | 0.636 | 0.629 |
   | 256 (`base`) | 18,432 | — | 0.619 |

   **32 dims/patch matches 256 dims/patch** — an 8x narrower encoder, already
   at the voxel baseline after one epoch. 8 dims is where it degrades. This
   halves as two separate findings: 256 was never necessary at the *readout*
   (per-patch embeddings are redundant), which is a different claim from
   whether 256 is necessary for the *model to train well* — these runs were
   only 3 epochs, so a longer run at `embed_dim=32` is the next cheap check
   before treating 32 as final.

   The 13-config sweep and its in-training-monitor ranking are **not**
   invalidated by this — they ruled out training length and most
   hyperparameters as an explanation for the (mistaken) plateau, and that
   conclusion still holds; only the plateau's cause changes, from "the
   representation is capped" to "the readout was capped."
4. **Then** full extraction of the 703 approved datasets, which is where the
   premise of the method (unlabeled fMRI is abundant) actually gets tested —
   see below. Step 3 now shows the representation **matches** raw voxels at
   this corpus size and readout; extraction is the test of whether it can
   *beat* voxels, which nothing here yet shows either way.

**The in-training probe was rebuilt to compute the real number, and now logs to
wandb.** It used to call `pool_spatial` and so inherited the pooling issue above
(compounded: capped to 400 windows, `split_by="subject"` not `"dataset"`).
Paired against `eval_probe` on identical weights it sat +0.11 to +0.24 high —
about what the protocol difference alone predicts — so its level was defensible
and its flatness over training was numerically correct, for the wrong reason.
Its *ordering* was not: at epoch 21 it ranked `lr-low` > `base` >
`mask-temporal` where `eval_probe` ranks them exactly reversed.

`train_jepa.py` now runs full leave-one-dataset-out over all six labeled sets at
`--spatial-pool 6x4x3` with `ridge-cv` — the same computation
`eval_probe.py --protocol dataset` performs, through the same shared
`collapse_spatial` — plus the `subject` protocol alongside, and logs every
dataset's `pearson_r`/`r2`/`euclidean` to wandb under
`probe/{dataset,subject}/…`. It embeds the labeled corpus once per evaluation
and splits the features into folds, so a 6-fold evaluation costs one pass, not
six. Defaults: `--probe-every 5` in the sbatch (~1 pass over 6.5k labeled
windows per evaluation), `--probe-windows 0` (no cap).

That removes the *discrepancy* but not the *rule*: it makes the monitor a
sharper contaminant, not a safe one. `probe/dataset/mean_r` is now literally the
reported quantity, so ranking configs on it is ranking on the test folds. Select
on `train/jepa_loss`, or retrain the winner. `slurm/eval_checkpoints.sbatch
--spatial-pool 6x4x3` remains the way to produce a number for the paper.

Still open and now cheaper to check: at random initialisation the *old* monitor
read r ≈ 0.51 where `eval_probe` on the same `subject` protocol and pooling read
0.005 (the 400-window cap was ruled out as the cause). The rebuilt probe reports
epoch 0 on the same code path as every later epoch, so whatever this was should
be visible as an epoch-0 outlier in the wandb curves, or gone.

**Evaluation cost, measured, so it is not relearned expensively again:** at full
resolution the readouts are cheap — `ridge-cv` 15s/fold, `pca-ridge` 6s/fold at
18,432 features — so a 6-fold `eval_probe` run should take minutes, not hours.
The four jobs that first tried this hit a 6h wall with the readouts barely
touched; the cause was `--feature-cache` calling `np.savez_compressed` on the
~3GB-per-arm activation matrices produced at this resolution, and zlib on
float32 activations does not pay for itself. **`--feature-cache` has been
removed from `eval_probe.py` and both sbatch scripts** — recomputing a forward
pass is faster than writing (or, worse, reading a half-written) multi-GB `.npz`.

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

- **Does JEPA beat voxels, or only match them — and does it beat an untrained
  encoder at all?** The second half of that question is now the live one.
  Settled that JEPA is not *below* voxels (see step 3): every config tried
  (widths 8-256, 7 learning rates spanning 30x, 4 mask schedules, weight decay)
  lands at 0.60-0.66 against voxels' 0.623, with no config clearing that band by
  more than run-to-run noise. But the corrected `random` control sits at **0.610
  / 0.635** in the same band, so *nothing measured so far distinguishes a
  trained encoder from an untrained one of the same shape*. Before pushing on
  `mask-temporal` or on more pretraining data, the question worth answering is
  which of these is true:
  - the readout is reading the patcher's random projection of raw voxels in both
    arms, and 100 TRs × 270 subjects of labeled data is enough for ridge to do
    the rest — in which case this evaluation cannot see representation quality
    at all, and needs a harder one (fewer labels, `pca-ridge`-style compression,
    or a paradigm-level split);
  - or training does help and the gap is smaller than the fold-to-fold spread,
    in which case it needs paired seeds rather than a single draw per arm.

  Cheapest next measurement: the `random` control at `embed_dim=256` and a
  trained-vs-random pair at matched seeds on the same folds.
- **`embed_dim=32` confirmed sufficient to train, not just to read out** —
  `slurm/sweep_configs_d32.tsv`, 7 configs at `embed_dim=32` fixed, 5 epochs,
  varying only `lr` (1e-5 to 3e-4, 30x) and the leading mask schedule / weight
  decay from the round-1 sweep, evaluated the same way (`--spatial-pool 6x4x3`,
  `ridge-cv`, 6 folds):

  | config | ep1 | ep2 | ep3 | ep4 | ep5 |
  |---|---|---|---|---|---|
  | `mask-temporal32` | 0.632 | 0.645 | 0.653 | **0.658** | 0.646 |
  | `lr-3e-4` | 0.642 | 0.636 | 0.625 | 0.644 | 0.641 |
  | `base32` | 0.618 | 0.626 | 0.634 | 0.633 | 0.655 |
  | `lr-1e-5` | 0.640 | 0.640 | 0.639 | 0.638 | 0.637 |
  | `lr-1e-4` | 0.613 | 0.626 | 0.634 | 0.635 | 0.614 |
  | `lr-3e-5` | 0.618 | 0.616 | 0.612 | 0.615 | 0.617 |
  | `wd-high32` | 0.617 | 0.599 | 0.602 | 0.619 | **0.623** |

  All seven sit at 0.60–0.66 from epoch 1 — matching `voxels` (0.623) — and LR
  moves nothing across a 30x range, same non-result as the width axis.
- **TR integration & static ratio sweep results** — TR (repetition time in seconds) is now integrated into `JEPAModel` positional embeddings as continuous physical time sinusoidal features ($t_{\text{sec}} = (t + 0.5) \cdot t_{\text{patch}} \cdot \text{TR}$) plus $\log(\text{TR})$ conditioning (`--use-tr` vs `--no-tr`). Dynamic epoch ramping was removed in favor of fixed static mask ratios across two 18-config sweeps (`embed_dim=32`, 5 epochs), evaluated at full spatial resolution (`--spatial-pool 6x4x3`, 6-fold LOO):

  **Pure Spatial Sweep (`s-ratio` 0.1–0.9, `t-ratio=0.0`)**:
  - `s-ratio` 0.1 to 0.9 lands in 0.612–0.638 mean r.
  - `--use-tr` prevents performance drop at low spatial ratios ($s=0.2$: 0.612 without TR vs **0.630** with TR) and reaches **0.638** at $s=0.9$.

  **Pure Temporal Sweep (`t-ratio` 0.1–0.9, `s-ratio=0.0`)**:
  - Temporal masking systematically outperforms spatial masking, reaching **0.655** mean r at $t=0.6$ (without TR) and **0.647** (with TR).
  - TR conditioning provides a positive boost at low ($t=0.2$: 0.632 vs **0.642**) and high ($t=0.8, 0.9$: 0.635) temporal mask ratios.
- **`pool_spatial` is now used only by `analyze_identifiability.py`'s CCA
  analysis** — the in-training monitor was moved off it (see above), that one
  was not. Whether the CCA numbers in `CLAUDE.md` (r≈0.75 mean) would change
  under `--spatial-pool none` is unmeasured.
- **Why did the old in-training monitor score a randomly initialised encoder at
  r ≈ 0.51 when `eval_probe` scored the same thing at 0.005 on the same
  protocol?** Both called the identical `pool_spatial` + `ridge-cv` +
  `aggregate_by_subject` path at the *old* pooled resolution; the window cap was
  ruled out. Unexplained. The rebuilt probe's epoch-0 numbers are computed the
  same way as every other epoch's, so the next run says whether this survived
  the rewrite; the monitor is not used for ranking either way.

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
