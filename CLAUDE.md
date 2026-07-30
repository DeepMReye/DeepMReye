# CLAUDE.md

Orientation for an agent picking up this project. Read this first, then
**`STATE.md`** for what is running right now and what to do next, then
`overview.md` for the method in depth.

## What this is

DeepMReye 2.0: decode eye gaze from fMRI without an eye tracker. The signal
around the eyeballs in a BOLD volume carries gaze position. Version 1 (published,
`frey2021deepmreye`) did this with a supervised regressor. This branch (`pytorch`)
rewrites it as a **self-supervised JEPA**: pretrain a representation on unlabeled
eye-region fMRI from OpenNeuro, then fit a light linear probe to gaze on the few
labeled datasets. The bet: unlabeled fMRI is abundant, gaze labels are scarce.

Status: pipeline runs end-to-end, tests pass, not yet trained at scale. The
immediate goal is to run the full pipeline on a cluster and iterate on the model.

## Pipeline (single entry point)

Everything runs through `python -m deepmreye <command>` (see `deepmreye/__main__.py`).
`run_pipeline.sh` is a thin `.venv` wrapper around the same calls.

1. `compile`    — sample ~2 subjects/dataset from OpenNeuro into `data/datasets.h5`.
2. `qa`         — Flask browser UI to label each subject eyes / no-eyes.
3. `preprocess` — download + extract all subjects of approved datasets.
4. `train`      — train the JEPA + linear probe.

Plus `fetch` to pull the corpus from HuggingFace up front, `export-labels` /
`restore-labels` for the label backup, and `merge-registry` to fold worker
sidecars into `datasets.h5` (see below).

```bash
python -m deepmreye compile --data-dir data --limit 5
python -m deepmreye qa --data-dir data
python -m deepmreye preprocess --data-dir data
python -m deepmreye train --data-dir data -- --epochs 50   # args after `--` go to train_jepa.py
```

`--data-dir` goes AFTER the command (subparser-only, by design).

**On a cluster, `compile` and `preprocess` do not work as single commands** —
they assume one machine with both network and memory. See "Running on Leonardo"
below for the split (stage on login, extract on compute) that replaces them.

## Layout

- `deepmreye/__main__.py` — the CLI. Every stage dispatches from here.
- `deepmreye/pipeline.py` — shared S3 download / coregister / extract / write.
  `compile` and `preprocess` both call `process_subject` here. Single source of
  truth for the ingestion logic and for `is_dataset_approved`.
- `deepmreye/preprocess.py` — coregistration, eye-mask extraction (ANTs) and
  `normalize_img`, which extraction now applies.
- `deepmreye/storage.py` — the per-participant HDF5 layout. Every read and write
  of an eye block goes through here.
- `deepmreye/registry.py` — sidecar records + `merge_pending`, so parallel
  workers never write `datasets.h5` directly.
- `deepmreye/qa_classifier.py` — 21-feature eye-detection & registration transform triage model (10 inner mask + 8 3-stage ANTs registration transform stats + 3 TR/volume metadata; 78.5% CV accuracy). Ranks and flags; never approves.
- `deepmreye/thumbnail.py` — the ~20 KB QA thumbnail every participant gets.
  Three ways in: `from_arrays` at extraction, `from_report` to backfill from an
  old HTML report, `from_block` when only the stored block survives.
- `deepmreye/labels.py` — CSV label backup (export / restore).
- `deepmreye/datasource.py` — finds the corpus (explicit path, `$DEEPMREYE_DATA`,
  `./data`, else HuggingFace) and decides what each stage downloads.
- `deepmreye/validation.py` — TR extraction/validation from NIfTI headers.
- `deepmreye/data/{jepa,probe}_dataset.py` — windowed HDF5 dataloaders.
- `deepmreye/models/{jepa,patcher}.py` — ViT encoders/predictor, patchify + masking.
- `deepmreye/evaluate/probe.py` — gaze probe metrics (R^2, Pearson r), temporal
  target binning, and `aggregate_by_subject` (the per-participant unit of
  analysis).
- `deepmreye/evaluate/baselines.py` — the readout zoo: mean, OLS, ridge,
  ridge-cv, PCA→ridge, PLS, RF, GBT. Pure sklearn, no torch, so it is testable
  on its own and every arm gets the identical readouts.
- `scripts/` — portable stages, runnable anywhere: the CLI-imported
  implementations (`run_compile`, `run_preprocess`, `run_labeler`),
  `train_jepa.py`, `eval_probe.py` (the baseline table),
  `analyze_identifiability.py` and `analyze_calibration.py` (paper analyses, not
  baselines — see below), `build_index.py` (writes `index.parquet`, validates
  every file), `train_qa_classifier.py`, `upload_to_hf.py`, `sync_labels.py`
  (label round trip via the Hub), `convert_labeled_to_h5.py` (source
  `labeled_data/` -> `dsL##_*` in the corpus), `backfill_thumbnails.py`.
- `docs/ssl_design_brief.md` — the SSL objective question, with measured numbers.
- `results/` — evaluation output (JSON + logs). Not the corpus.
- `slurm/` — everything cluster-specific and nothing else: `stage_downloads.py`
  (login node, network), `extract_staged.py` + `extract_array.sbatch` (compute,
  offline), `submit_extraction.sh`. Has its own README. Nothing outside this
  folder needs SLURM.
- `overview.md` — detailed method reference. `paper/` — ICLR 2026 draft.

## Data model

- `data/datasets.h5` — central registry. One group per dataset, one subgroup per
  subject. Manual QA labels live as the `approved` attribute:
  `1` eyes, **`4` eyes but faint**, **`3` eyes but cut off**, `0` no eyes / bad transform,
  `2` no eyes / good transform, `-1` unlabeled, `-99` whole dataset skipped.
  Constants and the approved set live in `deepmreye/pipeline.py`
  (`LBL_*`, `APPROVED_LABELS`) — use those rather than bare integers.
- `data/<ds>/<sub>.h5` — **one file per participant** (`deepmreye/storage.py`):
  `eye_block` `[X, Y, Z, T]` float32, plus `labels` `[T, 10, 2]` when gaze is
  known. Metadata (TR, source S3 key, `normalized`, `format_version`) sits in
  file attrs. Labeled and unlabeled participants use the identical container.
- Dataset names carry the subset: `ds######` is an OpenNeuro accession (keep the
  real accession, it is the provenance), `dsL##_<name>` is a gaze-labeled
  dataset. So `dsL*/*.h5` selects the probe set without opening a file — that is
  what `STAGE_PATTERNS["probe"]` is. The mapping from the source directory names
  in `labeled_data/` lives in `DATASET_ALIASES`
  (`scripts/convert_labeled_to_h5.py`).
- `data/<ds>/<sub>.png` — the QA thumbnail, beside the participant file. ~20 KB.
  This is what the labeling UI and the audit grid display.
- `data/<ds>/<sub>/report_*.html` — the full Plotly QA report. ~5 MB, opt-in
  (`--report html`), and only present for the original QA sample.
- `data/_pending/worker_*.jsonl` — sidecar registry records from extraction
  workers, folded in by `merge-registry`.
- `data/index.parquet` — one row per participant (`scripts/build_index.py`);
  the browsable overview of the whole corpus.
- `data/labels.csv` — append-only backup of every QA label (see below).

Eye blocks are `[47, 29, 18, T]`: the mask crop is fixed, so only `T` varies.

## Key decisions (context you won't get from the code)

- **Classifier removed as a gate; reintroduced as triage only.** An earlier
  design trained a decision tree to auto-QA coregistration quality
  (`transform_probability`) and it was deleted: approval is manual labels, and
  no model output may gate a dataset. That still holds.
  `deepmreye/qa_classifier.py` is deliberately *not* that. Two differences:
  it reads the extracted eyeball voxels (occupancy, centre/edge contrast,
  temporal SNR) rather than the registration's affine statistics, and its output
  never decides anything. It does two jobs — orders unlabeled subjects so the
  uncertain ones get labeled first, and flags likely no-eyes participants among
  the subjects pulled in by full extraction, which QA otherwise never sees
  (dataset approval samples 2 subjects, then every subject is downloaded).
  `is_dataset_approved` does not consult it and must not. If you find yourself
  wiring its probability into an approval path, that is the deleted gate coming
  back.
- **`3` = eyes visible but clipped, `4` = eyes visible but faint.** Count as approved:
  a partial or faint eyeball still carries gaze signal, and excluding them would drop whole
  datasets under the all-or-nothing rule. Kept as distinct labels rather than
  folded into `1` so the corpus can be filtered on them later if clipping or low SNR turns out to
  hurt the probe. The triage model predicts the exact label (not binary
  eyes/no-eyes) so the UI can pre-select these options.
- **Dataset-level all-or-nothing approval.** A dataset is used for training only
  if EVERY labeled subject shows eyes; one "no eyes" subject drops the whole
  dataset. This is intentional: the same scanner/experiment tends to fail the
  same way across subjects, and there are more datasets than we need. Logic
  lives in `is_dataset_approved` (`deepmreye/pipeline.py`), used by both
  preprocess and the training dataset. Note the QA sample is 2 subjects, so
  this decides a whole dataset from a small sample — deliberately conservative.
  It is also why `qa_classifier`'s `--flag` mode exists: the subjects pulled in
  by full extraction are never QA'd at all.
- **Labels are precious and backed up.** The QA UI mirrors every save into
  `data/labels.csv`. Re-running `compile`/`preprocess` never deletes labels. If
  the registry is rebuilt or corrupted, `python -m deepmreye restore-labels`
  replays the CSV. Labels are *not* in git — they live with the corpus, which is
  on scratch or in the HF cache — so they are versioned by pushing them to the
  Hub with `scripts/sync_labels.py`.
- **The Hub is the join between cluster and laptop.** Extraction needs a
  cluster; labeling needs a human at a browser. So the corpus is pushed once
  (~37 GB with reports) and after that only labels travel — a few MB, cheap
  enough to sync several times a day. `sync_labels.py pull` **merges**: it fills
  slots that are still `-1` locally and never overwrites a label made on this
  machine, reporting conflicts instead of resolving them. A pull can therefore
  not undo labeling, and pulling twice is a no-op.
- **The default upload is a working copy, not the artifact.** `upload_to_hf.py`
  includes every subject unless `--publish` is passed, no-eyes ones included.
  This was inverted at first and it was wrong: you cannot review or revise the
  label on a subject that was filtered out of the copy you are labeling from,
  and the excluded ones are exactly the ones worth a second look. The QA gate
  belongs to publication, which happens once, at the end.
- **A ~20 KB PNG replaced the ~5 MB HTML report as the QA artifact.** Measured
  on the QA sample: 1773 reports = 9.1 GB, the same 1773 thumbnails = 29 MB,
  **310x smaller**. At full extraction the reports would have cost >100 GB, and
  nobody opens twenty thousand Plotly pages anyway. The thumbnail is a strip of
  the panels QA actually decides on — the z=-30 brain slice with the eye mask in
  red, then the eye block from two sides — and it renders the *raw* volumes:
  normalization z-scores each voxel across time, so the temporal mean of a
  stored block is flat noise (measured 0.06 std against 0.50 for the variance
  map). Hence three constructors in `thumbnail.py`, and hence `from_block` uses
  the temporal **SD**, not the mean. `--report {png,html,both}` keeps the report
  available for a subject worth digging into; `png` is the default. The reports
  are also the *only* surviving record of the pre-normalization data, so
  backfill (`scripts/backfill_thumbnails.py`) before deleting any of them.
- **Each stage downloads only what it reads** (`STAGE_PATTERNS` in
  `datasource.py`). `qa` now takes the registry plus every thumbnail — ~30 MB,
  small enough to grab in one go. That is the payoff of the PNG switch: it used
  to stream 5 MB reports per dataset as you reached them, because taking them up
  front meant hours before the first label. `train` takes blocks and no images;
  `probe` takes `dsL*/*.h5` alone. The HF cache is topped up when a later stage
  needs more; a directory you pointed at explicitly is never touched over the
  network.
- **TR is validated but not yet used to resample.** Windows are a fixed number
  of TRs, not fixed duration, so datasets with different TRs give windows of
  different real length. Known limitation (see `overview.md` §Discussion). If you
  work on temporal handling, this is the open thread.
- **Extraction normalizes; it used to store raw BOLD.** `process_subject` now
  runs `normalize_img` (per-voxel z-score, per-volume z-score, clipped at 5 SD,
  float32). The labeled gaze datasets were already normalized this way, so
  before this change the JEPA pretrained on raw BOLD while the probe evaluated
  on z-scored data — two different input distributions through one encoder.
  `normalize_img` existed all along but nothing called it.
- **One HDF5 file per participant, not per dataset.** The old
  `data/<ds>/<ds>.h5` forced every subject of a dataset through one
  append-mode handle, which cannot be written in parallel and loses the whole
  dataset to a single corrupt write. Per-participant files make extraction
  embarrassingly parallel and contain any one failure. Writes go to a temp file
  and are renamed into place, so an interrupted job never leaves a file that
  reads as truncated.
- **Workers never write `datasets.h5`.** HDF5 allows one writer per file, and
  the labeling UI holds the registry open while you work. Extraction workers
  append to `data/_pending/worker_*.jsonl` instead, and `merge-registry` folds
  them in. This is what lets you label while extraction is still running.
  `merge_pending` never touches an `approved` attribute.
- **Each registration runs in a memory-capped child process.** ANTs `SyNAggro`
  occasionally diverges and consumes tens of GB, and *nothing about the input
  predicts it*: a 0.10 GB volume (64×64×30×195) blew past 32 GB while a 1.16 GB
  one finished fine. Sizing `--mem` cannot fix this, and the earlier
  `--max-input-gb` guard was aimed at the wrong signal. Without isolation the
  cgroup OOM killer takes down the entire array task and every subject queued
  behind it — 25 of 46 tasks died that way on the first real run.
  `extract_staged.py` forks a child per subject and the parent watches its
  **resident** memory (`/proc/<pid>/statm`), killing it just before the cgroup
  would (`--mem-limit-gb`, 100 under the 120 G task request). A runaway then
  costs one subject; casualties go to `staging/deferred_<task>.jsonl` for a
  rerun at higher memory, never silently dropped. The cap was 24 G at first and
  killed ~20% of subjects — it must sit just under the task's `--mem`, not at
  some "reasonable" fraction of it.

  Use RSS, **not `RLIMIT_AS`**. Address space is not resident memory: threaded
  ANTs reserves far more virtual than it faults in (17.6 GB RSS vs 18.6 GB VSZ
  on a healthy task, arenas higher still), so an `RLIMIT_AS` cap rejects
  allocations that were never a problem — it produced 309 spurious
  `itkImportImageContainer` failures, about a third of all attempts. If you see
  that ITK error en masse, an address-space limit is the cause.
- **Large datasets are trimmed, not skipped.** `MAX_SUBJECTS_PER_DATASET` used
  to drop any dataset exceeding it. Measured over the real corpus that threw
  away 28.9k of 48.7k available subjects (40%) to exclude 7% of datasets —
  and the large collections are the richest source of unlabeled pretraining
  data, which is the whole premise of the method. It now takes the first N
  (default 200), which keeps ~74% of subjects while still bounding how far one
  dataset can dominate. Only ~1206 of the 2394 datasets contain BOLD at all.
- **The gaze-labeled datasets are registry citizens, not just folders.** They
  used to exist only as directories on disk, so `is_dataset_approved` could not
  see them and they indexed with a null QA label. `convert_labeled_to_h5.py` now
  enters them with `approved = LBL_EYES` and `labeled = True` on the dataset
  group. That is not the classifier gate returning: eye tracking was recorded
  during the scan, so the eyeballs are in frame by construction. The `labeled`
  flag also keeps them out of the rapid audit grid, where a stray click would
  mark ground truth as no-eyes.
- **`ProbeDataset` walks the file system, `JEPADataset` walks the registry.** So
  a stale directory that is not in `datasets.h5` is invisible to pretraining but
  silently joins the probe. This bit: a pre-rename `dataset6_sequences/` was
  left on scratch beside the registered `dsL06_sequences/`, byte-identical files
  under both names, and the probe monitor duly reported the two as separate
  datasets. Leave-one-dataset-out would then hold out `dsL06` while training on
  the very same six participants — the identifiability trap, wearing a different
  directory name. If a dataset appears in probe output that is not in the
  registry, it is an orphan; delete it rather than adding it. `iter_subjects`
  taking the registry as its authority would close this off for good.
- **The labeled datasets' TRs live in code, not in the data.** No repetition
  time exists anywhere in the source `.npz` or nested `.h5` — only dataset 6
  encodes it per subject name (`S4_0004_TR1250_2MM`). `DATASET_TR` in
  `convert_labeled_to_h5.py` is the record: 0.80 / 0.87 / 1.02 / 1.00 / 1.00 s
  for datasets 1-5. Without it the control set would be the one part of the
  corpus with no TR, which is precisely the metadata the fixed-TR-window
  limitation needs.
- **Probe splits are per dataset, not a pooled shuffle.** The old `ProbeDataset`
  shuffled all subjects together, so subjects of one dataset landed on both
  sides of the train/test split and inflated the probe metrics that serve as the
  control. It now splits within each dataset (`split_by="subject"`) or holds out
  whole datasets (`split_by="dataset"`).
- **Metrics are aggregated per participant, then median across participants.**
  Pooling every row of every subject into one correlation is gameable: if one
  subject's gaze sits left of another's, a model that predicts only *which
  subject this is* scores a high pooled r with zero within-subject decoding.
  `aggregate_by_subject` computes r inside each participant, where the
  between-subject variance is constant by construction. `--pooled` prints the
  old number for comparison; the gap between them is the size of the trap.
- **Pearson r is the headline, not R².** Cross-dataset predictions are
  mis-calibrated in *gain* (measured 0.11–2.27 against the training scale) with
  offsets near zero, which destroys R² while leaving the correlation intact.
  The oracle affine correction lifts mean R² from 0.043 to 0.389; every
  unsupervised correction tested fails (z-match −0.921, quantile −0.973,
  feature-standardisation 0.003, mean-shift 0.071). The reason is
  identifiability, not effort: the required gain is ≈ `test_gaze_SD /
  train_gaze_SD`, and the target's marginal spread is exactly what differs
  between a fixation task and a free-viewing task. Degrees of visual angle
  depend on screen size and viewing distance, which are not in the BOLD.
  `scripts/analyze_calibration.py` is the measurement. Calibration is a separate
  problem from representation quality and should be reported as one.
- **CCA is an analysis, not a baseline.** `scripts/analyze_identifiability.py`
  fits CCA between the left and right orbit crops of a run and recovers gaze
  with no labels at r ≈ 0.75 mean (0.87–0.92 on pursuit and free viewing),
  against 0.57 for the *supervised* cross-dataset ridge. That is a strong
  result, and it is still not a baseline: it is fitted on the first half of the
  very run it scores, and it needs labels on that half to decide which variate
  is x, which is y, and their signs. Nothing about it could run on a new subject
  with no eye tracker. Its job is to separate "the representation cannot carry
  gaze" from "the readout does not transfer" — and it says the latter.
  `dsL03_pursuit` is the case in point: within-run supervised r = 0.895/0.865,
  unsupervised CCA r = 0.746/0.710, cross-dataset supervised r = 0.137/0.196.
  Keep it out of the baseline table; a reviewer who spots it there will
  reasonably assume the rest is oversold. Note also that the confound regression
  uses a crude mean-signal proxy — **realignment parameters are not stored** —
  so a canonical variate could still be partly head motion.
- **The baseline to beat is `pca-ridge` on raw voxels, and the random encoder is
  a real competitor rather than a floor.** The claim this bullet used to make —
  that a randomly initialised ViT scores *below* raw voxels everywhere and below
  the mean on three folds — came from a broken control: `eval_probe.py` built a
  separate random encoder for the train and test splits, so the arm measured
  nothing but basis mismatch. Corrected, one shared untrained encoder scores
  mean r 0.610 at `embed_dim=32` and full spatial resolution, against voxels'
  0.623 and a trained band of 0.60–0.66 (see STATE.md). Treat `random` as
  something to beat, not something to clear. `pca-ridge` is the
  honest competitor because it is the same shape of method — compress without
  seeing gaze, then fit a linear map. And ridge alpha is chosen by inner CV
  (`ridge-cv`), because a baseline pinned at `alpha=1.0` is the first thing a
  reviewer attacks.
- **The in-training probe is a monitor, not a model-selection criterion.**
  Picking the checkpoint or the config that maximises it would contaminate the
  headline number. Checkpoints are saved on a schedule (`last.pt`,
  `epoch###.pt`) and the reported result comes from a checkpoint chosen without
  reference to the monitor. Each checkpoint stores its own architecture, so
  `eval_probe.py` cannot load weights into a differently-shaped model.

  This got *sharper*, not looser, when the monitor was fixed to compute the real
  number (below): `probe/dataset/mean_r` is now the same quantity
  `eval_probe.py --protocol dataset` reports, so selecting on it is selecting on
  the test folds directly. Select on `train/jepa_loss`, which is unsupervised
  and leaks nothing, or retrain the winner and report that.
- **The in-training probe computes what `eval_probe.py` computes.** It used to
  mean-pool the encoder's 72 spatial tokens and split by subject only, and both
  were wrong for the job: pooling averages away the across-orbit contrast gaze
  lives in (r 0.45 pooled against 0.86 unpooled on dsL01), and the subject split
  is a looser protocol than the reported one. Together they put the curve +0.11
  to +0.24 above `eval_probe` and — the part that actually cost something —
  ranked three configs in exactly the reverse order. It now runs leave-one-
  dataset-out over all six labeled datasets at `--spatial-pool 6x4x3` with
  `ridge-cv`, plus the subject protocol alongside, and logs every dataset's
  scores to wandb. The pooling and grid logic is shared code
  (`deepmreye/evaluate/probe.py`: `collapse_spatial`), not a second
  implementation, so the two cannot drift apart again.

  It embeds the whole labeled corpus **once** per evaluation and splits the
  *features* into folds, rather than re-embedding 5/6 of it six times — the
  windows a fold trains on are exactly the windows of the other datasets, so
  this is the same computation for a sixth of the I/O. That is what makes a
  6-fold evaluation affordable inside the training loop at all
  (`ProbeDataset(split_by="all")` exists for this).

## Running on Leonardo (CINECA)

Two hard constraints shape everything here, and they pull in opposite
directions:

- **Compute nodes have no outbound network.** Anything touching S3 must run on
  a login node.
- **Login sessions are capped at 32 GB** (cgroup on `user.slice`, shared across
  all your shells on that node). ANTs memory scales steeply with run length:
  **~3.4 GB** for a 22 MB / 180-TR input, but a 2 GB / 650-TR input needs
  ~3.3 GB merely to *load* and was OOM-killed at **30 GB** during registration.
  Staged inputs average 155 MB and reach 2 GB.

  The cap is shared, so one heavy process kills everything else in the session.
  A registration test run alongside the staging job took the staging job down
  with it. Do not run memory-heavy work on a login node while a long job is
  going — measure on a compute node instead.

So the network-bound and memory-bound halves cannot run in the same place, and
`compile` / `preprocess` — which do both in one process — are unusable at scale
here. **Four parallel registrations on a login node were OOM-killed** with no
traceback (the process simply disappears; confirm with
`dmesg -T | grep -i "killed process"`). Do not raise the worker count to fix
slowness on a login node; that is what triggers the kill.

The split that works:

```bash
# 1. LOGIN NODE — download only (~2 s/subject, negligible memory)
setsid nohup python slurm/stage_downloads.py --data-dir $DATA \
    --staging-dir $STAGING --discover all --sample 2 --workers 3 \
    > logs/stage.log 2>&1 < /dev/null &

# 2. COMPUTE — register + extract, offline (~42-110 s/subject)
SLURM_PARTITION=boost_usr_prod ./slurm/submit_extraction.sh   # sizes the array from the manifest
# writes a ~20 KB thumbnail per subject; `--report html` is the >100 GB path, do
# not use it for a full run

# 3. LOGIN NODE — fold worker records in, then label
python -m deepmreye merge-registry --data-dir $DATA
python -m deepmreye qa --data-dir $DATA
```

Subjects that blow up are written to `staging/deferred_<task>.jsonl` and are
*not* lost — rerun them on a bigger allocation:

```bash
sbatch --mem=240G --array=0 --export=ALL,MANIFEST=$STAGING/deferred_0.jsonl \
    slurm/extract_array.sbatch
```

The old `--max-input-gb` size gate is disabled by default (`0`): measurement
showed input size does not predict ANTs memory at all, so it only threw away
large subjects that would have been fine. The per-child RSS watchdog below is
what actually contains a blowup.

`--sample 2` stages the QA sampling pass; drop it (and pass `--all-datasets`
only if you mean it) to stage every subject of the approved datasets. Both
stages are resumable: staging skips `.nii.gz` already present and downloads via
a `.part` rename, S3 listings are cached in `staging/resolved.jsonl` (~15 min of
work over the full corpus), and extraction skips participants already
extracted. Re-running after a timeout, a preemption, or a kill is always safe.

Long-running login-node jobs need `setsid nohup … &` — plain `nohup` has been
killed here when the session was recycled. After `setsid` the process is in a
different session, so the shell's job control no longer tracks it and `kill -0`
on `$!` tells you nothing. Check with `ps -u $USER | grep "[s]tage_downloads"`.
Do **not** use `pgrep -f` from inside a monitoring loop: the pattern matches the
monitor's own command line, so a dead job reports as alive.

Two distinct things fill that 32 GB, and both killed the staging job:

**1. Accumulating state.** Resolution died twice at exactly 1998/2394 while
holding all 2394 futures plus every subject→S3-key mapping (~42k entries; some
datasets have >1000 subjects). Memory grew with progress, so the kill looked
like a hang at a "bad dataset" — it was neither. Keep peak memory flat in the
corpus size: bounded submission waves, reduce each result immediately, never
hold the whole corpus in a dict.

**2. Page cache from your own writes.** This one is the real trap. Writing
hundreds of GB of staged NIfTIs fills the cgroup with page cache, which counts
against the limit. The download process was killed with **27 GB of cache and
~0 GB RSS** (`memory.failcnt` in the millions) — so it looks like the job is
leaking when it is doing nothing of the sort. `stage_downloads.py` calls
`posix_fadvise(POSIX_FADV_DONTNEED)` after each file to evict it; the files are
read back later on a compute node, so dropping them costs nothing.

Diagnose with:

```bash
C=/sys/fs/cgroup/memory/user.slice/user-$(id -u).slice
grep -E "^total_(cache|rss)" $C/memory.stat   # cache high + rss low = this
cat $C/memory.failcnt
```

A silent death with no traceback is one of these two, until proven otherwise.

Once ~200 subjects are labeled, the triage model helps with the remaining
~4600:

```bash
python scripts/train_qa_classifier.py --data-dir $DATA --rank   # label uncertain ones first
python scripts/train_qa_classifier.py --data-dir $DATA --flag   # screen the full download
```

It reports grouped (by dataset) cross-validated ROC-AUC. It ranks and flags
only — see the classifier note under Key decisions.

### GPU training

`slurm/train_jepa.sbatch` (one A100, `--gres=gpu:1`) and `slurm/eval_probe.sbatch`.
Both take account and partition on the command line, like the extraction
scripts, and both must be submitted **from the repo root**:

```bash
EPOCHS=150 sbatch -A AIFAC_S07_154 -p boost_usr_prod --time=16:00:00 slurm/train_jepa.sbatch
PROTOCOL=dataset ARMS="voxels random" sbatch -A AIFAC_S07_154 -p boost_usr_prod slurm/eval_probe.sbatch
```

Knobs on `train_jepa.sbatch`, all environment variables: `EPOCHS`, `BATCH_SIZE`,
`LR`, `EMBED_DIM`, `ENCODER_DEPTH`, `S_RATIO`/`T_RATIO` (default 0.0/0.6, the
best schedule measured), `PROBE_EVERY` (default 5), `PROBE_PROTOCOLS` (default
`"dataset subject"`), `SPATIAL_POOL` (default `6x4x3`), `WANDB_PROJECT`,
`WANDB_RUN_NAME`, and `EXTRA_ARGS` for anything else `train_jepa.py` takes.

**Both mask ratios at zero masks nothing**, the predictor gets an empty target
and `SmoothL1Loss` fails on a shape mismatch. `train_jepa.py` now refuses to
start in that case instead of dying on the first batch, and the defaults mask
something.

**wandb runs offline here and is synced afterwards.** Compute nodes have no
outbound network, so the sbatch scripts set `WANDB_MODE=offline` and
`WANDB_DIR=$OUT_DIR`; online mode blocks and then loses the metrics. Upload from
a login node once the job is done — the job prints the exact command:

```bash
.venv/bin/wandb sync $SCRATCH_DIR/runs/jepa/<jobid>/wandb/offline-run-*
```

**Metrics also go to `$OUT_DIR/metrics.jsonl`, one JSON line per epoch**,
written whether or not wandb is enabled. That is the copy that survives a job
killed at the wall clock and a sync nobody ran, and it needs no network and no
account.

**Probe cost, measured** (A100, 7 loader workers, `embed_dim=32`,
`--spatial-pool 6x4x3`, both protocols): the labeled corpus is 6,537 windows →
116,234 rows × 2,304 features. Embedding all of it takes **~55 s** — it is one
pass, not one per fold — and each `ridge-cv` fit takes **~55 s**, so an
evaluation of 6 dataset folds + 1 subject fold is **~7-8 min**, against ~1.5 min
for a training epoch at batch 32. The readouts dominate, not the I/O. At
`PROBE_EVERY=5` over 150 epochs that is ~4 h of probing on top of ~4 h of
training; trim with `PROBE_EVERY=10`, `PROBE_PROTOCOLS=dataset`, or
`--probe-windows`.

Logged per epoch: `train/jepa_loss`, and at every `PROBE_EVERY` epoch
`probe/<protocol>/<dataset>/{pearson_r,pearson_r_x,pearson_r_y,r2,euclidean,n_subjects}`
for each of the six labeled datasets, `probe/<protocol>/all/*` over every
held-out subject, and `probe/<protocol>/mean_r` (mean over datasets, the
headline). `probe/mean_r` is an alias for the first protocol's.

Five things about this environment that each cost a failed job:

- **The `.venv` ships CPU-only torch.** `torch==2.13.0+cpu` installs by default
  and fails silently on GPU — `torch.cuda.is_available()` is just False and
  `train_jepa.py` quietly picks the CPU. Boost nodes run driver 535.274.02, so
  install the cu126 build: `uv pip install --reinstall-package torch
  "torch==2.13.0+cu126" --index-url https://download.pytorch.org/whl/cu126`.
  Note that `uv pip install torch==2.13.0` will *not* replace the CPU build —
  the version compares equal, so only the local tag differs and uv keeps what is
  there. Name the `+cu126` local version explicitly.
- **Derive the repo root from `$SLURM_SUBMIT_DIR`, not `$BASH_SOURCE`.** Slurm
  copies the batch script into its spool directory before running it, so
  `dirname $BASH_SOURCE` is `/var/spool/slurmd/...` and `cd $REPO_DIR` lands
  somewhere unwritable — the job dies on `mkdir: cannot create directory 'logs'`
  before running a line of Python. `extract_array.sbatch` gets away with the
  `BASH_SOURCE` form only because the failing subshell leaves `REPO_DIR` empty
  and `cd ""` is a silent no-op that keeps the submit directory.
- **`export PYTHONUNBUFFERED=1`.** Under `srun`, stdout is a file rather than a
  terminal, so Python block-buffers it and a 16-hour run shows nothing in the
  log until it exits — or nothing at all if it is killed at the wall clock.
- **`export HDF5_USE_FILE_LOCKING=FALSE`.** Every data loader worker opens the
  same participant files read-only from Lustre, which does not serve the POSIX
  locks HDF5 asks for.
- **The probe's readout fits in float64, so `--mem` sizes off the feature
  matrix.** At `embed_dim=32` and `--spatial-pool 6x4x3` that is 2,304 features
  over ~130k rows, a few GB; at `embed_dim=256` it is 18,432 features and tens
  of GB, and sklearn holds more than one copy. `train_jepa.sbatch` asks for
  128 G. Above that, cap the probe (`--probe-windows 2000`) or probe less often
  rather than raising the request further.

Use `--qos=boost_qos_dbg` for test jobs: max 30 min and 8 nodes, but it clears
the queue far faster than `boost_usr_prod` for a two-minute smoke run.

Sizing, measured on an A100-SXM-64GB: the pretraining corpus is ~8k windows of
100 TRs (~250 steps at batch 32) and runs at **1.7 it/s, so ~2.4 min/epoch**.
The GPU is not the constraint — each batch is 314 MB of eye blocks off Lustre —
so data loader workers matter more than a second device, and DDP is not wired
up. 150 epochs is about 7 hours.

Other environment notes:

- **Large transfers belong on the data movers, not a login node.** Login
  sessions carry a 10-minute *CPU* time limit that will cut a long transfer off
  mid-stream, and there is no shell on the movers — only `scp`, `rsync`, `sftp`,
  `wget`, `curl`, `rclone`, `s3`, `aws s3`, invoked as
  `ssh -xt $USER@data.leonardo.cineca.it <cmd>` with **absolute paths** ($HOME,
  $WORK and $CINECA_SCRATCH are undefined there). Host-based auth works from a
  login node but *not* from inside a batch job. Pulling the 18 GB probe set with
  `huggingface_hub` survived on a login node because HF downloads are network-
  bound rather than CPU-bound, but that is luck, not a rule.

- **Submit under account `AIFAC_S07_154`, partition `boost_usr_prod`.** Not
  `EUHPC_D21_101` — that appears in the repo's `/leonardo_work` path but the
  budget expired 2026-03-24 at 104% consumed. And not `dcgp_usr_prod` (the
  natural choice for CPU work): `AIFAC_S07_154` has no allocation there.
  sbatch reports *both* failures as **"invalid account or expired budget"**,
  which sends you chasing the account when the partition is the problem. Test
  a pair with:
  `sbatch --test-only -A <acct> -p <part> --time=00:10:00 --nodes=1 --ntasks=1 --wrap=true`
  `submit_extraction.sh` now probes this way instead of trusting `sinfo`.
  Check budgets with `saldo -b`.
- `boost_usr_prod` is the GPU partition but takes CPU-only jobs fine (32 cores /
  4 GPUs per node; we ask for 8 cores). Its *estimated* start can read days out
  while array tasks actually backfill within seconds — don't be put off by the
  `--test-only` estimate.
- `sinfo` / `squeue` / `sbatch` intermittently hang from login nodes when the
  Slurm controller is unreachable, and `sacctmgr` fails separately with
  "No route to host" to slurmdbd. `sinfo --version` still works, so it is not a
  client problem — retry later rather than debugging it. This happened for
  several hours on 2026-07-25 (confirmed site outage, CINECA emailed users).
- Set `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS` and `OMP_NUM_THREADS` to the CPU
  allocation. Left unset, ANTs grabs every core on the node and concurrent array
  tasks fight for the same cores.
- Storage: `/leonardo_work` was 96% full; use
  `/leonardo_scratch/fast/AIFAC_S07_154/mfrey/dme/`. Staging the QA sample
  (2 subjects × ~1200 datasets with BOLD) is ~440 GB of `.nii.gz`, but
  extraction keeps only the eye bounding box and shrinks it ~13x — the whole
  extracted corpus is tens of GB. Pass `--cleanup` to `extract_staged.py` to
  delete each input once its output is written.
- OpenNeuro has **2394** datasets. A bare `list_objects_v2` returns only the
  first 1000 and sets `IsTruncated` — `list_datasets` in `pipeline.py` paginates.
  A handful of subjects 403 on download; these are skipped and reported.

## Dev

- Env: `uv` + `pyproject.toml` / `uv.lock` (no `requirements.txt`). The wrapper
  script expects a `.venv`. Use `.venv/bin/python` on this machine.
- Tests: `pytest deepmreye/tests/ -q` (jepa forward/masking, label round-trip,
  TR validation). Run before pushing.
- Device: training auto-selects cuda / mps / cpu in `train_jepa.py`.

## Paper sync

`paper/main.tex` Method section mirrors `overview.md`. When the pipeline or model
changes substantially, update `overview.md` AND the paper's Method, and note it
here if it changes a key decision above.
