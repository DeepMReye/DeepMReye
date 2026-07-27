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
- `deepmreye/qa_classifier.py` — eye-detection features and the triage model.
  Ranks and flags; never approves.
- `deepmreye/labels.py` — CSV label backup (export / restore).
- `deepmreye/datasource.py` — finds the corpus (explicit path, `$DEEPMREYE_DATA`,
  `./data`, else HuggingFace) and decides what each stage downloads.
- `deepmreye/validation.py` — TR extraction/validation from NIfTI headers.
- `deepmreye/data/{jepa,probe}_dataset.py` — windowed HDF5 dataloaders.
- `deepmreye/models/{jepa,patcher}.py` — ViT encoders/predictor, patchify + masking.
- `deepmreye/evaluate/probe.py` — gaze probe metrics (R^2, Pearson r).
- `scripts/` — portable stages, runnable anywhere: the CLI-imported
  implementations (`run_compile`, `run_preprocess`, `run_labeler`),
  `train_jepa.py`, `build_index.py` (writes `index.parquet`, validates every
  file), `train_qa_classifier.py`, `upload_to_hf.py`, `sync_labels.py` (label
  round trip via the Hub), `convert_labeled_to_h5.py`.
- `slurm/` — everything cluster-specific and nothing else: `stage_downloads.py`
  (login node, network), `extract_staged.py` + `extract_array.sbatch` (compute,
  offline), `submit_extraction.sh`. Has its own README. Nothing outside this
  folder needs SLURM.
- `overview.md` — detailed method reference. `paper/` — ICLR 2026 draft.

## Data model

- `data/datasets.h5` — central registry. One group per dataset, one subgroup per
  subject. Manual QA labels live as the `approved` attribute:
  `1` eyes, **`3` eyes but cut off**, `0` no eyes / bad transform,
  `2` no eyes / good transform, `-1` unlabeled, `-99` whole dataset skipped.
  Constants and the approved set live in `deepmreye/pipeline.py`
  (`LBL_*`, `APPROVED_LABELS`) — use those rather than bare integers.
- `data/<ds>/<sub>.h5` — **one file per participant** (`deepmreye/storage.py`):
  `eye_block` `[X, Y, Z, T]` float32, plus `labels` `[T, 10, 2]` when gaze is
  known. Metadata (TR, source S3 key, `normalized`, `format_version`) sits in
  file attrs. Labeled and unlabeled participants use the identical container.
- `data/<ds>/<sub>/report_*.html` — the QA report the labeling UI displays.
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
- **`3` = eyes visible but clipped by the bounding box.** Counts as approved:
  a partial eyeball still carries gaze signal, and excluding it would drop whole
  datasets under the all-or-nothing rule. Kept as its own label rather than
  folded into `1` so the corpus can be filtered on it if clipping turns out to
  hurt the probe. The triage model predicts the exact label (not binary
  eyes/no-eyes) precisely so the UI can pre-select this one — it is the
  distinction that is most tedious to make by eye.
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
- **Each stage downloads only what it reads** (`STAGE_PATTERNS` in
  `datasource.py`). `qa` takes the registry alone and streams reports per
  dataset as you reach them, because the reports total more than the eye blocks
  and downloading them up front means hours before the first label. `train`
  takes blocks and no reports. The HF cache is topped up when a later stage
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
- **Probe splits are per dataset, not a pooled shuffle.** The old `ProbeDataset`
  shuffled all subjects together, so subjects of one dataset landed on both
  sides of the train/test split and inflated the probe metrics that serve as the
  control. It now splits within each dataset (`split_by="subject"`) or holds out
  whole datasets (`split_by="dataset"`).

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

Other environment notes:

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
