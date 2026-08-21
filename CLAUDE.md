# CLAUDE.md

Orientation for an agent picking up this project. Read **`RESEARCH.md`** first —
it is the synthesis: what is established, what is closed and why, which
experiments are worth running, and the open question of how the project should
be framed. Then this file for the operating manual (layout, cluster constraints,
every design decision), then **`STATE.md`** for the dated experimental log, then
`overview.md` for the method in depth.

## What this is

DeepMReye 2.0: decode eye gaze from fMRI without an eye tracker. The signal
around the eyeballs in a BOLD volume carries gaze position. Version 1 (published,
`frey2021deepmreye`) did this with a supervised CNN regressor. This branch
(`pytorch`) is the data ingestion/QA/HuggingFace pipeline plus a **classic
regressor** baseline: read gaze straight off downsampled raw voxels with
sklearn readouts (ridge, PCA→ridge, PLS, random forest, gradient boosting, SVR,
LightGBM, MLP — `deepmreye/evaluate/baselines.py`), replicating the comparison
in `media/deepmreye_benchmarks.ipynb` on the current corpus. No representation
learning here.

A self-supervised JEPA pretraining approach was tried on this codebase and set
aside — its own corrected controls showed an untrained encoder scoring the same
as a trained one at every configuration tested, so there was nothing left to
build on. That work is preserved on the **`pytorch-jepa`** branch if it is ever
worth revisiting; do not resurrect it here without a reason the numbers on that
branch don't already rule out.

A **second** JEPA attempt — cross-orbit rather than masked-volume
(`deepmreye/orbitjepa.py`) — has since been rebuilt so that its untrained
control *is* `lr-cca:k` exactly, and measured properly: it ties that control
(0.823 against 0.825) and no configuration in a 27-checkpoint sweep beats it.
The reason is not tuning, and it is now measured: **gaze is linearly accessible
from these features**, so a non-linear encoder in front of a linear readout has
nothing to add (`scripts/analyze_nonlinear_ceiling.py` — every *supervised*
non-linear readout loses to ridge on identical features). Read that entry in
`STATE.md` before proposing any further non-linear encoder here. It also
retracts the `0.221` figure that circulated for this arm: that model was
**collapsed**, by an inverted SIGReg whose anti-collapse term was minimised by
Status: pipeline runs end-to-end, tests pass. Corpus is built (see `STATE.md` and `RESEARCH.md`):
- **337 gaze-labeled participants across 9 datasets** (`dsL01`–`dsL08`, `dsL11`), all cached locally in `~/.cache/deepmreye/` and mirrored on HuggingFace (`DeepMReye/eyeballs`). `dsL09_fearlearning` was **retired** on 2026-08-20 and folded back into `ds001242` as unlabeled data — read the vertical-convention entry under Key decisions before quoting any pre-repair number for `dsL08`, `dsL09` or `dsL12`.
- **2,009 eligible unlabeled participants across 655 distinct OpenNeuro datasets** (3,578 total `.h5` files, 96,000 functional TRs), used for unsupervised basis learning and scaling curves ($N=25 \to 2000$).
- **Winning Representations (N=2000 basis).** Re-measured 2026-08-20 on the
  repaired corpus: **9-fold LODO median over the 9 remaining datasets**, with
  `dsL09` and `dsL12` retired and `dsL11` at 37 participants. **Quote these, not
  the pre-repair figures**, and not the 10-dataset run that still had `dsL12` in
  the training pool:
  1. **1-TR mean gaze**: **`lr-cca:32`** reaches **$r = 0.841$** against the
     supervised reference `fold-pca:64` at **$0.835$** ($+0.005$); the
     Super-Basis (96 feats) is level at $0.839$.
  2. **Continuous sub-TR (10 pts/TR)**: **`lr-cca:32 + lags±2`** reaches
     **$r = 0.759$** against $0.746$ ($+0.013$).
  3. **Both margins are inside the ~0.02 noise floor** documented under Key
     decisions, so they are **ties, not wins** — state them that way. The
     ordering *among* the unsupervised arms is not stable either: retiring one
     dataset flipped `lr-cca:32` and the Super-Basis. The claim that survives is
     deployment: a frozen basis needing *no data from the target study* matches
     a fold-local supervised one.
  4. **Within dataset the result splits by resolution.** At sub-TR, `+lags±2`
     wins **7 of 10** datasets and the margins are real (`dsL03` $+0.073$ to
     $0.845$, `dsL01` $+0.059$ to $0.843$). At 1-TR the **supervised reference
     wins 6 of 10**. An earlier "9 of 11 datasets" claim here covered both and
     was too broad.

## Pipeline (single entry point)

Everything runs through `python -m deepmreye <command>` (see `deepmreye/__main__.py`).
`run_pipeline.sh` is a thin `.venv` wrapper around the same calls.

1. `compile`    — sample ~2 subjects/dataset from OpenNeuro into `data/datasets.h5`.
2. `qa`         — Flask browser UI to label each subject eyes / no-eyes.
3. `preprocess` — download + extract all subjects of approved datasets.

Then `scripts/eval_probe.py` reads gaze out of the labeled subset with the
readout zoo (see below) — there is no `train` stage on this branch.

Plus `fetch` to pull the corpus from HuggingFace up front, `export-labels` /
`restore-labels` for the label backup, and `merge-registry` to fold worker
sidecars into `datasets.h5` (see below).

```bash
python -m deepmreye compile --data-dir data --limit 5
python -m deepmreye qa --data-dir data
python -m deepmreye preprocess --data-dir data
python scripts/eval_probe.py --protocol dataset --readouts ridge-cv svr lgbm mlp

# The feature axis. `fold-pca:64` is the current best arm (median r 0.814).
python scripts/eval_probe.py --protocol dataset --readouts ridge-cv \
    --features raw fold-pca:64
python scripts/analyze_axis_conventions.py     # is a bad fold mis-aimed, or resolution-limited?
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
- `deepmreye/eyetracking.py` — ingest for OpenNeuro datasets that already ship
  gaze. Parses BIDS physio streams and bins them to `[T, 10, 2]`. The hard part
  is the time origin, so it is **recovered**, never assumed: three explicit
  anchor strategies (`starttime`, `trigger`, `message`), and it raises rather
  than accept a `StartTime` that turns out to be a raw tracker clock. Fed by
  `scripts/fetch_eyetracking.py`; every result is checked by
  `scripts/verify_gaze_sync.py`. See the entry under Key decisions.
- `deepmreye/datasource.py` — finds the corpus (explicit path, `$DEEPMREYE_DATA`,
  `./data`, else HuggingFace) and decides what each stage downloads.
- `deepmreye/validation.py` — TR extraction/validation from NIfTI headers.
- `deepmreye/data/probe_dataset.py` — windowed HDF5 dataloader over the
  gaze-labeled subset (`dsL*`), with `within`/`subject`/`dataset`/`paradigm`
  split protocols.
- `deepmreye/evaluate/probe.py` — gaze probe metrics (R^2, Pearson r), temporal
  target binning, and `aggregate_by_subject` (the per-participant unit of
  analysis).
- `deepmreye/evaluate/baselines.py` — the readout zoo: mean, OLS, ridge,
  ridge-cv, PCA→ridge, PLS, RF, GBT, SVR, LightGBM, MLP. Pure sklearn, no
  torch, so it is testable on its own.
- `deepmreye/evaluate/align.py` — unsupervised per-group feature alignment
  (Euclidean Alignment, CORAL). Measured and negative; see Key decisions.
- `deepmreye/evaluate/features.py` — the *feature* axis, crossed with the
  readouts above: `raw` (stride-4 voxels, the published baseline), `fold-pca`
  (full mask, PCA fitted on the training fold — the control), and the three
  frozen corpus bases.
- `deepmreye/unsupervised.py` — the linear bases over the unlabeled corpus. One
  streaming pass accumulates a 14236² second moment over the masked voxels **and
  the same over temporal differences**; every basis is a decomposition of those
  two, so adding one costs an eigendecomposition rather than another read of the
  corpus. `corpus-pca`, `diff-pca`, `lr-cca`, plus the temporally-selected
  family: `gev-fast`/`gev-slow` (generalized eigendecomposition of the two
  covariances — GED/CSP, Cohen 2022 NeuroImage), `band-pca` (a lag-1
  autocorrelation band) and `nuis-pca8`/`nuis-pca32` (PCA after projecting out
  the slowest high-variance directions). `lag1_autocorrelation` is the enabling
  trick: `sym(C_1) = C_0 - DC/2`, so per-direction lag-1 autocorrelation is free.
  Fitted by `scripts/fit_corpus_basis.py`, or across corpus sizes by
  `scripts/sweep_corpus_scaling.py`.
- `scripts/sweep_corpus_scaling.py` / `scripts/sweep_probe_scaling.py` — the
  corpus-size axis. The first fits every basis at a series of unlabeled-corpus
  sizes in **one incremental pass** (`Moments` is additive, so a snapshot per
  checkpoint costs the largest fit, not the sum) over a *shuffled* subject order
  — sorted order is grouped by dataset, so a prefix would confound "more
  participants" with "fewer acquisitions". The second probes them, and with
  `--budgets` crosses corpus size against the *labeled* budget instead.
- `deepmreye/temporal.py` — the causal next-TR objective (`ar-gru`) and its
  mandatory untrained control (`ar-random`). Trained by
  `scripts/train_ar_model.py`. Answered and negative; see Key decisions.
- `deepmreye/crossorbit.py` — the cross-orbit soft-argmax bottleneck (`xorb`),
  its untrained control (`xorb-random`) and its nuisance path (`xorb-nuis`).
  Trained by `scripts/train_crossorbit.py`. The only objective here whose
  training demonstrably helps the probe, though still short of `fold-pca`.
- `deepmreye/orbitcon.py` — the same cross-orbit constraint again, but
  **contrastive** rather than reconstructive: VICReg between the two orbits of
  the *same* TR (`ocon`, control `ocon-random`). The two reconstruction
  bottlenecks above grade a latent on repainting the other orbit, which pulls it
  toward appearance; this one grades agreement alone and never decodes. The probe
  feature is 2x32 = 64 dimensions, matched to `lr-cca:64` — the linear form of
  the same constraint and the arm it has to beat. Trained by
  `scripts/train_orbitcon.py`, which has a `--scaling` mode so "does more data
  help" is measured before a long run rather than assumed.

  **`split_orbits`' mirroring is correct for reconstruction and wrong for
  contrast — do not share it between the two without thinking.** The right orbit
  is flipped along x so both crops run lateral-to-medial, which is what
  `crossorbit`/`orbitrot` want: they repaint one orbit from the other, and that
  is easiest when the two look anatomically alike. But both eyes rotate
  conjugately, so horizontal gaze moves both eyeballs the same way in *global* x
  — and after the flip that becomes *opposite* local directions in the two
  crops. One shared encoder therefore reports horizontal gaze with opposite sign
  on the two orbits, and VICReg's invariance term is an MSE between them, so it
  penalises exactly the feature we are trying to learn. Vertical gaze is
  untouched, an x-flip leaving y alone.

  The fingerprint is an axis split, and it is what caught this: trained `ocon`
  scored r_y **0.829** against its untrained control's 0.512 while r_x collapsed
  to **0.390** against the control's **0.768** — vertical its best number,
  horizontal *below random*. `orbitcon.unmirror_right` undoes the flip, and the
  convention is stored in the checkpoint because the feature extractor has to
  reproduce the training geometry: feeding a mirrored orbit to un-mirrored
  weights raises nothing and simply scores worse. Un-mirroring also makes the two
  crops anatomical mirror images, which *helps* here — matching anatomy across
  them gets harder while matching gaze gets easier, and anatomy is the
  degenerate solution.

  **The degenerate solution here is anatomy, and the pooled agreement number
  cannot see it.** Anatomy is shared between the two orbits and varies across a
  batch, so "encode which participant this is" satisfies VICReg's invariance
  term perfectly — and *the shuffled control still reads ~0*, because a random
  re-pairing crosses subjects. Hence three defenses in the model (per-view
  random gain/bias, per-run re-centering, few runs per batch) and, decisively,
  `agreement_within_run`: inside one run the participant is constant, so
  anatomy contributes no variance and what remains has to move during the run.
  Read that column, not `agreement`.
- `deepmreye/orbitjepa.py` + `deepmreye/models/jepa_net.py` — the cross-orbit
  **JEPA** (`jepa`, `jepa-random`), and the one arm here whose untrained control
  is not a random projection. Each orbit is first projected onto the frozen
  corpus `lr-cca` basis's own 256 directions, then encoded as `s = z @ W_lin +
  MLP(z)` with `W_lin = I[:, :k]` and the MLP's last layer zero-initialised. So
  an untrained model reproduces `project("lr-cca", basis, x, k)` **bit for
  bit** — the control is the 0.825 arm, and the trained-minus-untrained margin
  is a margin over the best linear corpus basis.

  Two things not to redo. **A momentum/EMA target encoder is ill-defined here**:
  the two orbits are different voxel sets, so the encoders' parameter matrices
  index different anatomy and an EMA between them copies a column prefix across
  unrelated inputs. That is what the first implementation did, and it made the
  prediction target noise. The objective is symmetric instead, with a stop-grad
  on the target side. And **SIGReg's Epps-Pulley exponents are `/2` on the
  pairwise term and `/4` on the single term**; swapping them inverts the
  statistic so that it scores collapse (0.163) below its own target N(0, I)
  (0.285), i.e. the anti-collapse term becomes the collapse mechanism. A
  training log pinned at **0.16314** is that bug — it is `1 - sqrt(2) +
  1/sqrt(3)`, collapse toward zero, exactly.

  Feature extraction is **pure numpy** (`encode_numpy`), which sidesteps the
  LightGBM/OpenMP deadlock rather than working around it, and is parity-tested
  against torch. Trained by `scripts/train_orbitjepa.py`, screened by
  `scripts/sweep_orbitjepa.py` (calibrated fast LODO over the same
  pre-projection — check its `--calibrate` output before trusting an ordering),
  confirmed by `scripts/eval_orbitjepa.py`, which is a thin driver over
  `eval_probe.py` because the previous version's separate harness is how a
  `0.221` came to be compared against a `0.847`.
- `deepmreye/orbitrot.py` — the same cross-orbit objective through a **rotation**
  bottleneck instead (`xrot`, `xrot-random`, `xrot-nuis`): a 2-DOF rotation of a
  learned canonical orbit. Gaze rotates the eyeball rather than translating it,
  and a soft-argmax centroid is nearly blind to that — see the entry under Key
  decisions. Shares `encode`/`decode` signatures with `crossorbit`, so the
  training loop, the shuffle ablation and `OrbitExtractor` are the same code for
  both and the two arms differ in nothing but the bottleneck. Trained by
  `scripts/train_crossorbit.py --bottleneck xrot`.
- `scripts/` — portable stages, runnable anywhere: the CLI-imported
  implementations (`run_compile`, `run_preprocess`, `run_labeler`),
  `eval_probe.py` (the baseline table), `analyze_identifiability.py` and
  `analyze_calibration.py` (paper analyses, not baselines — see below),
  `visualize_corpus_embedding.py` (where the labeled participants sit inside
  the unlabeled corpus, and the proxy A-distance that goes with it),
  `eval_dme1.py` (the published DeepMReye 1.0 CNN on this corpus, using the
  authors' released OSF weights — runs in `.venv-tf`, see below),
  `build_index.py` (writes `index.parquet`, validates every file),
  `train_qa_classifier.py`, `upload_to_hf.py`, `sync_labels.py` (label round
  trip via the Hub), `convert_labeled_to_h5.py` (source `labeled_data/` ->
  `dsL##_*` in the corpus), `backfill_thumbnails.py`.
- `results/` — evaluation output (JSON + logs). Not the corpus.
- `slurm/` — everything cluster-specific and nothing else: `stage_downloads.py`
  (login node, network), `extract_staged.py` + `extract_array.sbatch` (compute,
  offline), `submit_extraction.sh`. Has its own README. Nothing outside this
  folder needs SLURM.
- `overview.md` — detailed method reference.

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

- **The corpus's gaze y grows DOWNWARD, and getting that wrong is invisible to
  every check that existed.** Screen coordinates, top-left origin -- so a
  top-left tracker (which is every EyeLink) needs `flip_y=False`, and a flip is
  the exception rather than the rule. `center_and_scale`'s docstring said the
  opposite for a long time and three ingested datasets were flipped on the
  strength of it: `dsL08`, `dsL09` and `dsL12` each decoded with a **positive
  r_x and a negative r_y** against a readout trained on the rest.

  **A lag sweep cannot catch this.** Negate y and every lag scores the same
  magnitude, so the peak stays at 0 and `verify_gaze_sync.py` says PASS -- which
  it did, for all three, with healthy margins. The time origin really was right.
  Nothing in the harness looked at the sign until a cross-dataset readout was
  pointed at it.

  **The convention is established from anatomy, not from another dataset** --
  checking a new dataset against the corpus is fine, checking the corpus that
  way is circular. `scripts/verify_gaze_sync.py --convention`: the template is
  stored L, A, S, so axis 2 grows superior; the eyeball is a bright vitreous
  sphere with a **dark lens at its anterior pole**, so looking up rotates that
  lens to higher z and corr(voxel, label y) becomes a dipole along z in the
  anterior half of the orbit. Six acquisitions vote positive with 83-100%
  per-participant agreement (`dsL01` .97, `dsL02` 1.00, `dsL03` .96, `dsL04`
  .83, `dsL07` .93, `dsL11` 1.00). That is what fixes the convention.

  **It is a corpus-level instrument, not a per-dataset gate**, and the tool
  reports a vote for that reason. On the other five datasets the participants
  split 33-65% and it is measuring noise; `dsL05` is the warning, splitting
  41/59 while its vertical axis decodes at 0.83. A minority-negative vote is not
  a flip. Use a cross-dataset readout for that, which is what caught these.
- **Label units are not uniform any more. Read `label_units`, and standardise
  the target per dataset before pooling.** Every dataset now reads
  `degrees_visual_angle` except `dsL09_fearlearning`, whose export is on a grid
  its sidecar's `degreePerPixel` does not describe (see its entry below) and
  which stays `pixel`. Every conversion is taken from a documented geometry, and
  where none exists the units stay native rather than being invented:
  studyforrest from Sengupta et al. 2016 (PMC5079121: 63 cm viewing distance,
  26.5 cm screen at 1280 px, movie subtending 23.75x13.5 deg -> 0.018555
  deg/px, cross-checked against 2*atan(26.5/2/63) = 23.77 deg); `dsL11` from the
  NNDb-3T+ display's 28.9 dva over 1920 px; `dsL12` from Szinte et al.'s 77.3 x
  44.5 cm screen at 1.2 m (35.71 dva over 1920 px); `dsL07` from Kling et al.'s
  18 dva calibration square over 1080 px.

  **A wrong `center` is free; a wrong `degrees_per_unit` is nearly free; a wrong
  sign is fatal.** Pearson r is invariant to translation and scale, so an
  imperfect centre or an approximate conversion costs nothing measurable -- which
  is exactly why an off-centre mean is a *diagnostic* rather than a defect, and
  why it is worth looking at. `dsL12`'s transposed columns were found that way:
  20 participants told to fixate a central dot sat 420 px off centre.

  **The consequence is not cosmetic.** `--protocol dataset` fits ONE readout
  over the pooled training folds, so the loss follows whichever dataset has the
  largest target variance. With the per-fold Euclidean scale spanning 21
  (`dsL01`) to 595 (`dsL12`), the 10-dataset probe collapsed to median r
  **0.131**. `--standardize-targets dataset` (the default now) z-scores each
  training dataset's gaze before pooling; it uses training data only and Pearson
  r is invariant to it. **R² and Euclidean error are meaningless in that mode**
  (predictions in z-units against raw test targets) and the report says so.
  An earlier note here claimed "Pearson r is invariant, so nothing downstream is
  lost" -- true for *evaluation*, false for *training*, and that is the gap.

  The original six still reproduce exactly (`fold-pca:64` 0.814, `raw` 0.703),
  so neither the ingest nor the eval changes moved the baseline:
  `results/probe_orig6_control.json`.
- **`ProbeDataset` takes any participant with labels — the `dsL*` glob does not
  gate it.** `_discover()` walks every directory under the corpus and accepts
  anything carrying a `labels` dataset; `dsL*` only controls what
  `STAGE_PATTERNS["probe"]` *downloads*. Renaming a rejected dataset out of the
  prefix therefore does **not** keep it out of the probe, and
  `dsX10_visseq_unaligned` silently ran as its own fold until that was caught
  (it is `ds007532` now, with no labels, which is what actually retires it).
  To retire a labeled dataset, **remove its `labels`** (blocks can stay as
  unlabeled corpus data) or move it out of the corpus root. Do not rely on a
  name. The same rule applies to a dataset mid-ingest: `dsL11_backtothefuture`
  is parked at `~/.cache/deepmreye_pending/` for exactly this reason.
- **Gaze/BOLD alignment is recovered and then proved, never assumed.** The
  corpus can be extended with OpenNeuro datasets that already recorded eye
  tracking — a scan of all 2409 accessions found 382 paired participants across
  18 datasets, against the original 270 across 6. **Independent acquisitions are
  the scarce resource**, not participants: every leave-one-dataset-out claim
  here rests on six folds and the temporal-envelope law on twelve cells.

  The whole difficulty is the time origin. A tracker runs on its own clock, and
  a constant offset against the scanner is close to invisible — labels still
  look like gaze, the decoder still trains, it just scores lower, which reads as
  "harder dataset" rather than "broken labels". So `deepmreye/eyetracking.py`
  supports exactly three anchors and records which one was used in every
  participant's attrs: BIDS `StartTime`, a scanner-pulse column (whose period
  must equal the TR, or it raises), and a sync message in a `physioevents` file.

  **Reading `StartTime` blindly is not one of them.** ds006833 and ds005166 both
  write the tracker's own first timestamp into a field defined as an offset from
  volume 0 — self-referential, and worth nothing. Believing it would have put
  volume 0 **58.5 s early** for ds006833. The guard is that `StartTime` equal to
  the first sample timestamp is an error, *except* when times were synthesised
  from the sampling rate (ds000113 has no timestamp column and a legitimate
  `StartTime: 0.0`), which is why `anchor_seconds` takes `times_from_column`.

  **Proof is a lag sweep, with the old corpus as the control.**
  `scripts/verify_gaze_sync.py` decodes gaze at every TR shift and finds the
  peak. The eyeball signal is not hemodynamic, so a correct alignment peaks at
  **lag 0** — there is no BOLD delay to absorb an error. Sweeping the six
  original datasets first is what makes the verdict mean anything (5/6 peak at
  exactly 0). It reports a **margin** over lags >= 2 away, because gaze is smooth
  and neighbouring lags score nearly as well; the sign convention is calibrated
  by injecting known shifts into real data (injected +k gives peak +k). A y-axis
  sign error survives this test, so orientation is checked separately per axis.

  Two things it caught that nothing cheaper would have. **ds001107 is
  byte-identical to ds000113** (240 files matching on size and ETag, same 30
  subjects) — ingesting both would have put the same participants on both sides
  of a "leave-one-dataset-out" split. And **dsL01 peaks at −1** (see below).

  **Result: 82 participants and 3 datasets added, 1 rejected.** The corpus is
  **352 labeled participants across 9 datasets**, from 270 across 6 — a 50%
  increase in independent acquisitions. (As of the 2026-08-20 repair the corpus
  is **337 participants across 9 datasets**: `dsL09` and `dsL12` retired, `dsL11` and
  `dsL12` added.) `dsL07_deepmreye_calib` (15, message
  anchor, offset +0.00 s, mean r 0.785), `dsL08_studyforrest_movie` (15,
  −0.75 s, 0.667), `dsL09_fearlearning` (52, +0.50 s, 0.291); all peak at lag 0.

  **All three of those lag-0 verdicts were correct and all three datasets were
  still wrong**, because the sweep does not look at the sign — see the vertical
  convention entry at the top of this section. What the sweep established (the
  time origin) held up: 15/15 of `dsL08` and 17/20 of `dsL12` still peak at
  lag 0 in a per-subject sweep after the repair. Where the LODO numbers landed
  once the sign was fixed (11 folds, 1-TR targets, `basis_n2000`,
  `results/repair_after.json`): `dsL08` −0.03 → **+0.19** with `fold-pca:64` and
  +0.07 → **+0.35** with `lr-cca:32`; `dsL12` −0.05 → +0.09. Every previously
  negative fold is positive — and `dsL06_sequences`, whose labels nobody
  touched, gains **+0.08** purely because `dsL09` left the *training* pool.
  A broken dataset does not only cost you its own fold.

  **The sub-TR sweep is not optional.** The integer sweep passed studyforrest at
  lag 0 while its profile was lopsided (−1 at +0.46 against +1 at +0.12);
  re-binning the raw gaze at fractional offsets found the optimum **half a TR
  away**, worth +0.12 r, with 15/15 participants agreeing on the sign. `dsL07`
  peaking at exactly +0.00 s is what proves the binning itself is right and
  localises each offset to its own dataset. Both fitted offsets are one scalar
  per dataset, estimated *with the decoder* — fine for method comparisons,
  worth stating for absolute-decodability claims.

  **`ds001242` (`dsL09_fearlearning`) was rejected too, on re-examination, and is
  folded back into `ds001242`.** Its config is parked as `_ds001242_excluded`
  with the evidence; its 52 label arrays are archived at
  `results/dsX09_fearlearning_unaligned_labels.npz` and stripped from the files.
  Use `scripts/retire_labeled_dataset.py` -- it archives before it strips,
  defaults to a dry run, and folds the participants into their own accession.

  **There is no `dsX##_*_unaligned` category any more, and there should never
  have been one.** A dataset whose gaze recording failed is not a third kind of
  thing: what is left is eye blocks with no labels, which is exactly what every
  other unlabeled participant in the corpus is, and the naming rule already says
  to keep the real accession because that is the provenance. `dsX09` and `dsX10`
  are now `ds001242` and `ds007532`. The old prefix read as a *status* rather
  than a source, and nothing in the pipeline understood it.

  **Folding back also deduplicated 11 participants that the corpus had been
  carrying twice.** The QA sample extracted a handful of subjects of each
  accession long before the eye-tracking ingest extracted all of them under a
  `dsL##` name, so those participants were counted twice in every unlabeled
  basis fit. The two copies are the same run and *not* bit-identical (ANTs is
  not reproducible run to run -- measured voxelwise r 0.83 and 0.99 on the same
  input), so the QA-sampled file is kept, because a human looked at its
  thumbnail and its `approved` label is keyed to it, and the gaze provenance
  attrs are merged onto it. **Any future ingest has this collision by
  construction** -- check the accession folder before writing a new `dsL##` one. Three independent disqualifications: **per-subject timing** (in a
  within-subject lag sweep only 21 of 48 participants peak at lag 0 and 12 peak
  at −3 — the ds007532 signature, and one offset per subject would be equally
  circular); **37–40% of all samples are the tracker's `(0,0)` track-loss code**
  with values running to ±3276, four participants >90% lost; and **the dataset
  says so itself** — its sidecar reads "Reliability would not be guaranteed",
  "more than 25% data loss", and names the 8 of 52 participants its own authors
  excluded. Cleaning does not rescue it: restricted to the 31 participants
  passing coverage, off-grid *and* the authors' list, median within-subject
  decoding is **0.128**, against 0.15 for all 52. There is no signal under the
  noise. To retire it, remove its `labels`; the eye blocks are fine as unlabeled
  corpus data.

  **ds007532 was rejected, and rejecting it was the point.** Its `StartTime`
  mixes proper offsets and raw tracker clocks run by run; a second anchor
  (`TRIGGER SENT`) helped three subjects and left the rest scattered over lags
  −3..0; the sub-TR sweep is flat, so no dataset-level offset exists. One offset
  per *subject* would have fixed it and would have been circular — 36 free
  parameters fitted on the decoding target. The eye blocks are kept as
  `ds007532` with their labels stripped -- ordinary unlabeled corpus data -- and
  the dataset is `LBL_DATASET_SKIPPED` in `labels.csv`. A labeled dataset nobody
  can trust is worse than one that is absent.

  Units are recorded, not invented — including when that meant retracting a
  claim. ds001242 looked fully documented (`degreePerPixel: 0.034`,
  `ScreenVisualAngle: [22, 16.5]` → a 647×485 screen), but gaze clusters at
  **(127, 100)** across subjects and a calibrated tracker does not put everyone
  6.7° left of centre; the export is a ~256×200 grid, not the sidecar's pixel
  space. Since that is inferred from where people fixate rather than documented,
  the centre is set to the observed fixation point and **no degree conversion is
  applied**. No dataset in this batch claims degrees. Nothing is lost: Pearson r
  is scale invariant and cross-dataset R² was already unidentifiable.
- **`dsL01`'s labels are stimulus positions, not measured gaze — and they lead
  the BOLD by one TR.** 11 of 12 subjects peak at lag −1 in the sweep above.
  The labels give it away: within-TR SD exactly **0.0000**, only **9 distinct x
  values**, changing every **5 TRs** — a 9-point fixation grid held 4 s per
  target, which is precisely what v1's `load_label` builds for `calibration_run`
  (`np.repeat(labels, 5, axis=0)`). `dsL05_free_viewing` by contrast has 1502
  distinct values changing every TR. So the −1 is the eye arriving after the dot
  jumps — saccadic latency at a 0.8 s TR — not corruption. It still costs
  accuracy on the **largest** labeled dataset (170 participants): within-subject
  r **0.65 at −1 against 0.60 at 0**. Shifting its labels +1 TR recovers that,
  and it has deliberately **not** been done: editing existing ground truth is a
  decision to take on purpose. The broader point is that dsL01 is
  *stimulus-locked* rather than gaze-measured, which makes it qualitatively
  unlike the other five — remember this when it behaves oddly, as it does in the
  corpus embedding (most isolated dataset) and as the one fold where
  `corpus-pca` beats `fold-pca`.
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
- **Every reported number averages 50 gaze samples per target, and that is a
  choice nobody has revisited.** `temporal_targets` reduces labels
  `[B, W, 10, 2]` to one coordinate per temporal bin by `nanmean` over *both* the
  10 sub-TR samples and the `--temp-patch-size` TRs in the bin — at the default
  patch of 5, that is **5 TRs x 10 samples = 50 gaze samples collapsed into one
  target**. Averaging a target makes it smoother and therefore more predictable,
  so the headline r's are partly a property of the binning. (`eval_dme1.py`
  already takes this seriously in the one place it would have been unfair —
  it applies the identical binning to the published CNN, because doing it on one
  side only "would have handed us the win".)

  **Measured, and the obvious guess was backwards.** The expectation was that
  heavy smoothing *compresses* the differences between arms and that finer targets
  would spread them out. The reverse happens — the default patch of 5 is the
  setting that flatters `fold-pca` most:

  | arm | patch=5 | patch=2 | patch=1 |
  |---|---|---|---|
  | `fold-pca:64` | **0.847** | 0.840 | 0.830 |
  | `corpus-pca:64` | 0.821 | **0.837** | **0.827** |
  | `lr-cca:32` | 0.825 | 0.836 | 0.810 |
  | `raw` | 0.725 | 0.780 | 0.772 |

  `fold-pca` falls monotonically as the target sharpens while `corpus-pca` rises
  and holds, so the frozen corpus basis is **tied with the fold-local one at
  per-TR resolution** (0.003, inside the noise floor) having trailed by 0.026 at
  patch=5. Since temporal resolution is the whole selling point of MR-based eye
  tracking, that is the regime worth reporting a no-labels-needed basis in.
  `--temp-patch-size` is one flag.

  **One confound, not yet controlled:** `--max-train-windows` caps *windows*, not
  rows, so a finer patch hands the readout proportionally more training rows from
  the same windows. The direction is mechanistically sensible (a frozen basis has
  nothing left to estimate, so extra rows go straight into the readout, while
  `fold-pca` refits its basis from a fixed `--basis-fit-windows` budget either
  way) but a row-matched control is needed before this is quoted as a pure
  resolution effect.

  The sub-TR samples are also *real information the pipeline discards* —
  `dsL05_free_viewing` has within-TR SD 1.18, against `dsL01`'s exactly 0.0000,
  which is the tell that its labels are stimulus positions rather than gaze.
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
- **A retry loop that resumes by skipping finished work will starve behind one
  deterministic failure.** Off-cluster there is no watchdog, so a long ingest is
  usually wrapped in "retry N times"; extraction resumes by skipping
  participants already on disk, so every attempt restarts *at the subject that
  killed the last one*. `dsL11`'s `sub-24` (1522 TRs against everyone else's
  1360) OOM-killed ANTs on a 48 GB laptop twice in four minutes, and eight
  attempts would have reached none of the 16 subjects behind it. The tell is a
  progress count that is identical at every attempt boundary. Fix it by
  ordering, not by memory: run the rest with `--subjects` first, then the
  blocker alone where it can only cost itself.
- **Registration must cover what is on disk, not what this run wrote.**
  `fetch_eyetracking.run_dataset` registers as its last statement, so anything
  that stops the loop early leaves participants extracted but absent from
  `datasets.h5` — `dsL11` sat at 22 files against 4 registry entries, silently,
  because two runs were killed mid-loop. It now enumerates every intact
  participant on disk and runs even when it wrote none (`written == 0` is the
  fully-resumed case, which is exactly when the registry most needs repairing).
  `register()` only sets attributes, so this heals rather than duplicates.
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
- **The noise floor on a 7-fold median is ~0.02, and the data says so itself.**
  In the labeled-budget sweep `fold-pca:64` reads **0.847 at 1000 training
  windows and 0.828 with all of them** — a method whose basis and readout can
  only improve with more labeled data, reporting that it got worse. That is a
  direct measurement of run-to-run variation across a 7-fold median, and it is
  the resolution limit for every comparison in this file. Consequences: a
  difference under ~0.02 is a tie no matter how suggestive its direction (the
  low-label "crossover" where `lr-cca:32` read 0.816 against `fold-pca`'s 0.812
  is exactly this trap), and `fold-pca:64` should be quoted as **0.83-0.85**
  rather than as a point value. Differences that *are* real here are the ones an
  order of magnitude larger: `lr-cca`'s +0.15 to +0.27 corpus-size gain,
  `gev-slow`'s -0.34, the k=16-to-24 cliff of +0.28.
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
- **The unlabeled corpus is NOT redundant — it was only ever measured at one
  size. Two scaling laws, and `k` must be retuned with the corpus.**
  `scripts/sweep_corpus_scaling.py` + `scripts/sweep_probe_scaling.py`. Every
  "the unlabeled half buys nothing" conclusion below was drawn from a basis
  fitted once on ~1005 participants, never from a curve. On the curve (7 verified
  folds, labeled budget fixed at 1000 windows):

  | basis | N=25 | N=100 | N=400 | N=800 | N=1039 |
  |---|---|---|---|---|---|
  | `lr-cca:64` | 0.661 | 0.749 | 0.784 | **0.811** | 0.809 |
  | `corpus-pca:64` | 0.758 | 0.813 | 0.810 | 0.818 | **0.821** |
  | `band-pca:64` | 0.749 | 0.793 | 0.811 | 0.815 | **0.820** |
  | `gev-slow:64` | 0.578 | 0.305 | 0.320 | **0.242** | — |
  | `fold-pca:64` | 0.847 | — | — | — | 0.847 |

  **`lr-cca` gains +0.150 and rises at every step** — it is the most data-hungry
  basis here (two 7000-dim whitenings plus a cross-covariance), which is why it
  benefits most. The gap to `fold-pca:64` closes from 0.19 to **0.022**. It does
  **saturate** between N=800 and 1039, so do not extrapolate it; the earlier
  straight-line projection to parity at N~1800 was wrong.

  **The optimal component count falls as the corpus grows**, which is the reverse
  of the obvious guess: `corpus-pca` peaks at k=256 when N=25 and at k=64 when
  N=800; `lr-cca` peaks at k=64 at N=800 and at **k=32 (0.825, the best corpus
  arm)** at N=1039. With few participants each component is a noisy mixture and
  ridge needs many to recombine; a well-estimated basis is compact. So retune k
  whenever the corpus size changes, and read every k conclusion below as
  conditional on the corpus it was measured at.

  **`gev-slow` is the control that makes the temporal axis credible**: it
  *degrades* by 0.336 as the corpus grows, because more data localises the slow
  nuisance subspace better and that subspace cannot carry gaze. One end of an axis
  improving with data while the other degrades is what a real axis looks like.

  **Lag-1 autocorrelation is free from the existing accumulators** —
  `sym(C_1) = C_0 - DC/2`, so `rho(w) = 1 - (w' DC w)/(2 w' C_0 w)`
  (`lag1_autocorrelation`). Measured: the nuisance is concentrated (top 2
  directions at rho 0.82-0.88 against a median 0.39), and at small N the
  directions are white (median rho 0.059) — i.e. noise — which is the mechanism
  behind the scaling.

  **Nuisance projection is tested and negative**, closing the standing "project
  out the global/motion components" suggestion for the basis: `nuis-pca8` ties
  `corpus-pca`, `nuis-pca32` *degrades with data* (-0.131). Gaze reaches lag-1
  0.851 (dsL02) while the corpus nuisance sits at 0.83-0.87, so they overlap and
  cutting the slow end cuts gaze. `gev-fast` also disappoints for the reason its
  docstring predicts — white noise maximises that objective.

  **`lr-cca` has a threshold in k, not an optimum**: 0.476 / 0.523 / **0.803** /
  **0.825** / 0.808 at k = 8 / 16 / 24 / 32 / 48. A cliff of **+0.280 between
  k=16 and k=24** — below ~24 canonical variates the projection cannot span gaze
  at all. Unlike `corpus-pca`'s smooth inverted U, do not economise here.

  **And the corpus basis adds NOTHING to `fold-pca` — all four concatenations
  lose** (`fold-pca:64` 0.847 against `+band-pca:16` 0.834, `+lr-cca:16` 0.829,
  `+lr-cca:32` 0.823, `fold-pca:32+lr-cca:32` 0.823), at the per-part budgets
  this file requires. With the scaling curve that settles the mechanism as
  **redundancy, not domain mismatch**: the corpus estimates the *same subspace*
  a fold-local PCA does, only less efficiently. The blunt statement to carry
  forward is that **a 64-dimensional linear subspace of a 14236-voxel eye mask is
  easy to estimate** — ~200 labeled participants across 6 acquisitions already
  suffice, so a larger unlabeled corpus can approach that ceiling and has no
  headroom above it. Corpus scaling is real *and* bounded by the target being
  easy. What survives is the deployment argument, now quantified: `lr-cca:32`
  needs no data at all from the target study (0.825 against 0.847).
- **The win is the full eye mask, not the unlabeled corpus — and 64 components,
  not 256.** The published baseline reads gaze off a stride-4 subsample — 480 of
  the 14236 masked voxels. Replacing that with a PCA over the *whole* mask beats
  it on **6/6** leave-one-dataset-out folds, and it is the cheapest change
  available: unsupervised, linear, seconds to fit. The component count matters
  more than expected — median r by budget is 0.744 / 0.792 / 0.808 / **0.814** /
  0.807 / 0.779 at k = 8 / 16 / 32 / **64** / 128 / 256. Use
  **`--features fold-pca:64`**: 0.703 → 0.814, and 256 components lets ridge fit
  directions specific to the training datasets. That is the result to keep.

  **Judge any basis at k=64, not k=256** — the corpus arms were originally
  measured at 256 and that understated them badly: re-run at 64, `corpus-pca`
  goes 0.775 → 0.796 and `lr-cca` 0.759 → 0.798. Folding the labeled
  participants' *voxels* into the unsupervised fit adds a further +0.008 on
  **6/6** folds, putting `corpus-pca:64` at **0.810** against `fold-pca:64`'s
  0.814 — 1/6 folds better, mean −0.009, the gap almost entirely `dsL06`. So the
  ordering below still holds, but the margin is 0.004 and the honest statement is
  that a frozen corpus basis *ties* a fold-local one at the right k. That makes
  `corpus-pca:64` the better **deployment** artifact (one precomputed projection,
  no refitting per study) while `fold-pca:64` stays the right paper baseline
  (no external file needed).

  The unlabeled corpus does **not** add on top of it. `corpus-pca` (fitted on
  1005 unlabeled participants across 613 datasets) ties or loses to `fold-pca`
  at every labeled-data budget tested — 100/300/1000/3000/all training windows,
  median r 0.670/0.732/0.762/0.776/0.775 against fold-pca's
  0.716/0.763/0.780/0.782/0.779 — including the low-data regime where
  pretraining is supposed to pay. `diff-pca` and `lr-cca` behave the same.
  The reason was recorded here as domain mismatch — a fold-local PCA is
  estimated on the very acquisitions it is applied to, while the corpus basis
  orders its components by variance in OpenNeuro scans whose scanners and
  protocols differ from the labeled gaze sets. **That explanation has now been
  measured and it does not hold** (see the embedding entry below). This remains
  the second independent finding on this corpus (after JEPA, see the
  `pytorch-jepa` branch) that the unlabeled half does not improve gaze decoding,
  but the mechanism is open, so "address the domain gap" is not the brief.

  Folding the gaze-labeled datasets' **voxels** into the unsupervised fit
  (`--include-labeled`) does not change this, in either available form: a
  per-fold basis excluding the held-out dataset gives `corpus-pca` 0.772, and a
  transductive basis that saw the held-out dataset's own voxels gives 0.775 —
  both still under `fold-pca`'s 0.779. Over 3 basis scopes, 5 labeled-data
  budgets and 5 concatenation budgets, the best unsupervised arm ever managed
  was `fold-pca+lr-cca:16` at 0.783, winning **3 of 6 folds** with a mean delta
  of **+0.001**. That is noise. Treat this line as closed.

  Two caveats worth keeping. `lr-cca` is the most *robust* basis: it scores
  0.671 on `dsL06_sequences` against `fold-pca`'s 0.593 and `corpus-pca`'s
  0.409, giving it the best mean across folds (0.693 vs 0.687) while losing on
  the median. Requiring a direction to be shared between the two orbits is a
  stronger constraint than variance, and it degrades more gracefully where
  variance ordering transfers badly — that is the one thing here worth
  revisiting. And none of the bases touches `dsL03_pursuit` (r 0.180 →
  0.196–0.204) under any readout — which is expected, since that fold is a
  resolution limit rather than a representation or transfer problem (see its own
  entry below).
- **The rotation bottleneck learns where the position bottleneck mostly did
  not — and still loses to it in absolute terms.** `deepmreye/orbitrot.py`,
  `--bottleneck xrot`. Same cross-orbit objective, same cache, split, optimizer
  and selection rule as `xorb`; the only difference is that the latent is a
  2-DOF rotation of a learned canonical orbit instead of `K` soft-argmax
  positions.

  | arm | dims | median r | mean r |
  |---|---|---|---|
  | `fold-pca:64` | 64 | **0.814** | 0.707 |
  | `xorb` | 24 | 0.389 | 0.352 |
  | **`xrot`** | **4** | **0.293** | 0.294 |
  | `xorb-random` | 24 | 0.273 | 0.253 |
  | **`xrot-random`** | **4** | **0.052** | 0.108 |

  **The controls are the result.** `xorb` scores 0.389 but its *untrained*
  control already scores 0.273 — so only **30%** of its number comes from
  training; the rest is what a random 3D conv plus a centroid gives you for
  free. `xrot` scores less in absolute terms but its control is 0.052, so
  **82%** of it is learned. Training beats the control on **6/6** folds
  (mean +0.186) against `xorb`'s 5/6 (mean +0.099), and it does it from **4
  numbers rather than 24**. This is the same lesson as the JEPA and next-TR
  entries, one level down: an untrained control is the only thing that
  distinguishes a bottleneck that learned from a bottleneck that is a lucky
  random projection. Never report `xorb`-style numbers without one.

  It is nonetheless **not** a win. 0.293 is below `xorb`'s 0.389 and nowhere
  near `lr-cca`'s 0.798 or `fold-pca:64`'s 0.814, and — importantly — it is far
  below the temporal envelope above, so the shortfall is a genuine
  representational deficit, not a data limit it could hide behind. Two things
  to try before concluding, because neither was ruled out: the run was still
  improving at its 4000-step limit (`best_step` = 4000, contribution 0.125 and
  rising against `xorb`'s 0.222), and 4 dimensions may simply be too few —
  `--angles 3`, `--parts` and more `--template-channels` are one flag each.

  **Encoder/decoder capacity is not the limit.** `--width 32
  --template-channels 24` (4x the conv width, 3x the template) tracks the
  original within noise and lands at **0.1268 against 0.1254** at step 4000 —
  +0.0014 for ~4x the compute, 2.25 s/step against 0.9. Both plateau around
  0.12-0.13. That was run as a deliberate *screen* with three variables moved at
  once, which is only sound because it failed: all three are ruled out together.
  Had it won it would have needed ablating before anyone could say which part
  mattered. The remaining untested axis is the **bottleneck width itself**
  (`--parts`), and `--parts 6` is the matched comparison — 12 dimensions per
  orbit, exactly `xorb`'s K=4 x 3.

  **Bottleneck width was the binding constraint, and at matched width the
  rotation latent wins.** `--parts 6` (12 dims/orbit, 24 features, identical to
  `xorb` K=4):

  | arm | dims | median r | untrained | learned margin | folds |
  |---|---|---|---|---|---|
  | `fold-pca:64` | 64 | **0.814** | — | — | — |
  | **`xrot` parts=6** | 24 | **0.422** | 0.208 | **+0.214** | **6/6** |
  | `xorb` K=4 | 24 | 0.389 | 0.273 | +0.116 | 5/6 |

  It also wins the objective it was trained on (contribution **0.248** vs
  0.222, val R2 **0.111** vs 0.096) and beats `xorb` on **4/6** probe folds.
  So the earlier 0.293-vs-0.389 gap was **dimensionality, not the kind of
  latent** — 4x the encoder bought +0.001, 6x the latent bought +0.12. `xrot`
  is now the best self-supervised arm on this corpus and the one whose score is
  most nearly *earned*: it starts from a lower untrained floor (0.208 vs 0.273)
  and its trained-minus-untrained margin is ~1.8x. It is still nowhere near
  `fold-pca:64`. Caveat: on `dsL05` training adds only +0.006, so that fold's
  "win" is nominal.

  **The control must be built from the trained model's own attributes.** Adding
  `--parts` updated the model and the trainer but not `eval_probe`'s
  `*-random` branch, which still read a stale `meta` and built a **4**-feature
  control against a **24**-feature model — inflating the reported margin to
  +0.370 before it was caught. `build_orbit_extractor` now derives every
  constructor argument from the loaded model and raises if the widths disagree.
  A control assembled from configuration rather than from the thing it controls
  will drift again the next time a field is added.

  **`scripts/analyze_orbit_bottleneck.py` explains the gap, and the probe table
  cannot.** Four measurements per arm, each against its own untrained control:

  | | dims/orbit | within-subj r | latent travel | L/R agreement |
  |---|---|---|---|---|
  | `xorb` trained | 12 | 0.600 | 0.0493 | +0.492 |
  | `xorb` untrained | 12 | **0.474** | 0.0005 | **+0.201** |
  | `xrot` trained | 2 | 0.393 | 0.1152 | **+0.739** |
  | `xrot` untrained | 2 | **0.221** | 0.0004 | **-0.033** |

  *Within-subject r* fits the readout inside one participant, where anatomy is
  constant — the bottleneck's own ceiling, separate from transfer. *L/R
  agreement* is the correlation between the two orbits' latents: both eyes
  rotate conjugately, so this is the cross-orbit objective's own success
  criterion, and it is invisible in the probe table.

  **The untrained control is mandatory for L/R agreement, not optional.** Both
  orbits sit in one volume, so global signal, motion and drift are common to
  them, and a random centroid model already agrees with itself at **+0.201**.
  `xrot`'s control is **-0.033**, so its 0.739 is entirely learned while a
  chunk of `xorb`'s 0.492 is not. Same for the headline: `xorb` untrained is
  already at 0.474 of its 0.600 within-subject, `xrot` untrained at 0.221 of
  0.393. So the rotation bottleneck wins on every measure of *learning* —
  agreement from a true zero, larger trained-minus-untrained margin, two
  dimensions instead of twelve — and loses on the probe only because it starts
  from a much lower architectural floor. State it that way: the position
  bottleneck is the better random projection, the rotation bottleneck is the
  better representation learner.

  One thing that did **not** work out: the learned canonical orbit renders as
  high-frequency texture, not an eyeball
  (`media/visualizations/09_template_*.png`). That is functionally sensible —
  texture makes a rotation more identifiable than a smooth blob does — but the
  interpretability this design was partly sold on is not delivered, and the
  figure should not be presented as "what the model thinks an eyeball looks
  like". The bright volume edge is a `padding_mode="border"` artifact.
- **The published DeepMReye 1.0 CNN is beaten on the one clean fold, and its
  gap is entirely vertical.** `scripts/eval_dme1.py`. The authors released model
  weights on OSF (https://osf.io/mrhk9/, `model_weights/`), so the head-to-head
  needs no retraining and no reimplementation a reviewer would have to trust.
  On `dsL06`, scored with the *identical* 5-TR binning `eval_probe` uses
  (`_reduce` is equivalence-tested against `evaluate.probe.temporal_targets`):

  | arm | r_x | r_y | mean r |
  |---|---|---|---|
  | `fold-pca:64` + ridge-cv | 0.947 | **0.343** | **0.645** |
  | `corpus-pca:64` | 0.922 | 0.250 | 0.586 |
  | `lr-cca:64` | 0.939 | -0.008 | 0.465 |
  | **DeepMReye 1.0** (`datasets_1to5`) | 0.946 | **-0.047** | 0.449 |

  Horizontal gaze is a **dead heat** (0.946 vs 0.947). The entire margin is the
  vertical axis, where the published CNN recovers nothing at all and the linear
  arm recovers some. Report it decomposed; a single mean r hides the only thing
  that is actually happening.

  **Which checkpoints are legitimate.** The labeled datasets here *are* the
  DeepMReye paper's training data, so `datasets_1to6.h5` has seen every
  participant we would score and must never be reported as held out —
  `CONTAMINATED` in the script refuses it without `--allow-contaminated`. Usable:
  `datasets_1to5.h5` (held out on `dsL06` only) and the six `dataset<N>_*.h5`
  single-dataset checkpoints (held out on the other five). That is one clean
  leave-one-dataset-out point plus 30 train-on-one/test-on-others points, and
  the other five folds have **no** clean published checkpoint — do not quietly
  fill them with `datasets_1to6`.

  Two implementation notes. The weights are Keras 2.4 HDF5, which Keras 3 cannot
  read: `TF_USE_LEGACY_KERAS=1` plus `tf-keras`, in a **separate `.venv-tf`**
  because TF's numpy pin fights the sklearn/torch stack. And v1's source is
  vendored at run time from `main` via `git show` rather than copied into this
  branch — which also means the script must never import this branch's
  `deepmreye`, since that would both pull in ANTs and register v2 under the name
  the vendored v1 needs.
- **`dsL06`'s vertical axis is broken in the data, not in the model.** Every
  other fold decodes y at 0.80-0.87; `dsL06` reads 0.343 for `fold-pca:64` and
  **-0.047 for the published CNN on its own held-out fold**. An earlier note here
  flagged this as a possible bug in our features worth chasing. It is not:
  it reproduces with the authors' weights, their preprocessing and their
  training data, so it is a property of `dsL06`. Note also that OSF names this
  dataset `dataset6_openclosed` where our source directory was
  `dataset6_sequences` — if it is an eyes-open/closed paradigm then vertical
  gaze may be barely sampled, which would explain the whole thing. Worth
  resolving before the paper, but it is not a modelling problem and further
  representation work aimed at it is wasted — the same conclusion `dsL03`
  reached, for a different reason.
- **The labeled participants are *not* out of domain, so "domain mismatch" is
  not why the corpus basis fails.** `scripts/visualize_corpus_embedding.py`
  embeds all 1450 fully-covered participants (246 labeled, 1204 unlabeled) and
  runs the standard multi-site batch-effect protocol on them: per-participant
  descriptors, t-SNE, k-means, and a held-out domain classifier scored as proxy
  A-distance `d_A = 2(1 - 2 eps)` (0 = indistinguishable, 2 = separable).
  Four results, and they point the same way:

  - **Anatomy is identical.** On per-voxel temporal SD, `d_A` = **-0.01** —
    exactly chance, grouped by dataset. Registration, coverage and eyeball
    geometry put the labeled sets squarely inside the corpus.
  - **Dynamics differ moderately, and the number is inflated.** On the
    per-participant log-variance and Fisher-z correlation of the corpus-PCA
    coordinates, `d_A` = **0.67** of a possible 2.0. But see the entry below:
    most of that is the 6-acquisitions-vs-684 asymmetry, not a domain gap.
  - **Nothing clusters by acquisition.** k-means at k=12 scores ARI **0.043**
    against dataset identity, and 1/12 clusters is >90% labeled (that one is
    `dsL01`). This is not the batch-effect regime the multi-site literature
    describes; there, embeddings organise by site.
  - **Decisively: distance does not predict the loss.** Per fold, distance from
    the corpus against `corpus-pca:64 - fold-pca:64` gives Spearman **-0.37,
    p=0.47 (n=6)**, and the sign is carried by the wrong dataset. `dsL01` is the
    *most* isolated set on every measure (nearest-neighbour mix 0.47 against a
    chance of 0.83; its own k-means cluster) and is the **one fold where the
    frozen corpus basis beats the fold-local one** (+0.012). If mismatch were the
    mechanism, that fold would lose hardest.

  So the corpus basis ties rather than wins for some other reason, and the
  obvious remaining candidate is that it is simply *redundant* — 64 variance
  directions over an eye mask are recoverable from a few hundred labeled
  windows, so a second estimate of them adds nothing. Do not build
  domain-adaptation machinery on the strength of the old story; `align.py`
  already measured that adaptation of this kind is actively harmful.

  Caveats, both in the script's output. The full-mask coverage filter drops 29%
  of participants, leaving `dsL02` and `dsL06` with 5 and 6 — their rows are
  printed with a `*` and are not measurements. And every labeled-vs-labeled
  `d_A` is ungrouped (one acquisition per side), so those cells are upper
  bounds. Only the pooled labeled-vs-unlabeled numbers are dataset-grouped, and
  those are the two the conclusion rests on.
- **The labeled datasets are ordinary acquisitions; the DeepMReye 1.0 pipeline
  did not process them differently.** Worth checking, because the labeled blocks
  come from a 1.0 run (`main`) via `convert_labeled_to_h5.py`, which copies
  without re-normalising — so any 1.0-vs-current difference would sit in the
  corpus unflagged and would masquerade as a domain gap. It does not:

  - `normalize_img` is **byte-identical** between `main` and `pytorch`, and
    nothing in the `preprocess.py` diff touches voxel values.
  - The invariants agree on the stored blocks: per-voxel mean ~0, per-volume SD
    ~1.0, `max|x|` exactly 5.0 (the clip) on both sides. Labeled per-voxel SD
    spans 0.71-0.87 across the six datasets, with the unlabeled 0.78 in the
    middle rather than outside.
  - **Decisive:** mean within-dataset cosine distance is **0.940** over the 520
    unlabeled datasets with two participants, against **0.90-0.95** for five of
    the six labeled datasets (`dsL06`, n=6, is the exception at 0.354). Two
    participants of a labeled study are no more alike than two participants of
    an arbitrary OpenNeuro study. Random pairs sit at 1.001.

  So the `d_A` = 0.67 above is mostly **structural**: the corpus is 684
  acquisitions of 1-2 participants each, while the labeled half is 6
  acquisitions of up to 158. Separating those groups only requires recognising
  six specific studies, which any six studies would permit. Do not read it as
  evidence that the gaze data is preprocessed differently — that was checked and
  it is not.

  One difference *is* real and was not previously quantified: **repetition time.
  Labeled median TR is 0.80 s against the corpus's 2.00 s** (242 of 246 labeled
  participants are <= 1.3 s, against 317 of 1204 unlabeled). That is a genuine
  property of which studies record eye tracking, and it is the concrete form of
  the fixed-TR-window limitation logged above. It does **not** explain the
  `d_A`, though: restricting the corpus to TR <= 1.3 s *raises* it to 0.758,
  while splitting the unlabeled corpus on TR alone gives 0.534.
- **The cross-orbit bottleneck is the one self-supervised objective that
  learns gaze — and it still loses to a linear basis.** `deepmreye/crossorbit.py`
  reconstructs each orbit from the *other* orbit's soft-argmax coordinate plus
  its own nuisance code taken from a different TR. Training beats its own
  untrained control on **6/6** folds at K=2 (0.316 vs 0.122, mean +0.132) and
  5/6 at K=4 (0.389 vs 0.273), from **12-24 label-free dimensions**. The
  reconstruction ablation agrees: shuffling coordinates costs 0.19-0.22 R²
  against 0.000 untrained. That is a real first here — JEPA had trained =
  untrained, next-TR had trained *worse*.

  It nonetheless does not reach `fold-pca` (0.779) and adds nothing to it
  (`fold-pca+xorb` 0.777, 2/6 folds). Most telling: **`lr-cca` at 0.759 is the
  linear version of the same cross-orbit constraint**, so making that constraint
  non-linear and routing it through a bottleneck bought nothing over the linear
  form. If this line is revisited, the thing to beat is `lr-cca`, not `raw`.

  Two implementation notes that cost time. The orbit split must **drop the
  midline trough at x=24** — it is the boundary between the lobes, and the
  halves must share no slice or the objective can predict an orbit partly from
  itself. And the orbit cache stores raw volumes, so it is only valid for the
  geometry that built it; it now records its `orbit_shape` and refuses to load
  against a different one, after a stale cache silently survived a split change.

  `xorb-nuis` scoring above `xorb` is **not** evidence the two paths failed to
  separate. At probe time the nuisance encoder is applied to the current TR, so
  it is just a wider learned embedding of that volume; the t/t' decoupling
  constrains only what that path was *useful* for during training. The
  comparison that means something is `xorb` vs `xorb-random` at matched
  dimensionality.
- **Cross-orbit *contrastive* learning: trained beats untrained, more data makes
  it worse at gaze, and it never reaches `lr-cca`. Closed.**
  `deepmreye/orbitcon.py`, `--features ocon ocon-random`. VICReg between the two
  orbits of the same TR, 64 dims matched to `lr-cca:64`. At a matched 400-window
  budget: `ocon` peaks at **0.785** (dsL02) / **0.666** (dsL05) against its
  untrained control's 0.646 / 0.576 — so training genuinely helps, +0.08 to +0.14
  at every scale — and against `lr-cca:64`'s **0.922** / **0.809**. Making the
  cross-orbit constraint non-linear and contrastive bought nothing over its linear
  form, which is the same verdict `xorb` reached.

  **The scaling curve is the finding.** From 100 to 884 pretraining runs the
  objective improves *monotonically* (val loss 29.47 → 28.24, within-run
  agreement +0.616 → +0.732) while the probe **peaks at 200 runs and then falls**
  on both folds (dsL02 0.785 → 0.723, dsL05 0.666 → 0.658). This is the next-TR
  result in a new objective: what the two orbits share is dominated by global
  signal, motion and drift — all common to both orbits, all varying within a run
  — so more data buys more nuisance. An eighth of the corpus is as good as all of
  it. Do not answer this by scaling; the untested escape is projecting out the
  leading global/motion components *before* the contrastive loss.

  **`agreement_within_run` excludes anatomy but not motion, and that is its
  limit.** It was built because pooled L/R agreement cannot distinguish gaze from
  anatomy (both orbits encode the subject; the shuffled control still reads ~0
  because re-pairing crosses subjects). Within a run the subject is constant, so
  it does rule anatomy out — and it rose monotonically with data while gaze
  decoding did not, which is how we know the learned shared signal is
  within-run nuisance rather than anatomy *or* gaze.
- **A voxel network warm-started at the incumbent matches it exactly and cannot
  beat it — and the reason is not capacity.** `deepmreye/voxelnet.py`,
  `scripts/train_voxelnet.py`. The objection that a universal function
  approximator should be at least as good as `lr-cca:32 + lags` read off voxels
  is **correct**, and it is delivered constructively rather than hoped for: the
  network *is* the incumbent at initialisation (`W_cca` as a frozen linear
  layer, `make_lags` as a fixed conv, the head warm-started from that fold's own
  `RidgeCV`), and a learned voxel branch is adopted only if it clears a margin on
  held-out **datasets**. Result: **+0.0000 on 8 of 9 folds**, all nine per-fold
  incumbent values reproducing `ridge-L1` to four decimals. Heavy augmentation
  (shift ±3, mixup, noise, voxel dropout) makes it *worse* (−0.0202): it raises
  the selection score without improving transfer, so it causes more bad
  adoptions. In 9/9 folds, adopting the learned branch either did nothing or hurt.

  **Do not attribute this to under-training.** On four participants both encoders
  drive the residual from 0.1978 to **0.0002** (low-rank linear) and **0.0008**
  (3-D CNN) — ample capacity, working optimisation. Unregularised at `lr 3e-3`
  the full-corpus loss still does not fall, because there is nothing *systematic*
  to fit across 400k rows and 8 datasets. The residual is memorisable
  participant-by-participant and carries nothing that crosses a dataset boundary.
- **Two initialisation traps in this design, both of which report a beautiful
  number instead of an error.** `alpha = 0` **and** a zero-initialised branch head
  is a **saddle**: each gradient is proportional to the other, so nothing ever
  trains and the arm reports a flawless `+0.0000` that reads exactly like the
  warm-start guarantee working (measured: both gradients exactly `0.000e+00`).
  Zero the head only. And **never `AdaptiveAvgPool3d` in a gaze encoder** — gaze
  *is* the eyeball's spatial position, so global pooling discards the signal;
  symptom is a training loss flat at 0.46–0.49 with the selection metric pinned to
  four decimals. DeepMReye 1.0 flattens its feature map for this reason. Both are
  regression-tested in `deepmreye/tests/test_voxelnet.py`.
- **Early stopping must hold out a DATASET, and adoption needs a margin.**
  Validating on held-out *participants of training datasets* selects branches that
  do not cross a dataset boundary — it cost `dsL01` 0.7678 → 0.7263 while its
  validation loss was still improving. Select on held-out datasets, scored by the
  reported metric (sub-TR r), not by residual MSE; and require a real margin
  (0.005–0.01), because any-improvement adoption on a median over a handful of
  participants cost `dsL08` 0.2859 → 0.1545.
- **The same is true in TIME, and more voxel rank is actively harmful. Both
  ceilings are now measured.** `scripts/analyze_temporal_ceiling_supervised.py`,
  `deepmreye/temporal_probe.py`. The spatial ceiling below tested only static
  per-bin features; the incumbent `lr-cca:32 + lags±2` is a *linear temporal*
  model, so the temporal version was a live gap. It is closed. Over 9-fold LODO
  at sub-TR resolution, **a linear ridge on a 3-TR window (`ridge-L1`, 0.768) is
  the best supervised model tested** and nothing beats it on more than 3/9 folds:
  `mlp-L2` 0.738, `tcn-L2` 0.737, `poly-time-L2` 0.750, `mlp-L4` 0.719.

  **Two of the refutations are convex and cannot be blamed on tuning.** Longer
  linear windows lose, and `BandedRidge` with one penalty per lag recovers only
  +0.006 of the 0.028 drop from L=1 to L=4 — so the decline is missing
  information, not a shared-alpha artifact. And rank *monotonically* hurts:
  k = 32/64/128/256 gives 0.759/0.750/0.745/0.730, with a per-block penalty
  making it worse, and concatenating the two orbits (0.746) losing to averaging
  them (0.759). That is the direct answer to "a voxel model is not restricted to
  the frozen canonical span": the room exists and **the gaze is not in it**.

  The weight-sharing argument for a temporal conv — one kernel at every offset
  instead of `k` new ridge columns per lag — was tested directly and is refuted:
  `tcn-L4` (0.542) is *worse* than `tcn-L2` (0.737), the opposite of the
  prediction. Do not build a temporal encoder here without a result that
  overturns this table.
- **`lags±1`, not `lags±2`, for sub-TR — and `lags±0` for 1-TR.** The shipped
  benchmark arm is a suboptimal setting: 0.7678 against 0.7585 at sub-TR, while
  no lags at all is best at 1-TR (0.8406). Temporal context interpolates
  within-TR motion and blurs the 1-TR mean, so the optimal window depends on the
  target resolution. Inside the ~0.02 noise floor, but two independent protocols
  agree on the peak and it costs one integer.
- **`eval_probe.py` cannot score sub-TR gaze — use `deepmreye/temporal_probe.py`.**
  `evaluate.probe.temporal_targets` nanmeans `[B, W, 10, 2] → [B, n_t, 2]`, so
  every number `eval_probe` has produced is 1-TR mean gaze at 5-TR bins. The
  sub-TR headline lived only inside `scripts/benchmark_all_11_datasets.py`, which
  also uses a different basis (`n2000` against `n1039`), alpha grid and NaN rule.
  `temporal_probe.lodo_subtr` is the single audited implementation and returns
  **both** resolutions so no arm can quote one and imply the other; it reproduces
  the benchmark to 0.0001/0.0005 and `--calibrate` refuses to be trusted
  otherwise. Its cache guard includes a **corpus fingerprint**, which the
  `sweep_orbitjepa` labeled cache lacks — that one still loads a 285-participant,
  7-dataset, pre-repair corpus without complaint.
- **Gaze is *linearly* accessible from these features, so a non-linear encoder in
  front of a linear readout cannot help. Measure this before building another
  one.** `scripts/analyze_nonlinear_ceiling.py`. The argument is short: the probe
  readout is linear, so a non-linear encoder only pays if gaze depends
  non-linearly on its input — and that is upper-bounded by what a *supervised*
  non-linear readout gets on the same features, which is generous, since it sees
  the labels the encoder never does and optimises the exact quantity scored. On
  the k=32 corpus canonical coordinates, 7 verified folds:

  | supervised readout | median r | vs ridge |
  |---|---|---|
  | **ridge (linear)** | **0.820** | — |
  | poly-ridge (squares + leading cross terms) | 0.808 | −0.012 |
  | gbt | 0.800 | −0.020 |
  | ridge on all 256 directions | 0.789 | −0.031 |
  | mlp (256, 128) | 0.777 | −0.043 |

  **Nothing non-linear wins, with labels.** That is a one-command ceiling for the
  entire non-linear program on this corpus, and it explains the whole run of
  negatives — JEPA, next-TR, CompositeNet, ContrastiveNet, `ocon`, `xorb`,
  `xrot` — without appealing to tuning in any of them. It also predicts, and is
  consistent with, the Orbit-JEPA result: 27 checkpoints across three learning
  rates and two widths, **none** beating a warm start that equals `lr-cca:32`.
  Note the corollary about width: `ridge` on 256 directions *loses* 0.031 to
  `ridge` on 32, which is the same "optimal k falls" law from the corpus-scaling
  entry showing up in the readout rather than the basis.
- **Next-TR prediction learns, and it destroys gaze.** A causal GRU predicting
  TR *t+1* from TRs ≤ *t* (`deepmreye/temporal.py`) reaches held-out R² **+0.230**
  against **−0.047** for the same architecture untrained — so unlike JEPA, the
  objective genuinely optimises. Probed, the trained hidden state scores
  **0.530** against **0.686** for its own untrained control and **0.775** for
  `corpus-pca`, which is literally the model's input. Training helped on
  **0 of 6 folds**, mean **−0.145**; the raw-variance-weighted checkpoint
  behaves the same (0.589 vs 0.721).

  The reason is measurable and worth keeping: over corpus-PCA coordinates the
  next TR is predictable at R² 0.32 (linear AR(4)), but that predictability is
  concentrated in components 0–8 (38% of variance, R² 0.59) versus 128–256
  (R² 0.09). The leading components are global signal, motion and drift; gaze at
  a 0.8–2.0 s TR is nearly white frame-to-frame because saccades outpace the
  sampling. **The predictable part of an eye block is the nuisance**, so a
  predictive objective spends its state there and evicts gaze. Whitening the
  targets per component was an explicit attempt to prevent this and did not
  help. Note too that an untrained GRU already loses ground to its own input
  (0.686 vs 0.775) — a recurrent bottleneck discards gaze before any training.

  Do not retry plain predictive pretraining here. An objective that could work
  has to avoid being dominated by the predictable nuisance: contrastive between
  the two orbits (`lr-cca` is the linear version, and is the best-behaved
  unsupervised arm), or prediction after projecting out the global/motion
  components.
- **EyeLink `.edf` is readable now, and the reader needs no SR Research SDK.**
  `read_edf` in `deepmreye/eyetracking.py`, backed by `eyelinkio` (one added
  package, no other resolution changes). Three datasets ship gaze only in that
  form and were invisible to the ingest until it existed. Two things about it:

  - **It refuses a file without the `SR_RESEARCH` magic**, because EEG's
    *European Data Format* shares the extension and would otherwise parse into
    plausible-looking numbers unrelated to gaze.
  - **It returns the header's `screen_coords`, and `build_labels` prefers it to
    a configured `center`.** Not knowing the display resolution is exactly what
    let `ds004158` ship with x and y transposed; here the file states it, so
    that guess is gone.
- **`ANCHOR_EVENTS` is the only anchor here that validates itself — prefer it
  where a dataset has an `events.tsv`.** The other three take one number from
  one place and trust it. This one fits the tracker's stimulus messages against
  the BIDS `events.tsv` onsets (already relative to volume 0), so 60 trials
  constrain 2 parameters and the fit reports whether the match is real: the
  **slope** is the ratio of the two clocks and must be ~1, and the **residual**
  must be small. Both are enforced (`EVENT_CLOCK_TOL`, `EVENT_RESID_TOL`), not
  merely recorded. On `ds004283` it reads slope 1.000102, residual SD 0.28 ms,
  and the three usable subjects agree to the fifth decimal.

  **The clock-ratio guard is not decoration.** `ds001840` fails on it: its
  `events.tsv` onsets are the *design*, the stimulus actually ran ~2.5% fast,
  and over an 859 s run that is 21 s of drift — not something one origin can
  fix. Do not widen the tolerance to admit it.
- **Coverage says a recording spans the run; nothing said it contains an eye.**
  `ds004283`'s `sub-04` has *five* runs with 738 TRs of perfect temporal
  coverage and **zero finite gaze samples**, and every gate passed them — they
  would have been written as a labeled participant with no labels.
  `MAX_NAN_FRACTION` (0.95) rejects that and moves to the subject's next run,
  which finds one at NaN 0.19. Deliberately loose: 73% track loss is a poor but
  usable recording (`dsL08`'s `sub-05`), and the case being excluded is the
  degenerate one.
- **A labeled dataset can be rejected on its *imaging* rather than its gaze, and
  `register()` must never be what decides.** `ds004283` (ingested as
  `dsL13_lokicat`, retired the same day) is the case: perfect anchor -- the
  best-validated in the corpus, slope 1.000102 -- and gaze that is simply not in
  the volumes. Within subject, where transfer cannot be blamed, it decodes at
  **0.232** against 0.79-0.86 for usable datasets, with **r_x +0.071**: the
  *easier* axis everywhere else, absent. Its orbits are clipped in the **raw**
  BOLD (zero-fraction 0.48-0.59 against the **0.4197** mask baseline that is
  identical in every healthy participant; one participant at 0.998, an empty
  block).

  The trap is that `register()` enters any `dsL##` dataset as `LBL_EYES` on the
  reasoning that gaze was recorded during the scan so eyeballs are in frame *by
  construction*. That is sound about the **acquisition** and says nothing about
  **coverage** -- and two of these participants were already in the corpus,
  QA-sampled and labeled `approved = 2`, "no eyes", by a human. Ingesting under
  a new name would have silently overwritten that judgement on the same runs.
  Check the accession folder before writing a `dsL##` one, and when a new
  dataset's fold looks bad, run the within-subject test before blaming transfer.

  Do not reach for "it is an fMRIPrep derivative, so the orbits were stripped":
  that was the obvious explanation here and it is **measured and false**. Across
  the 156 of 2096 registered participants sourced from `derivatives/`, median
  zero-fraction is exactly the 0.4197 baseline, the same as raw; extra zeroing
  tracks the QA verdict, not the source.
- **`dsL08_studyforrest_movie`'s labels are good; the 7T acquisition is what
  limits it. Do not spend more on its labels.** After the sign repair it decodes
  at 0.19 (`fold-pca:64`) / 0.34 (`lr-cca:32`) across datasets, and four
  measurements say the residual is not a labeling problem:

  - **The labels are as good as `dsL11`'s.** Inter-subject gaze correlation --
    everyone watching the same movie at the same time, computed without touching
    the BOLD, so it grades the labels alone -- is **0.55 / 0.40** against
    `dsL11`'s **0.56 / 0.50**.
  - **The gaze is in the eye blocks.** Within subject it decodes at **0.73**
    (r_x 0.77, r_y 0.69) and 15/15 participants peak at lag 0.
  - **It degrades at every step away from the subject**: 0.73 within subject,
    **0.52** leave-one-subject-out inside the dataset, 0.19-0.34 across
    datasets. `dsL11`, same paradigm, runs 0.85 / 0.79 / 0.70.
  - **Its within-dataset registration consistency is the worst in the corpus**:
    pairwise temporal-SD-map r **0.915**, centroid scatter 0.30/0.43/0.13 vox,
    against 0.99 and 0.03-0.08 for `dsL05`/`dsL07`/`dsL11`/`dsL12`. 7T EPI in
    the orbits, and it varies per participant.

  Ruled out, so do not re-test them: a mirrored registration (flipping the block
  along x, y or z rescues no subject and helps none systematically) and a
  truncated FOV (per-slice dead-voxel profiles are within 5% of every other
  dataset's, with no superior or inferior cut). The honest statement is that its
  gaze is recoverable *within* study and only partly *across* studies.
- **`dsL12_rest` was RETIRED on 2026-08-20, and its labels were never the
  problem.** Resting state with a central fixation dot: per-participant gaze SD
  is **0.26-1.3 deg**, against 2.3-2.7 for the pursuit and movie sets. It
  decodes at 0.11 within dataset and 0.05 across datasets, because a readout
  fitted on +-10 deg gaze cannot resolve a half-degree wobble. Everything that
  *could* be checked came back clean -- registration is the corpus's best
  (pairwise SD-map r 0.987), timing is right (17/20 peak at lag 0), units are
  documented. There is simply no gaze variance in the paradigm to decode, so
  the fold measured the task rather than the method.

  It had already been excluded from the headline median for that reason;
  retiring it makes the exclusion structural instead of a footnote, and also
  takes it out of the **training** pool, which the median footnote never did.
  Labels are archived at `results/dsL12_rest_labels.npz` and the 20 eye blocks
  live on as unlabeled corpus data in `ds004158` -- the same reversible path as
  `dsL09` and `ds004283`. Do not re-ingest it expecting a different answer: the
  ceiling here is the stimulus, which is the temporal-envelope law in another
  guise.
- **`dsL11_backtothefuture` is finished: 37 of 39 participants.** It was parked
  as a pending ingest at 4; its labels verify at lag 0 with margin +0.55, its
  gaze ISC matches `dsL08`'s, and it is the **best** non-original fold. All 37
  have coverage **1.000** and NaN median 0.031, one anchor throughout, and it
  decodes within subject at **0.855** -- the highest in the corpus.

  The missing two are a machine limit, not a data problem. `sub-24` and `sub-36`
  are the only participants at **1522 TRs** (everyone else is 1360) and both
  OOM-kill ANTs on a 48 GB laptop, at a measured peak RSS of 28 GB, **including
  single-threaded**. Both aligned cleanly first, so only registration is
  outstanding. Rerun them on a bigger allocation (`sbatch --mem=240G`) rather
  than locally; that is the same deferred-subject path `slurm/README.md`
  documents. And do not wrap that rerun in a plain retry loop -- see the
  starvation entry above.

- **`dsL03_pursuit` is a resolution limit, not a transfer failure. Stop
  targeting it.** It was long recorded here as a calibration/transfer problem.
  It is not. Pearson r is invariant to affine rescaling, so a gain mismatch
  cannot lower r — and dsL03 has r 0.20 *and* R² −0.64, so the direction is
  wrong, not the scale. Held-out **subjects within dsL03 itself** decode at
  0.142, the same as cross-dataset (0.159/0.196), so it is not about crossing
  datasets. The (pred, true) 2×2 correlation matrix is diagonal and positive, so
  it is not an axis swap or sign convention. Eyeball-centroid spread across
  subjects is mid-pack (0.885 voxels; dsL01 is worse at 1.236 and works), so it
  is not registration.

  What it is: the gaze trace's **lag-1 autocorrelation is 0.141**, against
  0.56-0.85 everywhere else. `dsL02_pursuit` is the control that settles it —
  same paradigm, same within-subject gaze SD (2.33 vs 2.35 deg), autocorrelation
  0.849, decodes at 0.911. dsL03's gaze simply moves faster than its acquisition
  can resolve. Every feature source, readout and alignment tried has left it
  between 0.18 and 0.21; that is what a resolution limit looks like, and further
  representation or domain-adaptation work aimed at it is wasted.
  `scripts/analyze_axis_conventions.py` is the diagnostic.

  **This generalises into a law over the whole corpus**
  (`scripts/analyze_temporal_ceiling.py`). Over the 12 (dataset, axis) cells,
  the gaze trace's lag-1 autocorrelation predicts the decoded correlation at
  **Pearson r = +0.977** (Spearman rho = +0.797, p = 0.002 — quote the Spearman,
  since three low cells against nine high ones flatter the Pearson and the two
  axes of a dataset are not independent). Ordered by autocorrelation the cells
  run dsL03.x 0.128 → r 0.181, dsL03.y 0.163 → 0.234, dsL06.y 0.253 → 0.343,
  then everything else from 0.598 → 0.811 up to dsL02.y 0.851 → 0.874. Both
  "failures" on this corpus are the same phenomenon, and it is a property of the
  stimulus, not of the decoder.

  **The evidence that makes it a mechanism rather than a correlation is
  `dsL06`'s two axes.** A between-dataset trend confounds TR, scanner, subjects,
  paradigm and registration all at once. dsL06 dissociates *within the same
  scans*: lag-1 0.761 on x decoding at 0.947, against 0.253 on y decoding at
  0.343 — same subjects, same TR, same preprocessing, same model. It is the only
  dataset whose axes differ at all (ratio 0.33; the other five sit at 0.98-1.27).
  Note this is also why the raw autocorrelation is not merely a proxy for TR.

  **Call it an envelope, not a ceiling.** The fit is
  `decoded_r = 1.03 * lag1 + 0.085` with residual SD **0.063** and one cell
  (`dsL05.x`) sitting **+0.111 above** it, so it is not a bound no method can
  pass. What is true is that `fold-pca:64` *achieves* it everywhere while weaker
  arms fall below: on `dsL06.y`, `fold-pca` reads 0.343 (on the line) against
  `lr-cca` -0.008 and the published CNN -0.047. So the useful statement is that
  the acquisition sets the scale and the readout is already at it — and the
  practical consequence is that **any representation improvement on this corpus
  is bounded to roughly 0.06-0.10 r on a couple of cells**, not a wholesale
  gain. The two with real headroom are `dsL01.y` (-0.098 residual) and
  `dsL02.y` (-0.086), both vertical axes of high-autocorrelation datasets.
  Read any new SSL arm against that budget before calling its score a
  disappointment.
- **Unsupervised feature alignment (Euclidean Alignment, CORAL) hurts.**
  `deepmreye/evaluate/align.py`. The standard cross-subject corrections in
  EEG/BCI, applied per subject and per dataset: `ea` 0.686 and `coral` 0.644
  against 0.779 unaligned (per dataset), 0.651 / 0.627 against 0.808 (per
  subject at 32 components). EA buys +0.014 on dsL03 and costs 0.19 / 0.14 /
  0.18 on dsL01 / dsL02 / dsL06. The between-component covariance of these
  features is **signal, not shift** — whitening it per group removes gaze.
  Mean-centring is free and neutral because the blocks are already per-voxel
  z-scored within subject; everything past that is harmful. `zscore` here is the
  same diagonal correction `analyze_calibration.py` already found useless as
  `feat-std`, kept as the reference the full-covariance methods must beat.
- **Concatenating features needs a per-part component budget.** Every readout
  wraps its features in a `StandardScaler`, so gluing 256 corpus components onto
  256 fold-local ones hands ridge 512 equally-scaled dimensions under a single
  alpha: it cannot downweight the added block, and unbudgeted concatenation
  *loses* (0.737 against 0.779). `--features fold-pca+lr-cca:16` is the fair
  form; the `:k` suffix caps that part alone.
- **Crossing a basis feature with `svr`/`lgbm`/`mlp` needs `--n-components`
  raised.** Those three are built as `StandardScaler -> PCA(--n-components,
  default 32) -> model`. On `raw` that is sensible compression of 480 correlated
  voxels; on an already variance-ordered basis the scaler whitens the components
  and the second PCA then truncates to 32 near-arbitrary directions. The
  difference is not subtle: `fold-pca`+lgbm reads 0.105 at the default and 0.517
  at `--n-components 256`. Neither beats `ridge-cv`'s 0.779, but only the second
  is a measurement of the model rather than of the truncation.
- **The covariance accumulator must be Fortran-ordered, and this fails
  silently.** `scipy`'s `syrk` wrapper only honours `overwrite_c` when `c` is
  already in BLAS's layout; handed a C-ordered array it updates a *copy* and
  returns it, so the accumulator stays at zero and nothing raises. The
  covariance then comes out as `-mu mu^T` — rank one, negative trace — and still
  yields a plausible-looking leading component, which is how it survived a first
  pass. `Moments` allocates with `order="F"` and raises if `syrk` ever returns a
  different object. The tell is a variance share of exactly −1.0, or canonical
  correlations of 0.99 followed by zeros.
- **Nothing in the feature path may use torch.** LightGBM and PyTorch each load
  their own OpenMP runtime, and a threaded torch reduction that runs *after* a
  LightGBM fit in the same process deadlocks — no error, no traceback, the
  process stops. `eval_probe` reaches exactly that ordering with
  `--readouts lgbm` on any multi-fold protocol, since fold 2's extraction
  follows fold 1's fit. `pool_time` is therefore numpy, and
  `test_feature_path_survives_a_lightgbm_fit_in_the_same_process` guards it.
  `OMP_NUM_THREADS=1` also masks it, which is why it can look environment-
  specific.
- **`pca-ridge` is the readout to compare everything else against, not just
  another entry in the zoo.** It is unsupervised dimensionality reduction (no
  peeking at gaze) followed by a linear map, so it is the honest floor for any
  non-linear readout: if SVR/LightGBM/MLP do not clear it, the extra
  complexity is not earning its keep. Ridge alpha is chosen by inner CV
  (`ridge-cv`) rather than pinned at `alpha=1.0`, because an under-tuned ridge
  baseline is the first thing a reviewer attacks.

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
- Tests: `pytest deepmreye/tests/ -q` (probe splits, readout zoo, label
  round-trip, TR validation). Run before pushing.
- `scripts/eval_probe.py` is CPU-only sklearn — no GPU, no `.venv` device
  selection to worry about.
