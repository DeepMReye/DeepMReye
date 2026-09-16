# What we tried, and what it taught us

This is the record. The package is small because most of what was built got
deleted, and the point of this file is that the deletions were *measured* rather
than abandoned. Read it before proposing an improvement -- several of the
obvious ones are closed, and a few are closed by evidence that took months to
get.

Everything below is leave-one-dataset-out over 337 gaze-labeled participants in
9 datasets unless stated otherwise.

---

## The one number that explains almost everything

**Gaze is *linearly* accessible from these features.** The readout is linear, so
a non-linear encoder in front of it only pays if gaze depends non-linearly on
its input -- and that is upper-bounded by what a *supervised* non-linear readout
gets on the same features, which is a generous bound, since it sees the labels
the encoder never does and optimises the exact quantity being scored.

On the k=32 canonical coordinates:

| supervised readout | median r | vs ridge |
|---|---|---|
| **ridge (linear)** | **0.820** | -- |
| poly-ridge (squares + leading cross terms) | 0.808 | -0.012 |
| gradient boosting | 0.800 | -0.020 |
| ridge on all 256 directions | 0.789 | -0.031 |
| MLP (256, 128) | 0.777 | -0.043 |

**Nothing non-linear wins, with labels.** That is a one-command ceiling for the
entire non-linear program on this corpus, and it explains every negative below
without appealing to under-tuning in any of them.

Note the corollary in the last two rows: ridge on 256 directions *loses* 0.031
to ridge on 32. Extra capacity is not neutral here, it is harmful.

---

## Representation learning: eight attempts, all closed

Each of these was built, trained, and scored against **its own untrained
control** -- which is the only thing that separates a bottleneck that learned
from a bottleneck that is a lucky random projection.

| arm | what it was | verdict |
|---|---|---|
| **JEPA (masked volume)** | self-supervised prediction of masked eye-region patches | untrained control scored the same as trained, at every configuration |
| **JEPA (cross-orbit)** | predict one orbit's latent from the other's | ties its control (0.823 vs 0.825); no configuration in a 27-checkpoint sweep beat it |
| **next-TR prediction** | causal GRU predicting TR *t+1* | the objective genuinely learns (held-out R² +0.230 vs -0.047 untrained) and **destroys** gaze: 0.530 trained vs 0.686 untrained |
| **cross-orbit reconstruction** | soft-argmax position bottleneck | trained beats untrained 6/6, but only 30% of its score is learned, and it never reaches the linear form of the same constraint |
| **cross-orbit rotation** | 2-DOF rotation of a learned canonical orbit | the best *learner* here -- 82% of its score is earned, agreement from a true zero -- and still far below `lr-cca` |
| **cross-orbit contrastive** | VICReg between the two orbits | trained beats untrained by +0.08 to +0.14, peaks at 200 pretraining runs and then **falls** with more data |
| **voxel network from scratch** | 3-D CNN, warm-started *at* the incumbent | **+0.0000 on 8 of 9 folds**; in 9/9, adopting the learned branch either did nothing or hurt |
| **supervised temporal models** | TCN, MLP, polynomial-in-time, banded ridge | a linear ridge on a 3-TR window beats all of them; nothing wins more than 3/9 folds |

### Why next-TR prediction fails, specifically

This one is worth understanding because it generalises. Over corpus-PCA
coordinates the next TR is predictable at R² 0.32, but that predictability is
concentrated in components 0-8 (R² 0.59) versus 128-256 (R² 0.09). The leading
components are global signal, motion and drift. Gaze at a 0.8-2.0 s TR is nearly
white frame to frame, because saccades outpace the sampling.

**The predictable part of an eye block is the nuisance.** A predictive objective
spends its capacity there and evicts gaze. The contrastive arm fails the same
way from the other direction: what the two orbits *share* is also dominated by
global signal and motion, which is why more pretraining data made it worse at
gaze while monotonically improving its own objective.

### Two traps that report a beautiful number instead of an error

- **A zero-initialised branch head *and* a zero mixing coefficient is a saddle.**
  Each gradient is proportional to the other, so nothing ever trains and the arm
  reports a flawless `+0.0000` that reads exactly like a warm-start guarantee
  working. Zero the head only.
- **Never global-pool in a gaze encoder.** Gaze *is* the eyeball's spatial
  position, so `AdaptiveAvgPool3d` discards the signal. The symptom is a
  training loss flat at 0.46-0.49 with the selection metric pinned to four
  decimals.

Both were regression-tested before the network was deleted.

---

## The unsupervised corpus: real, and bounded

**The corpus basis works, and the reason is not what we first wrote down.**

Re-measured on the current nine-fold protocol (`scripts/analysis_unlabeled.py`,
sub-TR, the shipped `lr-cca:32 + lags1` arm) over the eleven bases in
`results/scaling/`. The earlier table here was seven folds and a different k, so
quote these:

| unlabeled participants | 25 | 100 | 200 | 400 | 800 | 1039 | 2000 |
|---|---|---|---|---|---|---|---|
| `lr-cca:32+lags1` | 0.616 | 0.672 | 0.669 | 0.739 | 0.761 | 0.766 | **0.770** |
| `corpus-pca:64` | 0.714 | 0.726 | 0.737 | 0.743 | 0.739 | 0.745 | 0.746 |
| `gev-slow:64` (control) | 0.580 | 0.339 | 0.263 | 0.274 | 0.248 | 0.160 | 0.272 |

With no corpus at all: `raw+lags1` **0.729**, `fold-pca:64+lags1` **0.766** --
the supervised reference, fitted on the labeled data itself and given 200 TRs
per participant against the corpus bases' 48, so that it cannot lose on TR
budget. The sweep reproduces `probe.CALIBRATION` exactly at N=2000 (0.7703 and
0.7408), so it is the shipped number rather than a second implementation of it.

**The two bases scale completely differently, and that is the finding.**
`corpus-pca` is ordinary variance and gains **+0.03** across an 80-fold increase
in corpus size -- it barely needs the data. `lr-cca` gains **+0.154**, and
*loses to plain variance PCA below about N=400*: it is a cross-covariance
between two ~7100-voxel spaces, so it has vastly more to estimate, and it only
becomes the best arm once the corpus is large. State the corollary plainly --
with 200 unlabeled participants you would have shipped `corpus-pca` and had no
bilateral mechanism to describe.

**It is not one fold.** All **9 of 9** held-out datasets improve from N=25 to
N=2000, by +0.117 to +0.181 against a 0.02 noise floor, across fixation,
pursuit, free viewing and movie paradigms alike. That agreement is what stands
in for an error bar, because the sweep is **incremental over a single shuffled
participant order** -- one draw, not a mean over draws.

**And the control moves the other way.** `gev-slow` comes out of the same
accumulators at the same rank and *degrades by about 0.31* over identical data.

Then `lr-cca` **saturates**: 800 to 2000 is worth +0.009, inside the noise floor.

One confound to state rather than hide: participants, TRs and OpenNeuro
accessions all grow together (24 accessions at N=25, 655 at N=2000), so the axis
is "more unlabeled data", not "more studies" specifically. An older matched-N
comparison of one-vs-two participants per accession found no consistent
advantage for accession count, which is weak evidence that sample size rather
than study diversity is what is being bought. **The participant-versus-TR half
of that confound is now measured** -- see *Participants or TRs?* below -- and it
says rows, not people, down to a floor of a few hundred participants.

### What the corpus is worth in labeled participants

Crossing corpus size against the **labeled** budget -- at most `B` participants
per training study, with every fold still scored on its whole held-out dataset
(`probe.lodo(train_filter=...)`; subsetting the test side would move the
metric's denominator along with the budget). Median over three participant
draws, sub-TR:

| labeled participants per study | 1 | 2 | 4 | 8 | 16 | all |
|---|---|---|---|---|---|---|
| `lr-cca:32+lags1` @ N=2000 | **0.695** | 0.727 | 0.764 | 0.766 | 0.773 | 0.770 |
| `fold-pca:64+lags1` (supervised) | 0.707 | 0.732 | 0.754 | 0.754 | 0.754 | 0.766 |
| `raw+lags0` (stride-4 voxels) | 0.575 | 0.631 | 0.657 | 0.677 | **0.700** | 0.700 |
| `lr-cca:32` @ N=25 | 0.517 | 0.571 | 0.590 | 0.594 | 0.604 | 0.598 |

Two statements survive, and a third does not.

**The corpus buys labeled participants.** The N=2000 basis with *one* labeled
participant per training study reaches 0.695, which is what stride-4 voxels need
*sixteen* for. A frozen basis leaves only a ridge on 96 columns to estimate.

**A small corpus is worse than no basis at all.** `lr-cca` fitted on 25
participants sits below the raw voxel baseline at **every** budget and never
catches up -- more labels cannot repair an under-estimated basis. This is the
budget-axis form of the crossover above.

**What does *not* survive: the corpus basis does not beat the fold-local
supervised one at low budget.** They are within 0.012 at B=1 and the ordering
flips across participant draws. The corpus does not rescue a labeled-data
shortage relative to a *supervised* basis -- it removes the need for target-study
data, which is a deployment claim, not an accuracy one. Do not state it as the
latter.

**The optimal component count *falls* as the corpus grows**, which is the
reverse of the obvious guess: `corpus-pca` peaks at k=256 when N=25 and at k=64
when N=800; `lr-cca` peaks at k=64 at N=800 and at **k=32** at N=1039 and above.
With few participants each component is a noisy mixture and the ridge needs many
to recombine; a well-estimated basis is compact. Retune k if the corpus changes.

**`lr-cca` has a threshold in k, not an optimum**: 0.476 / 0.523 / **0.803** /
**0.825** / 0.808 at k = 8 / 16 / 24 / 32 / 48. A cliff of **+0.280 between
k=16 and k=24** -- below about 24 canonical variates the projection cannot span
gaze at all. Do not economise here.

### Why a bigger corpus stops helping

We first recorded this as *domain mismatch* -- the corpus basis orders its
components by variance in scans whose scanners and protocols differ from the
gaze datasets. **That was measured and it is false.** Embedding all 1450
fully-covered participants and running the standard multi-site batch-effect
protocol:

- On anatomy (per-voxel temporal SD), proxy A-distance is **-0.01** -- exactly
  chance. The labeled sets sit squarely inside the corpus.
- k-means at k=12 scores ARI **0.043** against dataset identity. Nothing
  clusters by acquisition, which is not the batch-effect regime at all.
- Decisively, **distance from the corpus does not predict the loss**
  (Spearman -0.37, p=0.47, n=6), and the sign is carried by the wrong dataset:
  `dsL01` is the *most* isolated set on every measure and is the one fold where
  the frozen corpus basis *beats* the fold-local one.

The real mechanism is **redundancy**. A 64-dimensional linear subspace of a
14236-voxel eye mask is easy to estimate -- a few hundred labeled participants
already suffice -- so a larger unlabeled corpus can approach that ceiling and has
no headroom above it. Do not build domain-adaptation machinery on the old story.

### Bases that lost

Six were fitted from the same accumulators and all lost, so they were deleted:

- `diff-pca` -- PCA of temporal differences.
- `gev-fast` / `gev-slow` -- generalized eigendecomposition of the two
  covariances. `gev-slow` *degrades by 0.336* as the corpus grows, which is the
  control that makes the temporal axis credible: one end of an axis improving
  with data while the other degrades is what a real axis looks like.
- `band-pca` -- selection on a lag-1 autocorrelation band. Ties `corpus-pca`.
- `nuis-pca8` / `nuis-pca32` -- PCA after projecting out the slowest
  high-variance directions. Ties, then *degrades with data* (-0.131).

**Nuisance projection is closed**, and the reason is instructive: gaze reaches
lag-1 autocorrelation 0.851 while the corpus nuisance sits at 0.83-0.87. They
overlap, so cutting the slow end cuts gaze.

Deleting these six also removed the temporal-difference accumulator, which is
why the basis-fitting pass is now half the memory and half the cost.

---

## Against the published DeepMReye 1.0 CNN

`scripts/eval_dme1.py`, using the authors' released OSF weights
(https://osf.io/mrhk9/) rather than a reimplementation, with the v1 source
vendored at run time from `main`. v1 emits `[T, 10, 2]` directly, so it is
scored through the *same* `probe._resolutions` -> `metrics.score` ->
`_summarise` path as every other arm -- no second implementation of the number,
which is what the deleted version of this script had (its own `_reduce` at
5-TR bins).

**The contamination structure is the whole design.** `dsL01`-`dsL06` *are* the
DeepMReye paper's training datasets 1-6, so the shipped `datasets_1to6.h5` can
only be reported as held out on `dsL07` (ds006833, Kling et al. -- a later
acquisition), `dsL08` and `dsL11`. `datasets_1to5.h5` adds `dsL06`.
`CLEAN_FOLDS` in the script encodes this and `predict` refuses a contaminated
fold without `--allow-contaminated`.

Sub-TR, on the folds v1 has not seen:

| arm | dsL07 | dsL08 | dsL11 | dsL06 | median |
|---|---|---|---|---|---|
| **`lr-cca:32+lags1` @ N=2000** | **0.744** | **0.283** | **0.671** | **0.673** | **0.671** |
| `fold-pca:64+lags1` (supervised) | 0.766 | 0.195 | 0.666 | 0.737 | 0.666 |
| `corpus-pca:64` @ N=2000 | 0.718 | 0.179 | 0.653 | 0.642 | 0.653 |
| **DeepMReye 1.0** | 0.728 | 0.097 | 0.625 | 0.485* | 0.625 |
| `raw+lags1` | 0.711 | 0.127 | 0.598 | 0.704 | 0.598 |

`*` `dsL06` via `datasets_1to5.h5`; the other three via `datasets_1to6.h5`. The
first three columns are one checkpoint and are the quotable median (0.625
sub-TR, 0.805 at 1-TR against our 0.671 / 0.837).

**Three things worth keeping.**

The axis conventions agree -- `r_y` is positive on every clean fold, so there is
no sign mismatch between v1's output and this corpus's y-down labels. That was
worth checking rather than assuming, given the corpus-level flip documented
below.

**0.625 is not a broken run**, and two independent publications say so: a
zero-shot application of the same pretrained model to three movie datasets
reports individual-level r 0.24-0.37 (Psychoradiology 2026), and MoCET
(Nat Commun 2025) reports r_x 0.42-0.52 / r_y 0.03-0.59 with predictions
"clustering toward screen center" -- which is the gain mis-calibration recorded
under *Calibration is a separate problem* below, observed by someone else.

**`dsL06`'s vertical axis reproduces as broken under the authors' own weights**
(r_x 0.884, r_y 0.083). That is the third independent confirmation that the fold
is a property of the paradigm, not of our features.

**The one asymmetry to state rather than bury:** on these folds our ridge is
fitted on the other eight labeled datasets, which includes v1's six *plus*
`dsL07`/`dsL11` when they are not the held-out one. Neither method saw the
held-out fold, but the supervised material is not identical -- we had two
acquisitions available that did not exist when v1 was published.

### Running the shipped weights on all nine folds: measured, and it does not help us

The tempting version of this comparison is "run `datasets_1to6.h5` on everything
and note that it trained on six of them, so a loss holds a fortiori". The
asymmetry is sound -- an advantage handed to the baseline strengthens a loss and
proves nothing about a win -- but it was **measured, and v1 does not lose**:

| fold | DeepMReye 1.0 | `lr-cca:32+lags1` @ N=2000 | |
|---|---|---|---|
| `dsL01`* | **0.875** | 0.770 | v1 |
| `dsL02`* | 0.907 | **0.920** | ours |
| `dsL03`* | 0.763 | **0.776** | ours |
| `dsL04`* | **0.883** | 0.851 | v1 |
| `dsL05`* | 0.813 | 0.803 | tie (0.010) |
| `dsL06`* | **0.856** | 0.673 | v1 |
| `dsL07` | 0.728 | **0.744** | ours |
| `dsL08` | 0.097 | **0.283** | ours |
| `dsL11` | 0.625 | **0.671** | ours |
| **median** | **0.813** | 0.770 | |

`*` in-sample for v1. **The 0.813 median is not a held-out number and must never
be quoted as v1's performance** -- it is what a model scores partly on its own
training data. The claim that survives is the narrow one: *we win every fold
neither method has seen, and v1 leads only where it was trained.* We also beat
it on two of its six training datasets, which is worth a clause and not more.

**`dsL06` measures the memorisation directly**, and it is the most useful number
here. It is the one dataset scored by two checkpoints differing *only* in
whether it was in training:

| | `dsL06` sub-TR |
|---|---|
| `datasets_1to6.h5` (in-sample) | 0.856 |
| `datasets_1to5.h5` (held out) | **0.485** |

A drop of **0.371** on identical participants under an identical metric. That is
the size of the in-sample advantage on this corpus, and it is why the all-nine
table above cannot be used in either direction as a transfer result.

### What else there is to compare against: nothing runnable

- **MRGazer** (Wu et al. 2024, J Neural Eng) is the only competing
  architecture. Code is released; **weights are not**, and it decodes in
  *individual space* from un-coregistered fMRI, which this corpus (fixed
  47x29x18 coregistered crop) cannot supply. Retraining it would make it our
  reimplementation, which is the thing using released weights exists to avoid.
- **bidsMReye** is a BIDS wrapper around the same model and the same weights.
- **MoCET** corrects *camera* eye-tracker drift with head-motion regressors. Not
  a decoder; the right citation for motivation.

---

## The two decodable ceilings

Both are properties of the **acquisition**, not of the decoder, and both were
mistaken for modelling problems first.

### Temporal resolution sets the score

Over the 12 (dataset, axis) cells, the *gaze trace's* lag-1 autocorrelation
predicts the decoded correlation at **Spearman rho = +0.797, p = 0.002**.
Ordered by autocorrelation the cells run from `dsL03.x` (0.128 -> r 0.181) up to
`dsL02.y` (0.851 -> r 0.874).

The evidence that makes it a mechanism rather than a correlation is `dsL06`'s
two axes, which dissociate *within the same scans*: lag-1 0.761 on x decoding at
0.947, against 0.253 on y decoding at 0.343. Same subjects, same TR, same
preprocessing, same model. A between-dataset trend would confound all of those
at once; this cannot.

Call it an **envelope**, not a ceiling: the fit is
`decoded_r = 1.03 * lag1 + 0.085` with residual SD 0.063, and one cell sits
0.111 *above* it. What is true is that the linear arm already achieves it
everywhere, while weaker arms fall below. The practical consequence:
**any representation improvement on this corpus is bounded to roughly 0.06-0.10 r
on a couple of cells**, not a wholesale gain. Read any new arm against that
budget before calling its score a disappointment.

`dsL03_pursuit` is the case in point and was chased for a long time as a
transfer or calibration failure. It is neither: held-out *subjects within
dsL03 itself* decode at 0.142, the same as cross-dataset, and its gaze simply
moves faster than its acquisition can resolve. `dsL02_pursuit` is the control
that settles it -- same paradigm, same within-subject gaze SD, autocorrelation
0.849, decodes at 0.911. Stop targeting dsL03.

### Calibration is a separate problem from representation quality

Cross-dataset predictions are mis-calibrated in **gain** (0.11 to 2.27 against
the training scale) with offsets near zero. This destroys R² while leaving r
intact. An oracle affine correction lifts mean R² from 0.043 to 0.389; every
*unsupervised* correction tried fails badly (z-match -0.921, quantile -0.973,
feature standardisation 0.003, mean shift 0.071).

The reason is identifiability, not effort. The required gain is about
`test_gaze_SD / train_gaze_SD`, and the target's marginal spread is exactly what
differs between a fixation task and a free-viewing task. Degrees of visual angle
depend on screen size and viewing distance, and neither is in the BOLD.

This is why `metrics.py` calibrates on held-out participants of the same dataset
rather than pretending the problem away. Report it as a separate problem.

---

## Things that were wrong in the data, and how they were caught

Every one of these passed the checks that existed at the time.

**Gaze y grows DOWNWARD, and getting it wrong is invisible to a lag sweep.**
Screen coordinates, top-left origin, so an EyeLink needs no flip and a flip is
the exception. Three datasets were ingested flipped and each decoded with a
*positive r_x and a negative r_y* against a readout trained on the rest. Negate
y and every lag scores the same magnitude, so the peak stays at 0 and the sync
verifier says PASS -- which it did, for all three, with healthy margins. The
convention is established from **anatomy**, not from another dataset: the
eyeball is a bright vitreous sphere with a dark lens at its anterior pole, so
looking up rotates that lens to higher z. Checking a new dataset against the
corpus is fine; checking the corpus that way is circular.

**A dataset can be rejected on its *imaging* rather than its gaze.** One
candidate had the best-validated time anchor in the corpus (clock ratio
1.000102, residual SD 0.28 ms) and gaze that was simply not in the volumes --
within subject it decoded at 0.232 with **r_x +0.071**, the easier axis
everywhere else, absent. Its orbits were clipped in the raw BOLD. Two of its
participants were already in the corpus, labeled "no eyes" by a human, and
ingesting it under a new name would have silently overwritten that judgement.

**Two datasets were retired, and neither had a labels problem.** One was resting
state with a central fixation dot: per-participant gaze SD 0.26-1.3 degrees
against 2.3-2.7 for the pursuit sets. There is no gaze variance in the paradigm
to decode, so the fold measured the task rather than the method. The other had
37-40% of samples at the tracker's track-loss code and its own authors' sidecar
saying "reliability would not be guaranteed". Cleaning did not rescue it.

**A broken dataset does not only cost you its own fold.** After the sign repair,
`dsL06` gained **+0.08** purely because a retired dataset left the *training*
pool. Nobody had touched `dsL06`'s labels.

**Retiring a dataset means removing its `labels`, not renaming it.** The loader
accepts any participant carrying a `labels` array; the `dsL*` prefix only
controls what gets *downloaded*. A dataset renamed out of the prefix silently
ran as its own fold until that was caught.

---

## Protocol decisions that took a measurement to settle

- **Metrics are aggregated per participant, then median across participants.**
  Pooling every row of every subject into one correlation is gameable: if one
  subject's gaze sits left of another's, a model that predicts only *which
  subject this is* scores a high pooled r with zero within-subject decoding.
- **The noise floor on a 9-fold median is ~0.02, and the data said so itself.**
  In a labeled-budget sweep the fold-local reference read 0.847 at 1000 training
  windows and 0.828 with all of them -- a method that can only improve with more
  labeled data, reporting that it got worse. Differences under 0.02 are ties.
- **Screening folds inflate the median.** A tuned arm measured on the folds used
  to screen it read +0.0099; on the six folds never used for screening it read
  **-0.0111**. Report unseen folds.
- **`lags±1` for sub-TR, `lags±0` for 1-TR.** Temporal context interpolates
  within-TR motion and blurs the 1-TR mean, so the optimal window depends on the
  target resolution. Inside the noise floor, but two protocols agree on the peak
  and it costs one integer.
- **Targets are z-scored per training dataset.** Without it the 9-fold median
  collapses to **0.131**, because one pooled ridge follows whichever dataset has
  the largest target variance and the Euclidean scale spans 21 to 595.
- **Dataset iteration must be `sorted`.** Set iteration order over strings
  varies with PYTHONHASHSEED between processes, changing which rows the fit sees
  -- about 0.01 of avoidable noise in a comparison meant to resolve 0.02.
- **There was an unexplained 20000-row cap on the training fit.** It had no
  comment, no docstring and no justification in the history, and it was
  copy-pasted into eight scripts. Removing it is worth at most 0.0024 anywhere
  tested, so no conclusion changed -- but it also retracted a filtering result
  that had looked significant under the cap (9/9 folds, p=0.004) and was a tie
  without it (6/9, p=0.250). **Fit on all the rows.**
- **Unsupervised feature alignment hurts.** Euclidean Alignment and CORAL, the
  standard cross-subject corrections in EEG/BCI, cost 0.09 to 0.13 median r. The
  between-component covariance of these features is *signal, not shift*, and
  whitening it per group removes gaze.
- **Nothing in the feature path may use torch.** LightGBM and PyTorch each load
  their own OpenMP runtime, and a threaded torch reduction after a LightGBM fit
  in the same process **deadlocks** -- no error, no traceback, the process just
  stops. `OMP_NUM_THREADS=1` masks it, which makes it look environment-specific.

---

## Trying to beat `lr-cca`: seventeen arms, and what each one closes

`scripts/improve_lrcca.py`, `scripts/hybrid_lrcca.py`, `scripts/beyond_cca.py`,
`scripts/shift_lrcca.py`, `scripts/tradeoff_participants_trs.py`,
`scripts/fit_lrcca_variants.py`, `scripts/corpus_pc_timeseries.py`,
`scripts/build_labeled_voxels.py`, `scripts/report_lrcca.py`; results under
`results/lrcca_variants/`.

Two pieces of infrastructure made this affordable and are the part worth
keeping. `fit_lrcca_variants.py` saves the corpus second moment, the
per-participant and per-slab means, and the lag-1 cross moment **once**, so
every covariance in the family -- total, within-participant, within-slab,
temporal difference, temporal sum -- is a subtraction rather than another pass
over 77 GB. `build_labeled_voxels.py` writes the 337 labelled participants'
masked voxels as one 22 GB memmap, so applying a new basis is a matmul instead
of re-reading 21 GB of HDF5. A complete arm then costs about 25 seconds, which
is what makes a shrinkage sweep worth doing at all.

Both references reproduce before anything is compared against them: the
incumbent refits at **0.7692 / 0.8379** against the shipped 0.7703 / 0.8382
(randomized-SVD seed, not a protocol difference) and `fold-pca:64+lags1` refits
at **0.7658** against the recorded 0.766. Every delta below is against the
refit.

Read all of it the way the temporal-prior entry says to -- **a 9-fold median is
a poor instrument below 0.02** -- so each arm carries folds-improved and a sign
test beside its median.

**Nothing won.** The best arm is +0.0029 at sub-TR and negative at 1-TR. What
follows is therefore a list of closures, and several of them are worth more than
the win would have been.

**Scan this table first.** Every row is closed; the section below is the
evidence. If a proposal is one of these, it has been measured --- say which row
it is not.

| # | family | what was tried | verdict | sub-TR |
|---|---|---|---|---|
| 1 | estimator | `shrinkage` 1e-4..0.3, `n_reduce` 64..512 | defaults already optimal | 0.769 |
| 2 | criterion | whitening exponent PLS(0) -> CCA(1) | monotone **to CCA**; justifies the arm | 0.760 -> 0.769 |
| 3 | covariance | within-participant, within-slab | tie (data is z-scored; between-subject is 3.8% of trace) | 0.770 / 0.771 |
| 4 | covariance | temporal difference (`fast`), temporal sum (`slow`) | worse, 1/9 and 0/9 folds | 0.756 / 0.734 |
| 5 | symmetry | mirror-tied `w_R = M w_L` | tie at **half the parameters**; same k=32 optimum | 0.770 |
| 6 | symmetry | selecting modes by mirror eigenvalue *sign* | catastrophic, 0/9 folds | 0.24-0.47 |
| 7 | beyond 2nd order | ICA ranked by kurtosis; dictionary learning | beat variance, lose to bilateral | 0.760 / 0.753 |
| 8 | control | random orthogonal subspace, matched rank | the floor the shipped arm never had | 0.394 |
| 9 | corpus | participants vs TRs at fixed 96k rows | tie 2000->250; **-0.050 on 9/9** at 125 | 0.769 -> 0.720 |
| 10 | corpus | more TRs per participant at N=2000 | saturated: 8x data buys +0.004 | 0.773 |
| 11 | data | per-participant registration shift (27) | agreement rule hurts; **label oracle only +0.011** | 0.761 |
| 12 | data | fold-local + corpus, hybrid and transductive | corpus alone wins; concatenation *costs* | 0.766 / 0.765 |
| 13 | data | cross-orbit fitted on labelled voxels only | loses to corpus **and** to variance on the same rows | 0.755 |
| 14 | physics | analytic eyeball-rotation basis (no fitting) | 3 free directions reach 0.383 vs 0.161 random | 0.383 |
| 15 | temporal | lag-1 hard negatives (deflated cross-covariance) | peak dies under nested selection (2/9 folds) | 0.774* |
| 16 | temporal | cross-orbit CCA over a 3- or 5-TR window | **more agreement, less gaze**, 0/9 folds | 0.574 / 0.505 |
| 17 | temporal | z-band (slice-timing) features; sub-TR alignment | blind under multiband; alignment verified **correct** | 0.742 |

`*` screening-fold number; it does not survive nested selection.

### Reproducing any of it

`results/` is **gitignored**, so this file is the record, not the JSON. Every
number above is regenerable, and the two caches are what make that cheap:

| step | command | writes |
|---|---|---|
| corpus moments (2.2 min) | `fit_lrcca_variants.py --n 2000` | second moment + group means |
| lag-1 cross moment (2.5 min) | `fit_lrcca_variants.py --n 2000 --lag1` | for rows 4 and 15 |
| labelled voxel memmap (2.2 min, 22 GB) | `build_labeled_voxels.py --out <dir>` | every arm is then a matmul |
| corpus PC timeseries (2.5 min) | `corpus_pc_timeseries.py` | 96k x 512; needed for row 7 |
| rows 1-8 | `improve_lrcca.py --arms shrink,reduce,within,mirror,sign,whiten,random,fast,combine` | `sweep*.json` |
| rows 9, 10 | `tradeoff_participants_trs.py` | `tradeoff.json`, `more_trs.json` |
| rows 11-13 | `shift_lrcca.py --oracle`, `hybrid_lrcca.py` | `shift*.json`, `hybrid.json` |
| rows 14-17 | `anatomical_basis.py`, `temporal_contrast.py`, `slice_time.py`, `subtr_align.py` | one JSON each |
| fold-level test | `report_lrcca.py results/lrcca_variants/*.json` | median, mean, folds-won, sign test |


### The estimator inside `fit_lr_cca` had never been swept. It is at its optimum.

`n_reduce=256` and `shrinkage=1e-3` are bare constants in
`deepmreye/unsupervised.py`. If "`lr-cca` needs 400 participants before it beats
variance PCA" is an estimation-variance statement, these are the knobs that
should say so.

| | 1e-4 | 1e-3 | 1e-2 | 3e-2 | 1e-1 | 3e-1 |
|---|---|---|---|---|---|---|
| shrinkage, sub-TR | 0.7670 | **0.7692** | 0.7722 | 0.7702 | 0.7640 | 0.7612 |
| shrinkage, 1-TR | 0.8346 | **0.8379** | 0.8358 | 0.8290 | 0.8171 | 0.8184 |

| | 64 | 128 | 256 | 512 |
|---|---|---|---|---|
| `n_reduce`, sub-TR | 0.7613 | 0.7663 | **0.7692** | 0.7653 |

`n_reduce=256` is exactly optimal in both directions. Shrinkage at 0.01 is
+0.0029 sub-TR on **5 of 9 folds** and -0.0021 at 1-TR: a tie, and the two
resolutions disagree, which is what a tie looks like. **The regularisation is
not what limits this basis**, so the corpus-size requirement is not a
whitening-variance story.

### Whitening is the whole mechanism, and the dose-response says so

CCA whitens each orbit before taking the SVD of the cross block; **partial least
squares does not whiten at all** and orders directions by shared *magnitude*
rather than shared *correlation*. Nothing here had ever asked the PLS question.
Both are one estimator with an exponent on the eigenvalues, so the family can be
swept:

| whitening exponent | 0 (PLS) | 0.25 | 0.5 | 0.75 | 1 (CCA) |
|---|---|---|---|---|---|
| sub-TR | 0.7603 | 0.7604 | 0.7625 | 0.7686 | **0.7692** |
| 1-TR | 0.8168 | 0.8190 | 0.8201 | 0.8318 | **0.8379** |

**Monotone to the CCA end, on both resolutions.** This is the first positive
evidence in the project that the *correlation* criterion is the right one rather
than merely the one that was implemented: whitening is precisely what lets a
low-variance direction be promoted for agreeing across orbits, and removing it
costs 0.009. Quote this instead of asserting that CCA is the natural choice.

### The covariance it is fitted from does not matter, and the temporal ones hurt

`C_total = C_within + C_between` is exact from the saved means, so the
between-participant structure a cross-orbit CCA might be locking onto can simply
be removed.

| covariance | sub-TR | 1-TR | folds | p |
|---|---|---|---|---|
| total (incumbent) | **0.7692** | **0.8379** | -- | -- |
| within-participant | 0.7695 | 0.8364 | 2/9 | 0.180 |
| within-slab | 0.7711 | 0.8361 | 6/9 | 0.508 |
| temporal difference (`fast`) | 0.7557 | 0.7833 | 1/9 | 0.039 |
| temporal sum (`slow`) | 0.7344 | 0.7859 | 0/9 | 0.004 |

Note the within-participant row is a median artefact of exactly the kind the
temporal-prior entry warns about: +0.0003 on the median while improving **2 of
9** folds. It is a tie at best.

The within-participant arm is nearly moot for a reason worth writing down:
**`eye_block` is z-scored per voxel across time**, so the between-participant
scatter is only **3.8% of the trace** and there is very little to remove. Any
future "remove subject identity" proposal should start from that number.

`fast` is the interesting negative. The recorded explanation for why every
predictive objective failed is that *the predictable part of an eye block is the
nuisance* -- so a cross-orbit basis fitted on temporal **differences** should
have asked the bilateral question about the fast part only. It is worse on 8 of
9 folds. Gaze and the nuisance overlap in temporal band for the *cross-orbit*
basis exactly as `nuis-pca` showed they do for the variance one, and this now
closes that escape from the third direction.

### Mirror symmetry: a tie at half the parameters, and the sign is not the gaze

The two orbits are approximate mirror images, so `w_R = M w_L` is a real prior
and it halves the free parameters of the cross-covariance. The reflection plane
is *measured* (x = 23.5, maximising mask overlap with its own reflection),
pairing 6741 voxels covering 13482 of 14236, and unpaired voxels are dropped
rather than extrapolated. The SVD becomes a symmetric generalized eigenproblem.

| | sub-TR | 1-TR | folds | p |
|---|---|---|---|---|
| incumbent | **0.7692** | 0.8379 | -- | -- |
| mirror, shrinkage 0.01 | 0.7696 | **0.8432** | 7/9 | 0.180 |
| mirror, shrinkage 0.001 | 0.7684 | 0.8449 | 6/9 | 0.508 |

A tie -- 7 of 9 folds and +0.0064 mean is the same size as the
Savitzky-Golay effect, which did not survive either -- but **a tie reached with
half the parameters**, and one that peaks at the same k = 32 with the same cliff
below 24 (0.4915 / 0.7647 / 0.7696 / 0.7587 / 0.7633 at k = 16 / 24 / 32 / 48 /
64). Worth stating as a mechanism result: the bilateral constraint survives being
tied to the anatomy.

**The sign of the mirror eigenvalue is not a gaze detector, and the prediction
was clean enough to be worth recording as wrong.** Under the reflection,
conjugate vertical gaze should move the orbits together and horizontal gaze
oppositely, while the bilaterally coherent nuisance is symmetric -- so the
antisymmetric half of the spectrum should be gaze-enriched. It is not:
symmetric-only scores 0.40-0.45 and antisymmetric-only 0.24-0.47, **0/9 folds,
p = 0.004** each, and a 16+16 split of the two scores 0.5644 against 0.7696 for
the top 32 by `|lambda|`. The magnitude ordering carries everything; the sign
carries nothing.

### Past second order: independence and sparsity lose to correlation

Every basis this project has compared is a decomposition of a second moment, so
*independence*, *sparsity* and *non-Gaussianity* had never been used as
selection criteria. `corpus_pc_timeseries.py` makes them affordable by reducing
96,000 corpus TRs to their top 256 principal directions per orbit -- 212 MB
instead of 5 GB.

The framing that makes the comparison fair: **a linear readout sees only the
subspace.** Rotating k features among themselves is invisible to ridge, so ICA
cannot help by rotating, only by selecting a different k-dimensional subspace,
and it has to be scored that way.

| subspace of the same 512 orbit PCs, k=32 | sub-TR | 1-TR |
|---|---|---|
| cross-orbit correlation (`lr-cca`) | **0.7692** | **0.8379** |
| ICA, ranked by \|excess kurtosis\| | 0.7596 | 0.8175 |
| dictionary learning (atoms as a projection) | 0.7530 | 0.8041 |
| leading variance (`corpus-pca` restricted) | 0.6922 | 0.7325 |
| **random orthogonal** | **0.3938** | **0.3956** |

Two things to keep. **Non-Gaussianity is a real criterion** -- ICA beats
variance by +0.067 at matched rank, which is not nothing -- **and it still loses
to bilateral agreement by 0.010.** And the random-projection floor says what the
criterion is worth at all: **+0.375 over a random subspace of identical rank**.
That is the control `CLAUDE.md` asks for, for the shipped arm, and it had never
been run.

### The analytic eyeball basis: three directions, nothing fitted, 0.383

Recorded above as blocked, and it was not. `eye_block` is z-scored per voxel, so
the *corpus* carries no mean image -- but `m` never had to come from the corpus.
**`deepmreye/masks/dme_template.nii` is the template every participant is
registered to**, it is a real anatomical volume (16..2872), and cropping it with
`cut_mask`'s own edges lands it on **exactly the 14236-voxel grid**, non-zero
support matching the corpus mask voxel for voxel. Zero new preprocessing.

An eyeball is a rigid high-contrast sphere, so to first order rotating it
changes a voxel by `-grad m(v) . (omega x (v - c))`: **three directions per
orbit, written down rather than estimated** (`scripts/anatomical_basis.py`).

| basis | k | sub-TR | 1-TR |
|---|---|---|---|
| `lr-cca:32` | 32 | **0.7692** | **0.8379** |
| analytic **rotation**, orbit-averaged | **3** | **0.3831** | 0.4014 |
| analytic translation, orbit-averaged | 3 | 0.2622 | 0.2714 |
| random orthogonal | 3 | 0.1611 | 0.1778 |
| random orthogonal | 32 | 0.4311 | 0.4321 |

**Three parameter-free directions reach 0.383 against a matched-rank random
floor of 0.161**, and land within 0.05 of a random *32*-dimensional subspace.
The physics ordering is right too: rotation beats translation by 0.121, which is
what distinguishes "the sphere rotates" from "any smooth spatial filter".
Gradients of `log m` (the stand-in for the per-voxel sigma normalisation
discarded) are consistently *worse*, so that correction is not the missing
piece.

**It does not add to the fitted basis.** Concatenated with `lr-cca:32` it scores
0.7633 against 0.7692 -- and not because it is redundant: principal cosines
between the analytic triple and the `lr-cca:32` span are **0.66 / 0.38 / 0.14**,
mean r-squared 0.199, so four fifths of it lies *outside* what the corpus basis
found. The part outside is simply not decodable gaze. Treat the arm as what it
is -- a mechanistic check that the bilateral basis is reading eyeball rotation,
obtained for free -- not as a candidate.

Recovering the same thing **per participant** would need the ingest pipeline to
keep `masked_eye.mean(-1)` and `.std(-1)` before `pipeline.py`'s `normalize_img`
call (98 KB each, ~216 MB for the corpus), and therefore a re-run: ~2 s staging
plus 42-110 s of ANTs per participant and ~460 GB of raw BOLD. Note also that
only **67 of 337** labelled participants carry an S3 `source_key` -- `dsL01`
through `dsL06` exist locally only as already-normalised `.npz`, so their
anatomy is not recoverable from this repository at all.

### A temporal contrast instead of a spatial one: measured, and it is the same trap

`scripts/temporal_contrast.py`, `results/lrcca_variants/temporal_contrast.json`.

`lr-cca` maximises `corr(z_L(t), z_R(t))` -- two orbits, one instant -- so the
contrast it draws is **spatial**, and in the linear limit it is exactly
cross-view InfoNCE with negatives drawn at random times. Its negatives are easy
ones. The obvious missing arm is the **hard**-negative version: agree across the
orbits at *this* TR and **not** at the neighbouring one, which asks for
directions that are bilateral *and* fast and should separate gaze from the
bilaterally coherent slow nuisance the cross-orbit constraint inherits by
construction.

Two linear forms, both from moments already on disk, and they move in **opposite
directions along the same axis**, which is what makes the pair informative.

**Deflating the lag-1 agreement out**, with the whitening held on the ordinary
covariance so only the criterion moves (this is what the `fast` arm above could
not separate):

| `C_LR - lam * C_LR^(1)` | 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|
| sub-TR | 0.7692 | 0.7732 | **0.7742** | 0.7716 | 0.7684 |
| 1-TR | 0.8379 | 0.8390 | **0.8406** | 0.8377 | 0.8260 |
| folds (sub-TR) | -- | 7/9 | 7/9 | 7/9 | 4/9 |

**Widening the window instead** -- cross-orbit CCA between a 3- or 5-TR window of
each orbit, solved in corpus-PC space because a voxel-space lag stack would be a
42708^2 accumulator:

| window | lag 0 only | +-1 TR | +-2 TR |
|---|---|---|---|
| 32nd canonical correlation | 0.355 | 0.517 | 0.599 |
| sub-TR (readout lags+-1) | 0.7716 | 0.5742 | 0.5052 |

**More bilateral agreement, less gaze, monotonically.** A wider window finds far
*more* cross-orbit structure -- rho32 rises 0.355 -> 0.517 -> 0.599 -- and
decodes far worse, 0/9 folds at p = 0.004 both times. The extra agreement a
temporal window buys is slow shared nuisance, which is the same verdict `ocon`,
`gev-slow` and the contrastive arm returned, now from the one direction that had
not been tried. (`multilag0` is the implementation control: the incumbent
re-derived through an entirely separate PC-space path, 0.7716 / 0.8390 against
0.7692 / 0.8379.)

**And the deflation peak does not survive nested selection.** `lam = 0.5` reads
+0.0050 sub-TR on 7 of 9 folds and +0.0027 at 1-TR on 8 of 9 (p = 0.039), which
is exactly the shape the Savitzky-Golay entry warns about -- so choose `lam` on
the eight other folds and read the ninth:

| | nested median | d | mean d | folds |
|---|---|---|---|---|
| sub-TR | 0.7684 | -0.0008 | **-0.0079** | **2/9** |
| 1-TR | 0.8390 | +0.0011 | -0.0037 | 7/9 |

and the chosen `lam` is different on almost every fold (1, 1, 1, 1, 1, 0.75, 0,
0, 0.25 at sub-TR). Inner selection chasing noise, for the second time in this
file. **Report the axis, not the peak.**

The durable statement is the pair: agreement *added* across time is nuisance
(0/9 folds, decisively), and agreement *removed* across time is worth nothing a
fold-honest protocol can keep. The lag-0 restriction the incumbent already
imposes is the whole of the available temporal contrast.

### Slice timing: the right mechanism, the wrong axis, and why the test was blind

`scripts/slice_time.py`, `scripts/subtr_align.py`,
`results/lrcca_variants/{slice_time,subtr_align}.json`.

Ten gaze labels per TR is a meaningful target because of acquisition physics,
and the DeepMReye paper says so outright: *"different imaging slices are being
acquired at a different time within each TR and thus inherently carry some
sub-TR information"*. A volume is not an instant, it is a sweep -- and every
basis here pools all eighteen z slices of the eye crop into one spatial pattern,
which averages that sweep away. The slice-timing literature is entirely about
*correcting* the effect; decoding with it is unexplored.

**The arm fails.** Cross-orbit bases fitted inside contiguous z bands, then
concatenated:

| | sub-TR | 1-TR |
|---|---|---|
| `lr-cca:32` (incumbent) | **0.7692** | **0.8379** |
| 6 z bands concatenated (6 x 16) | 0.7421 | 0.7958 |
| incumbent + 6 z bands | 0.7572 | 0.8054 |

Per band alone the score tracks nothing but how much eyeball the band contains
(0.19 / 0.32 / 0.43 / 0.72 / 0.71 / 0.31 for z = 0-3 ... 15-18). Splitting the
basis by z costs 0.027 and buys nothing.

**And the diagnostic says why, though it took a control to see it.** Per band,
the decoded correlation against each of the ten within-TR gaze samples rises
monotonically from early to late (0.92 -> 1.03 of the band's own mean) -- but the
preference does **not order with z** (centroids 5.60 / 5.77 / 6.01 / 5.71 / 5.84
/ 6.04). So it is not a sweep. It looked instead like a global sub-TR
misalignment, which would have been a real finding: `verify_gaze_sync.py`
validates at **TR** resolution, so a sub-TR offset is invisible to it by
construction -- the same blind spot that let three datasets ship with a flipped
y axis.

It is not that either, and the control is the whole point:

| profile | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | centroid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `lags0` | 0.928 | 0.974 | 1.009 | 1.028 | 1.043 | 1.048 | 1.041 | 1.014 | 0.984 | 0.932 | **4.62** |
| `lags+-1` | 0.922 | 0.944 | 0.975 | 0.995 | 1.014 | 1.026 | 1.030 | 1.031 | 1.032 | 1.031 | 5.78 |

**At `lags0` the profile is symmetric about the TR midpoint (4.62 against an
expected 4.5).** The ramp exists only when the readout can see the volume at
`t+1` and use it to predict the late samples of TR `t`. It is the lag stack, not
the alignment. A shift sweep agrees: `d = 0` is the 1-TR optimum (0.8379 against
0.8329 at `d = -2` and 0.8200 at `d = +1`), and the apparent sub-TR climb toward
`d = +4` does not survive nested selection (mean **-0.0023**, **1/9** folds, `d`
chosen inconsistently across folds).

**So the corpus is correctly aligned at sub-TR resolution.** That is worth
recording as a positive: it is the first check of the alignment at a resolution
finer than the verifier operates at.

**Why the z test was blind, and what would not be.** `ds006833`'s sidecar reports
`SliceTiming` over **60 slices at multiband factor 4**. Under multiband the map
from z to acquisition time is *periodic*, not monotone -- a contiguous z band
mixes several acquisition times by construction -- and the factor, slice count,
order and orientation all differ across the nine datasets, so a pooled
cross-dataset readout averages maps of different periods and phases. The
diagnostic could not have fired even if the effect were there.

The test that would work needs voxels grouped by **acquisition time**, not by z,
which needs the per-dataset `SliceTiming` vector. The HDF5 does not store it.
That is the second one-field-at-ingest recommendation in this file, alongside the
anatomical mean: **store `SliceTiming` and the slice axis**, and the experiment
becomes a grouping change rather than a re-ingest.

### The sub-TR capacity is already reached, and two literatures agree on the number

Worth stating plainly, because it bounds the whole temporal program. The DCT
readout arm found that **two** DCT coefficients per axis match all ten sub-TR
samples (`c=2` 0.7690 against the baseline 0.7678), and reduced-rank ridge found
the coefficient matrix is already rank ~4 -- two structured regularisers
converging on "a mean and a slope per TR". The DeepMReye paper reports, from the
other side, that up to **three** positions per TR explain more gaze-path variance
than one, and no more.

**A linear readout on a frozen corpus basis is therefore already extracting the
sub-TR capacity a trained CNN reports.** Any temporal proposal has to say which
of the three it is beating.

### Participants or TRs? The confound the scaling curve cannot separate

The headline scaling result grows participants, TRs and accessions together.
Holding the **total** corpus TR count fixed at 96,000 and trading participants
against TRs each separates them:

| corpus | subjects | TRs | sub-TR | d | folds better |
|---|---|---|---|---|---|
| 2000 x 48 | 2000 | 96000 | **0.7692** | -- | -- |
| 1000 x 96 | 1000 | 96000 | 0.7652 | -0.0040 | 5/9 |
| 500 x 192 | 500 | 96000 | 0.7607 | -0.0086 | 6/9 |
| 250 x 384 | 250 | 95868 | 0.7563 | -0.0129 | 4/9 |
| 125 x 768 | 125 | 95856 | 0.7196 | **-0.0496** | **0/9** |

**The honest reading is not "participants win".** Between 250 and 2000 it is a
tie on every instrument -- the mean delta is ~0 and the fold counts are 4-6 of 9
-- so in that range the corpus is buying *rows* and it does not matter how many
people supply them. Below that it collapses: 125 participants lose 0.0496 on
**all nine folds**. Set against the scaling curve, where 200 participants at 48
TRs each read 0.669, the same 250 participants reach 0.7563 when each supplies
384 TRs, so TRs genuinely substitute for people -- down to a floor of a few
hundred distinct people, and no further.

State it that way in the paper. The 25 -> 2000 scaling claim is unaffected,
because at a fixed 48 TRs per participant the participant axis *is* the data
axis; what this rules out is the stronger reading that distinct people are
themselves the resource across the whole range.

### The corpus TR budget is saturated as well

The corpus basis has always been fitted at 48 TRs per participant and nobody had
asked for more, at fixed N = 2000:

| TRs per participant | 48 | 192 | 384 |
|---|---|---|---|
| sub-TR | 0.7692 | 0.7732 | 0.7733 |
| 1-TR | 0.8379 | 0.8371 | 0.8377 |

+0.004 for **eight times** the data, flat thereafter, and nothing at 1-TR. The
48-TR budget costs nothing. Both corpus axes -- participants past ~800 and TRs
past ~192 -- are now measured saturated.

### Residual misregistration is not costing this basis anything

A frozen linear filter has no translation tolerance, which is the one thing a
convolutional decoder gets for free, and the eyeball is the worst structure in
the head to coregister. The basis was re-read at each of 27 integer shifts, with
the shift for a participant chosen label-free by **the criterion the basis is
built on** -- the agreement of its own two orbits.

| | sub-TR | 1-TR |
|---|---|---|
| no shift | **0.7692** | 0.8379 |
| per-participant, agreement-selected | 0.7608 | 0.8333 |
| per-participant, **label-chosen oracle** | 0.7800 | 0.8454 |

156 of 337 participants keep `(0,0,0)` and the rest pick a neighbour, so the
criterion is doing something -- it is just not doing gaze. The number that
closes the question is the oracle: choosing each participant's shift **with its
labels**, over 27 candidates, is worth only +0.011, and taking the maximum of 27
noisy per-participant correlations buys much of that by itself. There is no
alignment headroom here for any rule to find. Do not build per-participant
registration machinery.

### The labelled voxels are not the corpus's equal, and concatenating them costs

A fold is entitled to the eight training datasets' **voxels** -- `fold-pca` is
built from exactly those -- so the three-way comparison is fittable. All at 200
labelled TRs per participant, one basis per fold, held-out dataset excluded from
its own fold's accumulator:

| basis | sub-TR | 1-TR |
|---|---|---|
| `lr-cca:32` corpus only (inductive) | **0.7692** | 0.8379 |
| `fold-pca:64` labelled voxels, variance | 0.7658 | 0.8308 |
| `lr-cca:32` **labelled voxels only** | 0.7549 | 0.8290 |
| `lr-cca:32` corpus + labelled (hybrid) | 0.7655 | **0.8451** |
| `lr-cca:32` corpus + labelled incl. held-out (transductive) | 0.7653 | 0.8471 |
| `fold-pca:64` + `lr-cca:32` corpus, concatenated | 0.7610 | 0.8326 |

**The cross-orbit criterion fitted on the labelled voxels alone loses to the
corpus by 0.014 and to variance-ordered `fold-pca` on the same rows by 0.011.**
So the criterion is not self-sufficient -- it needs the corpus, which is the
cleanest statement yet of what the unlabelled half is for.

**And the negative that has to be reported with it: concatenating the corpus
basis onto the fold-local one is worse than `fold-pca` alone (0.7610 vs
0.7658).** The corpus does not add a component the labelled voxels are missing;
the two span overlapping subspaces and the extra columns cost more than they
carry. The defensible claim stays the deployment one already recorded above --
the corpus removes the need for target-study data -- not an accuracy one.

Transductive adaptation (the held-out participants' own voxels in their own
basis, still no labels) is worth **-0.004 sub-TR / +0.009 1-TR**. The deployment
case does not need it.

---

## What is left worth trying

Honestly: not much on this corpus, and that is the finding rather than a
failure.

The two cells with real headroom against the temporal envelope are `dsL01.y`
(-0.098 residual) and `dsL02.y` (-0.086), both vertical axes of
high-autocorrelation datasets. That is a budget of roughly 0.06-0.10 r on two
cells.

The scarce resource is **independent acquisitions**, not participants, not
unlabeled data, and not model capacity. Nine folds is what every claim here
rests on. A tenth well-verified gaze dataset is worth more than any of the eight
architectures above.

After the sweep above, the closed list now also covers the estimator
(`shrinkage`, `n_reduce`, whitening exponent), the covariance it is fitted from
(within-participant, within-slab, temporal difference, temporal sum), the
anatomical symmetry prior, selection criteria past second order (ICA,
dictionary learning), per-participant registration, and both corpus budget axes.
So before proposing anything else, say which of these it is *not*.

Four specific things were not run and are the honest gaps:

- **Bagged bases** -- `lr-cca` on disjoint corpus subsets, concatenated. The
  shrinkage null argues against it, but bagging attacks *selection* variance
  rather than whitening variance, which is a different failure.
- **Sub-voxel and per-orbit-independent shifts.** Only common integer shifts
  were tried; the oracle bound of +0.011 says the ceiling is small either way.
- **Slice timing, grouped properly.** The z-band test above was blind by
  construction under multiband; grouping voxels by acquisition time instead
  needs the per-dataset `SliceTiming` vector stored at ingest. This is the one
  temporal idea the measurements did not actually close.
- **Per-participant anatomy.** The *template* version of the analytic basis is
  now measured (see above) and does not add to `lr-cca`. The per-participant
  version -- each subject's own mean and SD image, which would give an exact
  matched filter instead of a template one -- is the remaining variant, and it
  needs two lines in `pipeline.py` plus a full re-ingest. Given that the
  template triple lies 80% outside the `lr-cca` span and still adds nothing, and
  that the label-chosen shift oracle is worth only +0.011, do not expect much;
  but **add the two lines the next time anything is re-ingested**, because then
  it costs nothing.
