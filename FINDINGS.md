# What we tried, and what it taught us

This is the record. The package is small because most of what was built got
deleted, and the point of this file is that the deletions were *measured* rather
than abandoned. Read it before proposing an improvement -- several of the
obvious ones are closed, and a few are closed by evidence that took months to
get.

Everything below is leave-one-dataset-out over 337 gaze-labeled participants in
9 datasets unless stated otherwise.

---

## The corpus was refitted on 6 October 2026

**What changed.** The unlabelled corpus is now every eligible participant:
`unsupervised.corpus_subjects`, **2,060 participants from 652 datasets**
(98,880 volumes at 48 per participant), against a random 2,000 of the 2,064
eligible (649 datasets) before. Four of those 2,064 were **evaluation
participants under another name** (`ds006833` sub-01/02 = `dsL07`, `ds006642`
sub-01/02 = `dsL11`, found by a decoded-trace correlation of 0.997), and all
four had been drawn; `unsupervised.EVALUATION_SOURCES` now excludes those
accessions and `ds000113`. Eligibility is unchanged: fully covered eye mask,
at least 32 volumes, no gaze labels. The QA labels gate downloads, as before,
not the corpus: 53 corpus datasets (73 participants) carry a failing QA label.

**The corpus-size axis now has draws.** Below the full corpus there are many
ways to pick N participants, so each of N = 10, 100, 200, 400, 800 and 1,600 is
fitted on **20 independent random draws** (seed 0 is the nested main sweep in
`results/corpus_bases/`, seeds 1-19 are in `results/corpus_bases/draws/`) and
scored by `--stages scaling-draws` -> `scaling_draws.json`. Intervals below are
the central 95% of draws -- which participants are drawn, not which labelled
participants are scored. At N = 1,600 the draws share most of their
participants, so that interval is narrow by construction.

**What it did to the numbers: nothing outside the noise floor at the full
corpus.** Every stage of `scripts/analysis_unlabeled.py` was rerun
(`results/corpus_bases/`; the previous corpus is archived as
`results/archive/2026-10-05_*_v1`). Sections marked *(2,060)* below carry the
new numbers. The other sections quote the 2,000 draw: the closed experiments
of "Trying to beat `lr-cca`" were not rerun, since a 0.001 shift in the
headline cannot move a verdict that rests on a 0.02 floor, and the rerun
stages they cite (`budget`, `transfer`, `gap`, `overlap`, `fewshot`, `priors`,
`participants`) moved by at most 0.004 -- the table above has the ones the
paper uses.

| measurement (sub-TR median unless noted) | 2,000 draw | **all 2,060** |
|---|---|---|
| `lr-cca:32+lags1`, median / 1-TR / mean over datasets | 0.770 / 0.838 / 0.721 | **0.771 / 0.835 / 0.721** |
| `lr-cca:32`, no temporal context | 0.741 | 0.741 |
| `lr-cca:32+lags1` at N = 10 / 100 / 400, mean of 20 draws [95%] | -- | 0.647 [0.567, 0.712] / 0.635 [0.543, 0.703] / 0.744 [0.717, 0.757] |
| gain to the full corpus from N = 10 (mean of draws) | +0.154 from N = 25, one draw | **+0.124**, 9/9 folds (+0.10 to +0.17) |
| `corpus-pca:64+lags1`, N = 10 / 100 / full | 0.724 (N = 25) / -- / 0.754 | 0.732 / 0.751 / 0.754 (flat from 100) |
| corpus size at which `lr-cca` passes the same ridge on voxels | ~500 | **~450** (mean of draws) |
| lead over that ridge (2x2x2 voxels, 0.747, corpus-free), mean over datasets | +0.035 [0.021, 0.065] | **+0.035 [0.020, 0.064]**, 8/9 datasets |
| the same at 1-TR (against 0.784) | -- | +0.038 [0.018, 0.068], 9/9 datasets |
| lead with one labelled dataset (mean) | +0.097 to +0.11 | **+0.104** [99% CI 0.081, 0.127], every one of 20 draws |
| ... with 2 / 4 / 6 / 8 labelled datasets | -- | +0.069 / +0.051 / +0.037 / +0.033, every draw |
| lead over a PCA of the labelled scans with 1 / 2 / 8 labelled datasets | +0.05 / +0.05 / 0 | +0.055 / +0.044 / +0.001 |
| matched rank k = 32: `lr-cca` / whole-mask PCA | 0.770 / 0.744 | 0.771 / 0.743 |
| readout fitted inside the dataset | 0.830 | 0.831 |
| orbit agreement: AUROC for the worst fifth / keep 80% / oracle | 0.93 / 0.802 / 0.804 | 0.94 / 0.802 / 0.804 |
| on the 80% most reliable: lead over the same ridge | -- | +0.034 |
| consensus prior, mean over datasets | 0.721 -> 0.742 | 0.721 -> 0.742 |
| a new study's own K labelled participants (ridge on ours) at K = 1 / 2 / 4 | 0.682 / 0.728 / 0.789 | 0.680 / 0.726 / 0.785 |
| the published CNN, mean over the three datasets acquired after it | 0.483 against 0.566 | 0.483 against 0.567 |

`probe.CALIBRATION` is now 0.7412 / 0.7712 (2,000 draw: 0.7408 / 0.7703).

A QA-gated corpus (1,865 participants of the 570 datasets passing QA, with the
auto-labeller's labels dropped) was fitted and fully analysed the night
before, by mistake -- the request was to remove the four evaluation
participants only. It read 0.773 / 0.832 / 0.721, the same within the floor,
and is archived as `results/archive/2026-10-06_*_qa_gated`.

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

*(2,060)* Re-measured on the nine-fold protocol (`scripts/analysis_unlabeled.py
--stages scaling,scaling-draws,transfer`, sub-TR median over datasets, the
shipped `lr-cca:32 + lags1` arm). Each size below the full corpus is the mean
of **20 random draws** of that many participants, with the central 95% of
draws in brackets; the full corpus is one fit:

| unlabeled participants | 10 | 100 | 200 | 400 | 800 | 1,600 | 2,060 |
|---|---|---|---|---|---|---|---|
| `lr-cca:32+lags1` | 0.647 [0.567, 0.712] | 0.635 [0.543, 0.703] | 0.701 [0.629, 0.739] | 0.744 [0.717, 0.757] | 0.760 [0.752, 0.766] | 0.767 [0.763, 0.771] | **0.771** |
| ... at 1-TR | 0.676 [0.602, 0.739] | 0.666 [0.568, 0.748] | 0.743 [0.665, 0.786] | 0.790 [0.778, 0.812] | 0.810 [0.797, 0.829] | 0.827 [0.813, 0.837] | **0.835** |
| `corpus-pca:64+lags1` | 0.732 [0.694, 0.753] | 0.751 [0.741, 0.760] | 0.754 [0.748, 0.762] | 0.753 [0.750, 0.757] | 0.754 [0.751, 0.758] | 0.753 [0.749, 0.757] | 0.754 |
| `corpus-pca:64` (seed 0 only) | 0.723 | 0.736 | 0.741 | 0.740 | 0.744 | 0.745 | 0.746 |
| datasets in the seed-0 draw | 10 | 86 | 162 | 287 | 453 | 608 | 652 |

The 2,000 draw (one nested draw, sizes 25-2000): `lr-cca:32+lags1` 0.616 at 25
to 0.770, `corpus-pca:64` 0.714 to 0.746, and the `gev-slow:64` control -- the
same accumulators selected for slow drift -- fell from 0.580 to 0.272.
`gev-slow` needs a temporal-difference accumulator the package no longer builds
and was not refitted. The QA-gated corpus of 6 October (one draw) had read
0.602 at 25 and 0.554 at 100 -- single draws at small N sit anywhere in the
intervals above.

With no corpus at all: `raw+lags1` **0.729**, `fold-pca:64+lags1` **0.766** --
the no-corpus reference, a PCA fitted on the labeled scans themselves (its basis
sees no labels; the supervised voxel decoders are under "What the corpus buys")
and given 200 TRs
per participant against the corpus bases' 48, so that it cannot lose on TR
budget. The sweep reproduces `probe.CALIBRATION` exactly at the full corpus
(0.7712 and 0.7412), so it is the shipped number rather than a second
implementation of it.

**The two bases scale completely differently, and that is the finding.**
`corpus-pca` is ordinary variance and, with the same +-1 TR of context, is at
its level by 100 participants (0.751) and gains nothing after (0.754 at 2,060);
even 10 participants give it 0.732 -- it does not need the data. `lr-cca`
gains **+0.124** from 10 participants and **+0.136** from 100, and *loses to
variance PCA below about N = 600*: it is a cross-covariance between two
~7100-voxel spaces, so it has vastly more to estimate, and it only becomes the
best arm once the corpus is large. State the corollary plainly -- with 200
unlabeled participants you would have shipped `corpus-pca` and had no
bilateral mechanism to describe.

**Small corpora are a lottery.** At N = 10 and 100 the 95% range of draws is
0.14-0.16 wide (0.543 to 0.703 at 100): which participants a small corpus
holds matters more than how many it holds, and the mean is not even monotone
between 10 and 100 (0.647, 0.635; the ranges overlap almost entirely). From
400 the range is under 0.04 and from 800 under 0.015. A single nested sweep,
which is all this file had before, can put N = 25 or 100 anywhere in that
range.

**It is not one fold.** All **9 of 9** held-out datasets improve from N = 10
(mean of draws) to the full corpus, by +0.10 to +0.17, and from N = 100 by
+0.08 to +0.19, against a 0.02 noise floor, across fixation, pursuit, free
viewing and movie paradigms alike.

**And the control moves the other way** (2,000 draw). `gev-slow` came out of
the same accumulators at the same rank and *degraded by about 0.31* over
identical data.

Then `lr-cca` **saturates**: 800 to 2,060 is worth +0.011 (2,000 draw, 800 to
2,000: +0.009), inside the noise floor.

One confound to state rather than hide: participants, TRs and OpenNeuro
accessions all grow together (10 accessions at N=10, 652 at N=2,060), so the axis
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
| `fold-pca:64+lags1` (no corpus) | 0.707 | 0.732 | 0.754 | 0.754 | 0.754 | 0.766 |
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
PCA at a low participant budget.** They are within 0.012 at B=1 and the ordering
flips across participant draws. The corpus does not rescue a labeled-data
shortage relative to a basis fitted on the labelled scans -- it removes the need for target-study
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

### Datasets or participants, at a matched total *(2,060)*

`--stages spread` -> `label_spread.json`. The two labelled budgets above move
the total with the axis (B participants from each of 8 datasets; all
participants of m datasets), so neither says which buys more. Here the total
is fixed: the readout trains on T = 8, 16 or 32 labelled participants drawn
from m = 1, 2, 4 or 8 of the other datasets (T / m from each, only datasets
with that many can be drawn), ten draws per cell, every fold scored on its
whole held-out dataset. Mean over datasets, sub-TR, mean of draws [99% CI]:

| T | from 1 dataset | 2 | 4 | 8 |
|---|---|---|---|---|
| ours, 8 | 0.585 [0.517, 0.653] | 0.609 | 0.648 | **0.664** [0.646, 0.682] |
| ours, 16 | 0.648 | 0.663 | 0.665 | 0.684 |
| ours, 32 | 0.633 [0.606, 0.660] | 0.684 | 0.688 | **0.696** [0.684, 0.708] |
| same ridge on voxels (3x3x3), 8 | 0.512 | 0.537 | 0.576 | 0.599 |
| same ridge on voxels, 16 | 0.570 | 0.590 | 0.604 | 0.633 |
| same ridge on voxels, 32 | 0.569 | 0.634 | 0.632 | 0.646 |

- **Spreading the same labels over more datasets helps, for every decoder.**
  1 -> 8 datasets is worth +0.04 to +0.08 at a fixed total; per doubling of
  datasets +0.020 (ours) and +0.025 (supervised). **8 participants from 8
  datasets beat 32 from one** (0.664 against 0.633; supervised 0.599 against
  0.569).
- **But doubling the participants from the same datasets helps about as much
  in this range** (+0.025 and +0.032 per doubling, T = 8 -> 32). Read with the
  budget sweep above, participants per dataset stop paying at about four
  (B = 4: 0.712 on the mean, all: 0.721), datasets have not stopped at eight.
- Ours leads supervision in every cell, by +0.04 to +0.08.
- Caveats: one dataset is a lottery (the 99% CI at T = 8 is 0.14 wide: it
  depends which dataset is drawn), and at T = 32 only `dsL01`, `dsL04` and
  `dsL11` have 32 participants. The training volumes are not matched (runs
  differ in length): T = 32 trains on 43k volumes from one dataset against
  55k from eight, T = 8 on 11k against 14k, T = 16 on 34k against 27k -- so
  at 16 the spread wins on fewer volumes.

The paper's Fig. c.

### Every supervised decoder at every labelled-dataset budget *(2,060)*

`--stages datasets-voxels` now runs every supervised decoder below eight
datasets too (the same ridge on 20 draws, the lag-0 ridges on 4, PLS and the
MLP on 2 -- they cost ~10 min a fold set). The draw of datasets moves the score
far more than the decoder does (the same ridge spans 0.43-0.52 at one dataset
across draws), so a decoder run on 2 draws must be compared **paired**, on the
draws it shares with the ridge, never by its raw mean: PLS's raw mean at one
dataset is 0.536 against the ridge's 0.476, but on the same two draws it is
-0.003. Mean over datasets, paired difference from the same ridge (3x3x3,
+-1 TR):

| decoder | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| PLS-64 then ridge | -0.003 | +0.008 | +0.009 | +0.007 | +0.005 |
| ridge, no context (3x3x3 / 2x2x2) | -0.001 / -0.006 | -0.009 / -0.007 | -0.010 / -0.019 | -0.009 / -0.013 | -0.015 / -0.014 |
| MLP | -0.066 | -0.025 | -0.009 | -0.015 | -0.008 |
| ridge on 2x2x2, +-1 TR | | | | | -0.001 |

PLS is the best supervised decoder from two datasets on, by under 0.01; the
lead of the frozen basis over the *best* supervised decoder is therefore
+0.10 at one dataset and +0.03 at eight, as against the same ridge. Fig. b
draws the others in grey at the ridge's curve plus their paired difference.

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
| `fold-pca:64+lags1` (no corpus) | 0.766 | 0.195 | 0.666 | 0.737 | 0.666 |
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

**Every metric, against selected supervised decoders** *(2,060)* (`--stages
against-dme1` -> `dme1_table.json`; the paper's Table 1). Mean over the three
datasets the shipped CNN never saw; sub-TR unless marked; R2 and median
Euclidean error (degrees) calibrated on each dataset's other participants:

| | r | r_x | r_y | R2 | error | r, 1-TR | r, all nine |
|---|---|---|---|---|---|---|---|
| DeepMReye 1.0 | 0.483 | 0.502 | 0.450 | 0.27 | 2.96 | 0.581 | -- |
| ridge on voxels (3x3x3, +-1 TR) | 0.517 | 0.568 | 0.445 | 0.27 | 3.00 | 0.615 | 0.687 |
| ridge on voxels, no context | 0.506 | 0.548 | 0.431 | 0.26 | 3.05 | 0.624 | 0.672 |
| PLS-64 on voxels | 0.519 | 0.561 | 0.444 | 0.28 | 2.98 | 0.618 | 0.692 |
| MLP on voxels | 0.517 | 0.551 | 0.439 | 0.25 | 3.01 | 0.617 | 0.680 |
| **ours** | **0.567** | **0.651** | **0.499** | **0.30** | **2.88** | **0.677** | **0.721** |

The supervised decoders sit between the CNN and ours on r; the calibrated
error barely separates anyone (2.9-3.0 degrees), because the per-dataset
affine absorbs most of what the CNN gets wrong (its gain). Ours leads on each
of the three datasets on r and on error except `dsL07`, where its error is
2.92 against the CNN's 2.86.

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

*Revised by "What the corpus buys, and where the rest is lost": these are LODO
scores, and a readout fitted inside the dataset exceeds the envelope on several
cells. Read it as a description of what transfers, not of what is decodable.*

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
number above is regenerable, and the two caches are what make that cheap. The
rows were measured on the 2,000 draw (`--n 2000` of the old shuffle); rerun
now, the same commands use all 2,060 participants.

| step | command | writes |
|---|---|---|
| corpus moments (2.2 min) | `fit_lrcca_variants.py` | second moment + group means |
| lag-1 cross moment (2.5 min) | `fit_lrcca_variants.py --lag1 --out results/lrcca_variants/lag1.npz` | for rows 4 and 15 |
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
| leading variance, 16 per orbit | 0.6984 | 0.7359 |
| **random orthogonal** | **0.3938** | **0.3956** |

*(The variance row is variance per orbit, which spends half its rank on the
second copy of what the first orbit already has. It read 0.6922 when the 32
were the left orbit alone; balanced it is 0.6984. A PCA over both orbits at
rank 32 reads **0.744 / 0.811** -- "Why the two bases tie inside a study" below
-- and that is the number to quote as variance at matched rank.)*

Two things to keep. **Non-Gaussianity is a real criterion** -- ICA beats
per-orbit variance by +0.061 at matched rank, and whole-mask variance by 0.016
-- **and it still loses
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

## What the corpus buys, and where the rest is lost

`scripts/analysis_unlabeled.py --stages transfer,gap,datasets`, all through
`probe.lodo`: `split=` trains the readout on the other eight datasets (`lodo`,
the protocol), on all nine (`pooled`) or inside the held-out dataset
(`within`); `readout=` swaps the ridge; `train_filter=` sets the labelled
budget. Same features, same scoring, so a difference between columns is the
cost of transfer and nothing else. `calibrate` still reads 0.7408 / 0.7703.

### The corpus-size curve is a transferability curve

Both arms with +-1 TR of context, sub-TR:

| N unlabelled | 25 | 100 | 400 | 800 | 2000 |
|---|---|---|---|---|---|
| `lr-cca:32+lags1`, within | 0.707 | 0.736 | 0.806 | 0.825 | 0.830 |
| `corpus-pca:64+lags1`, within | **0.804** | 0.814 | 0.824 | 0.826 | 0.829 |
| `lr-cca:32+lags1`, LODO | 0.616 | 0.672 | 0.739 | 0.761 | **0.770** |
| `corpus-pca:64+lags1`, LODO | 0.724 | 0.745 | 0.754 | 0.756 | 0.754 |

**Every linear basis converges on the same ~0.83 of decodable gaze**, and
variance PCA holds 0.80 of it at 25 participants. The corpus does not add
content; it lets the bilateral criterion be estimated, and what that buys is a
basis half the size that transfers slightly better.

**The margin over variance PCA is smaller than the scaling table above
states, and this corrects it.** That table sets `lr-cca:32+lags1` against
`corpus-pca:64` *without* lags. With the same context on both, N = 2000 reads
0.770 against 0.754 sub-TR, 0.837 against 0.823 at 1-TR and 0.721 against 0.706
on the fold mean: **+0.016, inside the noise floor**, carried by `dsL08`
(+0.10) and `dsL02` (+0.04), and the crossover is between N = 400 and 800
rather than at 400. What survives undiminished: the gain with corpus size
(+0.154 against +0.030), and the matched-*rank* comparison, which is
0.770 against 0.744 for a whole-mask PCA at k = 32 (fold mean 0.721 against
0.672) -- not the 0.692 quoted until 30 September, which was variance inside
one orbit. State the advantage as compactness and scaling, not as a win over
variance at its own best rank.

### The headroom is in the readout map, not the representation

The shipped arm, sub-TR / 1-TR:

| readout | within dataset | pooled, all nine | LODO |
|---|---|---|---|
| ridge | **0.830** / 0.866 | 0.778 / 0.839 | 0.770 / 0.838 |
| MLP (256, 256), early-stopped on held-out participants | 0.815 / 0.860 | **0.791** / 0.835 | 0.761 / 0.802 |

- **The features already carry 0.830.** `dsL06` reads 0.853 within against
  0.673 across, `dsL01` 0.845 against 0.770. This revises the temporal-envelope
  reading above, which was drawn from LODO scores only: several cells sit well
  above it once the readout is fitted inside the dataset.
- **No non-linear gaze information.** Where transfer cannot confound it, the
  MLP does not beat ridge. This is the supervised non-linear ceiling measured
  cleanly, and it is what closes the encoder programme at the root.
- **The feature->gaze map differs between acquisitions.** One ridge on all nine
  reaches 0.778 although 40% of its rows are `dsL01`, which it scores at 0.778
  against 0.845 within. The MLP given the same nine reaches 0.791 (`dsL06`
  0.791 against ridge's 0.681) by inferring the acquisition from the features;
  under LODO the acquisition is unseen and it falls below ridge.

### Labelled datasets are the axis still rising

Random subsets of the eight training datasets per fold, four draws, every fold
scored whole (`lr-cca:32+lags1`):

| labelled datasets | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| sub-TR median | 0.568 | 0.729 | 0.751 | 0.751 | **0.770** |
| 1-TR median | 0.654 | 0.773 | 0.808 | 0.827 | **0.837** |
| fold mean | 0.584 | 0.661 | 0.707 | 0.711 | **0.721** |

Decelerating but not flat: +0.014 to +0.029 from four datasets to eight,
against a corpus axis that stopped moving at N = 800. `corpus-pca:64+lags1`
matches `lr-cca` on the median from four datasets up and trails it on the fold
mean throughout (0.549 / 0.632 / 0.693 / 0.697 / 0.706), by 0.03 at one or two
datasets -- which is where a compact basis earns its keep.

**Against decoders that see only labelled participants, this is where the
corpus wins.** Two kinds of control, same draws, no unlabelled scan anywhere:

- **Supervised** (`--stages datasets-voxels`): voxels in, gaze out, no basis
  step. A ridge on every eye voxel (3x3x3 block means, 709 columns, +-1 TR) at
  every budget, solved by `ridge_by_covariance` -- the protocol's `RidgeCV`
  from `X'X`, identical to 1e-13. With all eight datasets also a PLS basis
  chosen *with* the gaze (64 components, then the ridge) and an MLP (256, 256)
  early-stopped on held-out participants.
- **Unsupervised step on labelled scans only** (`--stages
  datasets-supervised`): a PCA of the scans of exactly the datasets the readout
  may train on, then the ridge (`fold-pca:64+lags1`; at eight it reproduces the
  0.7658 / 0.8308 / 0.7194 of `foldpca_full.json`). Its basis uses no labels,
  so it is a no-corpus reference, not a supervised method.

Fold mean, sub-TR, mean over draws:

| labelled datasets | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| `lr-cca:32+lags1` @ N=2000 | **0.584** | **0.661** | **0.707** | **0.711** | **0.721** |
| PCA of the labelled scans (`fold-pca`) | 0.532 | 0.616 | 0.689 | 0.705 | 0.719 |
| supervised: ridge on voxels | 0.477 | 0.598 | 0.656 | 0.672 | 0.687 |
| supervised: PLS-64 on voxels | | | | | 0.692 |
| supervised: MLP on voxels | | | | | 0.680 |
| `lr-cca` minus supervised ridge, every draw | +0.08 to +0.17 | +0.04 to +0.09 | +0.04 to +0.06 | +0.039 | +0.034 |

With all eight datasets the medians are 0.770 / 0.837 against 0.741 / 0.788
(ridge), 0.752 / 0.792 (PLS), 0.739 / 0.771 (MLP) and 0.766 / 0.831 (PCA of
the labelled scans). `lr-cca` beats the supervised ridge in every draw at every
budget, on the fold mean, the median and 1-TR alike, and in 8 or 9 of 9 folds
in 16 of 17 draws (7 of 9 in the other). The resolution is not what holds the voxel arms back: at
lag 0, 2x2x2 and 3x3x3 block means read 0.673 and 0.672 on the fold mean.

**At 1-TR the best supervised decoder is the lag-0 ridge, and the margin is
small** (noticed 5 October, from the same `label_datasets_voxels.json`). The
supervised medians at all eight datasets, sub-TR / 1-TR: ridge 3x3x3 lag 0
0.722 / **0.823**, ridge 2x2x2 lag 0 0.740 / 0.806, ridge 3x3x3 +-1 TR
0.741 / 0.788, ridge 2x2x2 +-1 TR 0.747 / 0.784, PLS 0.752 / 0.792, MLP
0.739 / 0.771. Best of the supervised family at each resolution: **0.752
sub-TR (PLS), 0.823 at 1-TR (lag-0 ridge)**, against `lr-cca`'s 0.771 / 0.835
on the 2,060 corpus: +0.019 and +0.012, inside the noise floor at 1-TR, and the
corpus curve (mean of 20 draws) only crosses the 1-TR line at N ~ 1,350
(sub-TR, against the same ridge: ~450). The PCA of the same corpus reads
0.754 / 0.822 -- level with the best supervised decoder at both. Every
"0.838 against 0.784" quoted before this date compares against the +-1 TR
ridge, which is not supervision's best at 1-TR.

**Every base with every readout** *(2,060)* (`--stages readout-grid` ->
`readout_grid.json`). One protocol, all eight other datasets,
leave-one-dataset-out; sub-TR median (mean) / 1-TR median:

| readout | ours, `lr-cca:32` @ 2060 | corpus PCA:64 @ 2060 | voxels 3x3x3 | voxels 2x2x2 |
|---|---|---|---|---|
| ridge, +-1 TR (the protocol) | **0.771** (0.721) / 0.835 | 0.754 (0.705) / 0.822 | 0.741 (0.687) / 0.788 | 0.747 (0.686) / 0.784 |
| ridge, no context | 0.741 (0.700) / **0.838** | 0.746 (0.683) / 0.820 | 0.722 (0.672) / 0.823 | 0.740 (0.673) / 0.806 |
| ridge, +-2 TR | 0.761 (0.716) / 0.827 | 0.760 (0.703) / 0.823 | 0.737 (0.687) / 0.779 | not run (memory) |
| least squares, +-1 TR | 0.771 (0.721) / 0.835 | 0.754 (0.705) / 0.822 | 0.739 (0.686) / 0.785 | 0.744 (0.681) / 0.777 |
| PLS-64 then ridge, +-1 TR | 0.771 (0.721) / 0.835 | 0.754 (0.705) / 0.822 | 0.752 (0.692) / 0.792 | not run (memory) |
| MLP, +-1 TR | 0.765 (0.694) / 0.790 | 0.752 (0.675) / 0.781 | 0.739 (0.680) / 0.771 | not run (memory) |
| **best readout** | **0.771 (0.721) / 0.838** | 0.760 (0.705) / 0.823 | 0.752 (0.692) / 0.823 | 0.747 (0.686) / 0.806 |

The frozen basis is the best base under its best readout at every statistic.
Under the same readout it leads every other base in 6 of 6 rows on the mean and
at 1-TR, and in 5 of 6 on the sub-TR median, where it trails corpus PCA by
0.005 with no context (and ties it, +0.001, at +-2 TR). PLS is the ridge again
on the 32- and 64-dimensional bases (it keeps every direction); the MLP is
below the ridge on every base. At 1-TR the best readout for the frozen basis
and for voxels is the one without context. (The 2,000 draw: 0.770 / 0.721 /
0.838 for ours; the voxel columns do not depend on the corpus.)

**The penalty does nothing for anyone** (v1; the `readouts` stage it came from
is now the least-squares row of the grid above). Every arm with the protocol's
ridge and with plain least squares (no penalty), all eight other datasets,
sub-TR / 1-TR:

| arm | ridge | least squares |
|---|---|---|
| `lr-cca:32+lags1` @ N=2000 | 0.770 / 0.838 | 0.770 / 0.838 |
| `corpus-pca:64+lags1` @ N=2000 | 0.754 / 0.823 | 0.754 / 0.823 |
| voxels 3x3x3 +-1 TR | 0.741 / 0.788 | 0.739 / 0.785 |
| voxels 3x3x3 lag 0 | 0.722 / 0.823 | 0.722 / 0.822 |
| voxels 2x2x2 lag 0 | 0.740 / 0.806 | 0.735 / 0.799 |

With ~400,000 training rows and at most ~1,900 voxel columns, least squares is
well determined and the leave-one-row-out penalty is near zero anyway. So an
unpenalised supervised decoder is not a weaker baseline here (-0.005 at most),
and what holds supervision back across datasets is the map's transfer, not its
estimation.

Read in three steps:

1. **A low-rank basis beats decoding the voxels directly, and by more the
   fewer labelled datasets there are** (+0.03 at eight, +0.11 at one, against
   the supervised ridge). Choosing that basis *with* the labels (PLS) is worth
   only +0.005 over the plain ridge; choosing it by variance on the same
   labelled scans is worth +0.03. The bottleneck regularises the transfer, and
   the gaze is the wrong thing to select it by when it has to transfer.
2. **The corpus is worth +0.05 over the labelled scans' own PCA at one or two
   datasets**, in all eight draws (+0.015 to +0.108), +0.018 at four, level by
   six. `corpus-pca` sits between them (0.549 / 0.632), so a third of that is
   having a corpus at all and two thirds is the bilateral criterion.
3. **So the accuracy claim the unlabelled corpus supports against supervision
   is: it beats a supervised decoder at every labelled budget, and it stands in
   for labelled *datasets* when there are few** -- roughly one extra dataset's
   worth at two, against a PCA of the labelled scans. Against that PCA it is
   worth nothing once there are six.

**How sure is the lead over the supervised decoder?** `--stages participants`
keeps every participant's LODO r, and the figure's `_paired_boot` resamples
participants inside each dataset, the same ones for both arms. With all eight
labelled datasets, against the supervised voxel ridge at 3x3x3 (against the
finer 2x2x2 one, which is the fairer comparison, see the next section: +0.024
median, +0.035 mean, 8/9):

| arm | median over datasets, paired 95% | mean over datasets, paired 95% | datasets better |
|---|---|---|---|
| `lr-cca:32+lags1` @ N=2000 | +0.030 [-0.006, +0.059] | **+0.034 [+0.018, +0.058]** | 8/9 (sign test p = 0.039) |
| `lr-cca:32+lags1` @ N=800 | +0.021 [-0.020, +0.049] | +0.019 [+0.003, +0.044] | 7/9 |
| `corpus-pca:64+lags1` @ N=2000 | +0.014 [-0.004, +0.064] | +0.018 [+0.006, +0.041] | 8/9 |
| PCA of the labelled scans | +0.025 [+0.001, +0.060] | +0.032 [+0.015, +0.048] | 9/9 (p = 0.004) |

The median over nine datasets is too jumpy to separate 0.03 from zero; the mean
does, and so does the count of datasets. Quote the lead as the mean
(+0.034, interval +0.018 to +0.058) or as 8 of 9 datasets, not as the median.
And note what the last row says: a PCA of the labelled scans beats the
supervised decoder at least as surely. The lead over supervision belongs to
having a low-rank basis; what the corpus adds on top is the scaling and the
few-labelled-datasets lead above.

### Every decoding scheme against the supervised decoder

*(2,060)* `--stages schemes` -> `scheme_participants.json`. Six schemes -- what the readout is trained on
before it is tested -- and in each one every arm's readout sees the same
labelled rows. The supervised decoder is a ridge on every eye voxel (block
means, +-1 TR) at 2x2x2 and 3x3x3; it is compared at whichever scores higher in
that scheme. `labeled-pca` is a PCA of the training participants' own scans,
refitted inside each readout. `probe.lodo(split="halves")` trains on the first
quarter or half of a participant's run, in time order, and always tests on its
second half. Draws (one participant: six; one dataset: four) are averaged per
participant.

Sub-TR median / mean over datasets:

| trained on | `lr-cca` | corpus PCA | PCA of labelled scans | supervised 2x2x2 | supervised 3x3x3 |
|---|---|---|---|---|---|
| own run, first quarter | 0.799 / 0.707 | 0.789 / 0.710 | 0.814 / 0.751 | 0.812 / 0.751 | 0.802 / 0.736 |
| own run, first half | 0.824 / 0.741 | 0.816 / 0.754 | 0.835 / 0.791 | 0.833 / 0.787 | 0.819 / 0.774 |
| one other participant | 0.627 / 0.622 | 0.644 / 0.626 | 0.649 / 0.625 | 0.625 / 0.613 | 0.606 / 0.590 |
| all other participants | 0.831 / 0.776 | 0.829 / 0.776 | 0.843 / 0.795 | 0.843 / 0.791 | 0.834 / 0.785 |
| one other dataset | **0.595 / 0.582** | 0.564 / 0.544 | 0.554 / 0.525 | 0.477 / 0.485 | 0.503 / 0.483 |
| all eight other datasets | **0.771 / 0.721** | 0.754 / 0.705 | 0.753 / 0.700 | 0.747 / 0.686 | 0.741 / 0.687 |

`lr-cca` minus the best supervised decoder, paired 95% bootstrap over
participants (median over datasets | mean over datasets | datasets better):

| trained on | median | mean | datasets |
|---|---|---|---|
| own run, first quarter | -0.013 [-0.039, +0.007] | -0.043 [-0.059, -0.034] | 0/9 |
| own run, first half | -0.009 [-0.036, +0.008] | -0.046 [-0.058, -0.037] | 0/9 |
| one other participant | +0.002 [-0.032, +0.051] | +0.009 [-0.006, +0.022] | 5/9 |
| all other participants | -0.012 [-0.031, +0.011] | -0.014 [-0.034, -0.004] | 3/9 |
| one other dataset | **+0.118 [+0.070, +0.158]** | **+0.097 [+0.081, +0.118]** | **9/9** |
| all eight other datasets (vs 2x2x2) | +0.025 [-0.005, +0.051] | **+0.035 [+0.020, +0.064]** | 8/9 |

Three readings, and they are the whole story of this file in one table:

- **Inside an acquisition -- a participant's own run, or a dataset's other
  participants -- supervision is level or ahead.** Every participant scores
  better with the supervised decoder on its own run (0 of 9 datasets favour
  `lr-cca`, -0.04 on the mean). A frozen basis cannot learn what is specific to
  this head in this scanner; the voxels can.
- **Across acquisitions the frozen basis is ahead, and most where labels are
  scarcest.** Trained on one other dataset it leads by +0.10 to +0.12 in all
  nine folds, and leads both PCAs there (+0.031 over corpus PCA, +0.041 over
  the PCA of the labelled scans): this is the one cell where the bilateral
  criterion, the corpus and the low-rank basis each add something. With all
  eight datasets the lead is +0.035 on the mean (8/9) and +0.025 on the median,
  whose interval touches zero.
- **The finer supervised decoder** (2x2x2) is the one to compare against: it is
  better in five schemes of six. With it the LODO number is 0.747 / 0.784 (not
  0.741 / 0.788), and lr-cca overtakes it at N ~ 450 on the corpus-size curve
  (mean of 20 draws). The paper uses these.

### A new study with a few labelled participants of its own

`--stages fewshot`, `probe.lodo(split="within", shots=K)`: the readout is fitted
on at most K participants of the test participant's own dataset (`dsL06` has
six) and nothing else; the test side is the `within` one, so the denominator
does not move with K. Median over six draws of which participants; the draws
spread by +-0.04 at K = 1. K = 0 is the frozen decoder, leave-one-dataset-out.
Sub-TR:

| labelled participants in the new study | 0 | 1 | 2 | 4 | 8 | 16 | all |
|---|---|---|---|---|---|---|---|
| `lr-cca:32+lags1` @ N=2000 | **0.770** | 0.682 | 0.728 | 0.789 | 0.812 | 0.826 | 0.830 |
| `corpus-pca:64+lags1` @ N=2000 | 0.754 | 0.696 | 0.739 | 0.780 | 0.806 | 0.823 | 0.829 |
| `own-pca:32+lags1` (PCA of the study's own scans, no corpus) | -- | 0.707 | 0.727 | 0.780 | 0.806 | 0.822 | 0.833 |
| `own-pca:64+lags1` | -- | 0.669 | 0.739 | 0.757 | 0.795 | 0.825 | 0.836 |
| `raw+lags1` (stride-4 voxels, no basis) | 0.729 | 0.576 | 0.663 | 0.724 | 0.770 | 0.793 | 0.810 |
| frozen decoder, its output refitted on the K | 0.770 | 0.703 | 0.742 | 0.800 | 0.811 | 0.815 | 0.816 |

**Against decoders that see no unlabelled scan** (`--stages
fewshot-supervised`, same draws). `voxels` is the supervised one: a ridge
straight on 2x2x2 block means of all 14,236 voxels (an SVD of the whole
design, so one and two shots only). `labeled-pca` is a PCA of the K labelled
participants' own scans, refitted inside every readout from the same block
means, then the ridge; `fold-pca` is the PCA of the
other eight labelled datasets' scans, which holds 0.766 at K = 0.

| labelled participants in the new study | 0 | 1 | 2 | 4 | 8 | 16 | all |
|---|---|---|---|---|---|---|---|
| `labeled-pca:64+lags1` | -- | 0.673 | 0.760 | 0.791 | 0.821 | 0.838 | **0.843** |
| `labeled-pca:256+lags1` | -- | 0.664 | 0.741 | 0.791 | 0.821 | 0.837 | 0.844 |
| `voxels+lags1`, every voxel | -- | 0.668 | 0.738 | | | | |
| `fold-pca:64+lags1` | 0.766 | 0.689 | 0.753 | 0.793 | 0.815 | 0.831 | 0.837 |
| `lr-cca:32+lags1` @ N=2000, fold mean | 0.721 | 0.643 | 0.687 | 0.731 | 0.756 | 0.770 | 0.776 |
| `labeled-pca:64+lags1`, fold mean | -- | 0.636 | 0.707 | 0.746 | 0.779 | 0.791 | 0.795 |

- **The corpus basis has no few-label advantage over a supervised pipeline,
  and from two participants it is slightly behind one.** At K = 1 the PCA of
  that one participant's scans reads 0.673 against 0.682, a tie inside the
  +-0.04 of the draws. From K = 2 it leads on the fold mean by 0.019-0.022 in
  every draw (paired, same participants: +0.018 to +0.026 at K = 8), and with
  all participants it reads 0.843 / 0.889 against 0.830 / 0.866, ahead in eight
  folds of nine (`dsL08` +0.07, `dsL07` +0.04). A basis fitted on the study
  itself is adapted to its acquisition; a frozen one is not.
- **The `raw+lags1` row understates supervision.** Stride-4 keeps 233 voxels of
  14,236. A ridge on all of them reads 0.668 at K = 1, not 0.576, so the
  "+0.11 for any basis over raw voxels" below is what subsampling costs, not
  what a basis buys.
- **So the 0.83 within-dataset ceiling is the ceiling of a frozen basis.** A
  study-specific one reaches 0.843 / 0.889.

- **`lr-cca` is not better than variance PCA here, at any K.** The two corpus
  bases are within 0.014 of each other in every column (1-TR: 0.768 against
  0.754 at K = 1, 0.828 against 0.824 at K = 4; fold mean 0.643 against 0.647).
  This is the few-label form of "every linear basis holds the same gaze": no
  transfer is involved when the readout is fitted inside the study, and transfer
  is the only place the bilateral criterion has an edge.
- **The corpus is not needed for it either.** A PCA of the new study's own
  scans -- unlabelled data every study has -- matches both corpus bases from
  K = 1 up. Corpus *size* does matter when labels are this scarce (at K = 1,
  `corpus-pca` 0.625 at N = 25 to 0.696 at N = 2000, `lr-cca` 0.495 to 0.682),
  but the study's own scans buy the same.
- **Against stride-4 voxels any basis is worth +0.11 to +0.13 at K = 1**, +0.06
  at K = 4 and +0.02 with all of them -- but see above: that is the cost of
  keeping 233 voxels, and it is gone against a ridge on all of them.
- **The frozen decoder is worth two to four of the new study's own labelled
  participants.** With none it reads 0.770 / 0.838. A supervised decoder trained
  on one reads 0.67-0.69 / 0.75-0.76, on two 0.74-0.76 / 0.80-0.81, and at four
  it passes (0.791 / 0.842; fold mean 0.746 against 0.721). That is the
  deployment claim in labelled-participant units, and it belongs to the eight
  labelled datasets, not to the corpus: the no-corpus frozen decoder reads
  0.766 (`fold-pca`).
- **One or two labelled participants do not improve the frozen decoder.**
  Refitting its output (+-1 TR, 60 columns) on them reads 0.703 and 0.742, below
  leaving it alone; shrinking toward the frozen output instead of toward zero,
  and a per-sample gain and offset only, were tried in a scratch run and do not
  change that (0.70-0.74 at K = 1). From four it is level with a fresh readout,
  and with all of them plus the features it reads 0.842 / 0.875, +0.012 on the
  within-dataset readout and inside the noise floor.

### Both bases tuned on the readout side

*(2,060)* `--stages tune` -> `tuning.json`, leave-one-dataset-out, every
combination of rank, lag context (0, +-1, +-2 TR), whitening of the PCA scores
by the corpus eigenvalues, and a penalty grid to 1e7 (the protocol's stops at
1e4). Sub-TR median, protocol grid, no whitening:

| rank | 16 | 24 | 32 | 48 | 64 | 96 | 128 | 192 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| `corpus-pca`, +-1 TR | 0.574 | | 0.743 | 0.754 | 0.754 | 0.759 | 0.751 | 0.750 | 0.751 |
| `corpus-pca`, +-2 TR | 0.576 | | 0.734 | 0.753 | **0.760** | 0.751 | 0.744 | 0.751 | 0.749 |
| `lr-cca`, +-1 TR | 0.500 | 0.766 | **0.771** | 0.765 | 0.764 | 0.764 | 0.760 | | |
| `lr-cca`, +-2 TR | 0.500 | 0.754 | 0.761 | 0.752 | 0.752 | 0.756 | 0.752 | | |

- **Whitening and the penalty grid do nothing**: at most 0.0008 and 0.0006
  anywhere. With 270,000 training rows the leave-one-out choice of penalty
  barely matters, so the PCA scores' larger scale was never under-regularised.
- **Per-participant standardisation of the features hurts both** (scratch run):
  `corpus-pca` 0.754 -> 0.754 / 0.761 -> 0.755, `lr-cca` 0.770 -> 0.767.
- **Best of grid against best of grid**, equally optimistic for both:
  `corpus-pca` k = 64, +-2 TR, wide grid, **0.761 / 0.824 / fold mean 0.704**
  (its best fold mean is 0.712, at k = 128, +-1 TR; its best 1-TR is 0.838, at
  k = 128 with no context); `lr-cca` is its shipped configuration, **0.771 /
  0.835 / 0.721** (its best 1-TR is 0.842, at k = 24 with no context). Tuned,
  the margin is +0.010 on the median, +0.004 at 1-TR best against best and
  +0.009 on the fold mean -- all inside the noise floor, as the
  matched-context number already said. Nothing on the readout side was left
  on the table for either basis.
- The two bases want different context: `corpus-pca` gains from +-2 TR where
  `lr-cca` loses (0.761 against 0.771). Its 64 dimensions are dominated by slow
  variance, so a wider window helps it separate the faster gaze.

### Why the two bases tie inside a study, and where they do not

`--stages overlap` -> `basis_overlap.json`. Four measurements, N = 2000, both
arms with +-1 TR of context.

**They are mostly the same subspace.** Canonical correlations between the
`lr-cca:32` and `corpus-pca:64` features on the labelled scans: 16 of 32 at
0.95 or above, 21 at 0.90 or above, 25 at 0.79 or above, the last seven from
0.72 down to 0.37. What varies most in an orbit -- head motion, global signal,
the eye itself -- is also what the two orbits share, so the variance ordering
and the agreement ordering pick largely the same directions.

**They read the same signal and fail at the same moments.** Predictions of the
two readouts correlate at 0.96 inside a study and 0.97 across studies; their
*errors* correlate at 0.92 and 0.95. Concatenating the two bases reads 0.836
within (against 0.830 and 0.829), 0.696 at one shot, 0.759 LODO. There is no
gaze in one that is missing from the other; the 0.83 is a limit outside the
basis.

**Neither criterion ranks gaze first.** Sub-TR, by rank:

| k | 8 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|
| `lr-cca`, within | 0.544 | 0.580 | **0.830** | 0.830 | 0.830 | 0.825 |
| `corpus-pca`, within | 0.564 | 0.723 | 0.815 | 0.825 | 0.826 | 0.829 |
| `lr-cca`, one shot | 0.437 | 0.438 | 0.699 | 0.701 | 0.696 | 0.682 |
| `corpus-pca`, one shot | 0.401 | 0.503 | 0.697 | 0.701 | 0.703 | 0.700 |
| `lr-cca`, LODO | 0.469 | 0.505 | **0.766** | **0.770** | 0.764 | 0.762 |
| `corpus-pca`, LODO | 0.456 | 0.617 | 0.727 | 0.744 | 0.755 | 0.754 |
| `lr-cca`, LODO fold mean | 0.434 | 0.467 | 0.720 | 0.721 | 0.719 | 0.713 |
| `corpus-pca`, LODO fold mean | 0.415 | 0.556 | 0.675 | 0.672 | 0.704 | 0.706 |

The sixteen directions the two orbits agree on most hold 0.58 of a possible
0.83; the gaze arrives between the 17th and the 24th. Bilateral agreement is
not a gaze filter -- it keeps what is shared, and most of what is shared is
nuisance. This is the linear form of what the deep arms did when asked for
more agreement.

**One shot is short of people, not of rows or of columns.** One participant
with all of their rows reads 0.70 under either basis; *the same number of rows*
taken evenly from every training participant reads 0.809 under either. (Fifty
rows of one participant already read 0.61-0.63, scratch run.) The feature->gaze
map differs between people, and a smaller basis cannot supply the people. The
ridge is not short of data either: at one shot it spends about 65 of 96
effective degrees of freedom on `lr-cca` and 109 of 192 on `corpus-pca`, for
the same score.

**Where they differ is transfer at small rank.** Under LODO `lr-cca` leads by
0.039 at k = 24 and 0.026 at k = 32 on the median, and by 0.045-0.049 on the
fold mean, both outside the noise floor; variance needs 48 to 64 components to
close it to 0.01. `lr-cca:24` is already the shipped score (0.766 / 0.833 /
0.720). That is the whole of the bilateral criterion's advantage, stated in
the units it is real in: the same transfer from half to a third of the
dimensions -- not more gaze, and not fewer labels.

### Tried on top of it, measured, and deleted

September 2026. Eighteen arms, none a clean win; the code is removed and lives
only in the author's local `results/archive/`. Scores are sub-TR / 1-TR LODO
against the incumbent's 0.770 / 0.838 unless marked.

| arm | score | what it showed |
|---|---|---|
| deep CCA on all 2,064 participants (DCCA loss, separate orbit encoders, started at the incumbent), unregularised | 0.451 | memorises volumes: train sum-corr 17.4 -> 31.6, held-out 17.9 -> 11.7 |
| same, regularised (input noise, small nets, low lr, grouped batches); InfoNCE | 0.772-0.775 / 0.829-0.838 | the one run that raises held-out agreement (17.9 -> 20.3) lowers gaze to 0.753; a gradient-trained *linear* control gets the same +0.002 |
| same, from random weights | 0.772 (untrained 0.511) | rediscovers `lr-cca` in 1000 steps, then stops |
| per-participant cross-orbit CCA shrunk toward the corpus | 0.769 -> 0.636 | a single participant's strongest bilateral signal is their own motion |
| forward-model decoding with the participant's own feature covariance | 0.749 at 5% own | own covariance contains the signal; lifts only `dsL08`, `dsL11` |
| k = 64..256 with alpha up to 1e6 | 0.774, nested 0.769 | larger bases need ~10x the penalty and still lose |
| row weight per dataset / sqrt / participant | 0.765 / 0.771 / 0.740 | trades minority paradigms (+0.03) against pursuit (-0.02) |
| co-regularised L/R readouts on the corpus | nested 0.760 | agreement collapses onto shared nuisance |
| site-invariant CCA over 605 accessions | 0.769 -> 0.713 | uniform agreement across sites removes gaze |
| training sets weighted by decoded-dynamics similarity | 0.771 | tie |
| 2x decimated runs as synthetic acquisitions | 0.764 -> 0.749 | loses |
| TR-conditioned readout | 0.752 | the timing that fails to transfer is not TR |
| cross-orbit Noise2Noise temporal filter | 0.779 / 0.828 | smooths, cannot de-bias |
| lag-0 readout + linear interpolation to sub-TR, no model | 0.784 / 0.831 | no fold down by 0.02; the learned sub-TR kernel is worse than interpolating |
| lag-0 readout + per-study GP on the decoded trace | 0.793 / 0.838 | `dsL01` +0.03, `dsL03` +0.04, `dsL08` +0.05; `dsL06` -0.14 |
| joint space-time generative decoder | 0.787 / 0.826 | reduces to the incumbent with a pooled prior (0.773) only once noise is temporally correlated; dynamics are not identifiable from a new site's features |
| bilateral GP: dynamics from the two eyes' cross-covariance, decode the averaged trace | **0.794 / 0.843**, mean 0.728 | best arm; 3 folds up, 4 tied, `dsL07` -0.04, `dsL06` -0.08; chosen among four variants on all folds |
| whole-run GRU / transformer; transformer prior pretrained on corpus traces | 0.771 / 0.810; 0.700-0.727; 0.744-0.747 | a viewing prior from eight paradigms does not reach a ninth; corpus pretraining is below its untrained control (0.766) and flat in N |

Four things from that table are worth carrying forward.

**The part of the readout that does not transfer is temporal.** Fitting a
correction of the LODO *output* on the held-out dataset's other participants:
gain and offset per sample 0.771, x/y mixing 0.770, remixing the ten sub-TR
samples per axis 0.803, a +-1 TR kernel per axis **0.812 / 0.860**. How volumes
t-1, t, t+1 are weighted into each gaze instant is what differs between
studies; the pooled kernel is shaped by the pursuit datasets, which supply
~60% of training rows. The map also varies per participant -- a ridge on half of
one run beats the within-dataset readout on long runs (`dsL02` 0.954 vs 0.922,
`dsL04` 0.916 vs 0.866) -- and nothing label-free recovered that.

**Every temporal model was compared in one harness**, and the GP is best by
about 0.01 only: likelihood-selected GP 0.795 / 0.843, kernel-free Wiener
0.795 / 0.819, Matern-3/2 0.793, kernel smoother 0.789, OU 0.787 / 0.846,
**linear interpolation 0.784**, squared-exponential 0.781, total variation
0.776 (five folds down), cubic spline 0.772, per-volume value repeated 0.738.
Most of the gain over the shipped kernel is interpolation, not dynamics.

**`dsL06` is three acquisitions under one name.** Its six participants span TR
1.25 / 1.8 / 2.5 s and two resolutions, and its decoded trace carries a
periodic component at 3 TRs that the gaze does not have (lag-1 autocorrelation
0.24 decoded against 0.79 true) and that is present in both eyes. Every
dynamics-adaptive arm fails there and only there.

**Agreement and predictability are nuisance in time as they are in space.**
More cross-orbit agreement (deep CCA, co-regularisation, site-invariance) and
more predictability (masked pretraining of decoded traces) each lowered gaze.
Any further self-supervised arm needs a target only gaze can satisfy.

### Temporal stages on the decoded trace, remeasured *(2,060)*

8 October 2026. `--stages temporal` -> `temporal_stages.json`
(`scripts/unlabeled/temporal.py`). The September GP arms above were deleted
code, scored on the median only; this reimplements them through
`probe.score_predictions`. Each decoder is read out at lag 0 (one value per
TR, the mean of its ten outputs) and a stage produces the ten samples:
repeat, linear or cubic interpolation between TR centres, or the posterior
mean of a GP whose observation is the TR's *average* of the latent gaze plus
white noise. GP hyperparameters are label-free: marginal likelihood on the
held-out dataset's decoded traces (per study / per participant), the two
orbits' cross-covariance (bilateral), the length scale from the *other*
datasets' measured gaze, or -- a ceiling, not a method -- from the dataset's
own measured gaze. Paired difference of the mean over datasets from the
decoder's own +-1 TR readout [95% CI over participants], datasets up/down by
0.005:

| on ours (0.771 / 0.721) | sub-TR median / mean | delta mean | up/down |
|---|---|---|---|
| lag 0, repeated | 0.737 / 0.697 | -0.024 [-0.030, -0.020] | 1/8 |
| lag 0 + linear interpolation | 0.783 / 0.723 | +0.002 [-0.003, +0.007] | 3/2 |
| lag 0 + cubic spline | 0.774 / 0.724 | +0.003 [-0.001, +0.008] | 2/2 |
| lag 0 + GP Matern-3/2, ML per study | **0.794 / 0.668** | -0.053 [-0.068, -0.038] | 5/4 |
| ... Matern-1/2, -5/2, SE | 0.771-0.797 / 0.658-0.674 | -0.047 to -0.063 | |
| ... per participant; bilateral | 0.794 / 0.665; 0.790 / 0.689 | -0.055; -0.032 | |
| ... length scale <= 2 s | 0.793 / 0.718 | -0.002 [-0.008, +0.003] | 4/4 |
| ... length scale from training gaze | 0.765 / 0.720 | -0.001 | 2/4 |
| ... length scale from own gaze (oracle) | 0.790 / 0.725 | +0.004 [-0.001, +0.008] | 4/3 |
| shipped + GP on its TR means; average of shipped and GP | 0.779 / 0.683; 0.784 / 0.719 | -0.038; -0.002 | |
| stage chosen on the other eight datasets (cubic spline) | 0.774 / 0.722 | +0.002 [-0.002, +0.006] | 2/2 |

- **The GP's median gain is a median artefact.** Marginal likelihood picks
  length scales of 20-28 TRs on `dsL06` and `dsL08`, whose decoded traces
  carry slow non-gaze structure, and over-smooths them (-0.26, -0.22); five
  other datasets gain 0.005-0.04. Median up 0.023, mean down 0.053. Bounded,
  or with the oracle length scale, it is within +-0.004. Closed for our
  decoder: the residual error is map-and-timing error, correlated in time and
  shared by both orbits, not white noise a smoother removes.
- **The supervised voxel ridge gains, ours does not.** The 2x2x2 ridge at
  lag 0 + linear interpolation reads 0.778 / 0.711 against its +-1 TR 0.747 /
  0.686 (+0.025 [+0.016, +0.032], 8 up, 0 down; 3x3x3 +0.014). Its +-1 TR
  readout fits a 2,116-column spatial map per lag, and that kernel transfers
  worse than interpolating. **With each decoder's stage chosen on the other
  datasets, ours leads the 2x2x2 ridge by +0.015 [+0.002, +0.041] on the mean
  (5 up, 1 down) and +0.017 [+0.001, +0.049] at 1-TR**, against +0.035 /
  +0.038 with the protocol readout for both (3x3x3: +0.024, 8/9). On the
  median the interpolated ridge (0.778) is above our protocol 0.771 and below
  our interpolated 0.783. "No supervised decoder or readout exceeds 0.692 on
  the mean" holds for readouts; with a temporal stage the supervised mean is
  0.711. The labelled-dataset and corpus-size comparisons have not been
  remeasured with a stage on both.

---

## Viewing priors

October 2026. The question: can what people tend to look at -- other viewers,
the stimulus, the task -- add to what the eyeballs say? Literature and the
reasoning behind each item are in the author's `docs/LITERATURE_2026-10-02.md`.

### Only three of nine datasets share a stimulus clock, and one of them is its own labels

Leave-one-out correlation of each participant's *measured* gaze with the mean
of the others', at matched TR index (1-TR):

| `dsL01` guided fixations | `dsL08` studyforrest | `dsL11` Back to the Future | the six others |
|---|---|---|---|
| **1.000** | 0.703 | 0.739 | -0.01 to 0.11 |

**`dsL01`'s labels are its fixation-target sequence, not eye tracking**: all 170
participants carry the identical 270-TR sequence over 25 positions, with zero
variation inside a TR. So `dsL01` -- 40% of the training rows -- scores
following-the-target, its sub-TR score is the score of a step function, and any
prior that knows the task is trivially perfect there. Pursuit, free viewing,
sequences and calibration randomise order per participant; a prior across
viewers cannot help them.

### Other viewers' decoded gaze: +0.02 on the mean, decoder-agnostic

`--stages priors` -> `viewing_priors.json`. Label-free: each participant's
decoded trace mixed with the mean of the other participants' decoded traces
(z-scored per column), on the three shared-stimulus datasets only, scored by
`probe.score_predictions`:

| | median / 1-TR / mean | `dsL01` | `dsL08` | `dsL11` |
|---|---|---|---|---|
| `lr-cca`, no prior | 0.770 / 0.837 / 0.721 | 0.770 | 0.283 | 0.671 |
| `lr-cca`, own + consensus | 0.776 / 0.839 / **0.742** | 0.863 | **0.360** | 0.691 |
| `lr-cca`, consensus only | 0.776 / 0.837 / 0.740 | 0.909 | 0.393 | 0.595 |
| supervised voxels, no prior | 0.747 / 0.784 / 0.686 | 0.747 | 0.213 | 0.615 |
| supervised voxels, own + consensus | 0.747 / 0.810 / 0.707 | 0.851 | 0.260 | 0.655 |

The prior lifts both decoders by about 0.02 on the mean and leaves the lead
between them where it was (+0.035). Of the gain, `dsL01` is the trivial part;
the real one is the studyforrest film, `dsL08`, our worst fold: +0.08 to +0.11.

### A stimulus-derived prior is bounded by other viewers' *measured* gaze

The best any stimulus-driven prior (a saliency model, normative eye tracking of
the same film) could do is the mean measured gaze of the other viewers. Used as
the prior, sub-TR / 1-TR (scratch run):

| | decoded only | prior only | decoded + prior | decoded + 2 x prior |
|---|---|---|---|---|
| `dsL08` studyforrest (3T) | 0.281 / 0.348 | 0.614 / 0.730 | 0.484 / 0.588 | 0.562 / 0.680 |
| `dsL11` Back to the Future | 0.677 / 0.850 | 0.655 / 0.761 | **0.738 / 0.876** | 0.732 / 0.868 |

On a film the stimulus-locked part of gaze is large (ISC 0.70-0.74), so a good
prior adds +0.06 where decoding is good and dominates where it is poor. A real
saliency model sits well below this bound. This is a deployment lever for
naturalistic datasets, not a property of the basis, and this corpus has two
films to show it on.

### Two label-free alignments that hurt (scratch runs)

- **Euclidean alignment** (He & Wu 2019, the BCI standard: whiten each domain's
  features by its own covariance before a shared readout), per dataset or per
  participant, with shrinkage 0-0.5: 0.657 to 0.744, every setting below the
  incumbent's 0.770. Paradigms differ in how much gaze variance they have, and
  whitening erases exactly that.
- **Shared-response denoising** (SRM / multi-set CCA across the participants
  of a shared-stimulus dataset, keeping k = 2-16 shared time courses of the
  features): `dsL01` 0.77 -> 0.89, but the films get *worse* (`dsL08` 0.28 ->
  0.07-0.30, `dsL11` 0.67 -> 0.56-0.67). What viewers share in their eyeball
  features is not mostly gaze; what they share in their *decoded* gaze is.

## A confidence with no labels, and what pseudo-labels can do with it

October 2026, *(2,060)*. `--stages confidence` -> `confidence.json`. The question: does the
frozen decoder know when it is wrong, and can its own labels on unlabelled data
train something better? DeepMReye 1.0 had a confidence (a second output trained
to predict its own Euclidean error, used to keep the 80% most reliable
participants and drop 20-50% of volumes); `lr-cca` had none. Literature in the
author's `docs/LITERATURE_2026-10-04_pseudo_labels.md`.

**The confidence is orbit agreement.** The readout is linear in the average of
the two orbits' canonical coordinates, so the same weights applied to each orbit
alone (`probe.lodo(views=...)`) give two decodings that average to the scored
one exactly (to 5e-15). How well they correlate over a run
(`metrics.orbit_agreement`) needs no labels and no extra fitting.

### Per participant it is as good as knowing the answer

Leave-one-dataset-out, all 337 participants; Spearman with the participant's r
and AUROC for its dataset's worst fifth are medians over the nine datasets;
"keep" is the median over datasets of each dataset's most confident fraction's
median r:

| score | Spearman | AUROC worst 20% | keep 80% | keep 50% |
|---|---|---|---|---|
| **orbit agreement** | **0.66** | **0.94** | **0.802** | **0.825** |
| s.d. of the decoded trace | 0.66 | 0.93 | 0.804 | 0.826 |
| agreement through a random basis (control) | 0.43 | 0.82 | 0.785 | 0.809 |
| agreement of the raw canonical variates (no readout) | 0.32 | 0.72 | 0.772 | 0.775 |
| run length | -0.16 | 0.49 | 0.762 | 0.752 |
| oracle (sorted by true r) | 1 | 1 | 0.804 | 0.827 |
| random (mean of 200 draws) | | | 0.768 | 0.764 |

All participants: 0.771. Agreement is within 0.002 of the oracle at 80% and
at 50%; AUROC is 0.87-0.97 on seven datasets, 0.76 on `dsL01`, 0.62 on
`dsL06` (six participants). Two things carry it, and the
controls separate them: generic data quality (a random bilateral basis already
gets 0.43 / 0.82), and agreement *on the readout's gaze directions* (the trained
basis adds 0.22 Spearman over random, and weighting the variates by the readout
adds 0.30 over raw canonical agreement). The s.d. of the decoded trace carries
the same information (Spearman with agreement 0.69-0.94 on eight datasets);
agreement is the one to ship because it is a correlation and so does not depend
on the readout's units or the paradigm's gaze range.

### On the most reliable participants every arm rises, ours most

*(2,060)* `--stages reliable` -> `reliable_participants.json`: every participant's
r for the frozen basis and corpus PCA at every corpus size, the voxel ridge and
PLS, and the frozen basis's orbit agreement at each size (the 2x2x2 ridge is
per participant in `scheme_participants.json`). Keeping each dataset's 80%
most-agreeing participants -- the same participants for every arm, so no arm is
scored on different people -- at the full corpus, sub-TR median:

| | all participants | 80% most reliable |
|---|---|---|
| frozen basis | 0.771 | **0.802** |
| corpus PCA (k = 64, +-1 TR) | 0.754 | 0.785 |
| the same ridge on voxels (2x2x2 / 3x3x3) | 0.747 / 0.741 | 0.768 / 0.768 |
| PLS on voxels | 0.752 | 0.775 |
| lead over the same ridge | +0.025 | **+0.034** |

Below the full corpus the same statistic is in `scaling_draws.json`, per draw:
the 80% curve of Fig. a is the mean of 20 draws (0.675 at N = 10, 0.776 at
400, 0.801 at 1,600), with the supervised 80% line re-selected on each draw's
own agreement.

The lead holds on the reliable participants rather than growing: against
participants kept from 100% down to 20%, the lead over supervision on the mean
over datasets stays at +0.025 to +0.038 (0.033, 0.038, 0.036, 0.032, 0.025 at
100 / 80 / 60 / 40 / 20%), while the median jumps between +0.02 and +0.04 with
no trend; the lead over corpus PCA shrinks towards zero (+0.015 at 100%,
+0.001 at 20%).

The selection is the frozen basis's own, so it favours the participants it
decodes well; that it lifts PCA (+0.031) and supervision (+0.021) too says it
is picking participants whose eye signal is good, not only ones that suit it.

### It does not say how well a new dataset will decode

Read as a Spearman-Brown ceiling (`sqrt(2 rho / (1 + rho))`, what the averaged
trace would correlate with the orbits' common signal at), it overestimates every
dataset, and by how much is the transfer loss: `dsL02` 0.94 against 0.92, `dsL01`
0.97 against 0.77, `dsL08` 0.78 against 0.28. Across the nine datasets its
median ranks their medians at Spearman 0.23. What fails to transfer -- the
feature->gaze map, mostly its timing -- is shared by both orbits and invisible
to their agreement. It ranks participants *within* a study; it cannot replace
labelled data for judging a study.

### Per volume nothing works, and the reason is structural

Median over datasets of the within-participant Spearman between a score and the
1-TR error, and the squared error of the retained volumes relative to all
(1.00 = no better than dropping volumes at random):

| score | Spearman | keep 80% | keep 50% |
|---|---|---|---|
| orbit disagreement at that volume | 0.02 | 0.991 | 0.981 |
| eccentricity of the decoded gaze | 0.06 | 0.997 | 0.986 |
| decoded velocity | 0.06 | 0.972 | 0.889 |
| feature step / feature energy | 0.07 / 0.11 | 0.971 / 0.949 | 0.929 / 0.877 |
| learned error model, ridge / boosted trees (nested LODO, DME1-style) | 0.11 / 0.15 | 0.953 / 0.952 | 0.881 / 0.888 |
| oracle | 1 | 0.546 | 0.270 |

Dropping half the volumes by the best score removes 12% of the squared error;
by the oracle, 73%. Disagreement at a volume carries nothing, as it must: with
independent noise in the two orbits, the difference of their estimates is
independent of the error of their mean, and the error that remains is the
shared, map-and-timing part. Velocity is 0.31 on `dsL01` alone (errors sit at
target jumps -- timing again).

### Pseudo-labels from a linear decoder are a fixed point

The frozen decoder labels every participant of a dataset; a student readout is
fitted on its own dataset's participants against those labels (5-fold, the
`within` split, no gaze label of that dataset anywhere) and scored on gaze:

| | median / 1-TR / mean |
|---|---|
| frozen decoder (teacher) | 0.771 / 0.835 / 0.721 |
| student on its pseudo-labels | 0.771 / 0.835 / 0.721 |
| same, trained on the half of volumes the orbits agree on most | 0.771 / 0.835 / 0.721 |
| teacher + consensus prior (above) | 0.776 / 0.839 / 0.742 |
| student on consensus pseudo-labels | 0.776 / 0.835 / 0.724 |
| same + consensus prior at test | 0.776 / 0.838 / 0.742 |
| protocol readout without the 20% least-agreeing labelled participants | 0.772 / 0.831 / 0.719 |
| protocol readout without a random 20% (three seeds) | 0.768-0.773 / 0.818-0.837 / 0.720-0.725 |

(A scratch run with a richer student -- both orbits' 64 canonical coordinates
separately, +-1 TR -- gave the teacher to four decimals as well.) A pseudo-label
is a linear function of the very features the student reads, so the student's
best fit *is* the teacher, whichever volumes it is fitted on: selection by
confidence cannot move a linear fixed point. The only pseudo-labels that move
anything carry information from outside the decoder -- here other viewers'
decoded gaze -- and they do less as training targets (+0.003 on the mean) than
as a prior at test time (+0.021). The labelled side does not want cleaning
either: dropping the least confident labelled participants from the readout's
training is inside the spread of dropping random ones.

So: **ship orbit agreement as a per-participant confidence** (it is DME1's
"80% most reliable participants" at no cost, and within 0.002 of the oracle),
and close self-training with a linear student. Pseudo-labelling can only pay
with labels that know something the decoder does not -- other viewers, the
stimulus, the task's target layout.

---

## Decoding the archive *(2,060)*

`scripts/decode_corpus.py` -> `results/decoded/corpus_gaze.h5`: the shipped
basis with a readout fitted on all nine labelled datasets
(`probe.fit_readout`), applied to every participant's whole run -- 3,676
participants, 1.5 M volumes -- with each participant's orbit agreement.
`scripts/decode_saccades.py` -> `results/decoded/saccades.json` scores the
saccade studies below against their own `events.tsv`.
`scripts/figure_archive.py` draws both (draft, not in the paper).

**Two labelled datasets' own scans were in the 2,000 draw** -- fixed on 6
October. It included `ds006833` sub-01/02 and `ds006642` sub-01/02, the same runs as
`dsL07` sub-01/02 and `dsL11` sub-01/02 (154 and 1,360 volumes; decoded traces
correlate at 0.997). `unlabeled_subjects` excluded the `dsL*` folders but not
their source accessions; it now excludes `EVALUATION_SOURCES` too, and a
decoded-trace search over all 337 labelled participants found no other
duplicate.

**The confidence separates the QA outcome.** Orbit agreement on unlabelled
participants whose eye mask is fully covered: median 0.472 in datasets that
passed the eyeball QA (n = 1,994) against 0.243 in datasets that failed it
(n = 74), Mann-Whitney p = 2e-8. Restrict to covered participants: an uncovered
one decodes to a flat trace whose two orbits "agree" trivially (52 participants
of failed datasets sit at agreement ~1 with decoded SD ~0), so agreement is a
confidence about decoded gaze, not a substitute for the coverage check.

**Archived saccade studies decode trial by trial** (events from the datasets'
own `events.tsv`; no eye-tracking data used, and these participants are
unlabelled corpus participants). Change in decoded horizontal gaze 0.3-1.8 s
after target onset against the target's horizontal position, correct trials;
per participant, guided saccades r = 0.94-1.00 and antisaccades -0.87 to -0.93
in `ds000119`, -0.75 to -0.94 in `ds000121`:

| dataset (task) | trials | r |
|---|---|---|
| `ds000119`, visually guided saccades | 53 | **+0.96** |
| `ds000119`, antisaccades (same runs) | 42 | **-0.87** |
| `ds000121`, antisaccades | 105 | **-0.86** |
| `ds000121`, antisaccade errors (looked at the dot) | 4 | +0.97 |
| `ds000120`, antisaccades | 80 | +0.73 |

Gaze goes toward the target on guided saccades and away from it on
antisaccades, in the same participants' same runs. `ds000120` shows the
mirrored relation on every one of its five participants: a left-right flip in
that dataset's display or image orientation, unresolved (its registration
transforms show no reflection).

**Task categories are a weaker signal** (decoded with the 2,000-draw basis).
Decoded gaze SD by task (reliable
participants): fixation / retinotopy lowest (0.347), motor 0.372, other tasks
0.41-0.46; rest is not lower than movies. Inter-subject correlation of decoded
gaze is near zero except in the saccade tasks (0.23-0.28) and one pRF dataset.

## What is left worth trying

Honestly: not much on this corpus, and that is the finding rather than a
failure.

The representation is at its ceiling: every linear basis holds ~0.83 of gaze
within a dataset and no non-linear readout finds more. The 0.06 lost under
leave-one-dataset-out is the feature->gaze map, mostly its timing, differing
between acquisitions.

The scarce resource is **independent labelled acquisitions**, not participants,
not unlabeled data, and not model capacity, and it is now measured: the
labelled-dataset curve is still rising at eight while the corpus curve is flat
by N = 800. A tenth well-verified gaze dataset is worth more than any
architecture in this file.

The closed list now covers the estimator (`shrinkage`, `n_reduce`, whitening
exponent), the covariance it is fitted from, the anatomical symmetry prior,
selection criteria past second order, per-participant registration, both corpus
budget axes, deep and per-participant and site-invariant forms of the
cross-orbit criterion, readout weighting and co-regularisation, learned,
generative and pretrained temporal models, and self-training on the decoder's
own labels (a linear fixed point, with or without a confidence). So before
proposing anything else, say which of these it is *not*.

What is open:

- **More labelled acquisitions exist and are not in the corpus.** HCP 7T movie
  watching has EyeLink gaze for 176 participants (931 valid runs, 7T; our
  worst fold, studyforrest, is 3T); the NKI naturalistic-viewing EEG-fMRI set has 22
  participants with EyeLink on two films; HBN's PEER calibration scans give
  target-labelled fixations at three sites on two scanner models (labels are
  targets, as in `dsL01`). The labelled-dataset axis is the one still rising.
- **A stimulus-derived prior for naturalistic datasets.** Bounded above at
  +0.06 on a well-decoded film and much more on a poorly decoded one; needs the
  stimulus frames (or normative eye tracking of the same film, which exists for
  studyforrest's lab group and for the HBN / NKI films). Use it as a prior at
  test time: as pseudo-labels for a student, other viewers' gaze kept a fifth
  of its gain.
- **A label-free temporal stage is closed for our decoder** (remeasured 8
  October, "Temporal stages on the decoded trace"): GP and interpolation tie
  the shipped readout on the mean, even with an oracle length scale. What is
  open is the comparison with supervision *with* a stage on both, along the
  labelled-dataset and corpus-size axes -- interpolation lifts the voxel ridge
  by +0.025.
- **Using one or two labelled participants of the new study.** Measured above:
  a readout fitted on them loses to the frozen decoder, and refitting the
  frozen decoder's output on them loses too. Something that improves on 0.770
  with K = 1 is open; the ceiling is the within-dataset 0.830.
- **Bagged bases** -- `lr-cca` on disjoint corpus subsets, concatenated. The
  shrinkage null argues against it, but bagging attacks *selection* variance
  rather than whitening variance, which is a different failure.
- **Slice timing, grouped properly.** The z-band test above was blind by
  construction under multiband; grouping voxels by acquisition time instead
  needs the per-dataset `SliceTiming` vector stored at ingest.
- **Per-participant anatomy.** The template version of the analytic basis does
  not add to `lr-cca`; the per-participant version needs two lines in
  `pipeline.py` plus a full re-ingest. Do not expect much, but **add the two
  lines the next time anything is re-ingested**, because then it costs nothing.
