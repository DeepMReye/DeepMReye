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

`lr-cca` gains **+0.150** as the unlabeled corpus grows from 25 to 800
participants and is the most data-hungry basis tested (two 7000-dimensional
whitenings plus a cross-covariance). Then it **saturates**: 800 to 1039 is flat,
and extending to 2000 bought nothing measurable.

| corpus size | 25 | 100 | 400 | 800 | 1039 | 2000 |
|---|---|---|---|---|---|---|
| `lr-cca:64` | 0.661 | 0.749 | 0.784 | **0.811** | 0.809 | flat |
| `corpus-pca:64` | 0.758 | 0.813 | 0.810 | 0.818 | **0.821** | flat |

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
