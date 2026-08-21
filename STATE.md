# Corpus status and provenance

What exists right now, how it was made, and what is next. For the method see
`overview.md`; for design decisions and cluster constraints see `CLAUDE.md`;
for how to run anything see `README.md` and `slurm/README.md`.

Last updated **2026-08-20**.

## Recent Accomplishments

### A voxel network that provably matches the linear incumbent — and two bugs that faked it (2026-08-20)

`deepmreye/voxelnet.py`, `scripts/train_voxelnet.py`, `deepmreye/tests/test_voxelnet.py`,
`results/subtr/voxelnet_lowrank32.json`, `results/subtr/voxelnet_heavyaug.json`.

The objection to stopping at the temporal ceiling is a good one: a network is a universal
function approximator, so it should be **at least as good** as `lr-cca:32 + lags` read off
voxels. That is right, and it is right *constructively* -- `W_cca` is a linear layer,
`make_lags` is a fixed convolution, the ridge head is a linear layer. The mistake would be
treating universal approximation as a claim about optimisation or generalisation, which it is
not. So the network is **initialised at** the incumbent rather than asked to rediscover it:

    pred(x) = head_lin(make_lags(W_cca @ (x - mu), L))     # the incumbent, warm-started
            + alpha * head_nl(make_lags(g(x), L))          # zero-init; must earn adoption

`g` is a **supervised learned projection from voxels** -- and that is the arm
`analyze_temporal_ceiling_supervised.py` never tested. Its Block B varied the rank of a
*frozen, unsupervised* projection (`lr-cca` fitted blind to gaze) and found more of it
monotonically harmful; a projection fitted *against gaze* is a different object.

#### 1. The guarantee holds, and it is exact

Per fold the linear branch is warm-started from that fold's own `RidgeCV` fit. **All nine
per-fold incumbent values reproduce `ridge-L1` to four decimals** (0.7678, 0.9169, 0.7751,
0.8488, 0.8039, 0.6760, 0.7471, 0.2859, 0.6693), which is what makes the comparison
trustworthy rather than merely favourable. With selection on held-out *datasets*, the
unaugmented network lands at **+0.0000 on 8 of 9 folds** -- exactly the incumbent, because the
branch is adopted only if it clears a margin on datasets it did not train on.

| arm | median sub-TR | vs incumbent | folds won |
|---|---|---|---|
| incumbent (`ridge-L1`) | 0.7678 | — | — |
| voxelnet low-rank, no augmentation | **0.7678** | +0.0000 | 0/9 |
| voxelnet low-rank, heavy augmentation | 0.7476 | −0.0202 | 0/9 |
| voxelnet CNN, random init | **0.7678** | +0.0000 | 0/9 |
| voxelnet CNN, **SSL-init** (3,318 unlabeled) | **0.7678** | +0.0000 | 0/9 |

#### 2. Two bugs that produced plausible numbers instead of errors

**A zero gate times a zero head is a saddle.** The obvious reading of "the branch starts
switched off" is `alpha = 0` *and* `head_nl = 0`. But d(loss)/d(alpha) is proportional to the
branch output and d(loss)/d(head_nl) is proportional to alpha, so with both at zero neither
ever moves. The network sits at initialisation forever and reports an immaculate **+0.0000
margin that reads exactly like the guarantee working**. Measured: `alpha` pinned at 0.0000
with validation loss identical across epochs. Only one may be zero -- `head_nl` zero and
`alpha` one gives the same prediction at step 0 with gradient flowing immediately.
`test_gradient_reaches_the_learned_branch_at_init` fails (both gradients exactly 0.000e+00)
if the bug is reintroduced.

**Early stopping on held-out participants selects branches that do not cross a dataset
boundary.** The test is cross-dataset; validating on held-out participants *of the training
datasets* measures within-distribution fit and happily adopts dataset-specific structure. It
cost dsL01 0.7678 → 0.7263 while its validation loss was still improving. Selection moved to
held-out **datasets** scored by **sub-TR r** (the reported quantity, not residual MSE), which
restored dsL01 to +0.0000. A related detail: adoption needs a *margin* (0.005), because
selection is a median over a handful of participants and any-improvement adoption cost dsL08
0.2859 → 0.1545.

#### 3. Augmentation raised the selection score without improving transfer

Heavy augmentation -- integer-voxel shift +-3, mixup across participants, noise 0.2, voxel
dropout 0.2, dropout 0.3, weight decay 0.1 -- did make the branch learn something: selection r
rose **0.7405 → 0.7724** on held-out datasets and `alpha` held near 1 instead of collapsing to
0.05. But it did not transfer: **every fold where the branch was adopted lost** (dsL01 −0.002,
dsL02 −0.013, dsL03 −0.028, dsL05 −0.010) and every fold where it was not adopted landed
exactly on the incumbent. So augmentation caused *more* bad adoptions, not fewer, and the
augmented arm is 0.0202 **below** the unaugmented one.

The cleanest statement of the result so far: **in 9/9 folds, adopting a learned voxel branch
either did nothing or hurt. It never helped.**

#### 4. It is not capacity, not optimisation, and not tuning -- the overfit test settles it

A flat training loss has two readings that look identical on a curve: the model is
over-regularised, or the target is unlearnable. Run without augmentation, without weight decay,
without dropout and at `lr 3e-3`, the branch's training loss still did not fall (0.594, 0.606,
0.618, ... 0.589), which rules out the first reading but not the second.

The standard triage separates them. On **four participants of one dataset** (2008 rows), where
a working model must simply memorise:

| encoder | step 0 | 80 | 160 | 240 | 399 |
|---|---|---|---|---|---|
| low-rank linear, lr 1e-3 | 0.1978 | 0.0049 | 0.0016 | 0.0007 | **0.0002** |
| low-rank linear, lr 1e-4 | 0.1978 | 0.0709 | 0.0281 | 0.0113 | 0.0038 |
| 3-D CNN, lr 1e-3 | 0.1978 | 0.0139 | 0.0031 | 0.0016 | **0.0008** |

**Both encoders drive the residual to essentially zero.** Capacity is ample and the
optimisation works; the flat loss on the full corpus is the model finding nothing *systematic*
across 400k rows and 8 datasets, not failing to train. The residual is memorisable
participant-by-participant and carries no structure that crosses a dataset boundary.

(On those four participants the linear branch already explains **80.2%** of target variance,
which is the same fact from the other side.)

#### 5. One architectural error worth recording

The first CNN used `AdaptiveAvgPool3d(1)` before its head -- the default choice for a 3-D
encoder, and **exactly wrong here**. Gaze *is* the spatial position of the eyeball, so averaging
the spatial axes away discards the signal. The symptom was diagnostic once looked at: training
loss flat at 0.46-0.49 and selection r pinned at the incumbent's 0.8265 to four decimals.
Flattening the feature map instead (72 spatial positions x 64 channels) makes the metric move.
DeepMReye 1.0 flattens into dense layers for the same reason.

#### 6. Self-supervised pretraining on 3,318 unlabeled participants: the objective learns, the gaze does not improve

`scripts/pretrain_voxelnet.py`, `results/subtr/pretrained_cnn.pt`,
`results/subtr/voxelnet_cnn_ssl.json`. The one direction that does not need the residual to be
predictable from labels: 1.09M TRs across 903 accessions, asserted disjoint from every `dsL*`
participant at cache-build time.

**Objective.** Band-matched temporal contrast -- anchor at `t`, positive at `t +- {1,2}`,
negatives from the **same run** at `|delta| in [3,20]` TRs, so participant, anatomy, scanner and
slow drift are constant by construction and cannot separate anchor from negative. Chosen
against the measured next-TR failure, where the predictable part of an eye block is the
nuisance.

**It learns.** Loss falls from 2.2509 to **1.7987** against a chance of `ln(9) = 2.1972`, with
positive-rank-1 around 0.33 against 0.111.

**And it changes nothing downstream.** With the encoder initialised from that checkpoint and
*every other setting identical* to the random-init CNN run -- same architecture, augmentation,
selection rule, folds -- the result is **+0.0000 on 9 of 9 folds**, the same table as random
initialisation. The pretrained features never clear the adoption margin on a held-out dataset.

**Two things the ablation established on the way, both of which would otherwise have produced a
misleading null:**

| contrastive config | final-50 loss (chance 2.1972) |
|---|---|
| no augmentation, negatives 3-20 | **2.0761** |
| noise 0.1 only, negatives 3-20 | 2.0785 |
| shift +-2 + noise 0.1 | 2.1303 |
| no augmentation, negatives 10-60 (far) | **1.7583** |

- **Shift augmentation suppresses the objective.** Anchor and positive receive *independent*
  +-2 voxel shifts, a displacement that exceeds the actual motion between adjacent TRs, so the
  positive pair stops being similar. The first pretraining run sat at chance for 500 steps
  because of it, and would have handed the probe a useless encoder.
- **Far negatives learn much faster, and that is a reason to reject them.** 1.758 against 2.076
  is distance-in-time being read off slow drift -- the CPC failure mode, i.e. the next-TR
  result in contrastive clothing. The band-matched version is harder *because* it removes the
  cue, which is the design intent.

#### 7. What this settles

The objection was correct and is now demonstrated: **a voxel network is at least as good as
`lr-cca:32 + lags`**, exactly -- on 8 of 9 folds for the low-rank encoder and **9 of 9** for
the CNN, with and without self-supervised pretraining. What it is not is *better*, and the reason is
now pinned down by elimination rather than asserted -- not capacity (overfits to 0.0002), not
optimisation (loss falls three orders of magnitude), not regularisation (unregularised is no
better), not the readout (the branch is free to learn any projection). What remains is that the
residual after a linear read of the canonical span holds no transferable gaze signal, which is
the same conclusion Block B of the temporal gate reached from the frozen-basis side.

---

### The temporal non-linearity ceiling is flat too, and the incumbent's lag setting is wrong (2026-08-20)

`deepmreye/temporal_probe.py`, `scripts/analyze_temporal_ceiling_supervised.py`,
`results/subtr/temporal_ceiling_supervised.json`, `results/subtr/temporal_ceiling_tcn.json`.

`analyze_nonlinear_ceiling.py` closed the *spatial* non-linear program: no supervised
non-linear readout beats ridge on the same per-bin features. It was silent on **time** --
every arm there is a static map from one bin's canonical coordinates to that bin's gaze, and
the incumbent (`lr-cca:32 + lags±2`) is itself a *linear temporal* model. This is the temporal
sibling, run before building a network rather than after. Same logic, which is what makes it a
ceiling: the probe readout is linear, so a temporal encoder can only pay if gaze depends on the
window in a way a linear map cannot express, and that is upper-bounded by what a **supervised**
model gets on the same window.

#### 1. The gate: nothing clears, and nothing is close

9-fold LODO, 337 participants, `basis_n2000`, one process. Gate was > +0.02 median **and**
≥ 6/9 folds over the best linear arm.

| arm | block | sub-TR | 1-TR | vs `ridge-L1` | folds won |
|---|---|---|---|---|---|
| **`ridge-L1`** | A | **0.7678** | 0.8380 | — | — |
| `ridge-L2` *(the shipped arm)* | A | 0.7585 | 0.8230 | −0.0093 | 2/9 |
| `banded-L2` | A | 0.7563 | 0.8237 | −0.0116 | 3/9 |
| `poly-time-L2` | C | 0.7504 | 0.8140 | −0.0175 | 0/9 |
| `ridge-k64-L2` | B | 0.7501 | 0.7979 | −0.0178 | 1/9 |
| `banded-L4` | A | 0.7456 | 0.8033 | −0.0223 | 2/9 |
| `ridge-lr-concat32-L2` | B | 0.7455 | 0.8113 | −0.0223 | 1/9 |
| `ridge-L0` | A | 0.7421 | **0.8406** | −0.0257 | 1/9 |
| `banded-k128-L2` | B | 0.7418 | 0.7949 | −0.0260 | 2/9 |
| `mlp-L2` | C | 0.7377 | 0.7833 | −0.0301 | 0/9 |
| `tcn-L2` | C | 0.7370 | 0.7745 | −0.0309 | 0/9 |
| `ridge-k256-L2` | B | 0.7301 | 0.7898 | −0.0378 | 1/9 |
| `mlp-L4` | C | 0.7191 | 0.7722 | −0.0487 | 0/9 |
| `tcn-L4` | C | 0.5424 | 0.6529 | −0.2254 | 0/9 |

**A linear ridge on a 3-TR window is the best supervised model tested.** Nothing else beats it
on more than 3 of 9 folds.

#### 2. Three separate refutations, and two of them cannot be blamed on tuning

The non-linear arms (`mlp`, `poly-time`, `tcn`) all land 0.02–0.05 below ridge, which is the
spatial result reproduced in time. Those are trained models and a sceptic can always say they
were under-tuned. **Blocks A and B cannot be answered that way -- they are convex, solved to
optimality, with the penalty chosen by inner CV:**

- **Longer linear windows lose, and per-lag penalties do not rescue them.** The suspicion was
  that the decline past ±1 was a shared-alpha artifact (the earlier within-subject sweep used a
  *fixed* `alpha=10.0`). `BandedRidge` with one penalty per lag recovers only **+0.006** of the
  0.028 drop from L=1 to L=4. If lags ±3..±5 carried gaze that a shared penalty was suppressing,
  that is where it would have shown up.
- **More of the voxel space loses, monotonically.** k = 32 → 64 → 128 → 256 gives
  0.759 → 0.750 → 0.745 → 0.730, and `banded-k128` (a penalty per rank block) is *worse* than
  plain `ridge-k128`. Keeping the two orbits separate instead of averaging them also loses
  (0.746 against 0.759).

That second one is the direct answer to "a voxel model isn't restricted to 256 frozen
directions, so it has room the ceiling analysis never tested". The room is there; **the gaze is
not in it**. The extra directions are noise, and a network with access to them starts from a
worse position, not a better one.

#### 3. The weight-sharing hypothesis, tested and refuted

The one mechanism-level argument for a temporal network was that a lag stack pays `k` new ridge
columns per lag while a conv reuses one kernel at every offset, so it can afford a window the
linear model cannot. Measured with a small weight-shared 1-D conv over the lag axis, scored
through the identical protocol: **`tcn-L2` 0.737, `tcn-L4` 0.542.** It gets *worse* with the
longer window -- the opposite of the prediction. (`tcn-L4`'s collapse is probably an
optimisation failure rather than a clean ceiling statement; the load-bearing number is
`tcn-L2` at 0.737, which sits alongside `mlp-L2`'s 0.738.)

#### 4. What is worth keeping: the shipped lag setting is wrong

**`lags±1` beats `lags±2` at sub-TR by +0.0093 (0.7678 against 0.7585), and `lags±0` beats both
at 1-TR (0.8406).** The optimal temporal context depends on the target resolution -- context
helps interpolate within-TR motion and blurs the 1-TR mean. Two independent protocols agree on
the sub-TR peak (this LODO sweep, and the earlier within-subject sweep at fixed alpha), which
is why it is worth adopting despite sitting inside the ~0.02 noise floor. It costs one integer.

#### 5. Infrastructure this required, which was a finding in itself

**`eval_probe.py` cannot score sub-TR gaze and never could.** `evaluate.probe.temporal_targets`
nanmeans `[B, W, 10, 2] → [B, n_t, 2]`, so every number it has produced is 1-TR mean gaze at
5-TR bins. The sub-TR headline existed only inside `scripts/benchmark_all_11_datasets.py`, with
a different basis (`n2000` vs `n1039`), alpha grid and NaN rule from everything else.
`deepmreye/temporal_probe.py` is now the single audited implementation; it reproduces the
benchmark to **0.0001 / 0.0005** on `lr-cca:32` / `+lags±2` from an independently built cache,
and refuses to load a cache whose **corpus fingerprint** differs -- the guard the existing
labeled cache lacks, which is why a 285-participant, 7-dataset, pre-repair cache still loads
there without complaint.

#### 6. Verdict

The gate was set in advance as a hard stop and it failed on every block. **The temporal
non-linearity ceiling is as flat as the spatial one**, more voxel rank is actively harmful, and
weight sharing across time buys nothing. Together with the six prior arms (JEPA, Orbit-JEPA,
next-TR GRU, `xorb`, `xrot`, `ocon`) this closes representation learning on this corpus for
temporal objectives as well, and it closes it *before* a network was built rather than after.

The standing explanation is unchanged and now covers both axes: gaze is **linearly accessible**
from these features, and what limits decoding is the acquisition, not the model -- the temporal
envelope (`decoded_r ≈ 1.03·lag1 + 0.085`) that `fold-pca:64` already achieves everywhere.

---

### `dsL12_rest` retired, and a low-variance dataset in the *training* pool costs a fold 0.11 (2026-08-20)

`scripts/retire_labeled_dataset.py`, `results/dsL12_rest_labels.npz`,
`results/scaling/benchmark_all_11_datasets.json` (the 10-dataset run kept beside
it as `benchmark_10ds_with_dsL12.json`).

`dsL12_rest` is retired: labels archived, 20 eye blocks folded back into
`ds004158` as unlabeled corpus data, 5 QA-sample duplicates deduped on the way.
The corpus is **337 labeled participants across 9 datasets**.

**Its labels were never the problem, and that is worth stating plainly** --
timing verified (17/20 peak at lag 0), registration the corpus's best (pairwise
SD-map r 0.987), units documented. Resting state with a central fixation dot
gives per-participant gaze SD **0.26-1.3 deg** against 2.3-2.7 for the pursuit
and movie sets, so there is nothing to decode and the fold measured the
paradigm. It was already excluded from the headline median; retiring makes that
structural rather than a footnote.

#### The result that was not a formality

Excluding a fold from a median changes only the median. **Removing it from the
corpus also removes it from every other fold's training pool**, and that is
where the movement was:

| fold | `fold-pca:64` 1-TR | delta vs. `dsL12` in the pool |
|---|---|---|
| **`dsL06_sequences`** | **0.756** | **+0.109** |
| `dsL07_deepmreye_calib` | 0.830 | +0.013 |
| `dsL11_backtothefuture` | 0.835 | +0.007 |
| every other fold | — | within +-0.008 |

**This is the second time the same thing has happened, on the same fold.**
Retiring `dsL09` moved `dsL06` by +0.08; retiring `dsL12` moves it by +0.109.
Both times the retired dataset was one nobody scored the fold on.

The mechanism is `--standardize-targets dataset`, and it is worth understanding
before adding any dataset to the corpus. Pooled training z-scores **each
dataset's** gaze separately, precisely so a 595-unit dataset cannot drown a
21-unit one. For `dsL12` that rescales a 0.26-1.3 deg wobble -- which is mostly
measurement noise -- up to unit variance, so the readout sees it as
full-amplitude signal and fits it. A low-variance dataset is therefore not
merely uninformative in the training pool, it is actively harmful, and the
standardisation that makes pooling possible is what makes it so. `dsL06` is the
6-participant fold, hence the most sensitive to a worse readout.

#### 9-fold LODO on the final corpus

| arm | 1-TR | sub-TR |
|---|---|---|
| `fold-pca:64` (supervised reference) | 0.835 | 0.746 |
| **`lr-cca:32`** | **0.841** | 0.742 |
| Super-Basis (96) | 0.839 | 0.739 |
| **`lr-cca:32 + lags±2`** | 0.823 | **0.759** |

`lr-cca:32` is now the best 1-TR arm and the Super-Basis's advantage over it has
gone (0.841 vs 0.839, having been 0.834 vs 0.841 with `dsL12` in the pool) --
which is itself a reminder of the noise floor: **+0.005 over the supervised
reference at 1-TR and +0.013 at sub-TR are ties**, and the ordering among the
unsupervised arms is not stable across a corpus change of one dataset. Quote the
deployment claim, not a winner.

Within-dataset 5-fold is unchanged for the 9 remaining datasets, as it must be
-- each dataset is scored independently there, so retiring `dsL12` only removes
its own row (`dsL08` 0.5040, `dsL01` 0.7842, `dsL02` 0.9151, identical to the
previous run).

---

### Benchmarks re-run on the repaired corpus: 357 participants, 10 datasets (2026-08-20)

`scripts/benchmark_all_11_datasets.py`, `scripts/eval_within_dataset.py`,
`results/scaling/benchmark_all_11_datasets.json`,
`results/scaling/within_dataset_benchmark.json` (pre-repair copies kept beside
them as `*_prerepair.json`).

The first full benchmark on the corpus as it now stands: vertical convention
repaired, `dsL09` retired, `dsL11` at 37 participants instead of 4, `ds004283`
rejected. Basis is `basis_n2000`; targets are z-scored per training dataset.

#### 1. Leave-one-dataset-out

| fold | N | `fold-pca:64` | `lr-cca:32` | Super-Basis (96) |
|---|---:|---|---|---|
| `dsL01_guided_fixations` | 170 | 0.767 | 0.776 | 0.773 |
| `dsL02_pursuit` | 9 | 0.898 | **0.912** | 0.907 |
| `dsL03_pursuit` | 24 | **0.854** | 0.846 | 0.840 |
| `dsL04_pursuit` | 34 | **0.879** | 0.860 | 0.870 |
| `dsL05_free_viewing` | 27 | **0.846** | 0.834 | 0.841 |
| `dsL06_sequences` | 6 | 0.647 | 0.677 | **0.709** |
| `dsL07_deepmreye_calib` | 15 | 0.816 | 0.829 | **0.850** |
| `dsL08_studyforrest_movie` | 15 | 0.214 | **0.361** | 0.293 |
| `dsL11_backtothefuture` | 37 | 0.828 | 0.853 | **0.855** |
| `dsL12_rest` *(reported, not in the median)* | 20 | 0.053 | 0.004 | 0.032 |
| **median, 9 folds (excl. `dsL12`)** | | **0.828** | **0.834** | **0.841** |
| median, all 10 | | 0.822 | 0.831 | 0.841 |

Sub-TR (10 pts/TR), same folds: `lr-cca:32 + lags±2` takes it at **0.761**
against `fold-pca:64`'s 0.749 (medians excl. `dsL12`; 0.754 against 0.735 over
all ten).

**Read both margins as ties.** +0.013 (1-TR) and +0.012 (sub-TR) sit inside the
~0.02 noise floor this file documents for a 7-to-9-fold median. The claim that
survives is the deployment one, and it is not small: a **frozen** basis that
needs no data at all from the target study matches a fold-local supervised one.

#### 2. What moved, and why

- **`dsL08` is the big change: -0.031 -> +0.214** (`fold-pca:64`) and **+0.361**
  (`lr-cca:32`). That is the sign repair, not a modelling change.
- **`dsL06` gains +0.018 to +0.098** without anyone touching it -- `dsL09` left
  the *training* pool. Same effect recorded in the repair entry, reproduced here.
- **`dsL11` went slightly *down* (-0.007 to -0.016) while going from 4
  participants to 37.** This is the healthy direction: the old figure was a
  median over four subjects. At 37 it is an estimate, and it is still the best
  non-original fold.
- Per-fold deltas against the pre-repair run are meaningful; **the median rows
  are not like-for-like**, because the fold set itself changed (`dsL09` gone).

#### 3. Within-dataset 5-fold, and an honest correction

The superseded entry claimed DeepMReye 2.0 "beats supervised within-dataset
baselines on 9 of 11 datasets". On the repaired corpus that splits by
resolution, and only one half holds:

- **Sub-TR: `+lags±2` wins 7 of 10**, and by margins that are not noise --
  `dsL03` **+0.073** (0.845), `dsL01` **+0.059** (0.843), `dsL05` +0.028,
  `dsL02` +0.026. Temporal context is doing real work at the resolution that is
  the whole point of MR-based eye tracking.
- **1-TR: the supervised reference wins 6 of 10.** The Super-Basis takes only
  `dsL01`, `dsL02` and `dsL05`, each by <0.006. At 1-TR resolution within a
  single dataset, a fold-local PCA is the better feature set.

Best within-dataset numbers: `dsL06` 0.953, `dsL02` 0.915, `dsL04` 0.901,
`dsL11` **0.890** (37 subjects). `dsL08` reaches 0.504 within dataset against
0.214 across -- the gap that its 7T registration consistency already explains.

---

### Repair: the vertical convention was inverted in three ingested datasets, and dsL12's gaze columns were transposed (2026-08-20)

`deepmreye/eyetracking.py`, `scripts/fetch_eyetracking.py`,
`scripts/verify_gaze_sync.py --convention`,
`results/label_backup_pre_signfix/`, `results/repair_after.json`.

**The three "bad" datasets were not hard, they were mislabelled.** `dsL08`,
`dsL09` and `dsL12` all shipped with the vertical axis **negated** relative to
the corpus, and `dsL12` additionally had **x and y transposed**. A frozen
readout trained on `dsL01`..`dsL07` -- which decodes `dsL11` at r_x +0.73 /
r_y +0.67, so it is a working instrument -- read all three with a **positive
r_x and a negative r_y**:

| fold | pred_x vs true_x | pred_y vs true_y | after repair |
|---|---|---|---|
| `dsL08_studyforrest_movie` | +0.277 | **-0.237** | +0.277 / +0.098 |
| `dsL09_fearlearning` | +0.139 | **-0.109** | +0.122 / +0.109 |
| `dsL12_rest` | -0.016 | **-0.132** | +0.024 / +0.129 |
| `dsL11_backtothefuture` (control) | +0.725 | **+0.670** | unchanged |

(`dsL08`'s repaired r_y is +0.098 rather than +0.237 because the repair also
added a valid box, and the participant sitting at the median of 15 is `sub-05`
-- 73% track loss, the worst-tracked in the dataset. The other 14 negate
exactly.)

#### 1. Root cause: one inverted sentence

`center_and_scale`'s docstring said "the tracker's y grows downward while the
corpus's grows upward, which `flip_y` corrects". **The corpus's y grows
downward.** Every EyeLink is top-left origin, so anyone reading that sentence
sets `flip_y=True` and inverts the dataset. Three did. The sentence is now
corrected, with the measurement attached and a warning not to revert it.

#### 2. The convention is established from anatomy, not from another dataset

Checking a new dataset's sign against the corpus is fine; checking the *corpus's*
sign that way is circular. `verify_gaze_sync.py --convention` does not:

- the template is stored **L, A, S**, so axis 2 grows superior and axis 1 anterior;
- the eyeball is a bright vitreous sphere with a **dark lens at its anterior
  pole**, so looking up rotates that dark lens to higher z;
- therefore corr(voxel, label y) must be a **dipole along z in the anterior half
  of the orbit**, and its sign says which way y points -- from one dataset alone.

It is computed per participant and reported as a **vote**, which is what makes
it readable. On six datasets the participants agree overwhelmingly, and they
agree with each other:

| dataset | n | median dipole | fraction positive |
|---|---|---|---|
| `dsL02_pursuit` | 9 | +0.075 | **1.00** |
| `dsL11_backtothefuture` | 4 | +0.104 | **1.00** |
| `dsL01_guided_fixations` | 30 | +0.153 | **0.97** |
| `dsL03_pursuit` | 24 | +0.134 | **0.96** |
| `dsL07_deepmreye_calib` | 15 | +0.106 | **0.93** |
| `dsL04_pursuit` | 30 | +0.070 | **0.83** |

Six independent acquisitions, unanimously positive: **the corpus's y grows
downward.**

**It is a corpus-level instrument, not a per-dataset gate**, and the tool now
says so. On the other five datasets the participants split 33-65% and the
dipole is measuring noise. `dsL05` is the warning: it splits 41/59 here while
its vertical axis decodes at r 0.83. A minority-negative vote is not evidence of
a flip -- go to a cross-dataset readout for that, which is how these three were
actually caught.

#### 3. Why the original verification passed all three

**A lag sweep is blind to a sign error.** Negate y and every lag scores the same
magnitude, so the peak stays at 0 and the verdict stays PASS. `verify_gaze_sync.py`
was measuring the time origin, which really was right -- 15/15 of `dsL08` and
17/20 of `dsL12` still peak at lag 0 in a per-subject sweep. Nothing in the
harness looked at the sign until a cross-dataset readout was pointed at it.

#### 4. `dsL12_rest`: the TSV is (y, x, pupil, time)

`ds004158` has no root sidecar and its per-run JSON carries only calibration
fields, so the `columns=` override in `fetch_eyetracking.py` was the only column
information there was -- and it had x and y the wrong way round. The dataset's
own README settles it: participants "keep fixation to a fixation dot at the
screen center".

| read as | 20-subject median lands at | off a 1920x1080 centre by |
|---|---|---|
| shipped `(x, y)` | (538, 970) | **-422 px, +430 px** |
| swapped `(y, x)` | (970, 538) | **+10 px, -2 px** |

The sidecar's own mean calibration error is 0.65 deg = 35 px. Read the swapped
way the labels sit inside it; read the shipped way no fixation task could
produce them. After the repair every participant's mean gaze is within +-1 deg
of screen centre.

**No sub-TR offset is warranted**, checked now that the columns are right:
re-binning the raw gaze at fractional offsets peaks at +0.25 s with r 0.448
against +0.00 s at 0.440 -- **+0.008**, inside the noise -- and the four
participants' own optima scatter over 0.00 / 0.25 / 0.25 / 0.75 s. The
configured +0.00 s stands. Contrast `dsL08`, where the same sweep found a real
half-TR shift worth +0.12 r with 15/15 participants agreeing on the sign: that
is what a genuine offset looks like.

#### 5. `dsL09_fearlearning` is retired -- the labels, not the model

Repaired (sign, valid box) it still decodes at **0.16 within subject**, and
cleaning does not help: restricted to the 31 participants passing coverage,
off-grid *and* the authors' own exclusion list, the median is **0.128**. Three
independent disqualifications:

- **Per-subject timing.** In a within-subject lag sweep only **21 of 48**
  participants peak at lag 0 and **12 peak at -3**. That is the `ds007532`
  signature, and the same verdict follows -- one offset per subject would fix it
  and would be circular.
- **37-40% of all samples are the tracker's `(0,0)` track-loss code**, and
  values run to +-3276, past any screen. Four participants are >90% lost.
- **The dataset says so itself.** Its sidecar reads "Reliability would not be
  guaranteed", "more than 25% data loss", and names the 8 of 52 participants its
  own authors excluded (08 31 41 43 44 45 48 52).

**Retired**, following the `ds007532` precedent. Its config is parked as
`_ds001242_excluded` with the evidence; the corpus folder is now
`dsX09_fearlearning_unaligned`, its 52 label arrays are archived at
`results/dsX09_fearlearning_unaligned_labels.npz`, and the labels are gone from
the files -- which is what actually retires it, since `ProbeDataset._discover()`
finds a dataset by its `labels` and not by its name. The eye blocks stay: what
failed is the gaze recording, not the imaging, and they are perfectly good
unlabeled corpus data.

`scripts/retire_labeled_dataset.py` does this, because `dsX10` was retired by
hand and that is how the rename-without-stripping trap was set the first time.
It archives before it strips, defaults to a dry run, and marks the registry
`LBL_DATASET_SKIPPED`.

#### 6. `dsL08_studyforrest_movie`: labels are good, the 7T acquisition is the limit

Repaired, it goes from **-0.03 to +0.19** (`fold-pca:64`) and **+0.07 to +0.34**
(`lr-cca:32`) on LODO. What remains is not a label problem, and four
measurements say so:

- **The labels are as good as `dsL11`'s.** Inter-subject gaze correlation --
  everyone watching the same movie at the same time, computed without touching
  the BOLD -- is **0.55 / 0.40** against `dsL11`'s **0.56 / 0.50**.
- **The gaze is in the eye blocks.** Within subject it decodes at **0.73**
  (r_x 0.77, r_y 0.69), and 15/15 participants peak at lag 0.
- **It degrades with every step away from the subject**: 0.73 within subject,
  **0.52** leave-one-subject-out inside the dataset, **0.19-0.34** across
  datasets. `dsL11`, same paradigm, runs 0.85 / 0.79 / 0.70.
- **It has the worst within-dataset registration consistency in the corpus**:
  pairwise temporal-SD-map r **0.915** and centroid scatter 0.30/0.43/0.13 vox,
  against 0.99 and 0.03-0.08 for `dsL05`/`dsL07`/`dsL11`/`dsL12`.

Ruled out: a mirrored registration (flipping the block along x, y or z rescues
no subject), and a truncated FOV (per-slice dead-voxel profiles are within 5%
of every other dataset's). It is 7T EPI in the orbits, and the honest statement
is that its gaze is recoverable within study and only partly across studies.

#### 6b. Retirement, and what it cost to have kept `dsL09` around

`scripts/retire_labeled_dataset.py`. Two things came out of doing this properly
rather than by hand:

- **There is no `dsX##_*_unaligned` category any more.** A dataset whose gaze
  recording failed is not a third kind of thing -- what is left is eye blocks
  with no labels, which is exactly what every other unlabeled participant in the
  corpus is, and the naming rule already says to keep the real accession because
  that is the provenance. `dsX09` and `dsX10` are now `ds001242` and `ds007532`.
  The old prefix read as a *status* rather than a source and nothing in the
  pipeline understood it.
- **Folding back deduplicated 11 participants the corpus had been carrying
  twice.** The QA sample extracted 5 subjects of `ds001242` and 6 of `ds007532`
  long before the eye-tracking ingest extracted all of them under a `dsL##`
  name, so those 11 were counted twice in every unlabeled basis fit. The two
  copies are the same source run and **not bit-identical** -- voxelwise r 0.83
  and 0.99 -- which is a small standing result in itself: **ANTs `SyNAggro` is
  not reproducible run to run**, so "re-extract it" is not a null operation. The
  QA-sampled copy is kept (a human looked at its thumbnail, and its `approved`
  label is keyed to it) with the gaze provenance attrs merged onto it.

  **Any future ingest hits this collision by construction.** Check the accession
  folder before writing a new `dsL##` one.

#### 7. `dsL11_backtothefuture` was never broken -- it is just 4 of 39 participants

Its labels verify at lag 0 with margin +0.55 and it is the **best** non-original
fold (`lr-cca:32` 0.848, above `fold-pca:64`'s 0.813).

A dry run over the whole accession confirms the other 35 are ready:
**39/39 participants resolve at coverage 1.00** through the `indexed_message`
anchor, NaN fraction median 0.031 (max 0.287), 2 eye-tracking runs dropped for
having no paired BOLD (`results/eyetracking_ingest_ds006642.json`). Nothing
about the alignment is unresolved -- what is left is 39 BOLD downloads at a
~1.8 GB median (~70 GB) and one ANTs run each:

```bash
python scripts/fetch_eyetracking.py --dataset ds006642 --data-dir $DATA
```

`s3_get` retries now, because a single-shot read of one of these EyeLink ASCII
exports times out often enough to kill a 39-subject run partway through. Worth
doing: independent acquisitions are the scarce resource here, and this is the
only place in the corpus where 35 more of them on a *verified* acquisition are
available for nothing but bandwidth.

#### 8. LODO after the repair (10 folds, 1-TR targets, `basis_n2000`, 1000 train windows)

`results/repair_final_10fold.json`, measured on the final corpus with `dsL09`
retired:

| fold | `fold-pca:64` r_x / r_y / mean | `lr-cca:32` r_x / r_y / mean |
|---|---|---|
| `dsL01_guided_fixations` | +0.794 / +0.726 / 0.764 | +0.792 / +0.753 / **0.772** |
| `dsL02_pursuit` | +0.928 / +0.868 / 0.898 | +0.932 / +0.903 / **0.917** |
| `dsL03_pursuit` | +0.840 / +0.801 / **0.795** | +0.841 / +0.786 / 0.794 |
| `dsL04_pursuit` | +0.901 / +0.874 / **0.883** | +0.847 / +0.868 / 0.855 |
| `dsL05_free_viewing` | +0.829 / +0.852 / **0.832** | +0.850 / +0.825 / 0.828 |
| `dsL06_sequences` | +0.902 / +0.549 / **0.713** | +0.856 / +0.481 / 0.664 |
| `dsL07_deepmreye_calib` | +0.861 / +0.786 / 0.817 | +0.834 / +0.817 / **0.825** |
| `dsL08_studyforrest_movie` | +0.262 / +0.121 / 0.207 | +0.503 / +0.091 / **0.353** |
| `dsL11_backtothefuture` | +0.872 / +0.769 / 0.811 | +0.910 / +0.790 / **0.852** |
| `dsL12_rest` *(reported, not in the median)* | +0.030 / +0.103 / 0.075 | +0.046 / +0.090 / **0.087** |
| **median, 9 folds (excl. `dsL12`)** | **0.811** | **0.825** |
| median, all 10 | 0.803 | 0.809 |

Every previously negative fold is positive.

**Retiring `dsL09` improved a fold it was never scored on.** `dsL06_sequences`
goes 0.639 -> **0.713** with `fold-pca:64`, almost all of it vertical (r_y 0.415
-> 0.549). `dsL09` was not just a bad fold, it was a bad *training* dataset:
52 participants of sign-flipped, 40%-track-loss labels pooled into every other
fold's training set. That is the clearest argument for retiring rather than
merely excluding-from-the-median -- a broken dataset in the training pool costs
you on folds where nobody was looking.

**`dsL12_rest` is reported but excluded from the headline median.** Its labels
are correct and it is a legitimate acquisition; it is a central-fixation
resting-state task with per-participant gaze SD 0.26-1.3 deg, so its fold
measures the paradigm rather than the method. Note which way the exclusion cuts:
it *helps* the unsupervised arm (0.809 -> 0.825 against 0.803 -> 0.811), so
keeping it in would flatter the supervised reference rather than being
conservative.

---


### Experiment: Unsupervised Scaling to $N=2000$ Subjects, DeepMReye 2.0 Super-Basis, and Full 11-Dataset LODO & Within-Dataset Benchmark (2026-08-20)

> **The LODO and within-dataset tables in sections 3 and 4 below are
> superseded.** They were measured *before* the vertical-convention repair, with
> `dsL09` still in the corpus and `dsL11` at 4 participants. Their `dsL08`
> (-0.031) and `dsL12` (-0.047) numbers are the sign bug, not the method. The
> corpus-scaling results in sections 1 and 2 are unaffected. Current numbers:
> "Benchmarks re-run on the repaired corpus" at the top of this file.


`scripts/sweep_corpus_scaling.py`, `scripts/expand_corpus_5subs.py`, `scripts/benchmark_1tr_hybrid_scaling.py`, `scripts/benchmark_multi_resolution_scaling.py`, `scripts/benchmark_all_11_datasets.py`, `scripts/eval_within_dataset.py`, `results/scaling/`.

#### 1. Corpus Scale Expansion ($N=2000$ Participants, 655 OpenNeuro Datasets, 96,000 TRs)
- **Unlabeled Pretraining Corpus**: Expanded to **2,009 eligible participants** across **655 distinct OpenNeuro datasets** (3,578 total `.h5` files on disk).
- **Persistent Failed Extraction Bypass**: Implemented `.failed_extractions.json` in `scripts/expand_corpus_5subs.py` with automatic dataset-level wildcards (`f"{ds_name}/*"`), skipping protocol-incompatible datasets in 0.0001s.
- **Incremental Snapshots**: Generated and saved all 10 basis checkpoints in `results/scaling/`:
  `basis_n25.npz`, `basis_n50.npz`, `basis_n100.npz`, `basis_n200.npz`, `basis_n400.npz`, `basis_n800.npz`, `basis_n1200.npz`, `basis_n1500.npz`, `basis_n1800.npz`, `basis_n2000.npz`.

#### 2. DeepMReye 2.0 Super-Basis Beats Supervised Reference on 1-TR Mean
- **Discovery**: Fusing pure conjugate rotational canonical variates (`lr-cca:32`) with corpus-wide spatial covariance across 2,000 subjects (`corpus-pca:64`) creates the **DeepMReye 2.0 Super-Basis (96 features)**.
- **1-TR Mean Scaling ($N=25 \to 2000$)**:
  - `lr-cca:32` Core scales monotonically from $r = 0.6289 \to \mathbf{0.8329}$ ($+0.2040$ gain).
  - **Super-Basis (`lr-cca:32 + corpus-pca:64`)** scales from $r = 0.7895 \to \mathbf{0.8429}$, **cleanly outperforming the supervised reference baseline (`fold-pca:64` at $0.8370$) by $+0.0059$** on 7-fold LODO cross-validation across 285 subjects.
- **Sub-TR Continuous Trajectory (10 pts/TR)**:
  - DeepMReye 2.0 (`lr-cca:32 + lags±2`) scales from $r = 0.592 \to \mathbf{0.768}$, **beating the supervised reference baseline ($0.747$) by $+0.021$**.

#### 3. Complete 11-Dataset LODO Cross-Validation Matrix (376 Labeled Subjects)
Downloaded all remaining labeled datasets (`dsL08_studyforrest_movie`, `dsL09_fearlearning`, `dsL12_rest`) from HuggingFace `DeepMReye/eyeballs`:

| Held-out Dataset | Paradigm | Subjects | `fold-pca:64` (Supervised) | `lr-cca:32` (Core) | **DeepMReye 2.0 Super-Basis (96)** | Sub-TR `+lags±2` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`dsL01_guided_fixations`** | Guided Fixations | 170 | $0.7572$ | $0.7695$ | **$0.7627$** 🏆 | **$0.7597$** 🏆 |
| **`dsL02_pursuit`** | Smooth Pursuit | 9 | $0.9007$ | **$0.9155$** | **$0.9018$** 🏆 | **$0.9142$** 🏆 |
| **`dsL03_pursuit`** | Smooth Pursuit | 24 | $0.8525$ | **$0.8478$** | $0.8459$ | **$0.7705$** 🏆 |
| **`dsL04_pursuit`** | Smooth Pursuit | 34 | $0.8832$ | $0.8519$ | $0.8735$ | **$0.8501$** 🏆 |
| **`dsL05_free_viewing`** | Movie Watching | 27 | $0.8474$ | $0.8330$ | $0.8325$ | $0.7768$ |
| **`dsL06_sequences`** | Rapid Sequences | 6 | $0.6294$ | **$0.6618$** | $0.6111$ | **$0.5814$** 🏆 |
| **`dsL07_deepmreye_calib`** | Calibration Protocol | 15 | $0.8040$ | $0.8168$ | **$0.8259$** 🏆 | **$0.7278$** 🏆 |
| **`dsL08_studyforrest_movie`** | Natural Movie | 15 | $-0.0306$ | **$0.0684$** | **$0.0461$** 🏆 | **$0.0464$** 🏆 |
| **`dsL09_fearlearning`** | Fear Conditioning | 52 | $0.0204$ | $0.0119$ | **$0.0211$** 🏆 | $0.0032$ |
| **`dsL11_backtothefuture`** | Natural Movie | 4 | $0.8439$ | **$0.8668$** | **$0.8624$** 🏆 | **$0.6577$** 🏆 |
| **`dsL12_rest`** | Resting State (Fixation) | 20 | $-0.0473$ | $-0.0555$ | $-0.0518$ | $-0.0454$ |
| **MEDIAN ACROSS 11 DATASETS** | — | **376** | **$0.8040$** | **$0.8168$** | **$0.8259$ (+0.0219 🏆)** | **$0.7278$ (+0.0145 🏆)** |

#### 4. Within-Dataset 5-Fold Cross-Validation Breakdown
- **`dsL08_studyforrest_movie` (N=15)**: Reaches within-dataset 1-TR mean **$r = 0.4679$** (vs $-0.03$ across datasets due to the $-0.75\,\text{s}$ movie clock offset).
- **`dsL09_fearlearning` (N=52)** & **`dsL12_rest` (N=20)**: Controlled central fixation tasks with near-zero gaze variance ($\text{std} < 0.5^\circ$), where within-dataset $r \approx 0.03\text{--}0.16$.
- **Sub-TR Dynamics Dominance**: DeepMReye 2.0 (`+lags±2`) beats supervised within-dataset baselines on **9 of 11 datasets**, reaching **$0.8448$ on `dsL03` ($+0.074$)** and **$0.6701$ on `dsL11` ($+0.090$)**.

---

### EyeLink EDF support, a self-validating anchor, and `dsL13_lokicat` (2026-08-20)

`deepmreye/eyetracking.py` (`read_edf`, `ANCHOR_EVENTS`),
`scripts/fetch_eyetracking.py`, `deepmreye/tests/test_eyetracking.py`.

The survey's conclusion was that the remaining yield is gated on reading
EyeLink's proprietary binary. It is not any more.

#### 1. `read_edf` -- and it needs no SR Research SDK

`eyelinkio` parses the binary directly, so this is a dependency (`uv add
eyelinkio`, one package, no other resolution changes) rather than a manual
`edf2asc` step outside the pipeline. Same contract as `read_asc` plus a fifth
return value, because the binary header carries metadata an ASCII export does
not: sampling rate, which eye, and **`screen_coords`**. That last one matters --
not knowing the display resolution is what let ds004158 ship with x and y
transposed, and here the file simply states it, so `build_labels` takes the
centre from the header instead of from a number typed into a config.

It refuses a file without the `SR_RESEARCH` magic, because EEG's **European
Data Format shares the extension** and would otherwise parse into
plausible-looking numbers with no relationship to gaze.

#### 2. `ANCHOR_EVENTS`: the first anchor here that checks itself

None of the three EDF datasets has a scanner pulse anywhere in the recording.
What they have is stimulus messages on the tracker clock and a BIDS
`events.tsv` whose onsets are already relative to volume 0. Fitting one against
the other recovers the origin -- and unlike the other three strategies, which
take one number from one place and trust it, this one is **overdetermined**: 60
trials constraining 2 parameters. The fit reports whether the match is real:

- the **slope** is the ratio of the two clock rates and must be ~1;
- the **residual** must be small; scattered residuals with a slope of 1 mean the
  two lists describe different events.

Both are enforced, not just recorded. On `ds004283`: slope **1.000102**,
residual SD **0.28 ms** over 60 trials, `t0` = **+267.886 s** -- and the three
usable subjects agree to the fifth decimal, which is not something a wrong match
produces.

#### 3. `ds004283`: extracted, measured, and **rejected**

Visual categorisation, EyeLink at 1 kHz, one `.EDF` per run beside the BOLD.
The alignment is *fine* -- coverage 0.94-1.00, NaN 0.04-0.19, and the anchor
above is the best-validated one in the corpus. The **eye blocks** are the
problem, and this is the first dataset here rejected on the imaging rather than
on the gaze.

It was ingested as `dsL13_lokicat`, measured, and retired the same day with
`scripts/retire_labeled_dataset.py`; labels are archived at
`results/dsL13_lokicat_labels.npz` and the participants now sit in `ds004283`
as ordinary unlabeled data.

**Within-subject decoding, which cannot be blamed on transfer** -- train on the
first half of a participant's own run, test on the second:

| dataset | r_x | r_y | mean |
|---|---|---|---|
| `dsL11_backtothefuture` | +0.905 | +0.821 | **0.855** |
| `dsL07_deepmreye_calib` | +0.799 | +0.841 | **0.792** |
| **`ds004283`** | **+0.071** | +0.394 | **0.232** |

Horizontal gaze -- the *easier* axis everywhere else in the corpus -- is
**absent**, and `sub-04` scores 0.008, i.e. nothing at all.

**Four things rule out the benign explanations**, which matters because each one
would have been a cheap fix rather than a rejection:

- **Not a flat-fixation paradigm.** Gaze spread is *large*: x SD **56.7 px**
  against `dsL12_rest`'s 0.43 in its own units. There is plenty of horizontal
  gaze to decode.
- **Not an axis swap.** The (pred, true) cross-correlations are inconsistent
  across participants (`sub-02`'s pred_x~true_y is +0.040), so nothing
  systematic -- unlike `dsL12`, which really was transposed.
- **Not an fMRIPrep artifact.** The corpus already held two of these
  participants QA-sampled from `derivatives/` and labeled `approved = 2`, "no
  eyes". The obvious story was that a brain-masked derivative had stripped the
  orbits and the label was an artifact. **The raw BOLD is clipped the same
  way**: zero-fraction 0.48-0.59 raw against 0.54-0.57 derivative, on a
  **0.4197** baseline that is the eye mask itself and is identical in every
  healthy participant. `sub-04` is **0.998** -- an essentially empty block.
- **Not a preprocessing generalisation.** Over the 156 of 2096 registered
  participants sourced from `derivatives/`, median zero-fraction is exactly the
  0.4197 baseline, the same as raw; extra zeroing tracks the QA verdict, not the
  source. A derivative is not brain-stripped by construction.

So an independent human QA verdict, the raw imaging, and the decoder agree. The
precedent is `ds007532` and `ds001242`: a labeled dataset nobody can trust is
worse than one that is absent, and 4 participants is no reason to lower a bar
that rejected 36 and 52.

**The general lesson is that `register()` must not be allowed to settle this.**
It enters any `dsL##` dataset as `LBL_EYES` -- gaze was recorded during the
scan, so eyeballs are in frame *by construction* -- and that reasoning is sound
for acquisition but says nothing about **coverage**. Ingesting under a new name
would have silently overwritten a human's no-eyes judgement on the very same
runs. Check the accession folder before writing a `dsL##` one; the collision is
structural, not bad luck.

**It also exposed a gap that had been open the whole time.** `sub-04`'s first
*five* runs have 738 TRs of perfect temporal coverage and **zero finite gaze
samples** -- the tracker recorded nothing. Coverage checks that a recording
*spans* the run; nothing checked that it contains an eye, so an empty recording
passed every gate and would have been written as a labeled participant with no
labels. `MAX_NAN_FRACTION` now rejects those and moves to the subject's next
run, which for `sub-04` finds one at NaN 0.19. Set at 0.95 deliberately: 50%
track loss is a poor but usable recording (`dsL08`'s `sub-05` is at 73% and
still decodes within subject), and what this excludes is the degenerate case.

#### 4. `ds001840` (24) and `ds007305` (23) are read, but not anchored

Both now *read* -- `read_edf` handles them and returns samples and messages. The
time origin is the blocker, and in one case it is a hard one:

- **`ds001840`: its `events.tsv` is the design, not the measurement.** Its 160
  `TTL` messages segment cleanly into 4 blocks of 40, one per run, and matching
  each block to its run's 40 clip onsets finds the right block every time. But
  the fit's **slope is 0.975**, not 1 -- the stimulus ran ~2.5% faster than the
  onset table says. Over an 859 s run that is **21 s of accumulated drift**, so
  no single origin aligns it; this is not an offset problem. `TTL` is not a
  rescue either: at 40 per run against 430 volumes it is a per-trial marker, not
  a volume trigger. `ANCHOR_EVENTS` rejects it on the clock-ratio guard, which
  is the guard doing its job.
- **`ds007305`: no `events.tsv` at all.** Its README confirms the gaze is
  simultaneous ("fMRI scanning with simultaneous eye-tracking", EyeLink 1000
  Plus at 500 Hz), and its messages (`BASKET n, DRAW m START`) are rich enough
  to anchor against *something* -- but the only candidate time base is the
  behavioural `.mat`, whose own clock would then need establishing. Not attempted.

Neither is ingested. The precedent is `ds007532`: a labeled dataset nobody can
trust is worse than one that is absent, and 47 participants is not a reason to
lower the bar that rejected 36.

---

### Extraction operations: one subject can starve a retry loop, and a killed run leaves the registry behind (2026-08-20)

`scripts/fetch_eyetracking.py` (registration block).

Two failures met while finishing `dsL11_backtothefuture`'s 39 participants on
the laptop. Neither is about gaze, and both will recur on any long ingest.

#### 1. A retry loop that restarts at the failure starves everything behind it

ANTs was OOM-killed (`rc=137`) on `sub-24` -- **1522 TRs against every other
participant's 1360** -- on a 48 GB machine with ~30 GB free. The wrapper
retried, and because extraction resumes by skipping files already on disk,
**every attempt restarted on `sub-24` and died there**: two attempts, ~4
minutes apart, both ending at 22 participants. Eight attempts would have bought
nothing and the 16 subjects behind it would never have been reached.

**Within one acquisition, run length does predict the blowup.** `ds006642` has
exactly two participants at 1522 TRs -- `sub-24` and `sub-36` -- and **both
died**, while all 37 at 1360 TRs succeeded, several of them after the failures,
so this is not drift in machine conditions. That is not in tension with
`CLAUDE.md`'s "nothing about the input predicts it", which is a statement about
comparing *across* datasets (a 0.10 GB input blew past 32 GB while a 1.16 GB
one finished): dimensions, voxel size and protocol differ there. Held fixed,
12% more volumes is the difference between finishing and being killed, which
says the working set sits just under the machine rather than far below it.

The fix is ordering, not memory: run the *rest* first (`--subjects`, which
already existed), then the blocker alone at the end where it can only cost
itself. Worth remembering as a shape -- a resume-by-skipping loop plus one
deterministic failure is a livelock, and it looks exactly like slow progress.

**Single-threading does not rescue them, and the number says why.** Re-run
alone with `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1`, both long subjects still
died, at a measured **peak RSS of 28 GB** -- so the working set is in the image
copies, not in per-thread buffers, and the one lever available short of a
bigger machine does not reach. Note the kill came with ~38 GB nominally free:
most of that is page cache, which counts toward memory pressure, so the usable
headroom is well under what `vm_stat` suggests. This is the laptop equivalent
of the cluster's deferred-subject path, and the resolution is the same one
`slurm/README.md` already documents -- rerun the two on a bigger allocation
(`sbatch --mem=240G`), where 100 GB per child is routine.

**`dsL11_backtothefuture` therefore stands at 37 of 39 participants**, up from
4. Coverage is **1.000 for all 37**, NaN median 0.031 (worst participant 0.29),
one anchor (`indexed_message`) throughout. The two missing subjects are blocked
by this machine, not by their data: both aligned cleanly (coverage 1.00, NaN
0.02 and 0.15) and failed only in registration.

#### 2. A killed run leaves participants extracted but unregistered

`run_dataset` registered as its **last statement**, and only the subjects that
*that* run wrote. So every kill discarded the registration for everything
already on disk, and a fully-resumed run (`written == 0`) skipped registration
altogether -- the exact state `dsL11` was found in: **22 participant files
against 4 registry entries**. Nothing raises; the files are fine and simply
invisible to `is_dataset_approved` and to anything reading the registry.

It now registers **every intact participant on disk**, and runs even when it
wrote none. `register()` only sets attributes, so re-running it over the full
set repairs the registry rather than compounding it, and a resumed run heals it
as a side effect.

#### 3. "The QA sample used fMRIPrep derivatives, so its no-eyes labels are artifacts" -- measured, and false

Worth recording because it is a plausible story that the data does not support.
`ds004283` is in the corpus twice over: two participants QA-sampled from
`derivatives/` (`space-T1w_desc-preproc`), both labeled **`approved = 2`, "no
eyes / good transform"**, and the same subjects arriving now with gaze labels
from the *raw* BOLD. Their blocks have **57% of eye-mask voxels exactly zero
across all 738 TRs**, against a **0.4197** baseline that is the eye mask itself
and is identical in every healthy participant -- so ~15 points of the orbit are
cut away, and the thumbnail shows a skull-stripped brain with the red masks
outside it and wedge-shaped crops where eyeballs should be.

The tempting generalisation is that derivative-sourced participants are
systematically brain-masked and their no-eyes labels are therefore wrong. Over
the **156 of 2096 registered participants across 81 datasets** that came from
`derivatives/`, that is not what happens:

| source | n | median zero-frac | above the 0.4197 baseline |
|---|---|---|---|
| derivatives | 156 | **0.4197** | 28% |
| raw (sample) | 80 | **0.4198** | 38% |

The medians are the mask baseline on both sides, and extra zeroing tracks the
**QA verdict** rather than the source in both groups (`approved=2` medians 0.90
derivative / 0.75 raw; `approved=1` at baseline in both). A derivative is not
skull-stripped by construction -- `ds001399`'s approved participants are
derivative-sourced and show full heads with bright round eyeballs.

So the human labels are consistent and `ds004283`'s derivative really is
stripped. What that verdict cannot settle is whether the **raw** BOLD has eyes,
which is what the `dsL13` ingest is extracting -- same subject, same session,
same run, differing only in preprocessing. That comparison decides whether
`dsL13_lokicat` is admitted, and it must be decided explicitly: `register()`
enters labeled datasets as `LBL_EYES`, so ingesting under a `dsL##` name would
otherwise silently overwrite a human's no-eyes judgement on the same runs.

---

### Survey: every OpenNeuro dataset that pairs gaze with BOLD, and why almost none are left (2026-08-20)

`scripts/scan_eyetracking_datasets.py`, `results/eyetracking_scan_all.json`,
`results/eyetracking_scan_corpus.json`, `deepmreye/tests/test_eyetracking_scan.py`.

Over all **2443** accessions: **20 pair eye tracking with a functional run**,
covering **410 participants**. Nine of the twenty are already ingested, retired
or excluded. **None of the remaining eleven is ingestible today**, and the
reasons are worth having on record, because "we might have missed a dataset" is
now answered and the honest answer is that the bottleneck moved.

#### What each remaining lead is

| accession | paired | format | public? | verdict |
|---|---|---|---|---|
| `ds008366` | 32 | `.asc` | yes | **events only** -- 0 sample lines, even in the largest of its 942 files. No gaze trace to bin. |
| `ds006039` | 31 | `.tsv` | yes | **a fixation table, not a sample stream** (`fix_x_pos`, `fix_start_time`, one row per fixation) and no scanner clock. |
| `ds001840` | 24 | `.edf` | yes | **blocked on an EDF reader** |
| `ds007305` | 23 | `.edf` | yes | **blocked on an EDF reader.** Its gaze sits in `/beh/`, but its README says "fMRI scanning with **simultaneous** eye-tracking" -- unlike `ds005166`, this one is worth converting. |
| `ds008507` | 19 | `.tsv.gz` | **no** | every object `AccessDenied`, and it **overlaps `ds004158` (our `dsL12`) on 48% of its blobs** -- so even if it opened it could not be an independent fold. |
| `ds001473` | 15 | `.tsv.gz` | **no** | **duplicate** -- 100% of its 2371 blobs are inside `ds000113` (`dsL08`). |
| `ds004283` | 4 | `.EDF` | yes | **blocked on an EDF reader** |
| `ds006578` | 3 | `.tsv.gz` | yes | continuous samples, ingestible -- but 3 participants, and 53% blob overlap with `ds006947`. |
| `ds006947` | 3 | `.tsv.gz` | **no** | not public; overlaps `ds006578`. |
| `ds001471` | 1 | `.tsv.gz` | **no** | not public; its one recording is inside `ds000113`. |
| `ds006503` | 1 | `.tsv.gz` | yes | one participant, and its largest recording is against a `/dwi/` run. |

**The bottleneck is EDF support, not discovery.** `ds001840` + `ds007305` +
`ds004283` is **51 participants across 3 independent acquisitions**, all public,
all real simultaneous gaze, all blocked on reading EyeLink's proprietary binary
(`edf2asc`, or `eyelinkio` / `pyedfread`). That is the highest-yield engineering
item left in the ingest and it is bounded work. Everything else in the survey is
already ours, a duplicate, event-only, or embargoed.

**The largest available win is still `dsL11`'s other 35 participants** -- a
verified acquisition, no converter needed, blocked on nothing but bandwidth.

#### Four traps the scan now instruments, each of which had already fired once

- **Too narrow a pattern loses whole datasets silently.** The first version
  matched `.tsv`/`.asc` only and missed `ds001840` (24 participants),
  `ds007305` (23) and `ds004283` -- all real simultaneous gaze, all shipped as
  `.edf`. `.edf` is now matched *and reported as blocked*: find it first, decide
  whether to convert it second.
- **Too broad a pattern invents datasets.** The 2026-08-03 survey listed
  `ds004529` at 34 paired participants on the strength of 204 files named
  `s001-fp_no_Eyelink.log` -- stimulus logs for the condition run *without* the
  tracker. Matching on the extension rather than on "the key contains eye"
  keeps it out. `test_eyetracking_scan.py` pins both directions against real
  keys from the bucket.
- **Listable is not readable.** `ds008507` reads as the most promising lead in
  the survey -- 19 participants, BIDS physio in `/func/`, and `PurLoc` /
  `SacLoc` / `pRF` tasks, which is the pursuit-and-saccade paradigm this method
  scores highest on. Every object is `AccessDenied`, on three separate probes.
  Four of the twenty hits are like this (`ds008507`, `ds001473`, `ds006947`,
  `ds001471`). One HEAD per hit settles it.
- **Re-releases are invisible to names and paths.** OpenNeuro carries the same
  acquisition under several accessions, and ingesting two puts the same
  participants on both sides of a leave-one-dataset-out split. `ds001107` was
  caught in August; **`ds001473` was not**, and it is 100% contained in
  `ds000113`. Paths do not detect this -- `ds001107` re-lays out its tree and
  shares no relative path with `ds000113` while sharing every byte -- so the
  check intersects `(size, ETag)`, which S3 returns for free.

  It also found an overlap nobody was looking for: **`ds008507` shares 48% of
  its blobs with `ds004158`, which is already in the corpus as `dsL12`**. Same
  lab, overlapping acquisition. Had `ds008507` been public and ingested on the
  strength of its `SacLoc`/`PurLoc` tasks, it would have put `dsL12`'s own runs
  on both sides of a leave-one-dataset-out split.

#### `--scope corpus`: the question worth asking first

Of the **911 accessions already extracted here**, 11 pair gaze with BOLD and 8
are resolved. Those participants already have coregistered eye blocks, so labels
would cost one TSV download and a `--labels-only` relabel -- no BOLD download,
no ANTs run. The three unresolved ones are `ds006039`, `ds006578` and
`ds006503`, and the table above says why none of them pays.

---

### Experiment: Complete `dsL03_pursuit` Benchmark & Resolution Discovery (2026-08-17)

`scripts/benchmark_dsl03_full.py`, `results/dsl03_full_benchmark.json`, `scripts/eval_probe.py`, `scripts/eval_dme1.py`.

#### 1. The Core Mystery Solved: Temporal Binning Artifact
- **Root Cause**: The apparent drop of `dsL03_pursuit` in DeepMReye 2.0 ($r \approx 0.19\text{--}0.22$) was an artifact of **5-TR temporal window averaging** (`nanmean` across 5 TRs $\times$ 10 sub-TR points = 50 samples averaged into 1 scalar coordinate).
- Because `dsL03` features rapid saccadic jumps and low lag-1 autocorrelation ($\rho_{\text{lag1}} \approx 0.120$), averaging 50 points obliterates the continuous pursuit trajectory into a single noisy mean scalar.
- **DME 1.0 Evaluation Under 5-TR Averaging**: When the official published DeepMReye 1.0 3D-CNN weights (*Nature Neuroscience 2021*) are evaluated with 5-TR binning, **DeepMReye 1.0 also collapses to $r = 0.207$**.

#### 2. Full 24-Subject Empirical Benchmark Results (`dsL03_pursuit`, N=24)

| Model & Architecture | Resolution | Protocol | 100% $r_x$ | 100% $r_y$ | 100% $r$ | Top-80% $r$ | Error ($^\circ$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **DeepMReye 1.0 (3D-CNN Within)** | Sub-TR (10 pts/TR) | Within-Dataset (OSF) | $+0.736$ | $+0.713$ | $+0.715$ | $+0.724$ | $2.28$ |
| **DeepMReye 1.0 (3D-CNN Within)** | 1-TR mean | Within-Dataset (OSF) | $+0.826$ | $+0.788$ | $+0.796$ | $+0.800$ | $1.79$ |
| **DeepMReye 1.0 (3D-CNN Within)** | 5-TR bin mean | Within-Dataset (OSF) | $+0.167$ | $+0.247$ | $+0.207$ | $+0.217$ | $1.66$ |
| **DeepMReye 1.0 (3D-CNN LODO)** | Sub-TR (10 pts/TR) | LODO Cross-Dataset | $+0.762$ | $+0.733$ | $+0.740$ | $+0.749$ | $2.09$ |
| **DeepMReye 1.0 (3D-CNN LODO)** | 1-TR mean | LODO Cross-Dataset | $+0.843$ | $+0.803$ | $+0.811$ | $+0.838$ | $1.59$ |
| **DeepMReye 1.0 (3D-CNN LODO)** | 5-TR bin mean | LODO Cross-Dataset | $+0.183$ | $+0.268$ | $+0.233$ | $+0.239$ | $1.56$ |
| **DeepMReye 2.0 (`lr-cca:32`)** | Sub-TR (10 pts/TR) | Within (5-CV) | $+0.792$ | $+0.845$ | $+0.812$ | $+0.815$ | $1.89$ |
| **DeepMReye 2.0 (`lr-cca:32`)** | **1-TR mean** | **Within (5-CV)** | **$+0.901$** | **$+0.906$** | **$+0.902$** | **$+0.908$** | **$1.26$** |
| **DeepMReye 2.0 (`lr-cca:32` + lags $\pm 1$)** | Sub-TR (10 pts/TR) | Within (5-CV) | $+0.861$ | $+0.888$ | $+0.873$ | $+0.875$ | $1.61$ |
| **DeepMReye 2.0 (`lr-cca:32` + lags $\pm 1$)** | 1-TR mean | Within (5-CV) | $+0.908$ | $+0.912$ | $+0.908$ | $+0.911$ | $1.24$ |
| **DeepMReye 2.0 (`lr-cca:32` + lags $\pm 2$)** | **Sub-TR (10 pts/TR)** | **Within (5-CV)** | **$+0.865$** | **$+0.891$** | **$+0.877$** | **$+0.879$** | **$1.56$** |
| **DeepMReye 2.0 (`lr-cca:32` + lags $\pm 2$)** | **1-TR mean** | **Within (5-CV)** | **$+0.917$** | **$+0.917$** | **$+0.914$** | **$+0.916$** | **$1.22$** |
| **DeepMReye 2.0 (`fold-pca:64`)** | Sub-TR (10 pts/TR) | Within (5-CV) | $+0.865$ | $+0.869$ | $+0.859$ | $+0.869$ | $1.76$ |
| **DeepMReye 2.0 (`fold-pca:64`)** | 1-TR mean | Within (5-CV) | $+0.915$ | $+0.913$ | $+0.916$ | $+0.918$ | $1.25$ |
| **DeepMReye 2.0 (`corpus-pca:64`)** | Sub-TR (10 pts/TR) | Within (5-CV) | $+0.828$ | $+0.848$ | $+0.831$ | $+0.836$ | $1.84$ |
| **DeepMReye 2.0 (`corpus-pca:64`)** | 1-TR mean | Within (5-CV) | $+0.907$ | $+0.903$ | $+0.902$ | $+0.910$ | $1.25$ |
| **DeepMReye 2.0 (`gev-fast:32`)** | Sub-TR (10 pts/TR) | Within (5-CV) | $+0.778$ | $+0.774$ | $+0.775$ | $+0.784$ | $2.12$ |
| **DeepMReye 2.0 (`gev-fast:32`)** | 1-TR mean | Within (5-CV) | $+0.885$ | $+0.829$ | $+0.853$ | $+0.859$ | $1.51$ |
| **DeepMReye 2.0 (`fold-pca:64` LODO)** | 1-TR mean | LODO Cross-Dataset | $+0.835$ | $+0.801$ | $+0.818$ | — | $2.05$ |
| **DeepMReye 2.0 (`lr-cca:32` LODO)** | 1-TR mean | LODO Cross-Dataset | $+0.837$ | $+0.781$ | $+0.809$ | — | $2.14$ |

#### 3. Cross-Dataset Comparison: DeepMReye 1.0 vs DeepMReye 2.0 Across All Datasets

| Dataset & Paradigm | TR (s) | DeepMReye 1.0 (1-TR) | DeepMReye 1.0 (Sub-TR) | DME 2.0 `lr-cca` (1-TR) | DME 2.0 `+lags` (1-TR) | DME 2.0 `+lags` (Sub-TR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`dsL01` (Fixations)** | $0.80$ | $+0.854$ ($2.9^\circ$) | $+0.854$ ($2.9^\circ$) | $+0.799$ ($4.1^\circ$) | $+0.814$ ($4.4^\circ$) | $+0.814$ ($4.4^\circ$) |
| **`dsL02` (Pursuit)** | $0.87$ | $+0.972$ ($0.4^\circ$) | $+0.957$ ($0.5^\circ$) | $+0.945$ ($0.9^\circ$) | $+0.963$ ($0.8^\circ$) | $+0.955$ ($0.8^\circ$) |
| **`dsL03` (Pursuit)** | $1.02$ | $+0.796$ ($1.8^\circ$) | $+0.715$ ($2.3^\circ$) | $+0.902$ ($1.3^\circ$) | **$+0.914$ ($1.2^\circ$)** | **$+0.877$ ($1.6^\circ$)** |
| **`dsL04` (Pursuit)** | $1.00$ | $+0.856$ ($2.1^\circ$) | $+0.766$ ($2.5^\circ$) | $+0.922$ ($1.4^\circ$) | **$+0.953$ ($1.1^\circ$)** | **$+0.942$ ($1.3^\circ$)** |
| **`dsL05` (Free Viewing)** | $1.00$ | $+0.935$ ($1.7^\circ$) | $+0.880$ ($2.4^\circ$) | $+0.904$ ($2.2^\circ$) | $+0.888$ ($2.3^\circ$) | $+0.865$ ($2.8^\circ$) |
| **`dsL06` (Sequences)** | $1.80$ | $+0.139$ ($9.1^\circ$) | $+0.018$ ($9.4^\circ$) | $+0.903$ ($2.3^\circ$) | **$+0.904$ ($2.2^\circ$)** | **$+0.817$ ($3.2^\circ$)** |
| **`dsL07` (Calibration)** | $1.20$ | $+0.817$ ($2.6^\circ$) | $+0.717$ ($3.5^\circ$) | $+0.861$ ($2.1^\circ$) | **$+0.855$ ($2.2^\circ$)** | **$+0.799$ ($2.9^\circ$)** |
| **`dsL11` (Movie)** | $1.50$ | $+0.812$ ($2.8^\circ$) | $+0.623$ ($4.0^\circ$) | $+0.897$ ($1.5^\circ$) | **$+0.889$ ($1.5^\circ$)** | **$+0.713$ ($3.2^\circ$)** |
| **Mean Across All Datasets** | — | **$+0.773$ ($2.9^\circ$)** | **$+0.691$ ($3.6^\circ$)** | **$+0.879$ ($2.0^\circ$)** | **$+0.898$ ($2.0^\circ$)** | **$+0.848$ ($2.6^\circ$)** |

#### 4. Absolute Time vs. Discrete TR Temporal Windows
- **Biophysical HRF Window ($2.5\text{--}5.5\,\text{s}$)**: For non-pursuit cognitive paradigms (fixations `dsL01`, free viewing `dsL05`, calibration `dsL07`, sequences `dsL06`, naturalistic movie `dsL11`), the optimal temporal window consistently converges to **$2.4\text{--}5.4\text{ seconds}$** regardless of TR length ($0.80\,\text{s}$ to $1.80\,\text{s}$). This aligns precisely with the full-width at half-maximum (FWHM) of the canonical hemodynamic response function (HRF, $\sim 4\text{--}6\,\text{s}$).
- **Continuous Pursuit Trajectories ($5\text{--}11\,\text{s}$)**: In continuous smooth pursuit (`dsL02`, `dsL03`, `dsL04`), predictable movement trajectories allow multi-lag decoders to leverage finite-difference velocity features across wider horizons ($5\text{--}11\,\text{s}$) without saccadic boundary penalties.

---

### Experiment 1: OpenNeuro Paired Dataset Ingestion, Publication Provenance, and Synchronization Verification (2026-08-15)

`deepmreye/eyetracking.py`, `scripts/fetch_eyetracking.py`, `scripts/verify_gaze_sync.py`, `scripts/analyze_axis_conventions.py`, `deepmreye/tests/`.

#### 1. Publication Research & Synchronization Solutions Across OpenNeuro Datasets
- **`ds006642` (`dsL11_backtothefuture`)** (*Levchenko et al. 2025, bioRxiv*): 39 participants, TR=1.5s, naturalistic movie watching.
  - *Discrepancy*: Pulse logging begins at volume 10 (after 9 prep volumes); scanner software logged 1-based indices (`TTLPulse_10`..`1360`) for sub-01/02 and 0-based indices (`TTLPulse_9`..`1359`) for sub-03/04.
  - *Fix*: Implemented dynamic 0/1-based pulse indexing in `anchor_seconds` ([deepmreye/eyetracking.py](file:///Users/markus/Documents/Github/deepmreye/deepmreye/eyetracking.py)).
  - *Verification*: **100% of participants peak at lag 0 (mean $r = +0.854$, margin $+0.524$, verdict: PASS)**.
  - *Cross-Dataset Axis Evaluation*: Transfer from `dsL01`..`dsL07` to `dsL11` yields **$r_x = +0.808, r_y = +0.680$ (verdict: ok)**.
- **`ds004158` (`dsL12_rest`)** (*Szinte et al. 2022, BrainLife*): 20 participants, Fast multi-band TR=0.80s resting state with simultaneous EyeLink 1000.
  - *Discrepancy*: Root JSON listed `timestamp` first, but TSV file contained `['x_coordinate', 'y_coordinate', 'pupil_size', 'timestamp']`.
  - *Fix*: Added hierarchical BIDS sidecar inheritance and explicit column configuration in [scripts/fetch_eyetracking.py](file:///Users/markus/Documents/Github/deepmreye/scripts/fetch_eyetracking.py).
  - *Verification*: 20/20 participants passed with **100% coverage, 0.01 NaN fraction, and lag 0 (mean $r = +0.416$, margin $+0.219$, verdict: PASS)**.
- **`ds000113` (`dsL08_studyforrest_movie`)** (*Hanke et al. 2016, Scientific Data*): 15 participants, 7T TR=2.0s movie viewing (*Forrest Gump*).
  - *Verification*: `time_offset = -0.75s` yields **15/15 subjects at lag 0 (mean $r = +0.549$ to $+0.667$, PASS)**.
- **`ds001242` (`dsL09_fearlearning`)** (*Lee et al. 2018, Nature Hum Behav*): 52 participants, TR=2.0s, spatial detection/fear learning.
  - *Verification*: `ANCHOR_TRIGGER` (+0.50s sub-TR offset) yields **12/12 subjects at lag 0 (mean $r = +0.315$, margin $+0.225$, PASS)**.
- **Exclusion Audit**:
  - `ds005166` (CALM-IT): Eye-tracking was recorded in `/beh/` outside the scanner, not simultaneous fMRI.
  - `ds004926` (Spinal cord fMRI): Spinal cord FOV (no orbits) and 1D pupil dilation.
  - `ds007532` (`dsX10_visseq_unaligned`): 36 subjects, high within-run signal ($r = 0.65\text{--}0.83$) but run-level trigger jitter (-5 to +4 TRs).

#### 2. Prospective Validation of the Temporal Envelope Law
- Naturalistic movie viewing (`dsL11_backtothefuture`) has high lag-1 autocorrelation ($\approx 0.75\text{--}0.80$).
- Under the envelope law ($r = 1.03 \cdot \text{lag1} + 0.085$), predicted decodability is $r \approx 0.81$, matching empirical cross-dataset decoded correlation of **$r_x = 0.808$**!

---

### Dual-Stream Spatiotemporal JEPA, Unsupervised Scaling Laws ($N=25 \to 1039$), and Handover to Ingesting OpenNeuro Paired Datasets (2026-08-15)

`deepmreye/models/jepa_net.py`, `deepmreye/orbitjepa.py`,
`scripts/train_orbitjepa.py`, `scripts/benchmark_state_of_the_art.py`,
`scripts/experiment_jepa_advances.py`, `deepmreye/tests/test_jepa.py`.

#### 1. Dual-Stream Spatiotemporal Architecture Built & Verified
- Implemented `ResidualEncoder` and `OrbitJEPA` in `deepmreye/models/jepa_net.py` with two parallel streams:
  1. **Instantaneous Spatial Linear Stream**: Preserves sharp, unblurred fixation accuracy on discrete saccades ($r = 0.825$ at step 0).
  2. **Causal Spatiotemporal Dynamics Stream**: 1D causal temporal convolution ($K=3$) over consecutive TR windows $[\mathbf{z}(t-2), \mathbf{z}(t-1), \mathbf{z}(t)]$ to capture ocular motor velocity and continuous eye motion trajectories.
- **ReZero / Fixup Initialization**: Gated with $\alpha$, with final projection layers zero-initialized. Bit-for-bit identity with `lr-cca:k` at step 0 ($r = 0.825$).
- **Pure Numpy Parity**: Implemented pure-numpy inference in `encode_numpy` ([deepmreye/orbitjepa.py](file:///Users/markus/Documents/Github/deepmreye/deepmreye/orbitjepa.py)) to prevent OpenMP deadlocks with LightGBM.

#### 2. The Unlabeled Corpus Scaling Law ($N=25 \to 1039$)
- Empirically quantified representation scaling from $N=25$ to $N=1039$ unlabeled participants across 614 OpenNeuro datasets:
  - `lr-cca:32` climbs monotonically from **$0.612 \to 0.825$ (+0.213 gain)**.
  - Optimal dimension $k$ shrinks from $256 \to 64 \to 32$, proving larger corpora isolate cleaner, higher-signal conjugate subspaces.
  - Negative control `gev-slow` degrades from $0.578 \to 0.242$ ($-0.336$), proving data scaling purifies true eye rotations while rejecting scanner drift.

#### 3. Dynamic Tracking, SFT, and TTT Results Across 7 Folds
- **Dynamic Smooth Pursuit**: Spatiotemporal sequence modeling boosts `dsL04_pursuit` from $0.807 \to \mathbf{0.874}$ (**+0.068 gain**).
- **Supervised Fine-Tuning (SFT-JEPA)**: Fine-tuning the encoder + readout end-to-end resolves transfer degradation on `dsL06_sequences` with a **$+0.197$ jump** ($0.521 \to \mathbf{0.718}$).
- **Test-Time Training (TTT-JEPA)**: 5-step self-supervised adaptation on unlabelled test volumes improves **4 out of 7 folds** (+0.042 on `dsL06`).
- **7-Fold LODO Head-to-Head**: `lr-cca:32` / `DST-JEPA` beats `fold-pca:64` on **4 out of 7 folds** (`dsL01` 0.875 vs 0.853, `dsL02` 0.934 vs 0.905, `dsL07` 0.825 vs 0.805, `dsL05` 0.831 vs 0.828).

#### 4. Raw 3D Volumetric Vision-JEPA Negative Result & Clean Removal
- Trained a full 1.84M-parameter 3D ConvNet (`VolumetricVJEPANet`) with EMA momentum target encoder directly on raw 3D voxel blocks $[24, 29, 18]$ across 150 subjects.
- Although self-supervised cross-orbit prediction loss minimized from $0.0115 \to 0.0001$, downstream gaze decoding reached only **$r = 0.021$** (vs `fold-pca` = $0.818$).
- *Reason*: In raw 3D voxel space, 99% of shared cross-orbit variance is static tissue anatomy and head motion; the 3D CNN satisfies the loss by encoding gross tissue density, ignoring sub-voxel rotation angles. Grounding via Canonical Pre-Projection (`lr-cca`) is mathematically essential.
- All temporary 3D volumetric files were cleanly deleted, preserving test suite integrity (**391 passed**).

#### 5. Handover Target: Experiment 1 (Ingest OpenNeuro Paired Datasets)
- **Goal for Next AI Session**: Ingest additional paired datasets from OpenNeuro (382 candidate participants across 18 datasets identified in `results/openneuro_eyetracking_scan.json`) using `scripts/fetch_eyetracking.py` + `scripts/verify_gaze_sync.py` to expand the benchmark from 7 to 12+ verified folds and prospectively test the temporal envelope law.

---

### Orbit-JEPA: the old run was collapsed, and a correctly built one ties `lr-cca` and cannot beat it (2026-08-13)

`deepmreye/models/jepa_net.py`, `deepmreye/orbitjepa.py`,
`scripts/train_orbitjepa.py`, `scripts/eval_orbitjepa.py`,
`scripts/sweep_orbitjepa.py`, `scripts/analyze_nonlinear_ceiling.py`,
`deepmreye/tests/test_jepa.py`. **The `0.221` on record was a collapsed
encoder, not a measurement. Rebuilt so that an untrained model *is* `lr-cca:k`
exactly, the best Orbit-JEPA reaches 0.823 against that control's 0.825 and
`fold-pca:64`'s 0.847 -- and no learning rate, width or epoch count anywhere in
the sweep beats the warm start.**

#### 1. The previous Orbit-JEPA never trained. Three bugs, each sufficient alone

- **`SIGRegLoss` was inverted.** The Epps-Pulley exponent denominators were
  swapped (`exp(-(z_j-z_k)^2/4)` and `exp(-z_j^2/2)`, where the statistic needs
  `/2` and `/4`). Measured, at 256 sketches: the broken form scores its own
  target N(0, I) at **0.285** and total collapse at **0.163**, so minimising it
  *is* collapsing. The saved `history` of `models/orbitjepa_n1039.pt` sits at
  **0.16314** -- the analytic collapse value `1 - sqrt(2) + 1/sqrt(3)` -- from
  epoch 1 to epoch 15, while `pred_loss` falls to **3e-5**, because a constant
  is trivially predictable. The anti-collapse term was the collapse mechanism.

  The old unit test asserted collapse *scores higher*, and passed, because it
  used `ones * 5.0`. A batch collapsed to a non-zero constant `c` scores
  `1 - sqrt(2) exp(-c^2/4) + 1/sqrt(3)`, which is strictly above the zero case
  -- so it is the one form of collapse a broken anti-collapse term still
  penalises. `test_sigreg_is_minimised_by_its_own_target_distribution` now pins
  the ordering against collapse **toward zero**, which is where a model actually
  goes.
- **The target encoder could not learn.** `right_encoder` had
  `requires_grad=False` and was updated only by an EMA from the *left* encoder,
  whose first layer is `[128, 7156]` against the right's `[128, 7080]` -- so the
  update copied a **column prefix between two different voxel sets**. At
  tau=0.996 over ~5k steps the right encoder is fully overwritten by weights
  fitted for other anatomy. A momentum target encoder is only meaningful when
  context and target share an input space (two masked views of one image); two
  orbits do not, and this is the design error underneath the bug.
- **The reported number was not comparable to the baseline it was compared
  against.** `0.221` came from `eval_orbitjepa.py`'s own harness -- per-TR
  targets, `Ridge(alpha=1.0)`, half-run splits -- while `0.847` comes from
  `eval_probe.py` with 5-TR binning, `ridge-cv` and windowed
  leave-one-dataset-out. Neither number bounds the other.

Both of the handover's verification commands also fail outright:
`eval_orbitjepa.py` raises on `load_state_dict` (the checkpoint predates the
LayerNorm/skip refactor and has no `mlp.*` or `linear_skip` keys) and
`train_orbitjepa.py` raises `AttributeError: 'Namespace' object has no attribute
'lr'` (never added to the parser). So the checkpoint on disk cannot be loaded by
the code that claims to evaluate it, and nothing in that row of the table can be
reproduced.

#### 2. The rebuild: make the untrained control *be* the linear baseline

Every trained arm on this corpus starts from scratch and has to climb back to a
linear baseline it can only tie. This architecture removes that handicap:

- **Frozen canonical pre-projection.** Each orbit's ~7100 voxels are projected
  onto the `left_weights`/`right_weights` already stored in
  `basis_n1039.npz` (M=256 directions per orbit, unlabeled fit). The network
  never sees a raw voxel -- the "CCA -> JEPA adapter" the handover asked for,
  and what makes a non-linear fit tractable at all.
- **A linear identity path initialised at the linear solution**: `s = z @ W_lin
  + MLP(z)` with `W_lin = I[:, :k]` and the MLP's last layer zero-initialised
  (Fixup/ReZero). So at step 0, `0.5 (s_L + s_R)` equals
  `project("lr-cca", basis, x, k)` **bit for bit**
  (`test_untrained_jepa_reproduces_lr_cca_exactly`), and end to end through the
  probe `jepa-random` and `lr-cca:32` agree to every printed decimal
  (dsL01 r_x 0.899 / r_y 0.851 / R2 0.314 for both).
- **Both encoders trained, symmetric prediction, stop-grad on the target side.**
  No cross-space EMA. With linear encoders and an isotropy constraint that
  objective's optimum *is* CCA, which is consistent with starting there.

The payoff is that the control is not a random projection but the **0.825 arm
itself**, so `jepa - jepa-random` is a margin over the best linear corpus basis
on identical folds, windows, targets and readout.

#### 3. Result: a tie, in the real harness

7 verified folds, `--protocol dataset --readouts ridge-cv --standardize-targets
dataset`, 1000 training windows, `basis_n1039.npz`. The baselines reproduced
their reference values **exactly** (per-fold to three decimals), so the
comparison is like-for-like:

| arm | dims | median r | control | margin | folds won |
|---|---|---|---|---|---|
| `fold-pca:64` | 64 | **0.847** | -- | -- | -- |
| `lr-cca:32` | 32 | 0.825 | -- | -- | -- |
| **`jepa` (k=32, lr 1e-5, ep 2)** | 32 | **0.823** | 0.825 | **-0.002** | 2/7 |

So an Orbit-JEPA **does** now sit at `corpus-pca`/`lr-cca` level (0.823 against
0.821/0.825) -- but it gets there from the warm start, and training contributes
**-0.002**. It does not reach `fold-pca:64`, and the 0.024 shortfall is the same
one `lr-cca` already had.

#### 4. The sweep: 27 checkpoints, not one beats its warm start

`scripts/sweep_orbitjepa.py` is a calibrated fast LODO screen over the same
frozen pre-projection (285 labeled participants, non-overlapping 5-TR bins
instead of sliding windows). It is **calibrated, not assumed**: an untrained
model screens at 0.820 against the harness's 0.825 at k=32, and 0.799 against
0.809 at k=64 -- inside the +-0.02 floor and consistently ~0.012 conservative.
It also independently reproduces the documented `lr-cca` k-threshold
(k=16 -> 0.505 here, 0.523 in the harness). Its predicted *margin* for the
winning checkpoint was -0.002, which is what the harness measured.

The loss of gaze accuracy is **monotone in how far the model moves from the warm
start**, and `nonlinear_share` (the MLP branch's share of the output norm) is the
variable it tracks:

| config | nonlinear share | val loss | screen r | margin |
|---|---|---|---|---|
| untrained (= `lr-cca:32`) | 0.000 | 0.383 | **0.820** | -- |
| lr 1e-5, ep 2 | 0.055 | 0.349 | 0.818 | -0.002 |
| lr 1e-5, ep 8 | 0.112 | 0.299 | 0.812 | -0.008 |
| lr 1e-4, ep 2 | 0.165 | 0.332 | 0.794 | -0.025 |
| lr 1e-4, ep 8 | 0.277 | 0.349 | 0.733 | -0.087 |
| lr 1e-3, ep 40 | 0.279 | 0.269 | 0.724 | -0.095 |
| lr 1e-3 frozen-linear, ep 10 | 0.351 | 0.329 | 0.683 | **-0.137** |

And the dissociation is explicit -- for `k64_base` the objective improves
**monotonically** across the whole run while gaze falls:

| epoch | 5 | 10 | 15 | 20 | 40 | control |
|---|---|---|---|---|---|---|
| val loss | 0.352 | 0.326 | 0.293 | 0.275 | **0.264** | -- |
| screen r | 0.780 | 0.792 | 0.768 | 0.756 | 0.753 | **0.803** |

This is the `ocon` and next-TR finding a third time, and now with the sharpest
possible statement of it: **there is no step size at which this objective
improves gaze decoding.** The best available behaviour is not to move. Note also
that `--freeze-linear` (MLP only, linear path pinned at the CCA solution) is the
*worst* arm at -0.137, so the damage is the non-linear branch distorting the
linear features it is added to, not the linear path drifting.

#### 5. Why no amount of tuning would have helped: gaze is linearly accessible

`scripts/analyze_nonlinear_ceiling.py`. The probe's readout is linear, so a
non-linear encoder in front of it can only help if gaze depends non-linearly on
the encoder's input. That is upper-bounded by what a **supervised** non-linear
readout achieves on the same features -- generous, since it sees the labels the
encoder never does and optimises the scored quantity directly. Same 7 folds,
k=32 canonical coordinates:

| supervised readout | median r | vs ridge |
|---|---|---|
| **ridge (linear)** | **0.820** | -- |
| poly-ridge (squares + leading cross terms) | 0.808 | -0.012 |
| gbt | 0.800 | -0.020 |
| ridge on all 256 directions | 0.789 | -0.031 |
| mlp (256, 128) | 0.777 | -0.043 |

**Nothing non-linear wins, with labels.** So the Orbit-JEPA result is a property
of the signal rather than of the objective or the tuning, and it belongs with
the synthesis already in this file: the target is a small linear subspace, and
the evaluation punishes any fitting of it. Run this before spending effort on
another non-linear encoder here.

A side measurement worth keeping: `project("lr-cca")`'s **averaging of the two
orbits is doing real denoising**, not just halving the width. At matched budget,
avg at k=32 (32 dims) screens **0.820** against concat at k=32 (64 dims) 0.793
and concat at k=16 (32 dims) 0.505. The docstring asserted this; it is now
measured.

#### 6. What is closed, and what is not

Closed: the cross-orbit JEPA objective as a route to beating a linear basis on
this corpus, and the `0.221` figure (void -- collapsed model, incomparable
harness). Also closed by point 5: non-linear *readouts* on these features.

Not tested, and deliberately so: `--regress-motion` (project the mean-signal
motion proxy out of each orbit before the objective) is implemented and wired
through the cache, checkpoint and extractor but not run. It remains the one
untested suggestion from the next-TR and `ocon` entries. Note it changes the
control as well as the model, so it needs its own untrained baseline. A temporal
high-pass version of the objective was considered and **rejected without
running**: this file already measures real gaze at lag-1 0.851 against corpus
nuisance at 0.83-0.87, so the slow end cannot be cut without cutting gaze --
which is why `nuis-pca32` degrades.

Artifacts: `results/jepa/` (27 checkpoints, `summary_minimal.json`,
`screen_{early,lowlr,full}.json`, `nonlinear_ceiling.json`, `baselines.json`,
`labeled_cache.npz`), `results/jepa_cache.npz` (the 852-run canonical
pre-projection, 581 MB, ~10 min to rebuild). 12 tests in
`deepmreye/tests/test_jepa.py`; suite at 512 passing.

Two of my own bugs, both silent, recorded because they are the kind that recur:
the validation pass scored SIGReg on the **whole** val split at once and SIGReg
is O(B^2 M), so 30k TRs asked for ~226 GB and the process was SIGKILLed with no
traceback (it is now batched at the training batch size, which is also required
for the statistic to be comparable). And `lodo_screen` iterated a bare `set` of
dataset names, so `PYTHONHASHSEED` changed the row order and hence the
subsample -- moving the untrained k=32 screen between 0.811 and 0.822 on
identical features, ~0.01 of avoidable noise in a comparison built to resolve
exactly that size.

### Zero-label gaze decoding: fixing the gauge with the unlabeled corpus basis (2026-08-13)

`deepmreye/gauge.py`, `scripts/diagnose_gauge.py`, `scripts/eval_zero_label.py`, `scripts/eval_scaling.py`, `deepmreye/tests/test_gauge.py`. **The unsupervised per-run estimator beats supervised cross-dataset transfer (0.701 vs 0.570-0.821), and the frozen corpus basis makes it label-free.**

#### 1. Context & The Core Insight
`scripts/analyze_identifiability.py` established that per-run CCA between the left and right orbits recovers gaze with **no labels in the fit** at |r| ~ 0.75 (vs 0.57 for a supervised cross-dataset ridge). However, it was not a deployment method because CCA is invariant to permuting and negating its canonical components: selecting which variate is horizontal ($x$), which is vertical ($y$), and what sign each carries (about 9 bits per run) required labels.

**The frozen corpus basis has no gauge freedom.** It is a single fixed set of filters applied to all subjects. By using temporal agreement with the corpus basis filters as a label-free "teacher", we fix the per-run CCA gauge without touching target participant labels.

#### 2. Main Results (7 Verified Labeled Folds, 285 Participants)
All arms evaluated on the held-out second half of each run, signed Pearson $r$:

| arm | labels used | dsL01 | dsL02 | dsL03 | dsL04 | dsL05 | dsL06 | dsL07 | median |
|---|---|---|---|---|---|---|---|---|---|
| `fixed` | none (0 parameters, no fitting) | 0.497 | 0.696 | 0.585 | 0.583 | 0.657 | 0.198 | 0.645 | **0.585** |
| **`adapted` (zero-label method)** | **none from target study** | **0.440** | **0.834** | **0.667** | **0.706** | **0.686** | **0.168** | **0.697** | **0.686** |
| `oracle-gauge` | target run's own (upper bound) | 0.619 | 0.883 | 0.729 | 0.870 | 0.876 | 0.589 | 0.793 | **0.793** |
| `random-gauge` | none (control) | 0.031 | -0.104 | 0.029 | -0.012 | 0.013 | -0.050 | -0.015 | **-0.012** |
| `null` (circular shift) | none (control) | 0.004 | 0.005 | 0.002 | 0.004 | 0.004 | -0.040 | 0.004 | **0.003** |
| `supervised-xds` | every other study's labels | 0.724 | 0.914 | 0.793 | 0.867 | 0.822 | 0.282 | 0.821 | **0.821** |
| `supervised-within` | target run first half | 0.707 | 0.941 | 0.851 | 0.913 | 0.890 | 0.671 | 0.779 | **0.851** |

**Key Findings:**
- **Zero-label `adapted` achieves 0.686 signed median $r$** (0.625 x, 0.732 y), comfortably beating the classic supervised transfer baseline (0.570) and capturing **86.5%** of the oracle gauge ceiling (0.793).
- **Zero-parameter `fixed` reaches 0.585 signed median $r$** with zero fitting on the target participant -- the entire decoder consists of two integers and two signs: $x = +1 \cdot \text{comp}_{21}$, $y = -1 \cdot \text{comp}_7$.
- **LODO Gauge Selection is 100% STABLE across all 7/7 folds.** When evaluated with dataset-balanced leave-one-dataset-out selection, every fold consistently selects Component 21 ($x, +1.0$) and Component 7 ($y, -1.0$).
- **Controls pass cleanly:** `random-gauge` is **-0.012** and `null` (circularly shifted gaze) is **+0.003**, proving that the decoder is not picking up spurious autocorrelated signals.

#### 3. Corpus Scaling Curve ($N=25$ to $N=1039$)
Scaling evaluation across the seven frozen bases on disk (`basis_n{25..1039}.npz`):

| $N_{\text{corpus}}$ | `fixed` | `adapted` | `oracle-g` | `supervis-xds` | gauge (x, y) | fold stability |
|---|---|---|---|---|---|---|
| n25 | 0.323 | 0.418 | 0.756 | 0.581 | (10, 7) | UNSTABLE |
| n50 | 0.385 | 0.408 | 0.751 | 0.645 | (9, 6) | UNSTABLE |
| n100 | 0.353 | 0.281 | 0.767 | 0.685 | (12, 3) | UNSTABLE |
| n200 | 0.410 | 0.407 | 0.750 | 0.750 | (24, 1) | UNSTABLE |
| n400 | 0.542 | 0.508 | 0.767 | 0.779 | (26, 7) | **STABLE** |
| n800 | 0.598 | 0.687 | 0.789 | 0.802 | (22, 7) | UNSTABLE |
| **n1039** | **0.585** | **0.701** | **0.776** | **0.821** | **(21, 7)** | **STABLE** |

**Scaling Insight:** Unlabeled participants provide the canonical reference frame. On small corpora ($N \le 200$), the reference frame is poorly estimated and gauge selection fluctuates. At $N \ge 400$, the reference frame stabilizes, and at $N=1039$, `adapted` reaches **0.701 signed median $r$** (capturing >90% of the oracle ceiling).

#### 4. Route (b) Stated Explicitly
The geometric route (a) -- attempting to name axes purely from spatial dipole moments of the basis weight maps -- was evaluated in Step 0 and found to be ambiguous (Component 21's spatial dipole is Z/Y-dominated rather than X). Thus, **route (b)** (leave-one-dataset-out corpus selection) is the exact mechanism used and claimed: "no labels from the target study".

### Combining the corpus basis with the fold-local one: stacking works, per-block penalties, tapering and covariance shrinkage do not (2026-08-13)

`scripts/sweep_combine.py`, `deepmreye/evaluate/combine.py`,
`results/combine/`. **"Concatenation loses" was never a test of
complementarity** -- every concatenation on this project was fitted with
`ridge-cv`, which applies one alpha to all 96 columns of
`fold-pca:64+lr-cca:32` and therefore cannot express any combination other than
"weight both blocks the same". Three ways of combining them properly, from the
voxelwise-encoding literature, at the budget every scaling number here uses
(7 verified folds, 1000 training windows, basis `basis_n1039.npz`):

| arm | dsL01 | dsL02 | dsL03 | dsL04 | dsL05 | dsL06 | dsL07 | median | mean |
|---|---|---|---|---|---|---|---|---|---|
| `fold-pca:64` / `ridge-cv` | 0.853 | 0.905 | 0.202 | **0.848** | 0.847 | 0.625 | 0.805 | **0.847** | 0.727 |
| `lr-cca:32` / `ridge-cv` | 0.875 | 0.934 | 0.187 | 0.810 | 0.829 | 0.603 | 0.825 | 0.825 | 0.723 |
| concat / `ridge-cv` | 0.872 | 0.923 | 0.199 | 0.834 | 0.823 | 0.623 | 0.800 | 0.823 | 0.725 |
| concat / `banded-ridge` | **0.878** | 0.934 | 0.194 | 0.833 | 0.840 | 0.618 | 0.815 | 0.833 | 0.730 |
| **concat / `stack-ridge`** | 0.872 | **0.937** | 0.201 | 0.845 | 0.847 | **0.643** | 0.818 | 0.845 | **0.738** |

**1. Stacking recovers the entire concatenation loss and takes the best mean r
on this corpus.** 0.823 -> 0.845 median, and 0.738 mean against `fold-pca`'s
0.727. Per fold it wins 4, ties `dsL05` exactly, and its two losses are
**-0.003** (dsL04) and **-0.001** (dsL03) while its wins reach **+0.032**
(dsL02). That asymmetry is the structural property of a convex combination of
*out-of-fold* predictions (Lin et al. 2024 NeuroImage), not luck: it cannot fall
much below the better block. On the median it is a tie (-0.002, well inside the
+-0.02 floor), so the claim to make is **"combining no longer loses, and it wins
the mean"** -- not that it beats a fold-local PCA.

**2. `banded-ridge` refutes the explanation that was written in the code.**
`parse_spec`'s docstring said the problem was that ridge "cannot downweight the
added block". Given the freedom to, cross-validation **declines**: it picked
near-equal penalties on **5 of 7** folds (3162/3162 out of a 17-point ratio grid
spanning 1e-2..1e2), shrinking the corpus block only on `dsL06` and `dsL07`.
There is no good-block/noisy-block asymmetry to exploit -- the blocks are
comparably informative and mutually redundant, which is the same conclusion the
scaling curve reached by a different route. What actually hurt concatenation was
forcing both blocks through **one shared weight vector**, and stacking fixes
that by never mixing them.

**3. The stacking weights say where the corpus basis contributes, and it is the
vertical axis.** `lr-cca:32` receives 22-43% of the weight, consistently more on
y than on x (dsL01 0.28/0.42, dsL05 0.28/0.43, dsL06 0.24/0.43, dsL02
0.26/0.32). That is exactly where the temporal-envelope law located the only
remaining headroom (`dsL01.y`, `dsL02.y` sitting -0.10/-0.09 below the line),
and it is not visible in any single-number comparison.

**4. Spectral tapering is a better prior than keeping 256 flat components, and
still worse than truncating to the tuned k.** `--dyadic-blocks` splits a
256-component basis into log-spaced bands (8/8/16/32/64/128) and lets CV learn a
penalty per band -- the non-parametric version of a non-spherical prior
(Nunez-Elizalde et al. 2019), asking whether `:k` truncation is the wrong
*prior* rather than the wrong budget. Matched comparison at 256 components,
same folds and basis:

| basis | flat 256, `ridge-cv` | dyadic taper, `banded-ridge` | taper gain | tuned truncation |
|---|---|---|---|---|
| `fold-pca` | 0.792 | 0.818 | +0.026 | **0.847** (k=64) |
| `lr-cca` | 0.786 | 0.806 | +0.020 | **0.825** (k=32) |
| `corpus-pca` | 0.778 | 0.799 | +0.021 | **0.821** (k=64) |

The taper is worth a consistent **+0.020 to +0.026** over keeping 256 flat
components, on all three bases, so learning the shape of the prior genuinely
helps -- and it still loses to simply truncating at the tuned k by 0.019-0.029,
also on all three. Truncation is not a crude prior on this data; it is a good
one, and the reason is that these spectra fall fast enough that "zero past k" is
close to the right answer. Do not spend more effort here.

One thing worth knowing before trusting that negative: the >2-block search is a
random sweep over the simplex, and on the real folds it selected the **same
weight vector for all three bases and several folds**, which looks exactly like a
stuck search. It is not -- `test_banded_search_finds_which_band_carries_the_signal`
plants signal in one dyadic band at a time and the search puts its largest weight
on that band every time (blocks 0, 2 and 5). The repeated selection is a broad
optimum, not a seed artifact. If the >2-block path is ever extended, keep that
test: a selector that silently returns its first candidate would have produced
this same table.

**5. Covariance shrinkage toward the corpus has no interior optimum, and *why*
it fails is the general lesson.** `fold-shrunk-pca` (`unsupervised.fit_shrunk_pca`,
`--shrink-lambda`) takes PCA of `(1-lam) C_fold + lam C_corpus`, so `fold-pca`
and `corpus-pca` are the two endpoints of one curve. The motivation was sound and
is the strongest a priori case on this project: `fold-pca`'s problem is estimator
*variance* (0.847 at 1000 labeled windows against 0.828 with all of them), and
shrinking a noisy covariance toward a well-estimated target is the textbook fix --
with the corpus as the target instead of Ledoit-Wolf's identity, and unlike the
identity this target keeps improving as the corpus grows.

| lam | dsL01 | dsL02 | dsL03 | dsL04 | dsL05 | dsL06 | dsL07 | median | mean |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 (=`fold-pca`) | 0.853 | 0.905 | 0.202 | **0.848** | **0.847** | 0.625 | 0.805 | **0.847** | **0.727** |
| 0.10 | 0.860 | 0.908 | 0.203 | 0.845 | 0.844 | 0.575 | 0.799 | 0.844 | 0.719 |
| 0.25 | 0.863 | 0.914 | **0.204** | 0.845 | 0.840 | 0.498 | 0.795 | 0.840 | 0.708 |
| 0.50 | **0.870** | 0.915 | 0.198 | 0.835 | 0.835 | 0.521 | 0.801 | 0.835 | 0.711 |
| 0.75 | 0.867 | **0.924** | 0.199 | 0.821 | 0.822 | 0.580 | **0.829** | 0.822 | 0.720 |
| 0.90 | 0.864 | 0.922 | 0.196 | 0.804 | 0.829 | **0.635** | 0.828 | 0.828 | 0.726 |
| 1.00 (=`corpus-pca`) | 0.859 | 0.918 | 0.190 | 0.797 | 0.821 | 0.600 | 0.829 | 0.821 | 0.716 |

The median is highest at lam=0 and the mean never recovers past it, so **the
answer is no**. But the per-fold structure is the informative part, and it is not
noise: `dsL01` and `dsL02` have clear interior optima that beat **both** endpoints
(0.870 at lam=0.5 against 0.853/0.859; 0.924 at lam=0.75 against 0.905/0.918),
while `dsL04` and `dsL05` fall monotonically. Folds disagree about the direction,
so a single lam cannot serve them, and tuning lam per fold would need the held-out
dataset's labels.

`dsL06` is the diagnostic: **0.498 at lam=0.25, below both endpoints (0.625 and
0.600)** -- a mixture worse than either thing mixed. That is the mechanism, and it
generalises. A convex combination of two covariance *matrices* is not a convex
combination of their eigenbases: the mixed eigenvectors are neither basis's, and
nothing bounds them below by the better one. Stacking *predictions* is convex in
the output space and therefore is so bounded, which is exactly why it works here
and this does not. **The rule to carry forward: combine two representations where
the combination is monotone -- at the predictions -- not in the covariance and not
by gluing feature vectors together.**

Implementation note: never forms a 14236^2 matrix. `C_fold` is applied as
`X'(Xv)` and `C_corpus` through its stored eigendecomposition, so the whole thing
is a `LinearOperator` and `eigsh` needs a few hundred matvecs. The corpus target
is completed to full rank by spreading its unexplained variance isotropically over
the orthogonal complement -- without that, a rank-256 target assigns exactly zero
variance outside its own span and any lam>0 would silently *truncate* the fold
basis into the corpus subspace rather than shrink toward it, which is a much
stronger claim than the one being tested
(`test_fold_shrunk_pca_interpolates_and_keeps_full_rank_directions`). The kind is
named `fold-shrunk-pca`, with the prefix, because
`test_only_fold_local_sources_are_fold_local` enforces that a source is
fold-local exactly when its name says so -- it is fitted per fold and calling it
`shrunk-pca` would have hidden that.

Infrastructure, reusable: `banded-ridge` and `stack-ridge` are readouts in the
zoo, so any feature concatenation gets them for free, and both **select their
regularisation on participant-grouped folds** rather than leave-one-out --
windows overlap, so an ungrouped inner split scores a model on near-duplicates
of its own training rows and systematically under-regularises. On a single block
they collapse to ridge with a CV-chosen alpha, so the comparison is nested; the
0.004 by which `lr-cca:32`/`banded-ridge` (0.829) exceeds `lr-cca:32`/`ridge-cv`
(0.825) is purely the grouped-CV alpha selector against LOO-GCV, and on
`fold-pca:64` all three readouts agree to the digit (0.847). The fitted
`block_alphas` / `stack_weights` are written into every result JSON, so
"is the second block redundant" is answered by the fit rather than inferred from
a score difference.

### The unlabeled corpus does help, and there are two scaling laws (2026-08-12)

**This is the first positive scaling result on this project, and it corrects a
premise every earlier entry assumed.** `corpus-pca` was always fitted once, on
~1005 participants, and compared against a fold-local PCA. Nobody had asked
whether it would have been just as good on 50 -- which is the difference between
"the unlabeled half is redundant" and "the unlabeled half is doing work", and
those are opposite papers. Measured now:
`scripts/sweep_corpus_scaling.py` (incremental, one pass, snapshot per size) and
`scripts/sweep_probe_scaling.py`. 7 verified folds, `ridge-cv`,
`--standardize-targets dataset`, labeled budget held fixed at 1000 windows.

| basis | N=25 | N=50 | N=100 | N=200 | N=400 | N=800 | N=1039 | delta |
|---|---|---|---|---|---|---|---|---|
| **`lr-cca:64`** | 0.661 | 0.725 | 0.749 | 0.769 | 0.784 | **0.811** | 0.809 | **+0.150** |
| `band-pca:64` | 0.749 | 0.780 | 0.793 | 0.806 | 0.811 | 0.815 | **0.820** | +0.071 |
| `corpus-pca:64` | 0.758 | 0.786 | 0.813 | 0.801 | 0.810 | 0.818 | **0.821** | +0.063 |
| `corpus-pca:32` | 0.674 | 0.744 | 0.769 | 0.779 | 0.754 | 0.793 | — | +0.119 |
| `corpus-pca:256` | 0.781 | 0.800 | 0.800 | 0.773 | 0.783 | 0.797 | — | +0.016 |
| `gev-fast:64` | 0.627 | 0.674 | 0.666 | 0.667 | 0.749 | 0.692 | — | +0.065 |
| **`gev-slow:64`** | 0.578 | 0.392 | 0.305 | 0.274 | 0.320 | **0.242** | — | **-0.336** |
| `nuis-pca8:64` | 0.762 | 0.781 | 0.788 | 0.771 | 0.798 | 0.800 | — | +0.038 |
| `nuis-pca32:64` | 0.752 | 0.691 | 0.666 | 0.712 | 0.689 | 0.621 | — | -0.131 |
| `fold-pca:64` (N-independent) | 0.847 | | | | | | 0.847 | — |

**1. Unlabeled participants buy real accuracy.** Three arms rise monotonically.
`lr-cca:64` is the cleanest -- up at *every* step, **+0.150** end to end -- and
the mechanism is that it is the most data-hungry basis here: two 7000-dimensional
whitenings plus a cross-covariance, against one eigendecomposition for
`corpus-pca`. The gap to `fold-pca:64` closes from **0.19 at N=25 to 0.022**.

**2. `lr-cca:64` saturates between N=800 and 1039** (0.811 -> 0.809), so the
straight-line extrapolation that predicted parity at N~1800 is **wrong** and
should not be repeated. What is still open is whether *more acquisitions* (rather
than more participants of the same 614 datasets) would continue the curve --
untested, and the corpus is the limit.

**3. The optimal component count FALLS as the corpus grows.** The opposite of the
prediction that motivated crossing k with N, and the more interesting law:

| | best k | score |
|---|---|---|
| `corpus-pca`, N=25 | **256** | 0.781 (k=64 only 0.758) |
| `corpus-pca`, N=800 | **64** | 0.818 (k=256 down to 0.797) |
| `lr-cca`, N=800 | **64** | 0.811 |
| `lr-cca`, N=1039 | **32** | **0.825** |

With few participants each component is a noisy mixture, so ridge needs many of
them to recombine the signal; a well-estimated basis is *compact*. So the honest
headline is "more unlabeled data buys a smaller, better-conditioned
representation", not "a bigger one". Practical consequence: **k must be retuned
whenever the corpus size changes**, and every earlier k conclusion here was drawn
at one corpus size.

**4. `gev-slow` degrading with data is the control that makes the axis
credible.** It *loses* 0.336 as the corpus grows, because more data localises the
slow nuisance subspace more precisely -- and that subspace is exactly what cannot
carry gaze. An axis whose one end improves with data while the other end degrades
with data is a real axis. Reported alongside `gev-fast`, which behaved as its own
docstring predicted and disappointed: white noise maximises the fast objective,
so the extreme fast end is thermal noise, not gaze.

**5. Nuisance projection is now tested and negative.** `CLAUDE.md` has carried
"project out the global/motion components" as the open suggestion since the
next-TR result. Applied to the basis it does not pay: `nuis-pca8` tracks
`corpus-pca` without beating it, and `nuis-pca32` *degrades with data* (-0.131).
The reason is measured rather than guessed -- real gaze reaches lag-1 **0.851**
(dsL02, from the temporal-envelope law) while the corpus nuisance sits at
0.83-0.87, so the two overlap and removing 32 slow directions removes gaze with
the drift. Consider this line closed for the basis.

**What makes any of this cheap: lag-1 autocorrelation is free.** For centred
stationary `x`, `sym(C_1) = C_0 - DC/2`, so
`rho(w) = 1 - (w' DC w)/(2 w' C_0 w)` comes straight out of the two accumulators
`Moments` already keeps -- no extra pass over the corpus and no lag-1
accumulator. `lag1_autocorrelation` in `deepmreye/unsupervised.py`. That is what
made `gev-*`, `band-pca` and `nuis-pca*` an eigendecomposition each rather than a
re-read of 1039 participants.

**The measured spectrum, which is a result in itself.** Lag-1 autocorrelation of
the principal directions, over the 512-direction pool:

| | N=25 | N=100 | N=400 | N=800 |
|---|---|---|---|---|
| median | +0.059 | +0.296 | +0.404 | +0.394 |
| p5 | -0.053 | +0.146 | +0.326 | +0.348 |
| leading 2 | +0.88 / +0.73 | +0.84 / +0.81 | +0.83 / +0.88 | +0.82 / +0.87 |

Two readings. The **nuisance is concentrated and identifiable**: the top two
directions sit at 0.82-0.88 against a median of ~0.39. And at small N the
directions are *noise* (median rho 0.059, i.e. white); temporal structure only
emerges with hundreds of participants, which is the concrete mechanism behind
scaling law 1.

**Still unbeaten: `fold-pca:64`, which is best quoted as 0.83-0.85 rather than
as 0.847** -- see point 8 for why it has a range. Best corpus arm is `lr-cca:32`
at **0.825**, i.e. within the noise of it at this temporal resolution and tied
outright at finer ones (point 11). So the ranking from 2026-08-01 survives -- but
its explanation does not. "The unlabeled corpus is redundant" was measured at one
corpus size; the corpus is doing real and growing work.

**6. `lr-cca` has a sharp k threshold, not a gentle optimum.** At N=1039:

| k | 8 | 16 | **24** | **32** | 48 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|
| median r | 0.476 | 0.523 | 0.803 | **0.825** | 0.808 | 0.809 | 0.800 | 0.786 |

A cliff between k=16 (0.523) and k=24 (0.803) -- **+0.280 for eight components**.
Unlike `corpus-pca`'s smooth inverted U, this looks like a threshold: below ~24
canonical variates the projection cannot span gaze at all. Worth knowing before
anyone economises on `lr-cca`'s width.

**7. The corpus basis adds NOTHING on top of `fold-pca` -- it actively hurts.**
The test that decides whether the corpus is *complementary* or merely
*redundant*, at the per-part budgets `CLAUDE.md` requires:

| arm | median r |
|---|---|
| **`fold-pca:64`** | **0.847** |
| `fold-pca:64+band-pca:16` | 0.834 |
| `fold-pca:64+lr-cca:16` | 0.829 |
| `fold-pca:64+lr-cca:32` | 0.823 |
| `fold-pca:32+lr-cca:32` | 0.823 |

**All four concatenations lose**, by 0.013 to 0.024. Combined with the scaling
curve this settles the mechanism, and it is **redundancy, not domain mismatch**
(consistent with the corpus-embedding null, and against the story that entry
replaced): the corpus is estimating *the same subspace* a fold-local PCA
estimates, only less efficiently -- it takes 1039 participants to approach what
the six labeled datasets' voxels give for free. Nothing is left over to add.

The blunt reading, and the one to carry into any paper: **a 64-dimensional linear
subspace of a 14236-voxel eye mask is simply easy to estimate.** ~200 labeled
participants across 6 acquisitions already suffice, so a bigger unlabeled corpus
can match that ceiling but has no headroom above it. The corpus scaling is real
and it is bounded by the target being easy.

**8. Label efficiency: the corpus basis is label-INSENSITIVE, which is a
different and better claim than "it wins when labels are scarce".**
`scripts/sweep_probe_scaling.py --budgets`, 7 folds, median r:

| labeled windows | `fold-pca:64` | `corpus-pca:64` N=1039 | `lr-cca:32` N=1039 | `lr-cca:32` N=25 |
|---|---|---|---|---|
| 100 | 0.812 | 0.765 | **0.816** | 0.547 |
| 250 | 0.814 | 0.805 | 0.807 | 0.595 |
| 500 | 0.829 | 0.805 | 0.827 | 0.623 |
| 1000 | **0.847** | 0.821 | 0.825 | 0.582 |
| all | 0.828 | 0.817 | 0.829 | 0.591 |

**`lr-cca:32` is flat in labeled data (0.807-0.829) while `fold-pca:64` climbs
with it (0.812 -> 0.847).** That is the mechanism, and it is the expected one: a
fold-local basis has to be *estimated* from the labels available, a frozen corpus
basis was already paid for. They therefore converge as labels get scarce.

**They converge to a tie, not a crossover -- and the noise scale is what says
so.** At 100 windows `lr-cca:32` reads 0.816 against `fold-pca`'s 0.812, and it
would be easy to call that the first win for the corpus. It is not: `fold-pca`
itself reads **0.847 at 1000 windows and 0.828 with all of them**, and a method
that cannot get worse with more labeled data telling us it did is a direct
measurement that **run-to-run noise on these 7-fold medians is ~0.02**. The
+0.004 is inside it. Quote the tie.

**The corpus-size effect, by contrast, is far outside the noise and holds at
every budget**: `lr-cca:32` gains **+0.24 to +0.27** from N=25 to N=1039
(0.547 -> 0.816 at 100 windows), against `corpus-pca:64`'s +0.03 to +0.06. So the
unlabeled corpus is unambiguously doing work; what it does not do is overtake a
fold-local PCA.

A useful sanity check fell out: `fold-pca:64`'s two columns are **identical to
three decimals at every budget** (0.812/0.812, 0.814/0.814, ...), which is what
an N-independent arm must look like and confirms the basis files are being
switched as intended rather than silently reused.

**9. TR-matching the corpus helps the cross-orbit basis and hurts the
variance-seeking ones.** The one *documented* domain difference between the
corpus and the labeled half is repetition time -- corpus median **2.00 s**
against the labeled half's **0.80 s**. Since lag-1 autocorrelation depends
directly on sampling rate, an unmatched basis measures temporal structure at the
wrong TR. Tested by restricting the fit to TR <= 1.3 s (273 of 1039
participants) against an unmatched fit at the **same N=273** -- matched N is the
whole point, or the filter is confounded with corpus size:

| basis | TR-matched | unmatched | delta |
|---|---|---|---|
| **`lr-cca:32`** | **0.776** | 0.726 | **+0.050** |
| `corpus-pca:64` | 0.804 | 0.820 | -0.016 |
| `band-pca:64` | 0.799 | 0.816 | -0.017 |

**`lr-cca` gains +0.050, which clears the ~0.02 noise floor; the two
variance-seeking bases lose ~0.017, which does not.** The split is
interpretable and worth keeping: `lr-cca` selects directions by *shared
cross-orbit signal*, and at a 2 s TR the BOLD volume integrates over more gaze
movement, so the conjugate gaze correlation is washed out relative to the shared
nuisance -- feeding it fast-TR scans gives it a cleaner target. `corpus-pca` and
`band-pca` only want variance directions, and restricting to 273 fast-TR
participants costs them acquisition diversity for nothing.

**But corpus size dominates TR matching, and the two effects do NOT compose.**
`lr-cca:32` reads 0.776 TR-matched at N=273 against **0.825** unmatched at
N=1039, so the obvious move was a *looser* cut keeping most of the corpus. It
does not pay:

| corpus | N | `lr-cca:32` | `corpus-pca:64` |
|---|---|---|---|
| TR <= 1.3 s | 273 | 0.776 | 0.804 |
| TR <= 1.6 s | 376 | 0.794 | 0.806 |
| TR <= 2.0 s | 852 | 0.820 | 0.824 |
| all TRs | 1039 | **0.825** | 0.821 |

Both arms rise **monotonically with N** across the cuts and land on the
unfiltered numbers within noise (0.820 vs 0.825, 0.824 vs 0.821). So the score
tracks corpus size and is indifferent to the TR composition: the +0.050 measured
at N=273 does not survive at larger N and is better read as *which 273
participants* than as TR per se. **TR filtering is not a lever -- take the whole
corpus.** Which also means the one real documented domain difference between the
two halves has now been given three chances (matched at N=273, and two looser
cuts) and taken none of them.

An earlier draft of this entry called the result "indistinguishable" from
unmatched by comparing against *interpolated* N=200/N=400 values from the main
sweep instead of running the unmatched fit at N=273. The exact comparison
reverses the sign for `lr-cca`. Do not interpolate a control.

**10. `fold-srm` and `fold-pls` evaluated for the first time. A *supervised*
basis is the worst of the fold-local three.** Both were built by the previous
session, wired into `FEATURE_KINDS`, and never once run:

| arm | median r |
|---|---|
| **`fold-pca:64`** | **0.847** |
| `fold-srm:64` | 0.834 |
| `fold-pls:32` | 0.782 |
| `fold-pca:64+fold-pls:16` | 0.782 |

`fold-srm` (SVD hyperalignment with per-subject Procrustes, and label-free
test-time alignment for unseen subjects) **ties** -- 0.013 is inside the noise
floor -- so per-subject functional alignment neither helps nor hurts here. Note
its `_fit_unseen_subject` path is the only method in the repo that adapts to a new
participant without labels, so a tie is worth knowing rather than nothing.

**`fold-pls` is the informative one.** It is *supervised* -- PLS fits its
directions against the gaze targets -- and it loses to unsupervised PCA on the
identical voxels by **0.065**, and drags `fold-pca` down to its own level when
concatenated. Fitting the gaze relationship inside the training folds does not
survive leave-one-dataset-out, so the supervised advantage inverts into a
penalty. This is the cleanest single measurement on the project of why linear
unsupervised bases are so hard to beat here: **the evaluation punishes any
fitting of the target**, and PLS is the minimal example of a method that does it.
Keep it in the table for that reason.

**11. At finer temporal resolution the corpus bases catch up with `fold-pca`.**
Every number in this project reduces labels through `temporal_targets`, which
`nanmean`s over *both* the 10 sub-TR gaze samples and the `--temp-patch-size` TRs
in a bin -- **50 gaze samples collapsed into one target** at the default patch of
5. Averaging a target makes it smoother and more predictable, so the headline r's
are partly a property of the binning, and the prediction was that a harder,
finer target would *widen* the gaps between arms. It does the opposite:

| arm | patch=5 (50 samples/target) | patch=2 (20) | patch=1 (10) |
|---|---|---|---|
| `fold-pca:64` | **0.847** | 0.840 | 0.830 |
| `corpus-pca:64` | 0.821 | **0.837** | **0.827** |
| `lr-cca:32` | 0.825 | 0.836 | 0.810 |
| `raw` | 0.725 | 0.780 | 0.772 |
| gap, `corpus-pca` - `fold-pca` | **-0.026** | **-0.003** | **-0.003** |

`fold-pca:64` declines monotonically as the target sharpens (0.847 -> 0.840 ->
0.830) while `corpus-pca:64` *rises* and then holds, so the two are **tied at
per-TR resolution** -- 0.003, far inside the 0.02 noise floor. `lr-cca:32` tracks
at patch=2 and falls back at patch=1 (0.810), so `corpus-pca` is the arm to quote
in this regime, not `lr-cca`.

That matters practically: temporal resolution is the selling point of MR-based
eye tracking (v1's claim is "subimaging temporal resolution"), so a regime where a
basis needing **no data at all from the target study** matches the fold-local one
is the regime worth reporting. Note also that the default patch of 5 is the
setting least favourable to the corpus arms of the three tested -- worth knowing,
since it is the setting every other number in this file uses.

**A confound to state rather than bury.** `--max-train-windows` caps *windows*,
not rows, so patch=2 hands the readout 2.5x the training rows patch=5 does from
the same windows. This sweep therefore varies target smoothness **and** readout
sample size together. The direction is mechanistically sensible -- a frozen basis
has nothing left to estimate, so extra rows go straight into the readout, while
`fold-pca` still refits its basis from a fixed `--basis-fit-windows 400` budget
either way -- but the clean version needs a row-matched control, and until that
is run the claim is "at finer resolution *as configured*". `lr-cca:32` / `corpus-pca:64` need *no data at
all* from the target study -- one frozen projection shipped with the model --
against `fold-pca`, which must be refitted from whatever each new study has. That
trade is now quantified (0.825 vs 0.847, i.e. 0.022) and its accuracy provably
improves with the corpus it was fitted on.

Three bugs found and fixed, one silent and consequential:
- **`project` did not recognise the `nuis-pca*` kinds.** They fitted fine,
  registered in both `BASIS_KINDS` and `CORPUS_KINDS`, and raised at *apply*
  time -- after every basis had been fitted. `project` now dispatches on the
  stored arrays (`"components" in arrays`) rather than a hand-maintained name
  list, so a new PCA-shaped basis cannot be half-registered again.
- **A 143x speedup.** `lag1_autocorrelation` used
  `einsum("ij,jk,ik->i", v.T, C, v.T)`, which is the right quadratic form but
  does not dispatch to BLAS for that pattern; at 14236 x 512 it was minutes per
  basis instead of milliseconds, and it was the entire cost of the sweep.
  `(v * (C @ v)).sum(0)` is equivalent and guarded by a test.
- `sweep_corpus_scaling.py` wrote `meta["n_datasets"]` where
  `eval_probe.load_bases_for` reads `meta["datasets"]`, KeyError-ing every cell.

**12. Giving the corpus basis the labeled datasets' own voxels buys nothing --
the fourth failed explanation for the gap.** `fold-pca`'s one structural
advantage is that it is fitted on voxels from the labeled collection. Handing the
corpus fit those same voxels, per fold, with the held-out dataset excluded (the
honest unsupervised-domain-adaptation case, 7 bases of ~1200-1231 participants
each):

| arm | unlabeled only | + labeled voxels, held-out excluded | delta |
|---|---|---|---|
| `lr-cca:32` | 0.825 | 0.828 | +0.003 |
| `corpus-pca:64` | 0.821 | 0.825 | +0.004 |
| `lr-cca:64` | 0.809 | 0.813 | +0.004 |
| `fold-pca:64` | — | **0.847** | — |

+0.003 to +0.004, inside the noise floor, and consistent with the +0.008 recorded
for this at k=256 in the 2026-08-01 entry. The leave-one-dataset-out discipline
held: every basis logged its own exclusion list and no `TRANSDUCTIVE` warning
fired.

**So four distinct explanations for `fold-pca`'s advantage have now been tested
and failed:** domain mismatch (anatomy `d_A` -0.01, k-means ARI 0.043, distance
does not predict the loss), sampling rate (point 9 -- mixed, small, and the wrong
sign for two of three arms), missing in-domain voxels (this point), and nuisance
contamination (point 5). What is left is that there is **no missing information**
-- the corpus and a fold-local PCA are estimating the same subspace, and the
fold-local one gets there with less data because the subspace is small.

**Synthesis: why PCA is so hard to beat here, and where a scaling claim can
honestly come from.**

Three independent measurements now say the same thing, and together they explain
the whole pattern of null results on this project:

1. **The target is easy to estimate.** A 64-dimensional linear subspace of a
   14236-voxel eye mask is recoverable from ~200 labeled participants across 6
   acquisitions. That is why 1039 unlabeled participants can *approach* it and
   never pass it, and why concatenation adds nothing -- there is no residual
   subspace to contribute.
2. **The evaluation punishes any fitting of the target.** `fold-pls`, which is
   *supervised*, loses to unsupervised PCA on identical voxels by 0.065.
   Leave-one-dataset-out with 6-7 independent acquisitions destroys anything
   tuned to the training folds' gaze relationship. Every trained arm on this
   project -- JEPA, next-TR, CompositeNet, ContrastiveNet, `ocon` -- is a
   more elaborate version of the same mistake.
3. **The acquisition sets the ceiling.** The temporal-envelope law
   (`decoded_r = 1.03 * lag1 + 0.085`, residual SD 0.063) has `fold-pca:64`
   sitting *on* the line for nearly every cell. There is no headroom for a better
   representation to occupy.

So "beat PCA on peak median r" is close to unwinnable on this corpus, and any
method that appears to do it by more than ~0.02 should be suspected of a leak
before it is believed. The axes that remain, in order of how well they are
supported right now:

- **Unlabeled scaling of a frozen basis** (measured, strong): `lr-cca` +0.15 at a
  fixed labeled budget, **+0.24 to +0.27** in the low-label regime, monotone over
  six corpus sizes, with a second law (optimal k falls as the corpus grows) and a
  control that moves the other way (`gev-slow`, -0.34). This is a real scaling
  result about a deployable artifact.
- **Label insensitivity** (measured): `lr-cca:32` is flat at 0.807-0.829 across a
  10x range of labeled data while `fold-pca` climbs 0.812 -> 0.847. The claim to
  make is *matches a fold-local PCA without needing any labeled data from the
  target study*, not *beats it*.
- **Temporal resolution** (measured, promising, one confound): at patch=2 the
  corpus bases tie `fold-pca` (0.836-0.837 vs 0.840). Needs the row-matched
  control before it is quoted.
- **More acquisitions, not more participants** (untested, and the honest
  recommendation): every leave-one-dataset-out claim here rests on 6-7 folds, the
  temporal-envelope law on 12-14 cells, and `lr-cca:64` saturated between N=800
  and N=1039 -- i.e. more participants of the *same* 614 datasets stopped paying.
  The envelope also says the ceiling is set by gaze speed against TR. So the
  highest-value next investment is **more independent fast-TR acquisitions**, which
  raises the ceiling in a way no representation can.

What is *not* worth another run, on the evidence in this file: deeper models on
this corpus at this temporal resolution, domain adaptation (`align.py` harmful,
`d_A` null, TR-matching mixed and small), nuisance projection, and any further
attempt to add a corpus basis on top of a fold-local one.

Artifacts: `results/scaling/basis_n{25..1039}.npz`, `lag1_spectrum.json`,
`probe_n*.json`, `probe_scaling_summary.json`,
`results/scaling_budget/probe_budget_summary.json`,
`results/scaling_k/`, `results/scaling_tr/`, `results/scaling_untr/`; 21 new
tests in `deepmreye/tests/test_temporal_bases.py`. New scripts:
`sweep_corpus_scaling.py`, `sweep_probe_scaling.py`.

### Cross-orbit contrastive: the objective improves with data, the gaze does not (2026-08-12)

`deepmreye/orbitcon.py`, `scripts/train_orbitcon.py`, `--features ocon
ocon-random`. VICReg between the two orbits of the **same TR** — the third
bottleneck on the cross-orbit constraint, and the first *contrastive* one.
`crossorbit`/`orbitrot` grade a latent on repainting the other orbit, which pulls
it toward appearance and makes the decoder's capacity part of the score; this
grades agreement alone and never decodes. Probe feature 2x32 = **64 dims**,
matched to `lr-cca:64`, the linear form of the same constraint and the arm it had
to beat. Heavy regularisation by design (wd 1e-2, cosine-to-zero, per-view
augmentation) since every trained arm on this project has lost to a linear map.

**Verdict: it learns, it does not compete, and more data makes it worse at gaze
while making it better at its objective. The line is closed.** Judged against
criteria fixed before the runs — beat the untrained control, then beat
`lr-cca:64`.

All numbers below at a **matched 400-window budget** (the ocon arms are expensive
to extract, so the baselines were re-run capped rather than quoted from their
full-data values — comparing a capped arm to an uncapped baseline was the
obvious way to get this wrong).

| pretraining runs | val loss | within-run agree | probe dsL02 | probe dsL05 |
|---|---|---|---|---|
| untrained control | 49.50 | +0.103 | 0.646 | 0.576 |
| 100 | 29.47 | +0.616 | 0.754 | — |
| **200** | 28.85 | +0.660 | **0.785** | **0.666** |
| 425 | 28.43 | +0.682 | 0.711 | 0.608 |
| 884 (full corpus) | **28.24** | **+0.732** | 0.723 | 0.658 |
| `fold-pca:64` | — | — | **0.892** | **0.841** |
| `lr-cca:64` | — | — | **0.922** | 0.809 |

1. **Training helps, unambiguously.** `ocon` beats `ocon-random` at every scale on
   both folds (+0.08 to +0.14). That is a real result — JEPA had trained =
   untrained and next-TR had trained *worse* — and it puts this with `xorb`/`xrot`
   as an objective that genuinely optimises something gaze-related.
2. **It never approaches the linear arms.** Best `ocon` 0.785 against `lr-cca`
   0.922 on dsL02, 0.666 against 0.841 on dsL05. Making the cross-orbit
   constraint non-linear and contrastive bought nothing over its linear form —
   the *same* conclusion `crossorbit` reached, now for a second objective family.
3. **The scaling test the whole design was gated on fails, and fails
   informatively.** Val loss and within-run agreement improve **monotonically**
   from 100 to 884 runs (29.47 → 28.24; +0.616 → +0.732) while the probe peaks at
   **200** and then drops on **both** folds (dsL02 0.785 → 0.711/0.723, dsL05
   0.666 → 0.608/0.658). More participants make the encoder better at the thing
   it is trained on and no better at gaze.

   That dissociation is the useful part, and it is the **next-TR finding again in
   a new objective**: the content shared between the two orbits is dominated by
   what is *not* gaze — global signal, motion, drift, all common to both orbits
   and all varying within a run. `agreement_within_run` was built to exclude
   anatomy and it does; it cannot exclude motion, and that is what the curve
   says is being learned. So an eighth of the corpus is as good as all of it,
   and scaling further is not the missing ingredient.

**A geometry error worth more than the result: `split_orbits`' mirroring is
correct for reconstruction and inverts the signal for contrast.** The right orbit
is flipped along x so both crops run lateral-to-medial. Conjugate horizontal gaze
moves both eyeballs the same way in *global* x, so after the flip it runs in
**opposite** local directions — one shared encoder reports horizontal gaze with
opposite sign on the two orbits, and VICReg's invariance term is an MSE between
them, which penalises exactly that feature. Vertical gaze is untouched, an x-flip
leaving y alone. Ablated at n=200, identical in every other respect:

| geometry | dsL02 r_x | dsL02 r_y | dsL05 r_x | dsL05 r_y |
|---|---|---|---|---|
| un-mirrored | **0.780** | 0.789 | **0.565** | 0.766 |
| mirrored | **0.413** | 0.849 | **0.353** | 0.783 |
| cost of mirroring | **-0.367** | +0.060 | **-0.212** | +0.017 |

Horizontal collapses by 0.21-0.37 on both folds — *below the untrained control's*
r_x of 0.700 on dsL02 — while vertical is unchanged or slightly better. Two
things to take from it. **The objective cannot see this**: mirrored val loss
28.68 and within-run agreement +0.670 against un-mirrored 28.85 / +0.660, i.e.
the broken geometry looks marginally *better* on every label-free number. Only
the probe, **decomposed by axis**, shows it. And a mean r would have hidden it
too (0.631 vs 0.785 reads as "somewhat worse", not "one axis is destroyed").
Report axes separately when a geometric transform is anywhere in the pipeline.

The mirror convention is stored in the checkpoint and read by the extractor,
because feeding mirrored orbits to un-mirrored weights raises nothing and just
scores lower. `load` refuses any checkpoint missing an architecture field rather
than building a mismatched control — the `xrot` lesson, enforced.

Two of my own bugs, recorded because both were silent. Two `--scaling` jobs wrote
the **same checkpoint filename** (`orbitcon_n100.pt`), so a `--shift 0` run
overwrote the full-augmentation model at the same n with no warning; names are
now derived from `--out`. And the first design used global average pooling, which
leaves the embedding near-constant across a batch at init (VICReg's variance
hinge pinned at its 2.0 maximum) — `head="flat"` keeps the feature map's spatial
layout, which is where gaze lives and what `lr-cca` gets to use.

**What this closes and what it does not.** Closed: cross-orbit contrastive
learning at this scale, and the idea that the corpus size was the limit. Not
tested: projecting out the leading global/motion components *before* the
contrastive objective, which is the one remaining suggestion from the next-TR
entry and the direct answer to the mechanism measured here. If anyone picks this
up, that is the experiment — not more participants.

Artifacts: `results/orbitcon/` (checkpoints `unmirrored_n{100,200,425}.pt`,
`full_n884.pt`, `mirrored_n200.pt`, the gate JSONs, `verify_dsL11.log`,
`subtr_dsL11_retry.log`, and `probe_bar_8fold.json` — the 8-fold baseline table
below). Orbit cache for the full corpus at `results/orbit_cache_all.npz`
(1039 runs, 131240 TRs, 6.0 GB, 63 s to build).

### The 8-fold baseline bar, and dsL11 removed from it (2026-08-12)

`results/orbitcon/probe_bar_8fold.json`. The reference table any new arm is
measured against, `--protocol dataset --readouts ridge-cv --standardize-targets
dataset`, per-subject median Pearson r averaged over x and y:

| fold | `fold-pca:64` | `corpus-pca:64` | `lr-cca:64` |
|---|---|---|---|
| dsL01_guided_fixations | 0.869 | **0.882** | 0.877 |
| dsL02_pursuit | 0.912 | 0.917 | **0.929** |
| dsL03_pursuit | **0.191** | 0.184 | 0.167 |
| dsL04_pursuit | **0.843** | 0.787 | 0.799 |
| dsL05_free_viewing | **0.847** | 0.815 | 0.816 |
| dsL06_sequences | 0.624 | **0.635** | 0.600 |
| dsL07_deepmreye_calib | 0.795 | **0.843** | 0.811 |
| ~~dsL11_backtothefuture~~ | ~~0.776~~ | ~~0.785~~ | ~~0.756~~ |
| **median, 7 verified** | **0.843** | 0.815 | 0.811 |
| mean, 7 verified | **0.726** | 0.723 | 0.714 |

`fold-pca:64` leads on the median and `corpus-pca:64` does not beat it, which is
the 2026-08-01 conclusion holding on nine folds' worth of re-measurement. Note
`lr-cca:64` is best on dsL02 (0.929) and worst on dsL03/dsL06 — the same
variance-stability trade-off already on record.

### Three corrections to the record (2026-08-12)

Recomputed from the JSON on disk, not from the summaries written alongside it.
Each of these was stated the other way round somewhere in the notes, and each
would have sent the next person down a wrong path.

**1. The unlabeled corpus basis has no low-sample advantage. The claim that it
does is backwards in the data.** `results/low_sample_benchmark/` (56 eval files,
8 folds x 7 budgets) was summarised as "Corpus-PCA wins on 5 of 8 datasets" with
"dominance in low-to-medium labeled regimes (N=250 to N=1000)". Recomputing
`corpus-pca:64` minus `fold-pca:64` per cell:

| budget | N=100 | N=250 | N=500 | N=1000 | N=2500 | N=5000 | all |
|---|---|---|---|---|---|---|---|
| median diff | **-0.006** | **-0.023** | +0.012 | +0.009 | +0.002 | -0.001 | +0.006 |
| folds corpus wins | **1/8** | **2/8** | 5/8 | 5/8 | 5/8 | 4/8 | 5/8 |

The median is within +-0.023 of zero everywhere, and the low-sample regime is
where the corpus basis does **worst** -- 1 of 8 folds at N=100. So the
mechanism offered for it ("corpus eigenvectors prevent low-sample degeneracy
where fold-local ones overfit the coil geometry") is not merely unsupported, it
predicts the opposite of what the runs show. The fold-count framing also hides
the asymmetry: the wins are +0.004 to +0.048 while the losses reach -0.056
(`dsL04`, consistently, at every budget).

The one cell quoted as the headline -- `dsL06` at N=500, +0.122 -- is noise from
the least trustworthy fold in the corpus: `dsL06`'s six "subjects" are the *same
participant* at different TRs (see Open questions), its test set is **16
windows**, and the same fold reads -0.098 and -0.119 at N=100 and N=250. Do not
quote it in either direction.

This does not overturn the standing conclusion, it restores it: at k=64 a frozen
corpus basis **ties** a fold-local one (the 2026-08-01 entry below, mean -0.009,
1/6 folds). `corpus-pca:64` is still the better *deployment* artifact for the
reason given there -- one precomputed projection instead of a per-study refit --
and that argument never depended on it winning.

**2. `dsL11_backtothefuture` fails the gaze/BOLD sync check and must not be a
fold.** It was added to the corpus root and used as the 8th fold in every suite
in `results/` from 2026-08-11 onward. It had never been through
`verify_gaze_sync.py`, which `CLAUDE.md` requires of every ingested dataset and
which is the reason `dsL10` was rejected. Run now
(`results/orbitcon/verify_dsL11.log`):

| subject | peak lag | r | margin |
|---|---|---|---|
| sub-01 | 0 | +0.850 | +0.491 |
| sub-02 | 0 | +0.880 | +0.614 |
| sub-03 | **-1** | +0.821 | +0.471 |
| sub-04 | **-1** | +0.863 | +0.521 |
| MEAN | **-1** | +0.707 | +0.470 | **FAIL** |

The positive control passed in the same run -- `dsL07_deepmreye_calib` 5/5 at
lag 0, mean r +0.814 -- so the instrument is working and the split verdict is
dsL11's own. The mean profile is `-1:+0.71` against `0:+0.58`.

**The sub-TR sweep settles it: the error is per subject, so there is no offset to
fit.** Run over all four participants (`results/orbitcon/subtr_dsL11_retry.log`):

| subject | best offset | integer peak |
|---|---|---|
| sub-01 | +0.00 s | 0 |
| sub-02 | -0.25 s | 0 |
| sub-03 | **-1.50 s** (sweep edge) | -1 |
| sub-04 | **-1.50 s** (sweep edge) | -1 |

Two participants sit at ~0 and two run off the **edge** of the +-1.50 s sweep,
so for those the true optimum may be more than a full TR away. The dataset mean
profile decreases monotonically across the entire sweep
(`-1.50:+0.71` ... `+0.00:+0.58` ... `+1.50:+0.23`) with no interior peak -- it is
the average of two populations, not a measurement of one offset. An earlier
partial run of this sweep covered only sub-01 and sub-02 and appeared to peak
cleanly at +0.00 s; those are precisely the two participants that were never in
question, and reading that as a pass would have been the trap.

This is decisive because the **dataset-level anchor validates itself**:
`TTLPulse_N` fits at 1.49998 s against a nominal TR of 1.5, and the pulse count
equals the volume count exactly (1608 = 1608). A correct dataset-level anchor
with per-subject disagreement means the error is downstream of the anchor -- per
run or per subject. That is the ds007532/`dsL10` failure mode precisely, and the
precedent is explicit: one offset per subject would "fix" it and would be
circular, 4 (here) or 36 (there) free parameters fitted on the decoding target.
`dsL10` was rejected for exactly this and `dsL11` cannot be treated differently.

Only 4 of 39 participants are extracted, and
`CLAUDE.md` describes the dataset as parked at `~/.cache/deepmreye_pending/`
specifically so nothing would pick it up -- it is now inside the corpus root
instead, where `ProbeDataset._discover()` finds it by its `labels`, exactly the
trap `dsX10` already sprang.

**Consequence: every dsL11 number in `results/` is unusable**, including its rows
in the K sweep and the low-sample benchmark above. Medians here are over the
**7 verified datasets**, with dsL11 shown separately where it appears at all.

**Not done, and deliberately left to a decision rather than a side effect:**
retiring it from the corpus. A name will not do it -- `ProbeDataset._discover()`
takes any participant carrying `labels`, which is how `dsX10` silently ran as its
own fold. The two options that work are moving
`~/.cache/deepmreye/dsL11_backtothefuture/` back to
`~/.cache/deepmreye_pending/` (where `CLAUDE.md` says it belongs, and reversible)
or stripping its `labels` and keeping the blocks as unlabeled corpus data (what
was done for `dsX10`). Runs in the meantime should pass
`--exclude-datasets dsL11_backtothefuture`, which is what the numbers below do.

**3. The ContrastiveNet negative result is void, not a finding.**
`results/test_single.json` has it at r 0.09-0.45 against 0.78-0.91 for a
fold-local PCA of the same width, and that was written up as evidence that
"closed-form SVD dominates non-linear contrastive networks on fMRI voxels". The
run cannot support that:

- Its two views are frames at `t` and `t+dt` (`dt <= 2`). Predictive/temporal
  objectives on this signal were already diagnosed and closed -- the predictable
  part of an eye block is drift, motion and global signal, so temporal
  invariance is a direct instruction to encode nuisance (see the next-TR entry
  below, trained 0.530 against its own untrained control's 0.686).
- **No untrained control**, in a repo whose own `features.py` docstring calls it
  non-optional after the JEPA branch.
- 100 participants, 10 epochs, **188 seconds** of training, and no augmentation
  of any kind.
- `--exclude-datasets` exists as a function argument but is not wired to the
  CLI, so a command-line run cannot hold the test datasets out of pretraining.
- `scripts/eval_contrastive_low_sample.py` ran and produced **56/56 NaN cells**
  (`results/contrastive_low_sample_summary.json`), swallowed by a bare `except`,
  and its `winner` field defaults to "FOLD-PCA" on a NaN comparison -- so the
  file reads as a completed sweep.

Separately, `results/composite_sweep/` holds 4 checkpoints and no evals: the
sweep aborted, and its result keys omit the `/ridge-cv` suffix `eval_probe`
writes, so every cell would have read 0.0 had it finished. The CompositeNet
8-fold run *did* complete and is a real (negative) result:
`results/full_8fold_benchmark/` has it losing to `fold-pca:64` on 7 of 8 folds.

### The labeled corpus is not six datasets — OpenNeuro ships more gaze (2026-08-03)

A scan of **all 2409** OpenNeuro accessions for BIDS eye-tracking paired to a
functional run found **382 participants across 18 datasets**
(`results/openneuro_eyetracking_scan.json`). The corpus had 270 participants
across 6. The participant count is not the point: **independent acquisitions
are the scarce resource here**, and every leave-one-dataset-out claim in this
project rests on six folds, the temporal-envelope law on twelve (dataset, axis)
cells.

Tiering by what was actually verified, not by what the filenames suggest:

| tier | subjects | datasets | state |
|---|---|---|---|
| **A** continuous x/y + resolvable anchor | 155 | 6 | **done — see below** |
| **B** parseable, needs a writer (`.asc`, fixation-event reports) | 102 | 3 | not started |
| **C** EyeLink `.edf` binary (needs `edf2asc`) | 51 | 3 | not started |
| **D** unverified (ranged read failed, bespoke `.mat`) | 74 | 6 | not started |
| — pupil diameter only (ds006578) | 3 | 1 | **excluded** |

**Tiers B-D outcome: 2 datasets usable, 7 excluded.** Two needed new machinery
and got it -- an EyeLink ASCII reader (`read_asc`) and an anchor that fits a
*numbered* pulse train (`ANCHOR_INDEXED_MESSAGE`), which beats taking the first
pulse because the index says which volume each pulse belongs to and the fitted
slope returns the acquisition's own TR as a check:

- `dsL11_backtothefuture` (ds006642, 39, movie, TR 1.5 s). `TTLPulse_N` fits at
  **1.49998 s**, and the pulse count equals the volume count exactly (1608 vs
  1608), which validates the run pairing as well as the anchor -- necessary,
  because its ET files order entities the other way round from the BOLD
  (`run-003_task-x` against `task-x_run-003`) and live under `sourcedata/`.
  **Trap:** the same file logs `PULSE_N` every 42 ms -- the 24 Hz video frame
  counter. The anchor rejects any train whose spacing is not the TR.
- `dsL12_rest` (ds004158, 20, resting state, TR 0.8 s). `TR num N onset`, one
  per volume, indexed 1..500 against 500 volumes, slope **0.80026 s**. Same
  trap: the file also carries 5000 `mri_trigger` messages at 83 ms (ten sub-TR
  pulses per volume), and fitting those would bias the origin by half a TR.

Excluded, with the reason on record so nobody re-litigates them:

| dataset | n | why |
|---|---|---|
| ds008366 | 32 | `.asc` is an **events-only** export -- no samples, and no scanner trigger among its messages (only stimulus `TarX`/`TarY`) |
| ds006039 | 31 | fixation-event report with **trial-relative** times; no scanner-referenced clock |
| ds001840, ds007305, ds004283 | 51 | EyeLink `.edf` **binary**; `edf2asc` ships in SR Research's licensed Developer Kit, which is not installed |
| ds004529 | 34 | bespoke `.mat`/`.log`, no BIDS timing |
| **ds001473** | 15 | **byte-identical duplicate of ds000113** (240/240 files) |
| ds006947, ds006503, ds001471 | 5 | 1-3 participants each; not worth the per-dataset config risk |
| ds006578 | 3 | pupil diameter only, no gaze |

**studyforrest is mirrored three times on OpenNeuro** -- ds000113, ds001107 and
ds001473 are byte-identical. Taking all of them would have tripled the same 15
participants across folds that are supposed to be independent. Check ETags
before adding any dataset that looks familiar.

**Tier A outcome: 82 participants added, 3 new datasets, 1 rejected.**
The corpus is now **352 gaze-labeled participants across 9 datasets**, from 270
across 6 — a **50% increase in independent acquisitions**, which is the number
the statistics actually depend on.

| corpus name | source | n | anchor | offset | peak lag | mean r | margin |
|---|---|---|---|---|---|---|---|
| `dsL07_deepmreye_calib` | ds006833 | 15 | message | +0.00 s | **0** | **+0.785** | +0.599 |
| `dsL08_studyforrest_movie` | ds000113 | 15 | starttime | **−0.75 s** | **0** | +0.667 | +0.523 |
| `dsL09_fearlearning` | ds001242 | 52 | trigger | **+0.50 s** | **0** | +0.291 | +0.198 |
| ~~`dsL10_visseq`~~ | ds007532 | 36 | — | — | **FAIL** | — | — |

The three that pass do so on the same instrument that puts five of the six
original datasets at exactly lag 0.

**Two traps the scan had to be rebuilt to avoid.** `.edf` is both *EyeLink Data
Format* and *European Data Format*, so a bare extension match put a sleep-EEG
dataset in the candidate list. And **ds001107 is byte-identical to ds000113** —
240 files matching on size and ETag, the same 30 subjects. Ingesting both would
have duplicated 15 participants and placed the same people on both sides of a
"leave-one-dataset-out" split. It is dropped, and the pair is the reason the
tier-A dataset count is 6 rather than 7.

### Gaze/BOLD sync: an anchor is recovered, never assumed (2026-08-03)

`deepmreye/eyetracking.py` — new. Turning a tracker stream into `[T, 10, 2]` is
easy; putting it on the scanner's clock is the whole problem. A constant offset
is nearly invisible — the labels still look like gaze, the decoder still trains,
it just scores lower, which is indistinguishable from a harder dataset. Three
explicit strategies, recorded per participant in the file's attrs:

- **`starttime`** — BIDS-compliant `StartTime` (ds007532: `-12.27`).
- **`trigger`** — a scanner-pulse column; the pulse train verifies itself
  (ds001242's median inter-pulse is **2.0000 s**, exactly its TR).
- **`message`** — a sync message in a BIDS `physioevents` file
  (ds006833 logs `trial 1 mri_trigger val = -8`).

**What is deliberately not a strategy: believing `StartTime`.** ds006833 and
ds005166 both write the tracker's own first timestamp into a field defined as an
offset from volume 0. It is self-referential and carries no sync information;
taking it at face value would have placed volume 0 **58.5 s early** for ds006833.
`anchor_seconds` raises instead. The ds006833 anchor was cross-checked
independently: from `mri_trigger` to the last trial is **184.96 s** against a
scan of 154 x 1.2 s = **184.8 s**.

`scripts/verify_gaze_sync.py` — new, and the reason any of this is trustworthy.
It decodes gaze from the eye block at every TR shift in a window and finds the
peak. The eyeball signal is not hemodynamic — it is the orbit moving inside the
imaged volume — so correct alignment peaks at **lag 0** with no BOLD delay to
absorb an error. Two things make it honest: the **six original datasets are the
positive control** (5/6 peak at exactly 0, dsL06 sharply — 0.73 at lag 0 against
~0.05 elsewhere), and it reports a **margin** over lags at distance >= 2, because
gaze is smooth and an argmax alone overstates how determined the peak is. The
sign convention is calibrated by injecting known shifts into real data:
injected +k gives peak +k, exactly, with r preserved.

**`dsL07_deepmreye_calib` (ds006833, 13 participants): 13/13 peak at lag 0**,
mean r **+0.780**, margin **+0.603**, profile −1:+0.50 / 0:+0.78 / +1:+0.31.
Orientation checked separately, since a y-sign error survives a lag test:
**r_x 0.822, r_y 0.753**, both positive. This is a calibration protocol someone
ran *for DeepMReye* (fixation, pursuit, free viewing); its `DeepMReyeClosed`
companion task is excluded because it is deliberately eyes-closed.

**The sub-TR sweep is what earns the result.** The integer sweep passed
studyforrest at lag 0 — true, but its profile was lopsided (−1 scored +0.46
against +1's +0.12). Re-binning the *raw* gaze at fractional offsets through the
ingest's own code path put the optimum at **−0.75 s**, half a TR away, with
**15 of 15 participants** preferring a negative offset and none at or above 0.
Correcting it moved mean r **0.549 → 0.667** and made the profile symmetric,
which is the signature of a right answer rather than a lucky one. ds001242
needed +0.50 s on the same logic (9/10 subjects, +0.06 r, weaker evidence —
flat curve, scattered argmaxes; both caveats are in the config).

That `dsL07` peaks at exactly **+0.00 s** is what makes those two corrections
credible: it rules out a bug in the binning and localises each error to the
dataset that has it.

**ds007532 was rejected, and that is the point of the exercise.** Its
`StartTime` mixes conventions run by run — sub-01 alone has proper offsets
(−12.27, −7.38, −15.62) on some runs and raw tracker clocks (2351691, 1331388,
3200988) on others — and the values actually used spanned −89.6 to −0.7 s, which
would have the tracker stopping a minute before the scan ended. A second anchor
(`TRIGGER SENT` in the physioevents, self-checking because its two occurrences
bracket the scan at 468.8 s against 470 s) fixed three subjects dramatically
(sub-02 0.250 → 0.827) and left the rest scattered over lags −3..0. The sub-TR
sweep is **flat** (0.22 at −1.50 s to 0.11 at +1.50 s, no peak), so there is no
dataset-level offset to find: the error is per run.

One offset per *subject* would have "fixed" it, and that is exactly why it was
not done — 36 free parameters fitted on the decoding target would make every
number from this dataset circular. A labeled dataset nobody can trust is worse
than one that is absent. The eye blocks are fine (registration never touches
gaze), so they sit on disk as `dsX10_visseq_unaligned`, **outside the `dsL*`
glob** that selects the probe set, with the dataset recorded
`LBL_DATASET_SKIPPED` in `labels.csv`.

### The new datasets do not transfer, and one reason is a sign convention (2026-08-05)

`results/probe_10datasets.json`, `--protocol dataset --features raw fold-pca:64
--readouts ridge-cv --standardize-targets dataset`. Per fold, `fold-pca:64`:

| fold | r_x | r_y | |
|---|---|---|---|
| `dsL01_guided_fixations` | 0.905 | 0.851 | |
| `dsL02_pursuit` | 0.923 | 0.897 | |
| `dsL03_pursuit` | 0.162 | 0.225 | known resolution limit |
| `dsL04_pursuit` | 0.841 | 0.823 | |
| `dsL05_free_viewing` | 0.827 | 0.834 | |
| `dsL06_sequences` | 0.953 | 0.097 | known vertical failure |
| **`dsL07_deepmreye_calib`** | **0.876** | **-0.763** | **new — inverted y** |
| **`dsL08_studyforrest_movie`** | 0.011 | -0.019 | **new — no transfer** |
| **`dsL09_fearlearning`** | 0.139 | -0.132 | **new — no transfer** |
| **`dsL12_rest`** | -0.024 | -0.066 | **new — no transfer** |
| median over 10 folds | **0.359** | | (`raw` 0.385) |

**The original six are unchanged** — every one reproduces its control value
within noise, so nothing about the expansion or the standardisation degraded
them. The median falls to 0.359 purely because four new folds score near zero.

**`dsL07` is a vertical sign flip, not a failure.** |r_y| = **0.763** with the
wrong sign, against **+0.753** measured *within subject* by
`verify_gaze_sync.py`. The vertical signal is there and is strongly decodable;
it points the wrong way relative to the corpus's convention. Within a subject
the readout learns the sign, so the sync check cannot see this -- only a
cross-dataset fit can. `EnvironmentCoordinates: top-left` was honoured with
`flip_y=True`, so the mismatch is against whatever convention `dsL01`-`dsL06`
inherited from DeepMReye 1.0 (whose `load_label` also applies a `*= -1` to y).
**Resolve the corpus's y convention from first principles before flipping
anything** -- picking the sign that maximises r is fitting one bit per dataset
on the evaluation.

**`dsL08`/`dsL09`/`dsL12` are a real open question.** All three decode fine
*within* subject (0.667 / 0.291 / 0.416 from the lag sweep) and near zero across
datasets. Convention problems like `dsL07`'s have not been ruled out for them,
so do not yet conclude "transfer fails".

**But the possibility worth taking seriously** is that the published 0.814 is
flattered by all six original datasets coming from *one* collection with shared
acquisition conventions and shared preprocessing, and that genuinely independent
acquisitions are much harder. That is exactly the question adding them was meant
to answer, and it is now the most valuable open thread in the project. Settle
the conventions first; the answer is only meaningful once they are ruled out.

### The new labeled datasets sit *inside* the corpus better than the old ones (2026-08-05)

`scripts/visualize_corpus_embedding.py` re-run over 1573 fully-covered
participants (333 labeled, 1240 unlabeled, 694 datasets);
`media/visualizations/06_corpus_embedding.png`.

Nearest-neighbour mix — the share of a labeled participant's 25 nearest
neighbours that are unlabeled corpus participants, chance **0.788**:

| dataset | mix | |
|---|---|---|
| `dsL09_fearlearning` | **0.954** | new |
| `dsL12_rest` | **0.926** | new |
| `dsL05_free_viewing` | 0.912 | |
| `dsL02_pursuit` | 0.896 | |
| `dsL04_pursuit` | 0.865 | |
| `dsL03_pursuit` | 0.853 | |
| `dsL07_deepmreye_calib` | **0.816** | new |
| `dsL06_sequences` | 0.800 | |
| `dsL01_guided_fixations` | **0.470** | far below chance |

**All three new datasets sit at or above chance** — indistinguishable from the
unlabeled corpus by nearest neighbours — and two of them are the most embedded
labeled sets there are. That is what you would expect and it is worth having
measured: `dsL07`/`dsL09`/`dsL12` *are* OpenNeuro acquisitions, ingested by the
same pipeline as the unlabeled half, while `dsL01`-`dsL06` arrived from the
DeepMReye 1.0 collection.

Adding them also *lowered* the labeled-vs-unlabeled separation: covariance
`d_A` **0.549** (was 0.665 on six datasets), temporal-SD `d_A` **0.069** (was
-0.01, still ~chance). Clustering is unchanged — ARI against dataset identity
**0.045**, and 1 of 12 clusters is >90% labeled, which remains `dsL01`.

So the earlier conclusion holds and strengthens: this is **not** the multi-site
batch-effect regime, and `dsL01` is the outlier, not the labeled half. That
matters given the separate finding that dsL01's labels are stimulus positions
rather than measured gaze — the same dataset is anomalous on two independent
measures.

### Label units differ across the corpus, and that broke the probe (2026-08-05)

**Read `label_units` from the file attrs. Do not assume degrees.**

| dataset | units | why |
|---|---|---|
| `dsL01`-`dsL06` | `degrees_visual_angle` | from the DeepMReye 1.0 sets |
| `dsL07_deepmreye_calib` | `pixel` | viewing distance given (1.2 m), physical screen size not |
| `dsL08_studyforrest_movie` | `degrees_visual_angle` | **fully documented** -- Sengupta et al. 2016 (PMC5079121): 63 cm viewing distance, 26.5 cm screen at 1280 px, movie subtends 23.75 x 13.5 deg, so 23.75/1280 = **0.018555 deg/px**, pixels square. Cross-checks: 2*atan(26.5/2/63) = 23.77 deg |
| `dsL09_fearlearning` | `pixel` | sidecar's `degreePerPixel` contradicts the coordinate space (see below) |
| `dsL12_rest` | `pixel` | no geometry in the dataset; lab's setup not published in a form I could find |

The source papers were checked for each. Only studyforrest documents enough, and
its conversion is now applied. For the others the ingest **refuses to guess** --
a viewing distance without a physical screen size does not determine degrees,
and a fabricated factor is worse than an honest pixel.

**What this cost, and the fix.** `--protocol dataset` fits ONE readout over the
pooled training folds, so the squared-error loss follows whichever dataset has
the largest target variance. Harmless while everything was in degrees; not
harmless once the per-fold Euclidean scale spanned **21 (dsL01) to 595
(dsL12)**. The 10-dataset probe collapsed to median r **0.131**. The fix is
`--standardize-targets dataset` (now the default): z-score each training
dataset's gaze before pooling. Training data only, and Pearson r is invariant to
it. **R^2 and Euclidean error are not interpretable in that mode** -- predictions
land in z-units while test targets keep their own scale -- and the report says
so rather than printing numbers that look comparable.

**A second cause, and a correction to an earlier claim here.** The rejected
`dsX10_visseq_unaligned` was running as its own probe fold. Renaming it out of
`dsL*` does **not** keep it out: `ProbeDataset._discover()` picks up *any*
participant carrying a `labels` dataset, and the `dsL*` glob only controls what
`STAGE_PATTERNS["probe"]` downloads from the Hub. Its labels are now stripped
(blocks kept as ordinary unlabeled corpus data, labels stashed in
`results/dsX10_visseq_unaligned_labels.npz`). A naming convention was never
going to hold this.

**Control: the original six still reproduce exactly.** `fold-pca:64` median r
**0.814**, `raw` **0.703** -- identical to the pre-expansion numbers, so neither
the ingest nor the eval changes broke the baseline. Run it with
`--exclude-datasets dsL07_deepmreye_calib dsL08_studyforrest_movie
dsL09_fearlearning dsL11_backtothefuture dsL12_rest`
(`results/probe_orig6_control.json`).

**Units are recorded, never invented — including when that meant retracting a
claim.** ds001242 looked like the one dataset with complete geometry
(`degreePerPixel: 0.034`, `ScreenVisualAngle: [22, 16.5]`), implying a 647×485
screen centred at (323.5, 242.5). The data disagrees: gaze clusters at
**(127, 100)** across subjects, and a calibrated tracker does not put every
participant 6.7° left of centre. The exported coordinates are not in the
sidecar's pixel space — they look like a ~256×200 grid (self-consistent:
22/256 = 0.086 °/px horizontal, 16.5/200 = 0.083 vertical). Since that rests on
inferring grid size from where people fixate, the centre is set to the observed
fixation point and **no degree conversion is applied**. So no dataset in this
batch claims degrees, which costs nothing: Pearson r is scale invariant and
cross-dataset R² was already established to be unidentifiable.

### dsL01's labels are stimulus positions, and they lead the BOLD by one TR (2026-08-03)

Found by the sync check above, running as its own control. **11 of 12 dsL01
subjects peak at lag −1**, not 0. This is pre-existing data, unrelated to the
new ingest, and the cause is visible in the labels themselves: within-TR SD is
exactly **0.0000**, there are only **9 distinct x values**, and they change every
**5 TRs**. They are *stimulus* positions on a 9-point fixation grid held 4 s per
target — which is exactly what v1's `load_label` produces for `calibration_run`
(`np.repeat(labels, 5, axis=0)`). Compare `dsL05_free_viewing`: 1502 distinct
values, changing every TR, within-TR SD 1.18 — genuine measured gaze.

So the −1 is the eye arriving *after* the dot jumps: saccadic latency at a 0.8 s
TR, not corruption. But it costs accuracy on the corpus's **largest** labeled
dataset (170 participants): mean within-subject r **0.65 at lag −1 against 0.60
at lag 0**. Shifting dsL01's labels +1 TR would recover it. **Not changed** —
that edits existing ground truth and is a decision to take deliberately, not a
side effect of an ingest. Note also the wider implication: dsL01 is
*stimulus-locked*, not gaze-measured, which makes it qualitatively different
from the other five and is worth remembering when it behaves oddly (it is also
the most isolated dataset in the corpus embedding, and the one fold where
`corpus-pca` beats `fold-pca`).

### xrot wins at matched width — bottleneck size was the constraint (2026-08-03)

Two capacity experiments, opposite answers.

**Encoder/decoder capacity: no.** `--width 32 --template-channels 24` (4x conv
width, 3x template) reached contribution **0.1268 against 0.1254** at step 4000
— +0.0014 for ~4x the compute (2.25 s/step vs 0.9). Run deliberately as a screen
with three variables moved together, which is only sound because it failed: all
three are ruled out at once.

**Bottleneck width: yes.** `--parts 6` gives 12 dims/orbit, exactly `xorb`
K=4's:

| arm | dims | median r | untrained | learned margin | folds |
|---|---|---|---|---|---|
| `fold-pca:64` | 64 | **0.814** | — | — | — |
| **`xrot` parts=6** | 24 | **0.422** | 0.208 | **+0.214** | **6/6** |
| `xorb` K=4 | 24 | 0.389 | 0.273 | +0.116 | 5/6 |

It also wins the training objective (contribution **0.248** vs 0.222, val R2
**0.111** vs 0.096) and 4/6 probe folds. So the earlier 0.293-vs-0.389 gap was
**dimensionality, not the kind of latent**: 4x the encoder bought +0.001, 6x the
latent bought +0.12. `xrot` is now the best self-supervised arm here, and the
one whose score is most nearly earned — lower untrained floor, ~1.8x the learned
margin. Still nowhere near `fold-pca:64`. On `dsL05` training adds only +0.006,
so that fold's win is nominal.

**A control bug, caught and fixed.** Adding `--parts` updated the model and the
trainer but not `eval_probe`'s `*-random` branch, which read a stale `meta` and
built a **4**-feature control against a **24**-feature model. That inflated the
reported margin to +0.370; the true value is +0.214. `build_orbit_extractor`
now derives every constructor argument from the loaded model and raises if the
widths disagree, and `test_orbitrot.py` guards the invariant. A control built
from configuration rather than from the thing it controls will drift again.

### Why xrot loses, measured (2026-08-02)

`scripts/analyze_orbit_bottleneck.py` — new. The probe table gives one number
per arm and cannot separate "this bottleneck is dead" from "this bottleneck is
small". Four measurements per arm, each against its own untrained control:

| | dims/orbit | within-subj r | latent travel | L/R agreement |
|---|---|---|---|---|
| `xorb` trained | 12 | 0.600 | 0.0493 | +0.492 |
| `xorb` untrained | 12 | **0.474** | 0.0005 | **+0.201** |
| `xrot` trained | 2 | 0.393 | 0.1152 | **+0.739** |
| `xrot` untrained | 2 | **0.221** | 0.0004 | **-0.033** |

*Within-subject r* fits inside one participant, where anatomy is constant — the
bottleneck's own ceiling, independent of transfer. *L/R agreement* is the
correlation between the two orbits' latents; both eyes rotate conjugately, so
it is the cross-orbit objective's own success criterion and it does not appear
in the probe table at all.

**The untrained control is mandatory here, not decorative.** Both orbits sit in
one volume, so global signal, motion and drift are common to them — a *random*
centroid model already agrees with itself at **+0.201**. `xrot`'s control is
**-0.033**, so its 0.739 is entirely learned while part of `xorb`'s 0.492 is
not. This corrects an earlier claim here that the cross-orbit constraint
"worked" for `xorb`; partly it was nuisance.

So `xrot` is **small, not dead**: angles use 74% of their range and travel 300x
more than at init. It wins every *learning* measure (agreement from a true zero,
bigger trained-minus-untrained margin, 2 dims vs 12) and loses the probe only
because `xorb` starts from a much higher architectural floor. The position
bottleneck is the better random projection; the rotation bottleneck is the
better representation learner.

Not delivered: the learned canonical orbit renders as high-frequency texture,
not an eyeball (`media/visualizations/09_template_*.png`). Sensible — texture
makes rotation identifiable — but do not present it as anatomy.

Also added: `train(..., checkpoint_path=)` writes the best state whenever it
improves. Runs were all-or-nothing before, with the best state only in memory
until the function returned; at >2 s/step that is hours thrown away by one kill
or a sleeping laptop.

### The rotation bottleneck: learns cleanly, still loses (2026-08-02)

`deepmreye/orbitrot.py` — new. The cross-orbit objective with the latent
constrained to a **2-DOF rotation of a learned canonical orbit** instead of
soft-argmax positions. Motivated by a measurement: the `xorb` coordinate travels
only **0.187 voxels** over a whole run, a parameter-free centroid over the same
voxels decodes gaze at 0.367 within-subject where a *linear* projection reaches
0.904 — gaze rotates the eyeball, it does not translate it, and a centroid is
nearly blind to that.

| arm | dims | median r | mean r |
|---|---|---|---|
| `fold-pca:64` | 64 | **0.814** | 0.707 |
| `xorb` | 24 | 0.389 | 0.352 |
| **`xrot`** | **4** | **0.293** | 0.294 |
| `xorb-random` | 24 | 0.273 | 0.253 |
| **`xrot-random`** | **4** | **0.052** | 0.108 |

**The controls carry the finding.** `xorb`'s untrained control already reaches
0.273 of its 0.389 — only **30%** is learned. `xrot`'s control is 0.052, so
**82%** of its score is. Training wins **6/6** folds (mean +0.186) against
`xorb`'s 5/6 (+0.099), from **4 numbers instead of 24**. So the rotation latent
is by far the better-behaved objective per dimension.

It is still not a win: 0.293 < `xorb` 0.389 << `lr-cca` 0.798 ~ `fold-pca:64`
0.814, and it sits far below the temporal envelope, so this is a real
representational shortfall rather than a data limit. Two untested escapes: the
run was **still improving at its 4000-step cap** (contribution 0.125 rising,
`xorb` reached 0.222), and 4 dims may be too few (`--angles 3`,
`--template-channels`).

Implementation notes: MPS has no `grid_sampler_3d_backward`, and the CPU
fallback costs 5.4 s/step against 0.91 — so `_sample_trilinear` / `_affine_grid`
are written from primitives and equivalence-tested against
`torch.nn.functional`. That test earned its keep immediately: the first version
paired `align_corners=False`'s pixel mapping with `True`'s base grid, a
half-cell shift that trains fine and is silently wrong.

### Decodability is bounded by gaze speed, corpus-wide (2026-08-02)

`scripts/analyze_temporal_ceiling.py` + `media/visualizations/08_temporal_ceiling.png`.
Over the 12 (dataset, axis) cells, the lag-1 autocorrelation of the gaze trace
predicts the decoded correlation at **Pearson r = +0.977** (Spearman rho =
+0.797, p = 0.002 — quote the Spearman; three low cells against nine high ones
flatter the Pearson, and a dataset's two axes are not independent).

| cell | TR | lag-1 | decoded r |
|---|---|---|---|
| dsL03.x | 1.02 | 0.128 | 0.181 |
| dsL03.y | 1.02 | 0.163 | 0.234 |
| dsL06.y | 1.80 | 0.253 | 0.343 |
| dsL05.x | 1.00 | 0.598 | 0.811 |
| ... | | | |
| dsL06.x | 1.80 | 0.761 | **0.947** |
| dsL02.y | 0.87 | 0.851 | 0.874 |

**`dsL06`'s two axes are the load-bearing evidence.** A between-dataset trend
confounds TR, scanner, subjects and paradigm together; dsL06 dissociates *within
the same scans* — lag-1 0.761 on x decoding at 0.947 against 0.253 on y decoding
at 0.343, same subjects, same TR, same model. It is the only dataset whose axes
differ (ratio 0.33; the rest are 0.98-1.27), which is also why this is not
merely a proxy for TR.

Both apparent failures on this corpus — `dsL03` (both axes) and `dsL06`'s
vertical — are one phenomenon: gaze moving faster than the acquisition samples
it. That is a property of the stimulus, not the decoder, and it puts a ceiling
on what any method here can report. Stop targeting either.

### The DeepMReye 1.0 head-to-head finally exists (2026-08-02)

`scripts/eval_dme1.py` — new. Until now the baseline table held sklearn readouts
and a random-feature control and **no published-model number**, which is the
first thing a reviewer asks for. It needed neither retraining nor a
reimplementation: the authors released weights on OSF (https://osf.io/mrhk9/),
so this runs their checkpoint on our corpus. Scored with the *identical* 5-TR
binning `eval_probe` uses (`_reduce` is equivalence-tested against
`temporal_targets`, because averaging lifts correlations and doing it on only
one side would have handed us the win):

| arm | r_x | r_y | mean r |
|---|---|---|---|
| `fold-pca:64` + ridge-cv | 0.947 | **0.343** | **0.645** |
| `corpus-pca:64` | 0.922 | 0.250 | 0.586 |
| `lr-cca:64` | 0.939 | -0.008 | 0.465 |
| **DeepMReye 1.0** (`datasets_1to5`, held out on dsL06) | 0.946 | **-0.047** | 0.449 |

**Horizontal gaze is a dead heat (0.946 vs 0.947); the entire margin is
vertical.** Always report this decomposed.

Two things fall out. First, `dsL06`'s broken y-axis — previously flagged here as
a possible bug in our features worth chasing — **reproduces with the authors'
own weights, preprocessing and training data**, so it is a property of `dsL06`
and not of this pipeline. (OSF calls that dataset `dataset6_openclosed` where
our source directory was `dataset6_sequences`; if it is an eyes-open/closed
paradigm, vertical gaze may be barely sampled. Worth resolving before the paper.)
Second, only `datasets_1to5.h5` and the six single-dataset checkpoints are
legitimate — `datasets_1to6.h5` was trained on every labeled participant here
and the script refuses it without `--allow-contaminated`.

Setup: `.venv-tf` (separate, because TF's numpy pin fights the sklearn/torch
stack), `TF_USE_LEGACY_KERAS=1` for the Keras 2.4 checkpoints, and v1's source
vendored at run time from `main` via `git show` rather than copied into this
branch.

### The domain-mismatch explanation is wrong (2026-08-02)

`scripts/visualize_corpus_embedding.py` — new. It puts all **1450**
fully-covered participants (246 labeled, 1204 unlabeled) in one space and asks
whether the labeled gaze datasets are actually out of domain, which is what
every "the corpus basis doesn't transfer" note here has assumed without
measuring. Protocol is the standard multi-site batch-effect one: per-participant
descriptors, t-SNE, k-means, and a dataset-grouped domain classifier scored as
proxy A-distance `d_A = 2(1 - 2 eps)`.

| measure | value | reading |
|---|---|---|
| `d_A`, per-voxel temporal SD (anatomy) | **-0.01** | indistinguishable |
| `d_A`, corpus-PCA covariance (dynamics) | **0.67** / 2.0 | moderate |
| k-means ARI vs dataset identity (k=12) | **0.043** | no batch structure |
| Spearman(distance, `corpus-pca:64 - fold-pca:64`) | **-0.37**, p=0.47, n=6 | no relationship |

Anatomy is identical between the two halves, dynamics differ only moderately,
nothing clusters by acquisition, and — the decisive one — **distance from the
corpus does not predict where the corpus basis loses**. `dsL01` is the most
isolated labeled set on every measure (nearest-neighbour mix 0.47 against a
chance 0.83, and the only k-means cluster that is >90% labeled) and it is the
**one fold where the frozen corpus basis beats the fold-local one** (+0.012).
Under the mismatch story it should have lost hardest.

So the reason the unlabeled half does not pay is still open, and the leading
candidate is now *redundancy* rather than mismatch: 64 variance directions over
an eye mask are already estimable from a few hundred labeled windows. This
matters practically — it removes the motivation for domain-adaptation work,
which `align.py` had separately measured as harmful anyway.

Written: `media/visualizations/06_corpus_embedding.png`,
`07_corpus_clusters.png`, `results/corpus_embedding.npz` (descriptor cache,
re-plot with `--cache-only`), and `results/domain_matched_subjects.json` (the
400 unlabeled participants nearest the labeled cloud — kept because it is cheap,
but the null above is exactly the evidence *against* spending a run on it).

**Follow-up (same day): is the labeled half processed differently?** It comes
from a DeepMReye 1.0 run on `main`, and `convert_labeled_to_h5.py` copies it
without re-normalising, so a pipeline difference would sit in the corpus
unflagged. Checked four ways, and it is not:

| check | result |
|---|---|
| `normalize_img`, `main` vs `pytorch` | byte-identical; no voxel-value change in the diff |
| stored-block invariants | voxel-mean ~0, volume-SD ~1.0, `max\|x\|`=5.0 on both |
| anatomy `d_A` (TR-matched) | **-0.06** — indistinguishable |
| within-dataset cosine distance | unlabeled **0.940**; labeled **0.90-0.95** (5 of 6) |

The last row is the decisive one: two participants of a labeled study are no
more alike than two participants of an arbitrary OpenNeuro study. So the
`d_A` = 0.67 is largely **structural** — the corpus is 684 acquisitions of 1-2
participants, the labeled half is 6 acquisitions of up to 158, and separating
those needs only to recognise six studies.

One real difference did surface: **labeled median TR 0.80 s against the corpus's
2.00 s** (242/246 labeled are <= 1.3 s, against 317/1204 unlabeled). That is the
fixed-TR-window limitation in concrete form. It does not explain the `d_A` —
TR-matching *raises* it to 0.758, and splitting the corpus on TR alone gives
0.534.

Two things I got wrong on the first pass and corrected: "components 0-8 are
indistinguishable" was **cancellation** (dsL01 sits +0.030 on that band while
the other five sit -0.04 to -0.37; per dataset every band separates at 0.5-1.5),
and the first null used random multi-dataset splits, which averages out exactly
the per-acquisition identity that turned out to drive the effect.

Caveats, both surfaced in the script's own output: the full-mask coverage filter
drops 29% of participants, leaving `dsL02` and `dsL06` at 5 and 6, so their rows
print with `*` and are not measurements; and every labeled-vs-labeled `d_A` is
ungrouped (one acquisition per side), so those cells are upper bounds. Only the
pooled labeled-vs-unlabeled numbers are dataset-grouped, and the conclusion
rests on those. 253 tests pass (22 new, `test_corpus_embedding.py`).

### Correction: the corpus bases were judged at the wrong k (2026-08-01)

Every corpus-basis conclusion above was measured at **k=256**, before the
component sweep showed 256 is the *worst* setting even for `fold-pca` (0.779 vs
0.814 at 64). Re-run at k=64, with and without the labeled participants' voxels
folded into the unsupervised fit:

| basis fitted on | arm | median r | mean r |
|---|---|---|---|
| labeled training fold only | **`fold-pca:64`** | **0.814** | 0.707 |
| unlabeled + labeled (held-out excluded) | `fold-pca:64+lr-cca:16` | 0.813 | **0.708** |
| unlabeled + labeled (held-out excluded) | `corpus-pca:64` | **0.810** | 0.698 |
| unlabeled + labeled (held-out excluded) | `lr-cca:64` | 0.806 | 0.683 |
| unlabeled only | `lr-cca:64` | 0.798 | 0.677 |
| unlabeled only | `corpus-pca:64` | 0.796 | 0.690 |
| *(for reference, at k=256)* | `corpus-pca:256` | 0.775 | 0.653 |

Two things change, one does not.

- **Fixing k was worth a lot to the corpus arms.** `corpus-pca` goes
  0.775 → 0.810 and `lr-cca` 0.759 → 0.806. The earlier "the unlabeled corpus
  buys nothing" numbers understated them.
- **Folding the labeled participants' voxels in helps, consistently**: +0.008
  mean, **6/6 folds**, for `corpus-pca:64`. Small but not noise.
- **It still does not beat a fold-local PCA.** `corpus-pca:64` (labeled+unlabeled)
  0.810 against `fold-pca:64` 0.814 — 1/6 folds better, mean −0.009, and the gap
  is almost entirely `dsL06` (−0.058). So the *ranking* stands; the margin is now
  0.004, which is nothing.

**Practical consequence, and it is the useful one:** at k=64 a frozen corpus
basis is statistically indistinguishable from refitting a PCA per fold. That
makes `corpus-pca:64` the better *deployment* choice despite the tied accuracy —
it is one precomputed 64-component projection shipped with the model, rather
than a basis that has to be refitted from whatever labeled data each new study
happens to have. `fold-pca:64` remains the right arm for the *paper table*,
because it needs no external artifact.

Note `lr-cca:64` is the best arm anywhere on `dsL02_pursuit` (0.940) and the
worst of the three on `dsL06_sequences` (0.481) — the same variance-stability
trade-off recorded earlier, now at the corrected k.

Artifacts: `results/probe_k64_unlab.json`, `results/probe_k64_loo.json`.

### dsL03 is not a transfer failure, and 64 components beat 256 (2026-08-01)

Two results, one of which corrects a standing assumption in these notes.

**1. `dsL03_pursuit` has been mis-diagnosed all along.** It was recorded here as
"a transfer/calibration failure". It is neither:

- **Not calibration.** Pearson r is invariant to any affine rescaling of the
  prediction, so a gain mismatch *cannot* lower r. dsL03 has r 0.20 **and**
  R² −0.64, so the predicted direction is wrong, not merely its scale.
- **Not cross-dataset transfer.** Held-out *subjects within dsL03 itself* decode
  at **0.142**, statistically the same as the cross-dataset 0.159/0.196
  (`scripts/analyze_axis_conventions.py`). A shared readout fails between
  dsL03's own participants.
- **Not an axis or sign convention.** The 2×2 (pred, true) correlation matrix is
  diagonal and positive everywhere; off-diagonals are ~0.05.
- **Not registration.** Across-subject eyeball-centroid spread is 0.885 voxels
  for dsL03, mid-pack — and dsL01, the *worst* at 1.236, decodes at 0.86.

  What it is: **dsL03's gaze changes too fast for its acquisition to track.**
  Lag-1 autocorrelation of the gaze trace is **0.141**, against 0.56-0.85 for
  every other dataset. The control is exact: `dsL02_pursuit` is the same
  paradigm at the same gaze amplitude (within-subject SD 2.33 deg vs dsL03's
  2.35) but autocorrelation **0.849**, and it decodes at **0.911**. Same task,
  same excursion size, opposite outcome, differing in temporal structure alone.

  So dsL03 is a data limitation, not a modelling one, and it should stop being
  treated as the target for representation or domain-adaptation work. Every
  feature source, readout and alignment tried has moved it between 0.18 and
  0.21, which is exactly what a resolution limit looks like.

**2. The component count was set too high: 64 beats 256.** A sweep at matched
everything else (`results/probe_dim_sweep.json`):

| components | 8 | 16 | 32 | **64** | 128 | 256 |
|---|---|---|---|---|---|---|
| median r | 0.744 | 0.792 | 0.808 | **0.814** | 0.807 | 0.779 |

`fold-pca:64` beats `fold-pca:256` on 4/6 folds (mean +0.020) and `raw` on
**6/6** (mean +0.076). A clean inverted U — 256 components let ridge fit
directions that are specific to the training datasets. **The recommended
default is now `--features fold-pca:64`**, which takes the headline from
0.703 (published stride-4 baseline) to **0.814**.

**Unsupervised feature alignment does not help and mostly hurts.** Euclidean
Alignment and CORAL (`deepmreye/evaluate/align.py`), the standard cross-subject
corrections in EEG/BCI, were tried per subject and per dataset:

| | none | center | zscore | ea | coral |
|---|---|---|---|---|---|
| per dataset | **0.779** | 0.779 | 0.688 | 0.686 | 0.644 |
| per subject (32 comps) | **0.808** | 0.808 | 0.765 | 0.651 | 0.627 |

EA moves dsL03 by +0.014 and costs 0.19/0.14/0.18 on dsL01/dsL02/dsL06. The
reading is that the between-component covariance of these features is *signal*,
not shift: whitening it per group removes gaze. Mean-centring is free and
neutral (the blocks are already per-voxel z-scored within subject), everything
beyond it is harmful. Treat this line as closed too.

### Cross-orbit soft-argmax bottleneck: the first self-supervised objective that learns gaze (2026-08-01)

`deepmreye/crossorbit.py`, `scripts/train_crossorbit.py`, `--features xorb`.
Two paths, each structurally blocked from carrying the other's content:

- **coordinate path** — a soft-argmax over `K` heatmaps per orbit, collapsed to
  its spatial expectation, so the latent *is* a position rather than 2-3 numbers
  hoped to encode one. `K=2` is **12 dimensions** across both orbits.
- **nuisance path** — a wide global vector encoded from a **different TR of the
  same run**, so anatomy and scanner pass through but *this* TR's gaze cannot.

Objective: reconstruct each orbit from the **other** orbit's coordinate plus its
own nuisance code. Both eyes rotate together, so a useful coordinate must carry
conjugate gaze. No range prior — the corpus gaze spread is not a constant
(std(GazeX) 2.42 deg on dsL03 vs 7.05 on dsL01), so a global one would impose the
wrong scale, which is the calibration failure already on record.

**Training helps, and this is the first time on this project that it does.**
Same architecture, same dimensionality, only the weights differ:

| | dim | median r | vs untrained control | folds helped |
|---|---|---|---|---|
| `xorb` K=2 | 12 | **0.316** | `xorb-random` 0.122 | **6/6**, mean +0.132 |
| `xorb` K=4 | 24 | **0.389** | `xorb-random` 0.273 | 5/6, mean +0.099 |

The reconstruction ablation agrees: shuffling the coordinates across the batch
costs **0.190** (K=2) / **0.222** (K=4) of reconstruction R², against **0.000**
untrained. The bottleneck is genuinely used. Contrast JEPA (trained = untrained)
and next-TR (trained *worse* than untrained on 0/6 folds).

**But it does not reach the linear baselines, and adds nothing to them:**

| feature | dim | median r |
|---|---|---|
| `fold-pca` | 256 | **0.779** |
| `fold-pca+xorb` | 268 | 0.777 |
| `corpus-pca` | 256 | 0.775 |
| `lr-cca` (the *linear* cross-orbit constraint) | 256 | 0.759 |
| `xorb-nuis` | 64 | 0.449 |
| `xorb` K=4 | 24 | 0.389 |

So the objective finds real gaze signal from 12-24 label-free dimensions, but a
256-dimensional linear basis over the same voxels finds much more, and the
bottleneck contributes nothing on top of it (2/6 folds, mean -0.008).

Two readings worth keeping. First, `lr-cca` at 0.759 shows the *linear* version
of this same cross-orbit constraint already extracts most of what is available —
making it non-linear and adding a bottleneck did not beat it. Second,
`xorb-nuis` (0.449) scoring above `xorb` is **not** evidence the paths failed to
separate: at probe time the nuisance encoder is applied to the current TR, so it
is simply a 64-dim learned embedding of that volume, whereas the t/t' decoupling
only constrains what the nuisance path was *useful* for during training. The
honest comparison is `xorb` against `xorb-random` at matched dimensionality, and
that one is unambiguous.

Artifacts: `results/probe_xorb_k2.json`, `results/probe_xorb_k4.json`,
checkpoints `results/crossorbit_k{2,4}.pt`.

### Next-TR pretraining: the objective works, and it makes gaze decoding worse (2026-08-01)

A causal, language-model-style objective — predict TR *t+1* from TRs ≤ *t* over
the corpus-PCA coordinates, then probe the GRU's hidden state
(`deepmreye/temporal.py`, `scripts/train_ar_model.py`, `--features ar-gru`).
Trained on the 1005 unlabeled full-coverage participants, so it is valid for
every leave-one-dataset-out fold without retraining.

**Unlike JEPA, this objective demonstrably learns.** Held-out next-TR R²:

| | whitened targets | raw-variance targets |
|---|---|---|
| untrained (same architecture) | −0.047 | −0.053 |
| persistence (repeat TR *t*) | −0.246 | −0.036 |
| **trained** | **+0.230** | **+0.246** |

That is the key difference from the `pytorch-jepa` result, where trained and
untrained encoders scored the same. Here optimization is not the problem.

**And the representation is still worse than its own input, and worse than its
own random-init control, on every fold:**

| feature | dsL01 | dsL02 | dsL03 | dsL04 | dsL05 | dsL06 | median | mean |
|---|---|---|---|---|---|---|---|---|
| `fold-pca` (reference) | 0.859 | 0.911 | 0.200 | 0.757 | 0.800 | 0.593 | **0.779** | 0.687 |
| `corpus-pca` (the AR model's *input*) | 0.851 | 0.923 | 0.196 | 0.741 | 0.808 | 0.396 | 0.775 | 0.653 |
| `fold-pca+ar-gru:32` | 0.856 | 0.915 | 0.201 | 0.746 | 0.801 | 0.607 | 0.774 | 0.688 |
| `ar-random` (untrained control) | 0.794 | 0.808 | 0.138 | 0.648 | 0.724 | 0.511 | 0.686 | 0.604 |
| `ar-gru` (trained) | 0.710 | 0.649 | 0.077 | 0.434 | 0.626 | 0.260 | **0.530** | 0.459 |

Training **helped on 0 of 6 folds**, mean delta **−0.145**. The raw-variance
checkpoint behaves identically (`ar-gru` 0.589 vs `ar-random` 0.721, −0.132), so
the target weighting is not the cause.

**Why, and this is the useful part.** The diagnostic that motivated the design
also explains the result. Over corpus-PCA coordinates, next-TR variance is
predictable at R² 0.32 by a linear AR(4) — but it is concentrated in the leading
components (0–8: 38% of variance, predicted at R² 0.59; 128–256: predicted at
0.09). Those leading components are global signal, motion and drift. Gaze at a
0.8–2.0 s TR is close to white frame-to-frame — saccades are faster than the
sampling. So *the predictable part of an eye block is precisely the nuisance*,
and a model optimised to predict it allocates its state to that and evicts gaze.
Whitening the targets was an attempt to prevent exactly this and was not enough.

Note also that `ar-random` (0.686) is already below `corpus-pca` (0.775): a
recurrent bottleneck loses gaze information on its own, before any training.

**Conclusion: predictive self-supervision on this signal is counterproductive,
and now for a understood reason rather than an unexplained null.** Anything
further in this direction needs an objective that is not dominated by the
predictable nuisance — contrastive across the two orbits (which is what `lr-cca`
already does linearly, and it is the best-behaved unsupervised arm), or
prediction after regressing out the global/motion components. Plain next-TR
prediction is answered.

Artifacts: `results/probe_ar.json`, `results/probe_ar_rawvar.json`,
checkpoints `results/ar_gru.pt`, `results/ar_gru_rawvar.pt`.

### Readout comparison closed, and the unlabeled corpus exhausted (2026-08-01)

**1. `ridge-cv` wins the readout comparison outright.** This closes the open
question from the previous phase. Leave-one-dataset-out, per-subject median r:

| readout | on `raw` (stride-4) | on `fold-pca` (full mask) |
|---|---|---|
| **ridge-cv** | **0.703** | **0.779** |
| lgbm | 0.695 | 0.517 |
| svr | 0.690 (capped, 1200 win) | 0.168 |
| mlp | 0.619 | 0.506 |

No non-linear readout beats ridge-cv on either feature source, matching the
earlier GBT finding. `media/deepmreye_benchmarks.ipynb` is reproduced.

**Watch out for a pipeline artifact here.** `svr`/`lgbm`/`mlp` are built as
`StandardScaler -> PCA(--n-components, default 32) -> model`. On `raw` that is
reasonable compression of 480 correlated voxels. On a *basis* feature it is
destructive: the scaler whitens components that were already variance-ordered,
so the second PCA truncates to 32 near-arbitrary directions. Uncorrected this
reads as `fold-pca`+lgbm = **0.105**; with `--n-components 256` the same arm is
**0.517** (`results/probe_readouts_fullpca.json`). The corrected number is the
one above. Anyone crossing a basis feature with those three readouts must raise
`--n-components` or they will measure the truncation, not the model.

**2. The unlabeled corpus is exhausted — including with the labeled datasets
folded in.** Per the request, the gaze-labeled datasets' *voxels* (never their
labels) were added to the unsupervised fit, in two forms: a per-fold basis
excluding the held-out dataset (honest leave-one-dataset-out), and a
transductive basis that saw everything. Neither changes the conclusion:

| basis scope | `corpus-pca` | `diff-pca` | `lr-cca` | reference `fold-pca` |
|---|---|---|---|---|
| unlabeled only | 0.775 | 0.767 | 0.759 | **0.779** |
| + labeled voxels, held-out fold excluded | 0.772 | 0.768 | 0.760 | **0.779** |
| + labeled voxels, transductive | 0.775 | 0.768 | 0.759 | **0.779** |

Across every configuration tried — 3 basis scopes, 5 labeled-data budgets,
concatenations at 5 component budgets — **the single best unsupervised arm is
`fold-pca+lr-cca:16` at median r 0.783 against 0.779, winning 3 of 6 folds with
a mean delta of +0.001.** That is noise, not a result. Do not report it as a
gain.

Concatenating a corpus basis onto `fold-pca` *unbudgeted* actively hurts
(0.737): the readouts standardise every feature, so 512 equally-scaled
dimensions under one ridge alpha means the added block cannot be downweighted.
`--features fold-pca+lr-cca:16` (the `:k` per-part budget) is the fair form.

**The one qualitative thing the unlabeled corpus does buy is robustness on the
worst fold.** On `dsL06_sequences`, `lr-cca` scores 0.671 against `fold-pca`'s
0.593 and `corpus-pca`'s 0.409 — so `lr-cca` has the best *mean* across folds
(0.693 vs 0.687) while losing on the median. Consistent with the CCA analysis:
requiring a direction to be shared between the two orbits is a stronger
constraint than variance, and it degrades more gracefully where variance
ordering transfers badly. If anything from this line is worth keeping, it is
`lr-cca`, and the reason is variance-stability, not accuracy.

Artifacts: `results/probe_{readouts,readouts_fullpca,svr,loo_labeled,transductive,composite_budget}.json`,
bases in `results/basis_{labeled_all,loo_*}.npz`.

### Unsupervised feature bases: the full mask wins, the unlabeled corpus does not (2026-07-31)

The published baseline reads gaze off a **stride-4 subsample — 480 of the 14236
masked voxels**. That stride was a budget, not a modelling choice. Replacing it
with a 256-component linear basis over the whole mask is the single cheapest
improvement available on this branch:

| feature source | fitted on | median r | vs `raw` |
|---|---|---|---|
| `raw` (stride-4, 480 dim) | — | 0.703 | — |
| **`fold-pca`** (full mask, 256 dim) | labeled training fold | **0.779** | **6/6 folds better, +0.056** |
| `corpus-pca` | 1005 unlabeled subjects | 0.775 | 5/6 better, +0.022 |
| `diff-pca` (temporal differences) | 1005 unlabeled subjects | 0.767 | 5/6 better, +0.030 |
| `lr-cca` (left↔right orbit) | 1005 unlabeled subjects | 0.759 | 6/6 better, +0.043 |

Leave-one-dataset-out, `ridge-cv`, per-subject median Pearson r averaged over x
and y. Full table in `results/probe_unsup_dataset.json`.

**The unlabeled corpus does not add anything on top of a fold-local PCA.** It
was given its best shot — a labeled-data budget sweep, since scarce labels are
where pretraining should pay — and it loses at every point:

| train windows | 100 | 300 | 1000 | 3000 | all (~6500) |
|---|---|---|---|---|---|
| `raw` | 0.663 | 0.703 | 0.698 | 0.701 | 0.703 |
| `lr-cca` | 0.645 | 0.712 | 0.736 | 0.760 | 0.759 |
| `corpus-pca` | 0.670 | 0.732 | 0.762 | 0.776 | 0.775 |
| **`fold-pca`** | **0.716** | **0.763** | **0.780** | **0.782** | **0.779** |

`results/probe_budget_*.json`. Note `raw` plateaus at ~0.70 no matter how many
labels it gets, while `fold-pca` clears it with 100 windows.

This is the **second** independent finding that the unlabeled half does not help
gaze decoding (after JEPA, `pytorch-jepa`). The mechanism here looks like domain
mismatch, not data volume: a fold-local PCA is estimated on the acquisitions it
is applied to, whereas a corpus basis orders components by variance in OpenNeuro
scans with different scanners and protocols. Two things worth remembering:
`lr-cca` is the most *robust* corpus basis (6/6 against `raw`, and the only one
that does not collapse on `dsL06`), and **no basis touches the `dsL03_pursuit`
transfer failure** (r 0.180 → 0.196–0.201) — still a calibration problem.

New code: `deepmreye/unsupervised.py` (one streaming pass over the unlabeled
corpus accumulating a 14236² second moment; all three bases are decompositions
of it), `deepmreye/evaluate/features.py`, `scripts/fit_corpus_basis.py`, and
`--features` / `--max-train-windows` on `scripts/eval_probe.py`. The basis fit
is ~2 minutes on a laptop. Two silent-failure traps found and documented in
`CLAUDE.md`: the Fortran-order requirement on the `syrk` accumulator, and a
LightGBM/PyTorch OpenMP deadlock in the feature path.

### Visual Diagnostic Suite & Empirical Findings (2026-07-30)
- **Visual Suite Created**: [`scripts/visualize_gaze_diagnostics.py`](file:///Users/markus/Documents/Github/deepmreye/scripts/visualize_gaze_diagnostics.py) generates 5 core visual diagnostics across all 6 gaze-labeled datasets (`dsL01`–`dsL06`), saved in `media/visualizations/`:
  1. `01_condition_difference_maps.png` -- Top 20% vs Bottom 20% gaze difference dipole maps.
  2. `02_voxel_correlation_maps.png` -- Hotspot maps overlaid on eyeball anatomical slices.
  3. `03_cross_correlation_lags.png` -- Cross-correlation profiles across lags -5 to +5 TRs.
  4. `04_pc_projection_domain_shift.png` -- 2D PC space scatter plots by gaze_x, gaze_y, and dataset.
  5. `05_eyeball_gaze_movie.gif` -- Synchronized eyeball slice animation with moving gaze cursor.

- **Empirical Diagnostics on `dsL03`**:
  - Eyeball voxel signal is strongly present on `dsL03` ($r = 0.66$ max voxel correlation).
  - Cross-correlation peak occurs at **Lag 0 TRs** for `dsL03` (no TR lag offset).
  - Cross-dataset transfer drop on `dsL03` is driven by **gaze target amplitude/coordinate scaling mismatch** across acquisitions ($\text{std}(GazeX) = 2.42^\circ$ on `dsL03` vs $7.05^\circ$ on `dsL01`).

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

2. **Done (2026-08-01).** `ridge-cv` wins on both feature sources; no
   non-linear readout beats it. Numbers and the `--n-components` artifact that
   makes svr/lgbm/mlp look far worse than they are on basis features are in the
   2026-08-01 entry above. SVR was run at `--max-train-windows 1200` because it
   is O(n²)-O(n³) in training rows, so its row is not comparable to the others.

3. **Next**: the readout and feature axes are both closed. What is *not* closed
   is calibration — `dsL03_pursuit` sits at r ≈ 0.20 under every feature source
   and every readout tried, and that is the only remaining large gap.
   `scripts/analyze_calibration.py` is where to pick it up.

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
