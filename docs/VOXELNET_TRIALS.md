# Voxelnet trial log

Persistent record of every voxel-network training run: what was tried, the per-epoch
trace, why it was killed, and what changed next. Newest entries appended at the bottom.

**Protocol.** All trials are scored by `deepmreye/temporal_probe.py` (sub-TR, 9-fold LODO).
The network is warm-started at the incumbent `lr-cca:32 + lags±1 -> RidgeCV`, so epoch -1 is
the incumbent **by construction**. The learned branch fits the *residual*.

**Reading the columns.** `train` is MSE on the residual target (per-dataset z-scored units),
averaged over `steps_per_epoch`. `sel r` is the median sub-TR r on held-out **datasets**.
`alpha` is `|alpha|.clamp(max=1)` -- NOTE it saturates at 1.0, so it cannot distinguish a
gate at 1.0 from one at 5.0. Treat it as a floor, not a measurement.

**Gating.** Diagnostic runs use `--adopt-margin -1 --patience 999`, which disables both the
adoption gate and early stopping, so the reported `net` is the RAW trained model. This is the
number that answers "did the network learn anything", as distinct from "did it clear the bar".
Production runs restore `--adopt-margin 0.005`.

## Incumbent, per fold (`ridge-L1`, sub-TR)

| fold | r |
|---|---|
| dsL01_guided_fixations | 0.7678 |
| dsL02_pursuit | 0.9169 |
| dsL03_pursuit | 0.7751 |
| dsL04_pursuit | 0.8488 |
| dsL05_free_viewing | 0.8039 |
| dsL06_sequences | 0.6760 |
| dsL07_deepmreye_calib | 0.7471 |
| dsL08_studyforrest_movie | 0.2859 |
| dsL11_backtothefuture | 0.6693 |
| **median** | **0.7678** |

## Prior runs (gated; `net == incumbent` means the branch was REJECTED, not that it was untrained)

| run | encoder | aug | adopt-margin | median | verdict |
|---|---|---|---|---|---|
| voxelnet_lowrank32 | lowrank r32 | none | 0.005 | 0.7678 | rejected 8/9; dsL08 adopted at alpha 0.054 and LOST (0.2859 -> 0.1545) |
| voxelnet_heavyaug | lowrank r32 | shift3+mixup+noise | 0.005 | 0.7476 | adopted on 5/9 and lost on all 5 |
| voxelnet_cnn | cnn w16 | shift3+mixup0.2+noise0.1 | 0.01 | 0.7678 | rejected 9/9; train loss flat 0.46-0.49 |
| voxelnet_cnn_ssl | cnn w16 (SSL init) | same | 0.01 | 0.7678 | rejected 9/9 |

**What these did not establish:** whether the branch trains at all. Gated runs report the
fallback, and the CNN run stopped after 5 epochs (patience 5) at lr 1e-3 with three
augmentations on. The trials below vary lr and augmentation with the gate OFF.

---

## Design change: warm start dropped (2026-08-21)

**Objection (Markus):** the warm start is cheating, and it may only ever find a local
minimum. Correct on both counts, and the second is the sharper one. With `head_lin` frozen at
a fitted RidgeCV and `head_nl` zero-initialised, the learned branch fits only the residual
`y - ridge(z_cca)`. The representation the head reads from is fixed at step 0, so gradient
descent can decorate a good linear solution but can never reorganise the map. "At least as
good as the incumbent" is then true by construction and worth nothing -- it is a property of
the initialisation, not a finding about networks.

All trials from `trial02` on are **from scratch**: `scripts/train_voxelnet_scratch.py`,
voxels -> encoder -> lag stack -> dense head -> 20 outputs. The incumbent is computed per
fold and printed as a reference line; it never enters the loss and is never added to a
prediction. Selection is on held-out training **datasets** by sub-TR r (unchanged), and the
best-validation state is restored, so a fold can now legitimately score BELOW the incumbent.
That is the point.

**New diagnostic: `fit r`.** Each epoch also scores 6 *training* participants the same way.
A flat `val r` alone cannot distinguish a model that generalises badly from one that is not
fitting at all, and those need opposite fixes (more regularisation vs more capacity/lr).

| trial | what changed | status |
|---|---|---|
| trial01 | warm-start CNN, gate off, lr 1e-3, no aug | KILLED before epoch 0 -- superseded by the from-scratch design |

### trial02 -- from-scratch CNN, lr 1e-3, no augmentation (fold dsL07)

```
--encoder cnn --rank 64 --width 16 --hidden 256 --dropout 0.1 --lags 1
--chunk 128 --batch-chunks 2 --epochs 40 --steps-per-epoch 150
--lr 1e-3 --weight-decay 1e-2 --patience 8 --val-datasets 3 --val-subjects 8
```

Incumbent on this fold 0.7471. Selection datasets dsL03/dsL04/dsL05 (24 participants),
fit pool 237 participants.

| epoch | loss | fit r | val r |
|---|---|---|---|
| 0 | 0.5780 | 0.7152 | 0.6685 |

**Immediately informative:** a from-scratch CNN reaches val r 0.6685 after ONE epoch
(150 steps, ~38k rows). The warm-start design could not have surfaced this -- it reports
"the incumbent" whenever the residual branch is rejected, so a model decoding gaze at 0.67
on its own and a model that never trained produce the same line in the JSON.

**Full trace (early-stopped at epoch 17, patience 8; 804 s for the fold):**

| epoch | loss | fit r | val r |
|---|---|---|---|
| 0 | 0.5780 | 0.7152 | 0.6685 |
| 2 | 0.4283 | 0.7612 | 0.6889 |
| 4 | 0.4128 | 0.8330 | 0.7313 |
| 6 | 0.3493 | 0.8294 | 0.7100 |
| 9 | 0.3231 | 0.8783 | **0.7328** (best) |
| 12 | 0.2489 | 0.9127 | 0.6465 |
| 14 | 0.2566 | 0.9113 | 0.6732 |
| 17 | 0.2182 | 0.9198 | 0.6869 |

**RESULT: dsL07 incumbent 0.7471 -> scratch net 0.6933 (-0.0538).**

**Diagnosis: overfitting, not an optimisation failure.** `fit r` climbs monotonically to
0.9198 while `val r` peaks at epoch 9 and then oscillates 0.64-0.71 -- a 0.23 train/val gap.
The network has ample capacity and the optimiser works; what it learns does not cross a
dataset boundary. This is the regime where regularisation and augmentation are the right
lever, so that is what the next trials vary.

Note also `val r` is noisy epoch to epoch (0.6433 -> 0.7328 -> 0.7047 in three consecutive
epochs). Selection on a single noisy epoch will cherry-pick; that is a real risk to watch.

### trial03 -- batch diversity (fold dsL07)

Changed ONE thing from trial02: `--chunk 16 --batch-chunks 16` instead of `--chunk 128
--batch-chunks 2`. Same rows per step (256), but drawn from **16 participants instead of 2**.
Rationale: with 2 contiguous 128-TR chunks per step the gradient is dominated by two
participants' anatomy, which is exactly the participant-specific structure the train/val gap
says is being learned.

| epoch | loss | fit r | val r |
|---|---|---|---|
| 0 | 0.5577 | 0.7581 | **0.6998** (best) |
| 5 | 0.3281 | 0.8347 | 0.7005 |
| 9 | 0.2752 | 0.8827 | 0.6546 |
| 13 | 0.2313 | 0.9331 | 0.6778 |

**RESULT: dsL07 0.6870 (-0.0601 vs incumbent).** Early stop at 13, 627 s.

**Verdict: batch diversity is NOT the mechanism. Ruled out.** 16 participants per step
scores 0.6870 against 2 participants' 0.6933, and best val fell 0.7328 -> 0.7005. The
train/val gap is if anything slightly worse (fit 0.9331). So the overfitting is not the
gradient being dominated by two participants' anatomy -- it survives a batch that mixes 16.
Reverting to `--chunk 128 --batch-chunks 2` for subsequent trials.

### trial04 -- heavy regularisation + augmentation (fold dsL07)

Back to trial02's batch shape, everything regularising turned up at once:
`--dropout 0.3 --weight-decay 0.05 --shift 3 --noise 0.1 --mixup 0.2 --vox-dropout 0.1`
(shifts now per-chunk, see the `per_sample` change to `shift_augment`).

Run as a deliberate SCREEN, not an ablation: five knobs move together. That is only sound if
it fails -- if it wins, it has to be ablated before any single knob can be credited. Justified
here because trial02/03 agree the failure is generalisation, and the question worth one run is
whether regularisation moves the val ceiling **at all**.

| epoch | loss | fit r | val r |
|---|---|---|---|
| 0 | 0.7631 | 0.5994 | 0.5792 |
| 3 | 0.6267 | 0.6615 | **0.6350** (best) |
| 7 | 0.5913 | 0.6454 | 0.5912 |
| 11 | 0.5761 | 0.6634 | 0.5987 |

**RESULT: dsL07 0.6543 (-0.0928 vs incumbent).** Worst so far. Early stop at 11, 544 s.

**Verdict: the screen fails, and informatively.** The train/val gap closed from 0.23 to 0.03
-- but by dropping `fit r` from 0.9198 to 0.6634, not by raising `val r`. The five knobs
together do not improve transfer; they remove the capacity to fit at all. Regularisation
strength trades along a curve here rather than lifting the ceiling:

| trial | regularisation | best fit r | best val r |
|---|---|---|---|
| trial02 | dropout 0.1, wd 1e-2, no aug | 0.9198 | **0.7328** |
| trial03 | same + 16-participant batches | 0.9331 | 0.7005 |
| trial04 | dropout 0.3, wd 0.05, shift+noise+mixup+voxdrop | 0.6789 | 0.6350 |

Since the screen failed, all five knobs are ruled out **together** and none needs individual
ablation to be blamed -- but the ladder is still worth walking to find whether any single one
helps, starting with the only augmentation that has a physical justification.

### trial05 -- registration jitter alone (fold dsL07)

`--shift 2 --dropout 0.2 --weight-decay 1e-2`, no noise, no mixup, no voxel dropout.
Rationale: a rigid translation of the eye crop is what a registration error looks like, and
it leaves gaze unchanged, so it is label-preserving by construction. Mixup and voxel dropout
have no such guarantee.

### trial06 -- heavy regularisation, trained LONG (fold dsL07)

**Markus's point, and it is right:** in trial04 train and val sat 0.03 apart, which is the
*underfitting* regime, and patience 8 stopped it at epoch 11 while the training loss was still
falling (0.7631 -> 0.5761). Under strong augmentation a model needs far longer to converge --
stopping it on a short patience measures the schedule, not the regulariser.

Same five knobs as trial04, but `--epochs 150 --patience 40 --cosine`.

**Second reason the long run needs cosine:** `val r` swings +-0.05 epoch to epoch (trial02:
0.6433 -> 0.7328 -> 0.7047 in three consecutive epochs). Patience on a metric that noisy both
terminates early and cherry-picks the peak epoch as "best". A decaying lr shrinks the step
size late in training, which is the cheapest fix; if it is not enough, the selection metric
itself needs smoothing or more validation participants.

### trial05 RESULT -- registration jitter alone: BEST SO FAR, and not converged

| epoch | loss | fit r | val r |
|---|---|---|---|
| 0 | 0.7085 | 0.6669 | 0.6714 |
| 10 | 0.5108 | 0.7413 | 0.6969 |
| 17 | 0.5126 | 0.7419 | 0.7107 |
| 25 | 0.4924 | 0.7818 | 0.7186 |
| 30 | 0.4577 | 0.7963 | **0.7317** (best) |
| 38 | 0.4172 | 0.7973 | 0.6783 |

**RESULT: dsL07 0.7037 (-0.0434 vs incumbent).** Early stop at 38, 1751 s.

Two things this establishes:

1. **`--shift` alone is the augmentation that works.** 0.7037 against 0.6543 for the same
   augmentation plus noise/mixup/voxel-dropout. The train/val gap is 0.065 against trial02's
   0.23 *without* the capacity collapse trial04 suffered (fit r 0.797, not 0.664). It is the
   only augmentation here that is label-preserving by construction, and it is the only one
   that helps.
2. **It had not converged.** `val r` was still climbing at epoch 30 (0.6714 -> 0.7317 over 30
   epochs) when patience 8 terminated it at 38. Every conclusion drawn from a short run under
   augmentation is therefore suspect -- augmented runs converge slowly, and trial04's "heavy
   regularisation fails" verdict was drawn from **11** epochs.

| trial | config | best fit r | best val r | test r |
|---|---|---|---|---|
| trial02 | no aug, dropout 0.1 | 0.9198 | 0.7328 | 0.6933 |
| trial03 | no aug, 16-participant batches | 0.9331 | 0.7005 | 0.6870 |
| trial04 | 5 knobs, 11 epochs | 0.6789 | 0.6350 | 0.6543 |
| trial05 | shift 2 only, 38 epochs | 0.7973 | 0.7317 | **0.7037** |
| *incumbent* | `lr-cca:32 + lags±1 -> RidgeCV` | -- | -- | *0.7471* |

### trial07 -- trial05's config on a long schedule (queued behind trial06)

`--shift 2 --dropout 0.2 --weight-decay 1e-2 --epochs 150 --patience 40 --cosine`.
The direct follow-up to "it was still improving when it stopped".


---

## How the record is kept (read this before adding runs)

Hand-written notes will not survive "we will run MANY more", so the table is **generated from
the run artifacts**:

```bash
python scripts/summarize_voxelnet_trials.py
```

- `results/subtr/trials.csv` -- one row per (trial, fold): every config knob, `test_r`,
  `incumbent`, `delta`, `val_r_best`, `fit_r_at_best`, `gap`, `best_epoch`, `last_epoch`.
- `results/subtr/trials_epochs.csv` -- one row per (trial, fold, epoch): the full curves,
  for plotting or for asking "was it still improving".
- Raw stdout per run: `results/subtr/logs/trialNN.log`. Raw JSON: `results/subtr/trialNN_*.json`.

Nothing in the summarizer re-scores anything, so the table cannot drift from what was run.
Pass `--note "..."` to the trainer to label a run; it lands in the JSON and in the CSV.

**Conventions for a new run.** Give it the next `trialNN_` prefix, write `--out
results/subtr/trialNN_<slug>.json` and log to `results/subtr/logs/trialNN.log`. Keep the fold
fixed at `dsL07_deepmreye_calib` while screening (13-30 min/fold); only promote a
configuration to all nine folds once it beats the incumbent on that one.

**Two traps the columns are designed to catch.**

1. **`es_fired` is not convergence.** trial04 stopped on patience at epoch 11 looking settled;
   trial06 ran the identical configuration for 150 epochs with cosine decay and improved val r
   from 0.6350 to 0.6874. Patience on a metric that swings +-0.05 fires on noise. The
   summarizer therefore flags runs that **peaked in their final 40% of epochs** -- those are
   evidence about a schedule, not about a configuration.
2. **`gap` distinguishes the two failure modes**, which need opposite fixes: `fit` high with
   `val` low is overfitting (regularise), both low is underfitting (train longer, raise lr,
   cut regularisation). trial02 and trial04 sit at opposite ends -- 0.1455 and 0.0265 -- and
   reading either without the other would prescribe exactly the wrong change.


### trial06 RESULT (KILLED at epoch 79/150 -- moving to Leonardo)

| epoch | loss | fit r | val r |
|---|---|---|---|
| 3 | 0.6267 | 0.6615 | 0.6350 <- where trial04 stopped |
| 30 | 0.5136 | 0.6926 | 0.6524 |
| 35 | 0.5248 | 0.7037 | 0.6874 |
| ~70 | -- | -- | **0.6964** (best) |
| 79 | 0.4190 | 0.7450 | 0.6430 |

**Markus was right and the trial04 verdict is RETRACTED.** The identical configuration, given
79 epochs with cosine decay instead of 11, improved best val r from 0.6350 to 0.6964 and fit r
from 0.6615 to 0.7456. trial04 measured its schedule, not its regularisation. The general rule
this establishes: **an augmented run that stopped before ~50 epochs has not tested its
configuration.**

It is still behind trial05's 0.7317 val with far less regularisation, so shift-only remains
the best arm -- but trial06 was killed while still climbing, so "heavy augmentation loses" is
NOT established. Rerun it to 150 epochs on the cluster.

### trial07 -- never started (queued behind trial06, killed with it)

---

## Moving to Leonardo (2026-08-21)

Laptop runs stopped here: a fold is 10-30 min on MPS and the remaining questions need tens of
folds. See **`HANDOFF_VOXELNET.md`** in the repo root for the full handoff -- state of the
arm, what to run next, how to fetch code and data, and the traps.

Two things recorded there that are easy to lose:

- **The labeled probe set is byte-identical to HuggingFace** (337/337 `dsL*` files verified by
  size against `DeepMReye/eyeballs`, 0 mismatches), so every number above is reproducible from
  the Hub.
- **The unlabeled corpus is NOT.** 1,470 local-only participants across 429 accessions
  (20.4 GB) from `expand_corpus_5subs.py` were never uploaded. Supervised work is unaffected;
  SSL pretraining on the server would run on a smaller corpus than the laptop's.

