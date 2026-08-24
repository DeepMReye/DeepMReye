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


---

## First Leonardo runs (2026-08-21)

Environment established on the cluster before any number was trusted. Both gates in
`HANDOFF_VOXELNET.md` §6 reproduce on Leonardo:

```
pytest deepmreye/tests/ -q        -> 444 passed (162.93 s)
python -m deepmreye.temporal_probe --calibrate
    lr-cca:32          sub-TR 0.7421  expected 0.7420  OK
    lr-cca:32+lags2    sub-TR 0.7585  expected 0.7590  OK
```

Corpus: 337/337 `dsL*` participants (`dsL07`, `dsL08`, `dsL11` pulled from HuggingFace;
`dsL01`-`dsL06` were already on scratch). Voxel cache 337 participants / 405,379 TRs /
11.5 GB, sub-TR cache fingerprint `4eb9867e1790`.

**Three environment traps, all of which produce a wrong or absent number rather than an
error at the point of use.** Recorded here because none is visible from the branch alone:

1. **`deepmreye/models/jepa_net.py` is gitignored** (`.gitignore:6`, `models/`) and was never
   committed on any branch. `orbitjepa.py` imports it, `temporal_probe.py` imports
   `orbitjepa`, so a fresh clone cannot import the probe **or** the trainer, and 7 of the
   test modules fail to collect. It had to be copied from the laptop by hand. Nothing else
   in `deepmreye/models/` is needed — with that one file present, collection is exactly 444.
   **This should be force-added to the branch**; a module the pipeline imports is source, not
   a model artifact, and `models/` was ignored to keep weights out.
2. **A bare `uv sync` breaks the GPU.** There is no `uv.lock` here, so it resolves fresh and
   installs `torch==2.13.0` (`+cu130`). Leonardo's driver is **535.274.02 / CUDA 12.2**, and a
   CUDA 13 wheel fails at runtime with "driver too old (found version 12020)" —
   `torch.cuda.is_available()` is False *on the GPU node*. Re-pin with
   `uv pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.13.0+cu126"`.
   A login-node check cannot catch this: login nodes have no GPU, so `is_available()` is
   False there either way. Check `torch.version.cuda`, or run a real GPU job.
   `uv sync` also drops packages the code imports but `pyproject.toml` never declares:
   `h5py`, `nibabel`, `pandas`, `pyarrow`, `huggingface_hub`, `matplotlib`, `flask`,
   `nilearn`, `antspyx`, `boto3`. (`deepmreye/__init__.py` imports `ants` at package level,
   so antspyx is required even to build a cache.)
3. **`results/` is gitignored, so the laptop's trial JSONs are not in the clone.** Running
   `summarize_voxelnet_trials.py` with its default `--out-dir docs` would have rewritten both
   CSVs from the single JSON on this machine, deleting trials 02-05 from the record. The rows
   below were generated to a temp directory and **appended**. Anyone continuing on a third
   machine has to do the same until the JSONs travel with the branch.

### trial07 -- trial05's configuration on a long cosine schedule (fold dsL07)

```
--shift 2 --dropout 0.2 --weight-decay 1e-2 --epochs 150 --patience 40 --cosine
--encoder cnn --rank 64 --width 16 --hidden 256 --lags 1 --chunk 128 --batch-chunks 2
--lr 1e-3 --steps-per-epoch 150 --val-datasets 3 --val-subjects 8
```

**Rationale (handoff §3, item 1).** trial05 was the best arm so far (test 0.7037) and it was
*still improving at epoch 30* when patience 8 stopped it at 38. trial06 had already shown
that an augmented run stopped before ~50 epochs has not tested its configuration. So this is
trial05's exact configuration given the schedule trial06 established as necessary: 150 epochs,
cosine decay, patience 40.

A100 timing: **665 s for 124 epochs**, against ~2 h for 150 epochs on the laptop MPS. The
screen is now cheap enough that schedule length is no longer the binding constraint.

| epoch | loss | fit r | val r |
|---|---|---|---|
| 0 | 0.7054 | 0.6607 | 0.6626 |
| 5 | 0.5889 | 0.7284 | 0.6530 |
| 10 | 0.5115 | 0.7379 | 0.6820 |
| 20 | 0.4559 | 0.7655 | 0.6799 |
| 30 | 0.4526 | 0.7838 | 0.6996 |
| 43 | 0.4201 | 0.8109 | 0.7357 |
| 50 | 0.4092 | 0.8096 | 0.7318 |
| 60 | 0.3517 | 0.8270 | 0.7144 |
| 64 | 0.3593 | 0.8403 | 0.7383 |
| 70 | 0.3632 | 0.8444 | 0.7238 |
| 80 | 0.3470 | 0.8529 | 0.7270 |
| 84 | 0.3520 | 0.8517 | **0.7514 (best)** |
| 90 | 0.3559 | 0.8655 | 0.7497 |
| 100 | 0.3304 | 0.8716 | 0.7339 |
| 110 | 0.3089 | 0.8809 | 0.7332 |
| 120 | 0.3420 | 0.8895 | 0.7260 |
| 124 | 0.3324 | 0.8887 | 0.7284 |

**RESULT: dsL07 incumbent 0.7471 -> scratch net 0.7076 (-0.0396).** Early-stopped at epoch
124 on patience 40 from the epoch-84 peak. `net_1tr` 0.7969.

**The long schedule is confirmed to help, and it is not enough.** Against trial05, the same
configuration run 124 epochs instead of 38 improves best val r 0.7317 -> **0.7514** (+0.0197)
and fit r 0.7963 -> 0.8517. This is the best from-scratch arm to date and it narrows the
deficit to the incumbent from -0.0434 to -0.0396. But that is a **+0.0039 move in test r** —
well inside the ~0.02 noise floor. The schedule bought a fifth of a noise floor on the number
that decides the question.

**The interesting number is the val/test gap, not the fit/val gap.** Read the three together:

| | trial05 (38 ep) | trial07 (124 ep) |
|---|---|---|
| fit r at best | 0.7963 | 0.8517 |
| val r best | 0.7317 | 0.7514 |
| test r | 0.7037 | 0.7076 |
| fit - val | 0.0646 | 0.1004 |
| val - test | 0.0280 | 0.0438 |

Both gaps widened. The longer schedule let the model fit more (fit +0.055) and that gain
propagated to validation (+0.020) but almost entirely failed to reach test (+0.004). Since
`val` is itself held-out *datasets*, this is not the ordinary train/val story — it says the
extra capacity the schedule unlocked is being spent on structure that generalises across the
three selection datasets but **not** to a ninth unseen one. More schedule will not fix that,
and the widening `val - test` means selection is now also mildly optimistic: picking the
epoch-84 peak overstates what dsL07 actually gets by 0.044.

**The summarizer flags this run** as peaking at 84 of 124 — inside its final 40% — which by
the handoff's own rule marks it as evidence about a schedule rather than a configuration. I
think that flag is a false positive here: patience 40 means 40 epochs passed with no
improvement, and val r is visibly flat-to-declining from ~90 onward while fit r keeps
climbing to 0.8887. This run did converge; it converged to overfitting. The flag is calibrated
for runs killed *at* their peak, which this was not.

**Next.** The shift sweep (handoff §3 item 2) is now the right move and is no longer
expensive: at 665 s/fold, `--shift 1/2/3/4/6` at this schedule is under 90 GPU-minutes total.
DeepMReye 1.0 used +-4, and every trial so far has sat at 2 or 3.

---

## Session 2 (2026-08-21): the screen was lying, and one new augmentation is real

Everything below is on Leonardo, A100, ~5-20 min per fold. The session did four things: it
established that the single-fold single-seed screen this arm has been steered by is not
sensitive enough to resolve the differences it was being asked about; it found two real
improvements (one new); it refuted two plausible ideas; and it promoted the winner to nine
folds, where the verdict reversed.

### The screen is a paired design, and nobody was pairing it

trial07 was run at three seeds to size the noise floor before trusting any ordering. The
same configuration gives **0.7262 / 0.7620 / 0.7073** -- a range of 0.055 and an SD of
**0.028**, against the ~0.02 the project quotes for a 9-fold median. Most differences the
shift sweep was being asked to resolve are smaller than that.

But the seed is not just jitter, and this is the useful part. `--seed` sets the model init,
the training-row subsample, **and which three datasets are drawn for selection**
(`train_voxelnet_scratch.py`, the `permutation(len(tr_ds))[:val_datasets]` line). So it
changes the *problem*, not only the optimisation -- which is why the incumbent moves with it
too (0.7471 / 0.7394 / 0.7469), and why seed 1 is the high seed for **every** arm tested.

That makes seed a **blocking factor**: unpaired means across seeds hide the effect, paired
differences within seed expose it. Same numbers, both ways:

| shift | mean over 3 seeds | paired vs shift 1 |
|---|---|---|
| 0 | 0.6968 +- 0.0444 | -0.0351, loses 3/3 |
| **1** | **0.7318 +- 0.0278** | -- |
| 2 | 0.7176 +- 0.0346 | -0.0143, loses 3/3 |
| 3 | 0.6961 +- 0.0177 | -0.0357, loses 3/3 |
| 4 | 0.6665 (1 seed) | |
| 6 | 0.6425 (1 seed) | |

Unpaired, shift 1 vs shift 2 is 0.0143 against SDs of 0.03 -- unresolvable. Paired, shift 1
wins **3/3 against every alternative**. **Report paired differences on this screen; a mean
+- SD over seeds throws the design away.** It also revises the handoff's finding 5: shift
does help (+0.035 over shift 0, 3/3), but the optimum is **1**, not the 2-3 every trial had
used, and DeepMReye 1.0's +-4 is clearly too large here (-0.08).

### A new augmentation: the left-right mirror

The eye crop contains both orbits, which are approximate mirror images about the midline.
Both eyes rotate conjugately, so reflecting the crop in x maps a real sample to another
*physically valid* one with **horizontal gaze negated and vertical gaze untouched** -- an
x-reflection leaves the superior-inferior axis alone. Unlike noise, mixup or voxel dropout,
this adds data without trading away label fidelity.

Two details that decide whether it works:

1. **The crop is not centred on the midline.** Voxel counts peak at x~9 and x~38 with the
   trough at x=24, so a bare `flip` misaligns the lobes. Measured over the corpus mask,
   `flip + roll +1` maximises self-overlap: **IoU 0.910, 95.3% of voxels matched**, against
   0.855 / 92.2% for the bare flip. `voxelnet.mirror_index` returns this as a gather index,
   so the augmentation costs one fancy-index per chunk and never materialises a grid.
2. **The label must be negated BEFORE per-dataset z-scoring.** The symmetry is
   `gaze_x -> -gaze_x` in the centred coordinates the labels are stored in. Negating the
   standardised value instead is wrong by `2*mean/sd`.

**It was verified against the data before being trained on**, which matters because the
argument above is a physical claim, not a measurement. Fitting the incumbent's readout on
ORIGINAL rows only and then scoring held-out `dsL07` participants both ways:

| scoring | median r |
|---|---|
| original | **0.7480** |
| mirrored, horizontal gaze negated | **0.7455** |
| CONTROL: mirrored, gaze NOT negated | **-0.0069** |

A readout that never saw a mirrored voxel transfers to mirrored data at -0.0025, and
collapses if the sign is not flipped. The symmetry is real. Per-subject deltas span
-0.027..+0.032 with mean ~0.

**Result: +0.0060, 3/3 seeds** on top of shift 1 (`--mirror 0.5`).

### Test-time mirroring is the bigger half

If the symmetry is exact then averaging the prediction over the input and its mirror (with
the horizontal outputs negated back) is an unbiased variance reduction, free at training
time. **`--tta-mirror`: +0.0118, 3/3 seeds** -- twice the gain of using the mirror in
training, for no training cost at all.

> One implementation note, because it produced a wrong number here first. The per-epoch
> scoring and the final test scoring were **two separate loops**, so an inference-time option
> added to `score_parts` applied to *selection only* and was then reported as a test result.
> They now share one `predict_slice`. If you add anything at inference time, check both paths
> or the run measures something other than what it claims.

Recovering the un-mirrored horizontal prediction exactly would also subtract `2*mean/sd`,
which is unknown for a held-out dataset -- but it is constant across a participant's rows and
Pearson r is translation invariant, so it cannot move the score.

### dsL07 after all of it

| arm | seed0 | seed1 | seed2 | mean | vs incumbent |
|---|---|---|---|---|---|
| trial05 (handoff best) | 0.7037 | -- | -- | 0.7037 | -0.0434 |
| cnn + shift 1 | 0.7262 | 0.7620 | 0.7073 | 0.7318 | -0.0126 |
| + mirror 0.5 | 0.7377 | 0.7663 | 0.7097 | 0.7379 | -0.0066 |
| **+ TTA mirror** | **0.7470** | **0.7699** | **0.7322** | **0.7497** | **+0.0053** |

**+0.046 over the handoff's best**, and on this fold the from-scratch net is level with the
incumbent (SEM 0.0133, so a tie rather than a win).

### Two ideas refuted

**Initialising the encoder at the unsupervised basis does not help (-0.0057, wins 1/3).**
The hypothesis was that the net's deficit is a missing prior: the incumbent's `lr-cca` basis
is fitted on 2000 unlabeled participants, while the net must rediscover that structure from
eight labeled datasets. `--init-basis` puts a `lowrank` encoder at exactly that basis with a
**random** head and nothing frozen -- which is *not* the rejected warm start (that froze a
fitted RidgeCV head, trained a zero-init residual branch, and reported the incumbent whenever
its gate rejected; here there is no gate and a fold can score below the incumbent). Against
an identically-shaped random-init control it is slightly **worse**. So the deficit is not a
missing unsupervised prior. Separately, the conv encoder beats the rank-32 linear one by
**+0.0190, 3/3** -- spatial structure in the encoder is worth more than a good linear
starting point.

**Smoothing the selection metric does not help (-0.0060, wins 1/3).** The handoff proposes a
trailing mean over `val r` as the fix for its noisy-peak trap. Implemented as `--val-smooth`
and measured at K=5, it loses on 2 of 3 seeds. Worth knowing, since it is the fix the file
recommends; the noise is apparently not the kind a trailing mean removes.

### The promotion, and the reversal

`--array=0-8`, the winning arm (`shift 1 + mirror 0.5 + TTA`), seed 0:

| fold | incumbent | net | delta |
|---|---|---|---|
| dsL01_guided_fixations | 0.7678 | 0.7730 | **+0.0052** |
| dsL02_pursuit | 0.9169 | 0.9114 | -0.0055 |
| dsL03_pursuit | 0.7751 | 0.7107 | -0.0644 |
| dsL04_pursuit | 0.8488 | 0.8350 | -0.0138 |
| dsL05_free_viewing | 0.8039 | 0.7879 | -0.0160 |
| dsL06_sequences | 0.6760 | 0.5603 | -0.1157 |
| dsL07_deepmreye_calib | 0.7471 | 0.7466 | -0.0005 |
| dsL08_studyforrest_movie | 0.2859 | 0.1276 | -0.1583 |
| dsL11_backtothefuture | 0.6693 | 0.6143 | -0.0550 |
| **median** | **0.7678** | **0.7466** | **-0.0212** |
| mean | 0.7212 | 0.6741 | -0.0471 |

**The net wins 1 of 9 folds. The dsL07 tie did not generalise, and dsL07 is the second-best
fold for this arm out of nine.** Screening on it and promoting when it ties is exactly the
procedure that produced this, so the convention in this file is not safe as written:
**dsL07 is an easy fold for the network, and a configuration that ties there is still
~0.02 (median) to ~0.05 (mean) behind over nine.**

The losses are not spread evenly. They concentrate on **dsL08 (-0.158)** and **dsL06
(-0.116)** -- the two smallest labeled datasets (15 and 6 participants) and the two with
documented pathologies in `CLAUDE.md` (7T orbit registration; a vertical axis that is broken
in the data). The mechanism is visible in the ordering: the incumbent's basis is fitted on
2000 unlabeled participants and is *independent of the fold*, so it degrades gracefully when
the held-out study is unlike the training pool, whereas the net has to learn its entire
representation from eight labeled datasets and collapses when the ninth is unfamiliar. The
mean-vs-median gap (-0.047 vs -0.021) is that tail.

### Ensembling: where the remaining headroom is

Five seeds of the winning arm on dsL07, predictions saved and combined offline
(`scripts/analyze_voxelnet_ensemble.py`):

| arm | median r |
|---|---|
| incumbent | 0.7479 |
| net, individual seeds | 0.6900 - 0.7696 |
| **net, 5-seed ensemble** | **0.7735** |
| 0.50 * ensemble + 0.50 * incumbent | 0.7773 |
| 0.75 * ensemble + 0.25 * incumbent | **0.7776** |

The ensemble beats **every** individual seed and the incumbent by +0.026, which is what a
0.028 seed SD predicts: most of the per-seed shortfall is variance, not bias. Mixing in the
incumbent adds only ~0.004 on top.

**Residual correlation between net and incumbent is 0.7396** -- they make substantially
different errors, so the two are partly complementary rather than the same model in
different clothes. (An earlier version of that number read 0.9892 and was an artifact:
z-scored predictions were differenced against **raw-unit** labels, which makes every residual
approximately `-y` and correlates them at ~1 by construction. The analysis script now
z-scores the target too, and says why.)

This is the one lever with real headroom left, and it needed testing where it matters: the
+0.026 above is on the single fold the arm is already best on. So a **9-fold x 3-seed grid**
(27 runs) was run to settle it.

### The ensemble grid: 27 runs, 9 folds, 3 seeds

`scripts/analyze_voxelnet_ensemble.py --by-fold`. Every fold gets a 3-seed ensemble of the
winning arm; `mix` additionally blends the frozen incumbent in.

| fold | n | incumbent | single net | 3-seed ens | 0.50*ens + 0.50*inc |
|---|---|---|---|---|---|
| dsL01_guided_fixations | 170 | 0.7679 | 0.7730 | 0.7784 | **0.7815** |
| dsL02_pursuit | 9 | 0.9169 | 0.9114 | 0.9244 | **0.9289** |
| dsL03_pursuit | 24 | 0.7753 | 0.7107 | 0.7862 | **0.7910** |
| dsL04_pursuit | 34 | 0.8489 | 0.8350 | 0.8822 | **0.8844** |
| dsL05_free_viewing | 27 | 0.8040 | 0.7879 | 0.8247 | **0.8332** |
| dsL06_sequences | 6 | **0.6756** | 0.5603 | 0.5668 | 0.6377 |
| dsL07_deepmreye_calib | 15 | 0.7479 | 0.7466 | 0.7582 | **0.7745** |
| dsL08_studyforrest_movie | 15 | **0.2851** | 0.1276 | 0.1712 | 0.2516 |
| dsL11_backtothefuture | 37 | 0.6691 | 0.6143 | 0.6757 | **0.6896** |

| arm | median | vs inc | **mean** | folds won | sign-test p |
|---|---|---|---|---|---|
| incumbent | 0.7679 | -- | 0.7212 | -- | -- |
| single net, seed 0 | 0.7466 | -0.0213 | 0.6741 | 1/9 | -- |
| 3-seed ensemble | 0.7784 | +0.0105 | 0.7075 | 7/9 | 0.18 |
| 0.50*ens + 0.50*inc | 0.7815 | +0.0136 | **0.7303** | 7/9 | 0.18 |
| 0.75*ens + 0.25*inc | **0.7832** | **+0.0153** | 0.7234 | 7/9 | 0.18 |

**Ensembling turns the arm around: 1/9 folds won becomes 7/9.** Almost the whole per-seed
shortfall was variance rather than bias, exactly as a 0.028 seed SD on a single model
predicts. That is the single most useful thing in this session for anyone continuing the arm.

**It is still not a demonstrated win, and it should not be written up as one.** Two reasons,
both visible above. The median gain of +0.0105 to +0.0153 sits **inside the ~0.02 noise floor
this project quotes for a 9-fold median**, and 7/9 folds is a sign test at **p = 0.18**. The
honest statement is that a seed-ensembled voxel network is the first arm on this corpus to
reach *parity* with `lr-cca:32 + lags -> RidgeCV` over nine folds -- not that it beats it.

**The mean is the more interesting column, and it separates the arms the median hides.** The
pure ensemble's mean (0.7075) is *below* the incumbent's (0.7212) even though its median is
above, because it still collapses on `dsL06` (-0.109) and `dsL08` (-0.114). Only the blend
lifts the mean (**0.7303**, +0.009), by cutting those two losses roughly in half while
keeping most of the gain elsewhere. So the deployable arm is the hybrid, not the network
alone -- and note what that means: with a residual correlation of 0.74 the network is
contributing information the linear readout does not have, but it is not a *replacement* for
it, because on a small or unusual held-out study the frozen basis is the safer half.

**Where it fails is where the theory says it should.** `dsL06` (6 participants) and `dsL08`
(15, 7T, the worst within-dataset registration consistency in the corpus) are the two folds
where the network has least to learn from and least resemblance to its training pool. The
incumbent's basis is fitted on 2000 *unlabeled* participants and does not depend on the fold
at all, so it degrades gracefully exactly where a learned representation cannot. That is the
same conclusion `CLAUDE.md` reaches from the feature side, arrived at here from the voxel
side and by a different route.

### What I would do next

1. **More seeds.** The ensemble is the whole effect and 3 is the minimum that shows it.
   5-10 seeds per fold is ~2 GPU-hours per fold and should tighten the median further; the
   dsL07 5-seed ensemble (0.7735) already beat its 3-seed one.
2. **Fix the two tail folds, or accept the hybrid.** A principled fallback for small/unlike
   held-out studies would rescue the mean. The blend is the version of that which needs no
   post-hoc rule, and picking the blend weight per fold from the test score would not be.
3. **Do not screen on one fold again.** Screen on at least three, chosen to span the range
   (`dsL07` easy, `dsL03` mid, `dsL08` hard), and always paired across seeds.

---

## Session 3 -- hyperparameter tuning of the single network (no ensembles)

The Session 2 headline was a **seed ensemble**, and that was rejected: the brief is to make
the *single* network better by tuning it, not to average several of them. Everything below is
one network per (fold, seed) cell.

### Protocol

Screening runs on three folds (`dsL03_pursuit`, `dsL05_free_viewing`, `dsL07_deepmreye_calib`)
x two seeds, and every configuration is compared to a base on the **identical six
(fold, seed) cells** -- seed selects both the initialisation and the validation datasets, so
it is a blocking factor and only paired differences mean anything.

**Selection is on `best_val`, never on the held-out fold.** The validation datasets are drawn
from each fold's *training* pool, so tuning on them leaves the final 9-fold number clean.
That this is usable at all was measured first, on 39 dsL07 runs:

| | Pearson | Spearman |
|---|---|---|
| all 39 runs | +0.710 | **+0.811** |
| within seed 0 (n=15) | +0.884 | +0.896 |
| within seed 1 (n=11) | +0.808 | +0.645 |
| within seed 2 (n=11) | +0.787 | +0.873 |

A rank correlation of +0.81 is enough to *order* configurations without ever looking at test.

Screening deliberately runs with **TTA off**: it is inference-only, roughly
configuration-independent, and doubles the cost of every run. It goes back on for the
finalists.

### Round 1 -- learning rate x weight decay

Base `--shift 1 --mirror 0.5 --epochs 150 --patience 40 --cosine`, 9 configs x 6 cells = 54
runs. `dVal vs base` is the mean paired difference against `lr 1e-3, wd 1e-2`; `win` counts
cells where the config beat the base.

| config | mean val | dVal vs base | wins |
|---|---|---|---|
| **lr 1e-3, wd 1e-1** | **0.7860** | **+0.0016** | **5/6** |
| lr 1e-3, wd 1e-2 (base) | 0.7845 | +0.0000 | - |
| lr 1e-3, wd 1e-3 | 0.7845 | -0.0000 | 3/6 |
| lr 3e-3, wd 1e-2 | 0.7797 | -0.0048 | 1/6 |
| lr 3e-3, wd 1e-3 | 0.7791 | -0.0054 | 1/6 |
| lr 3e-3, wd 1e-1 | 0.7771 | -0.0073 | 1/6 |
| lr 3e-4, wd 1e-1 | 0.7749 | -0.0096 | 0/6 |
| lr 3e-4, wd 1e-2 | 0.7739 | -0.0106 | 0/6 |
| lr 3e-4, wd 1e-3 | 0.7739 | -0.0106 | 0/6 |

**The learning rate was already at its optimum and weight decay barely matters.** Both
directions away from `1e-3` lose, and by more than the wd column ever moves: the lr effect
spans 0.012 and the wd effect 0.002. Raising wd to 1e-1 wins 5/6 paired cells, which is a
consistent direction at a magnitude far inside the noise floor -- it is adopted as the new
base because it is free, not because it is a result.

The practical conclusion is that **this grid is a plateau**, so the gains have to come from
capacity, regularisation and the temporal/batch shape, not from the optimiser's step size.

### Round 2 -- capacity, regularisation, temporal and batch shape

One factor at a time from `lr 1e-3, wd 1e-1, shift 1, mirror 0.5, cosine, 150 epochs`.
25 configurations x 6 cells = 150 runs, all complete, none failed.

| config | mean val | dVal vs base | wins |
|---|---|---|---|
| **batch-chunks 4** | **0.7896** | **+0.0036** | **4/6** |
| **width 32** | **0.7874** | **+0.0013** | **4/6** |
| base | 0.7860 | +0.0000 | - |
| steps-per-epoch 300 | 0.7854 | -0.0006 | 3/6 |
| width 24 | 0.7851 | -0.0009 | 1/6 |
| dropout 0.1 | 0.7844 | -0.0016 | 4/6 |
| hidden 128 | 0.7844 | -0.0017 | 2/6 |
| dropout 0.0 | 0.7832 | -0.0029 | 1/6 |
| vox-dropout 0.05 | 0.7827 | -0.0033 | 1/6 |
| chunk 256 | 0.7810 | -0.0050 | 2/6 |
| hidden 512 | 0.7805 | -0.0055 | 2/6 |
| noise 0.05 | 0.7803 | -0.0058 | 2/6 |
| mirror 0.75 | 0.7797 | -0.0063 | 0/6 |
| mirror 0.25 | 0.7789 | -0.0071 | 2/6 |
| lags 2 | 0.7777 | -0.0083 | 1/6 |
| noise 0.15 | 0.7775 | -0.0085 | 1/6 |
| lags 0 | 0.7749 | -0.0112 | 2/6 |
| batch-chunks 1 | 0.7747 | -0.0114 | 0/6 |
| vox-dropout 0.15 | 0.7745 | -0.0115 | 1/6 |
| chunk 64 | 0.7738 | -0.0123 | 2/6 |
| dropout 0.5 | 0.7737 | -0.0124 | 2/6 |
| dropout 0.3 | 0.7733 | -0.0127 | 1/6 |
| width 8 | 0.7733 | -0.0127 | 1/6 |
| mixup 0.2 | 0.7729 | -0.0132 | 1/6 |
| steps-per-epoch 75 | 0.7690 | -0.0170 | 1/6 |
| lags 3 | 0.7646 | -0.0214 | 1/6 |

**The inherited settings were already at or next to the optimum on every axis.** Dropout 0.2,
lags 1, chunk 128, hidden 256, no added noise, no voxel dropout, no mixup, mirror 0.5 -- each
is the best or within 0.002 of it. Only two configurations point up, both by less than 0.004,
and both are *more of the same*: a bigger batch and a wider encoder.

### Round 3 -- optimiser knobs, including four that did not exist

`--warmup`, `--loss huber`, `--ema` and an exposed `--clip` (previously hardcoded at 1.0)
were added to the trainer for this round, with three regression tests and a GPU smoke run
before any sweep job used them. One test asserts the new warmup+cosine `LambdaLR` reproduces
`CosineAnnealingLR` **exactly** at `--warmup 0`, so every trial already on record stays
comparable.

| config | mean val | dVal vs base | wins |
|---|---|---|---|
| epochs 250 | 0.7874 | +0.0014 | 4/6 |
| warmup 15 | 0.7863 | +0.0003 | 3/6 |
| base | 0.7860 | +0.0000 | - |
| clip 0.5 | 0.7858 | -0.0002 | 3/6 |
| warmup 5 | 0.7843 | -0.0018 | 2/6 |
| clip off | 0.7835 | -0.0025 | 2/6 |
| rank 128 | 0.7821 | -0.0039 | 2/6 |
| **ema 0.99** | 0.7814 | -0.0046 | 1/6 |
| loss huber | 0.7813 | -0.0048 | 1/6 |
| huber delta 0.5 | 0.7803 | -0.0057 | 1/6 |
| rank 32 | 0.7802 | -0.0058 | 2/6 |
| **ema 0.995** | 0.7800 | -0.0061 | 1/6 |
| clip 2.0 | 0.7791 | -0.0070 | 1/6 |
| **ema 0.999** | 0.7757 | -0.0103 | 1/6 |

**Weight EMA hurts, and monotonically in its horizon** (-0.0046 / -0.0061 / -0.0103 at
0.99 / 0.995 / 0.999). It was the round's best bet -- the validation curve swings +-0.05 epoch
to epoch and averaging the weights attacks that at its source. The likely reason it fails is
that cosine decay plus early stopping already ends on a slow-moving iterate, so the average
only adds lag toward worse earlier weights. Huber loss also loses, so the sub-TR outliers it
was meant to stop chasing are apparently not what limits this model. Grad clipping was
already at its optimum and rank 64 was already the right bottleneck.

### The finding that matters: the hyperparameter space is a plateau, and it is too small

Across all **48 configurations** (Rounds 1-3, 288 paired runs), against the base on the
identical six cells:

|  | SD | worst | best |
|---|---|---|---|
| config-level `dVal` | 0.0053 | -0.0214 | **+0.0036** |
| config-level `dTest` | 0.0056 | -0.0206 | **+0.0043** |

The selection metric is doing its job -- config-level `dVal` predicts `dTest` at Spearman
**+0.643** (p < 0.001, n = 48), and at cell level +0.375 (p = 5e-11, n = 288) -- so this is
not a case of tuning on a signal that does not transfer.

**It is that the signal has nowhere to go.** The best hyperparameter change found anywhere in
the space is worth **+0.004**, the full best-to-worst range is 0.025, and the gap the network
has to close against `lr-cca:32 + lags -> RidgeCV` over nine folds is **~0.021**. Tuning
cannot deliver that: the single most helpful knob is worth a fifth of it, and the knobs are
not independent enough to stack forty of them.

This is the same conclusion `CLAUDE.md` reaches from the feature side, arrived at from the
optimiser side. `analyze_nonlinear_ceiling.py` shows gaze is *linearly* accessible from these
features, so a non-linear model has nothing to add; the temporal-envelope law says the
acquisition sets the scale and `fold-pca` already achieves it. A network whose hyperparameters
are all at their optimum and which still trails a linear readout is what both of those
predict.

### Rounds 4 and 5 -- what is left

Only two things in pure hyperparameter space remain untested. **Round 4** asks whether the
three directions that pointed up *combine*, since all three are "more" (batch-chunks 4 and 8,
width 32 and 48, 250 epochs, and their crosses), run at **four** seeds rather than two,
because a +0.004 effect cannot be resolved on six cells.

**Round 5 is judged on test, and is flagged as such.** `--val-datasets` and `--val-subjects`
change *what `best_val` means*, so they cannot be ranked by it -- a larger validation set is a
less noisy selection signal bought with a smaller fit pool, and only the test score sees that
trade. Ranking it on the three screening folds contaminates them for that one decision, so the
confirmation must come from the **six folds never used for screening**, and any 9-fold number
quoted for a Round-5 winner has to say so.

### Round 4 -- the upward directions do not combine

Batch-chunks, width and schedule length crossed, at **four** seeds (12 cells per config).

| config | dVal vs base | wins | mean dTest |
|---|---|---|---|
| width 48 | +0.0033 | 8/12 | -0.0240 |
| batch-chunks 4 | +0.0030 | 8/12 | -0.0150 |
| bc4 + epochs 250 | +0.0027 | 6/12 | -0.0165 |
| width 32 | +0.0024 | 9/12 | -0.0248 |
| bc4 + w32 + ep250 | +0.0013 | 7/12 | -0.0169 |
| epochs 250 | +0.0008 | 7/12 | -0.0249 |
| **bc4 + width 32** | **+0.0005** | 7/12 | -0.0197 |
| batch-chunks 8 | +0.0001 | 6/12 | -0.0126 |
| base | +0.0000 | - | -0.0235 |
| bc8 + width 32 | -0.0001 | 7/12 | -0.0187 |

**The combinations are worse than their parts.** `bc4` alone is +0.0030 and `w32` alone
+0.0024, but `bc4 + w32` is +0.0005; every cross lands at or below the better of its two
components. Nothing here is additive, which is what a plateau looks like when it is sampled
in two directions at once -- these are fluctuations around one optimum, not gains to stack.
Doubling again (`bc8`, `w48`) does not extend the trend either.

### Round 5 -- the one real effect, and it is not a model hyperparameter

`--val-datasets` is not a property of the network: it decides how many training datasets are
withheld to select the checkpoint on, and every one withheld is one the model cannot fit. It
therefore cannot be ranked on `best_val` (the metric itself changes), and is judged on test --
**so the three screening folds are contaminated for this decision and the confirmation has to
come from the six folds never used for screening.**

| val datasets | fit pool | mean dTest |
|---|---|---|
| **2** | **7 of 8** | **-0.0078** |
| 3 (base) | 6 of 8 | -0.0235 |
| 5 | 3 of 8 | -0.0641 |

**Monotone across three levels, and worth +0.0157 over base** -- four times the best knob in
Rounds 1-4, and the only effect all night that is large relative to the 0.02 noise floor. The
mechanism is not subtle: there are only 8 training datasets, so holding out 5 leaves 3 to fit
on, and the selection signal is nowhere near worth that much data. `--val-subjects 16` also
loses (-0.0250), and combining it with `--val-datasets 5` is the worst arm tested (-0.0591).

This reframes the whole exercise. The binding constraint on the from-scratch network is **how
much labeled data it gets**, not how it is optimised -- which is consistent with everything in
Rounds 1-3, where 48 configurations spanning every knob moved it by at most 0.004. Round 6
pushes to `--val-datasets 1` and crosses the winner with the Round-4 directions.

### Round 6 -- `--val-datasets 2` is the optimum, and the tuning finally adds up

| config | mean dTest | dVal vs base | wins (val) |
|---|---|---|---|
| **val-datasets 2 + bc4 + width 48** | **+0.0042** | -0.0203 | 4/12 |
| val-datasets 2 + val-subjects 16 | +0.0014 | -0.0293 | 5/12 |
| val-datasets 2 + width 48 | -0.0065 | -0.0171 | 4/12 |
| val-datasets 2 | -0.0078 | -0.0188 | 4/12 |
| val-datasets 2 + bc4 | -0.0168 | -0.0230 | 4/12 |
| val-datasets 1 + val-subjects 16 | -0.0131 | -0.2073 | 3/12 |
| val-datasets 1 | -0.0219 | -0.1665 | 3/12 |
| base (val-datasets 3) | -0.0235 | +0.0000 | - |

**Two is the optimum, not one.** The Round-5 trend does not continue: selecting on a single
dataset collapses the validation signal (mean `best_val` 0.537 against 0.685 at two, a drop of
0.15 that dwarfs anything else in this file) and test follows it down to -0.0219. So the
trade has an interior optimum -- one dataset is too noisy a selection signal, five costs too
much training data, two is the balance.

**And on top of `--val-datasets 2`, the Round-4 directions finally combine.** `bc4 + w48`
was worth nothing over the base (Round 4: the crosses were all below their parts) but is worth
**+0.012** over `val-datasets 2`. Read with Round 4, that says the earlier plateau was partly
an artifact of a starved fit pool: with 6 of 8 datasets to fit on, extra capacity and a bigger
batch have nothing to work with; with 7 of 8 they do.

### Where this actually lands, stated carefully

Paired over the 12 screening cells, each arm against its own per-fold incumbent:

| arm | vs incumbent | wins | Wilcoxon p |
|---|---|---|---|
| starting base | **-0.0235** | 3/12 | 0.077 (t-test **0.048**) |
| tuned finalist | **+0.0042** | 7/12 | **0.569** |
| **finalist - base** | **+0.0277** | **12/12** | **<0.001** |

**The tuning is real and it is large: +0.0277, winning 12 of 12 paired cells.** That is larger
than the ~0.021 the network had to make up, and it is the answer to "play through the
hyperparameters" -- the network was not at its optimum, and moving it there took it from
*significantly worse* than `lr-cca:32 + lags -> RidgeCV` (t-test p = 0.048 against) to
**statistically indistinguishable from it** (p = 0.57).

**It is not a win over the incumbent, and should not be reported as one.** +0.0042 with 7/12
cells and p = 0.57 is a tie. It is also optimistically biased: `--val-datasets 2` was chosen
on test, on these same three folds. The clean read is the promotion below.

### Promotion -- full 9-fold LODO, single network, TTA back on

`scripts/promote_submit.sh` + `scripts/promote_report.py`. Three arms (base, tuned finalist,
`val-datasets 2 + val-subjects 16`) x 9 folds x 3 seeds = 81 runs, one network per cell and
**no ensembling anywhere** -- the across-seed spread is reported as a spread, never averaged
into a prediction.

The report prints the 9-fold median **and** the median over the six folds never used for
screening (`dsL01`, `dsL02`, `dsL04`, `dsL06`, `dsL08`, `dsL11`). Those six are the only
unbiased estimate of the tuned arm, because `dsL03`, `dsL05` and `dsL07` picked it.

### Promotion result -- the 9-fold median claims a win the unbiased folds do not support

81 runs: 3 arms x 9 folds x 3 seeds, one network per cell, TTA on, no ensembling.

| arm | 9-fold median | vs incumbent | **6 unseen folds** | vs incumbent |
|---|---|---|---|---|
| base | 0.7511 | -0.0198 | 0.7025 | -0.0203 |
| **tuned** (`valds2 + bc4 + w48`) | **0.7808** | **+0.0099** | 0.7117 | **-0.0111** |
| `valds2 + val-subjects 16` | 0.7659 | -0.0050 | 0.7133 | -0.0095 |

**Read the two right-hand columns together, because they disagree by 0.021 and that is the
whole point.** On all nine folds the tuned network's median is 0.7808 against the incumbent's
0.7709 -- a win. On the six folds that never took part in selecting it, it is 0.7117 against
0.7228 -- a loss. The difference is exactly the selection bias `dsL03`, `dsL05` and `dsL07`
injected, and **quoting the 9-fold number alone would have been a false claim.** This is why
`promote_report.py` prints both and why the screening folds are marked in its output.

What *is* confirmed on all nine folds, paired per cell, is that the tuning beat the starting
configuration: `base - tuned` = **-0.0044, p = 0.025**; `base - valds2_s16` = **-0.0251,
p = 0.004**. The hyperparameters were genuinely not at their optimum, and moving them was
worth a real, significant amount. It was not worth enough to pass the incumbent.

**The failure is concentrated in two folds, not spread across nine.** The tuned arm beats the
incumbent on 7 of 9 folds -- `dsL04` +0.0375, `dsL03` +0.0273, `dsL05` +0.0207, `dsL01`
+0.0101, `dsL07` +0.0086, `dsL02` +0.0047 -- and then loses `dsL06` by **-0.278** and `dsL08`
by **-0.115**. Those two are exactly the folds `CLAUDE.md` predicts: `dsL06` has 6
participants and a vertical axis that is broken in the data, `dsL08` is 7T with the worst
within-dataset registration consistency in the corpus. A learned representation has least to
work with there, while the incumbent's basis is fitted on 2000 unlabeled participants and does
not depend on the fold at all.

Note also that the median and the mean disagree about which tuned arm is better, and the mean
is the more informative one here: `valds2 + val-subjects 16` has the worse median (0.7659) but
much the better per-cell mean (-0.0136 against -0.0344), because it largely stops the `dsL06`
collapse (-0.046 against -0.278). More participants in the selection set makes checkpoint
choice on a small hard fold far more stable.

### Round 7 -- spending one more fold to chase the collapse

That lead cannot be pursued for free. Tuning against `dsL06` and `dsL08` while they are
"unseen" would destroy the only unbiased estimate in this file, so **`dsL06` is deliberately
promoted to a screening fold** and the unbiased set shrinks to five (`dsL01`, `dsL02`,
`dsL04`, `dsL08`, `dsL11`). `dsL08` is deliberately *kept* unseen: it is the other collapse
fold, so it tests whether a fix found on `dsL06` generalises to the failure mode rather than
to one dataset.

Six configurations x 4 folds x 4 seeds, all on `--val-datasets 2`, varying what plausibly
stabilises selection on a small fold: `--val-subjects` 8/16/24, more dropout, less width,
and less patience.

### Round 7 -- the collapse fold is a VARIANCE problem, and no knob removes it

Six configurations x 4 folds (`dsL06` now screened) x 4 seeds = 96 runs, all on
`--val-datasets 2`.

| config | dsL03 | dsL05 | **dsL06** | dsL07 | mean |
|---|---|---|---|---|---|
| `sub16 + patience 20` | +0.0075 | -0.0003 | **-0.0932** | -0.0024 | -0.0221 |
| `sub24` | +0.0002 | +0.0022 | -0.1239 | +0.0050 | -0.0292 |
| `sub16` | +0.0094 | +0.0156 | -0.1857 | -0.0024 | -0.0408 |
| `sub16` + width 16 | -0.0088 | +0.0005 | -0.1602 | -0.0096 | -0.0445 |
| tuned (`bc4 + w48`) | +0.0005 | +0.0161 | **-0.2024** | -0.0042 | -0.0475 |
| `sub16` + dropout 0.3 | +0.0044 | +0.0016 | -0.2123 | -0.0073 | -0.0534 |

**Every configuration is within 0.02 of the incumbent on the three large folds and none is
within 0.09 on `dsL06`.** The knobs are doing almost nothing on `dsL03`/`dsL05`/`dsL07` --
the spread across six configurations is 0.018, 0.016 and 0.015 respectively -- while the
`dsL06` column spans 0.11. Whatever separates these configurations, it is only visible on the
small fold.

**And on that fold it is variance, not bias.** The four seeds of a single configuration score
`0.30 / 0.43 / 0.50 / 0.64` on `dsL06`, an SD of ~0.13, against an incumbent of **0.6731**
that has no seed variance at all because it is a deterministic ridge. The best individual
seed *reaches* the incumbent (0.6821); the mean does not come close. With 6 participants the
checkpoint choice is a lottery, and the reason the configurations differ is how wide they
make the lottery, not where they centre it.

Consistent with that, **the two mitigations that work are both "do less"**: stopping earlier
(`--patience 20`, -0.093 against -0.202) and dropping the extra capacity. The promotion's
`valds2 + val-subjects 16` -- no `bc4`, no `w48` -- reached **-0.046** on `dsL06`, better than
all six configurations here. So the capacity that buys the large folds their gains is exactly
what destabilises the small one; the tuned arm is trading `dsL06` and `dsL08` robustness for
`dsL01`-`dsL05` accuracy, and the LODO median hides the trade because it only ever reports the
middle fold.

`sub24` is the only arm that wins on `best_val` (+0.0056, 10/16), and it is *not* the best on
test (-0.0292 against `pat20`'s -0.0221). That is the one place in this file where the
selection metric and the test metric disagree in direction, and it is the small fold that
breaks them apart -- another way of saying that `best_val` measured on 2 datasets cannot see a
6-participant failure mode.

### Final promotion

Two arms, chosen for robustness rather than for peak: `p_pat20` (the tuned configuration plus
`--val-subjects 16 --patience 20`) and `p_safe` (`--val-datasets 2 --val-subjects 16
--patience 20`, no capacity increase). 9 folds x 3 seeds each.

`dsL06` is now a screening fold, so the unbiased set is **five**: `dsL01`, `dsL02`, `dsL04`,
`dsL08`, `dsL11`. `dsL08` is the one that matters -- it is the other collapse fold and was
deliberately kept out of Round 7, so it tests whether a fix found on `dsL06` generalises to
the failure mode or only to one dataset.
