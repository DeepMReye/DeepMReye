# Handoff: from-scratch voxel network, laptop → Leonardo

Written 2026-08-21. Everything below was measured on an M-series laptop GPU (MPS) and is
being moved to Leonardo because a single fold takes 10–30 min there and the question needs
tens of folds. Read `CLAUDE.md` first for the project as a whole; this file covers only the
voxel-network arm and what the server needs to continue it.

---

## 1. The question

The incumbent is **`lr-cca:32 + lags±1 → RidgeCV`**, a frozen unsupervised linear basis with a
linear readout, at **9-fold LODO median sub-TR r = 0.7678**. The question is whether a network
trained on voxels beats it.

**This has to be answered from scratch, not warm-started.** An earlier design
(`scripts/train_voxelnet.py`) initialised the network *at* the incumbent — a frozen RidgeCV
linear branch plus a zero-initialised learned branch gated by a ReZero `alpha`. That
guarantees "at least as good as the incumbent" by construction, and it is worth nothing:

- the learned branch only ever fits the residual `y − ridge(z_cca)`, so the representation the
  head reads from is fixed at step 0 and gradient descent can decorate a good linear solution
  but never reorganise the map;
- it may only ever find a local minimum around the linear solution;
- and it **reports the incumbent's score whenever the branch is rejected**, so a model that
  never learned and a model that learned something rejected by the gate produce byte-identical
  JSON. Four gated runs each returned exactly `0.7678` — that number is the adoption gate
  firing, not a model score, and the artifacts do not distinguish the two.

The from-scratch trainer is **`scripts/train_voxelnet_scratch.py`**. The incumbent is computed
per fold and printed as a reference line; it is never in the loss and never added to a
prediction, so a fold can legitimately score below it. That is the point.

```
pred(x) = head( make_lags( g(x), L ) )       # g = 3-D conv over the eye block, or low-rank linear
```

Protocol identical to `deepmreye/temporal_probe.lodo_subtr`: leave one dataset out,
per-training-dataset target z-scoring, selection on held-out **training datasets** scored by
sub-TR r, median over test participants.

---

## 2. What is established so far

All on **one fold** (`dsL07_deepmreye_calib`, incumbent 0.7471) used as a screen. Nothing has
been promoted to nine folds yet.

| trial | config | best fit r | best val r | test r | vs incumbent |
|---|---|---|---|---|---|
| trial02 | no aug, dropout 0.1, 17 ep | 0.9198 | 0.7328 | 0.6933 | −0.0538 |
| trial03 | + 16-participant batches, 13 ep | 0.9331 | 0.7005 | 0.6870 | −0.0601 |
| trial04 | dropout 0.3, wd 0.05, shift+noise+mixup+voxdrop, 11 ep | 0.6789 | 0.6350 | 0.6543 | −0.0928 |
| **trial05** | **shift 2 only, dropout 0.2, 38 ep** | 0.7973 | **0.7317** | **0.7037** | **−0.0434** |
| trial06 | trial04 config, 150-ep cosine, **killed at ep 79** | 0.7456 | 0.6964 | — | — |
| trial07 | trial05 config, 150-ep cosine | *never started* | | | |

**Findings, in order of confidence.**

1. **A from-scratch CNN decodes gaze from voxels.** val r 0.6685 after a single epoch
   (150 steps). This is the fact the warm-start design structurally could not surface.
2. **It does not yet beat the incumbent.** Best is 0.7037 against 0.7471, a deficit of 0.043
   — well outside this project's ~0.02 noise floor.
3. **The failure mode is generalisation, not optimisation.** trial02 reaches fit r 0.9198 with
   val r 0.7328: a 0.23 train/val gap. Capacity and optimisation are fine; what is learned
   does not cross a dataset boundary.
4. **Batch diversity is not the mechanism — ruled out.** 16 participants per step scored
   *worse* than 2 (0.6870 vs 0.6933) with an unchanged gap.
5. **`--shift` (registration jitter) is the augmentation that works, and it is the only one
   tried with a physical justification** — a rigid translation of the eye crop is what a
   registration error looks like and leaves gaze unchanged, so it is label-preserving by
   construction. Alone it gives the best result and cuts the gap from 0.23 to 0.065 *without*
   the capacity collapse that mixup/noise/voxel-dropout cause together.
6. **Augmented runs converge slowly, and short runs give wrong verdicts.** trial04 stopped on
   patience at epoch 11 and read 0.6350, which I wrote up as "heavy regularisation removes the
   capacity to fit". trial06 ran the *identical* configuration for 79 epochs with cosine decay
   and reached 0.6964. That verdict was retracted. **Any conclusion drawn from a run under
   augmentation that stopped before ~50 epochs should be treated as untested.**
7. **`val r` is noisy — ±0.05 epoch to epoch** (trial02: 0.6433 → 0.7328 → 0.7047 in three
   consecutive epochs). Patience fires on that noise, and "best epoch" cherry-picks its peak.
   This is a live methodological weakness, not a solved problem — see §7.

---

## 3. What to run next

In priority order. The first is the direct follow-up to the best result.

1. **trial07 — trial05's config on a long schedule.** It was still improving at epoch 30 when
   patience 8 killed it at 38.
   `--shift 2 --dropout 0.2 --weight-decay 1e-2 --epochs 150 --patience 40 --cosine`
2. **Shift strength sweep**, the one knob known to help: `--shift 1/2/3/4/6` at the trial07
   schedule. DeepMReye 1.0 used ±4 voxels.
3. **Finish trial06** (heavy reg, 150 ep) — it was killed at 79 and was still climbing.
   Settles whether the full augmentation set eventually overtakes shift-only or plateaus
   around 0.70.
4. **Capacity**, only once the schedule is right: `--width 32 --rank 128 --hidden 512`. Do not
   test this on a short schedule; a bigger model needs longer, and the result would be
   uninterpretable.
5. **Promote the winner to all 9 folds** (`--array=0-8`) and compare against the incumbent's
   per-fold table in §6. Only a 9-fold median is quotable.
6. **SSL pretraining** (`scripts/pretrain_voxelnet.py`, band-matched temporal contrast) — only
   after a supervised configuration is competitive. It demonstrably learned on the laptop
   (InfoNCE 2.2509 → 1.7987 against chance ln(9) = 2.1972) but gave *exactly* the incumbent
   downstream, which under the warm-start design means "the gate rejected it" and is
   uninformative. It has never been tested from scratch. **Note the unlabeled cache cannot be
   rebuilt from HuggingFace — see §5.**

---

## 4. Getting the code

Branch **`pytorch_unsup`**. Everything for this arm is committed there.

```bash
git clone https://github.com/CYHSM/deepmreye.git      # or your remote
cd deepmreye
git checkout pytorch_unsup

uv venv .venv && uv sync                              # pyproject.toml / uv.lock
# torch must be a CUDA build on Leonardo, not the macOS wheel:
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Files that matter for this arm:

| path | what |
|---|---|
| `scripts/train_voxelnet_scratch.py` | **the trainer** — from scratch, no warm start |
| `scripts/build_voxel_cache.py` | builds the voxel memmaps (see §5) |
| `scripts/summarize_voxelnet_trials.py` | rolls all trial JSONs into `trials.csv` |
| `slurm/train_voxelnet.sbatch` | GPU job; array mode = one fold per task |
| `deepmreye/temporal_probe.py` | the audited sub-TR LODO protocol + `--calibrate` |
| `deepmreye/voxelnet.py` | cache builder, `cca_matrix`, augmentations |
| `docs/VOXELNET_TRIALS.md` | the narrative trial log (rationale + epoch tables) |
| `docs/voxelnet_trials.csv` | generated: one row per trial x fold |
| `scripts/train_voxelnet.py` | **the warm-start version — superseded, do not use for new work** |

---

## 5. Getting the data — and the one place it diverges from HuggingFace

**The labeled probe set is byte-identical to HuggingFace.** Verified 2026-08-21 against
`DeepMReye/eyeballs` (dataset repo) by comparing every file's size: **337/337 `dsL*`
participants match exactly, 0 mismatches, 0 missing.** So every voxelnet number in §2 is
reproducible from the Hub.

| dataset | n |
|---|---|
| dsL01_guided_fixations | 170 |
| dsL02_pursuit | 9 |
| dsL03_pursuit | 24 |
| dsL04_pursuit | 34 |
| dsL05_free_viewing | 27 |
| dsL06_sequences | 6 |
| dsL07_deepmreye_calib | 15 |
| dsL08_studyforrest_movie | 15 |
| dsL11_backtothefuture | 37 |
| **total** | **337** |

`dsL09_fearlearning` and `dsL12_rest` are correctly **absent** from both — they were retired
and folded back as unlabeled data under their accessions (`ds001242` 52, `ds004158` 20).

```bash
export DEEPMREYE_DATA=/leonardo_scratch/fast/AIFAC_S07_154/mfrey/dme/data
python -m deepmreye fetch --data-dir $DEEPMREYE_DATA          # login node: has network
```

> **Download on a login node.** Leonardo compute nodes have no outbound network. Login
> sessions are capped at 32 GB *including page cache from your own writes*, which is what
> killed the staging job twice — see the Leonardo section of `CLAUDE.md`.

**The unlabeled corpus does NOT match the Hub.** Local has **3,347** unlabeled participants;
the Hub has **1,878**. **1,470 local-only files across 429 accessions, 20.4 GB**, from
`scripts/expand_corpus_5subs.py` (the expansion to 5 subjects/dataset), which was never
uploaded.

Consequences, and they only bite for step 6 of §3:

- Anything **supervised** (trials 02–07, everything in §3 steps 1–5) uses only `dsL*` and is
  fully reproducible from the Hub.
- **SSL pretraining is not.** A cache rebuilt from the Hub would hold ~1,878 participants
  against the laptop's 3,348, so laptop SSL checkpoints are not comparable to server ones.
  Either push the 20.4 GB first (`scripts/upload_to_hf.py`) or state the corpus size with
  every SSL number.

### Building the caches

The trainers read memmaps, not HDF5. These were originally built ad hoc from an interactive
session; `scripts/build_voxel_cache.py` is that step as a command.

```bash
# labeled: 337 participants, ~405k TRs, ~11 GB fp16. Also writes z_cca_k32.npy.
python scripts/build_voxel_cache.py --out results/subtr/voxels

# unlabeled (SSL only): ~29 GB on the full local corpus, less from the Hub
python scripts/build_voxel_cache.py --out results/subtr/voxels_unlabeled --unlabeled
```

The unlabeled build **asserts no `dsL*` participant enters it**. Keep that assertion: one
labeled participant leaking into pretraining puts the same person on both sides of a LODO
split and nothing downstream would notice.

You also need the frozen basis **`results/scaling/basis_n2000.npz`** (used for the incumbent
reference and `cca_matrix`). It is not in git — copy it from the laptop or refit with
`scripts/fit_corpus_basis.py`. **Pin `n2000`**, not `n1039`: the 0.759 benchmark used `n2000`
and mixing them silently changes the baseline.

---

## 6. Verify before trusting any number

```bash
pytest deepmreye/tests/ -q                       # 444 pass as of this handoff
python -m deepmreye.temporal_probe --calibrate   # must print lr-cca:32 = 0.742, +lags2 = 0.759
```

**Do not believe any ordering until `--calibrate` passes.** It reproduces the benchmark to
0.0001/0.0005; if it drifts, the cache, the basis or the corpus has changed.

Per-fold incumbent (`ridge-L1`, sub-TR) — the reference for every comparison:

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

---

## 7. Traps — every one of these already produced a wrong number here

1. **A warm-started or gated run reports the incumbent when it rejects the branch.** Identical
   four-decimal values across configurations mean the gate fired, not that the models tied.
   `train_voxelnet_scratch.py` has no gate, which is why it is the one to use.
2. **`es_fired` in `trials.csv` is not convergence.** trial04 looked converged at epoch 11;
   the same config ran to 0.6964 by epoch 79. The summarizer flags runs peaking in their final
   40% of epochs — treat those as evidence about a *schedule*, not a configuration.
3. **`val r` swings ±0.05 between epochs**, so patience fires on noise and "best epoch"
   cherry-picks a peak. Cosine decay helps late in training. If it is not enough, the fix is
   more validation participants or a smoothed selection metric — **not** a shorter patience.
4. **Read `fit r` and `val r` together.** High fit + low val = overfit (regularise). Both low =
   underfit (train longer, raise lr, cut regularisation). trial02 and trial04 sit at opposite
   ends and reading either alone prescribes exactly the wrong change.
5. **Never `AdaptiveAvgPool3d` in a gaze encoder.** Gaze *is* the eyeball's spatial position;
   global pooling deletes it. Symptom is a training loss flat at 0.46–0.49 with the selection
   metric pinned. The conv flattens instead, as DeepMReye 1.0 does.
6. **`--adopt-margin -1 --patience 999`** is how you disable gating in the *old*
   `train_voxelnet.py` to see a raw trained score. Only relevant if you revisit that arm.
7. **`eval_probe.py` cannot score sub-TR gaze** — `temporal_targets` nanmeans the sub-TR axis.
   Every sub-TR number must come from `deepmreye/temporal_probe.py`.
8. **PYTHONHASHSEED drift is worth ~0.01 r**, half the noise floor. The sbatch pins it to 0.
9. **A retry loop that resumes by skipping finished work starves behind one deterministic
   failure.** If a fold OOMs, it will OOM again at the same place on every attempt. Reorder or
   isolate it; do not wrap the job in `for i in 1..8`.
10. **Leonardo submit:** account `AIFAC_S07_154`, partition `boost_usr_prod`. `EUHPC_D21_101`
    is expired and `dcgp_usr_prod` has no allocation — sbatch reports **both** as "invalid
    account or expired budget", which sends you chasing the account when the partition is the
    problem. Test with
    `sbatch --test-only -A AIFAC_S07_154 -p boost_usr_prod --time=00:10:00 --nodes=1 --ntasks=1 --wrap=true`.

---

## 8. Running it

```bash
# single fold, interactive-ish screen
sbatch -A AIFAC_S07_154 -p boost_usr_prod slurm/train_voxelnet.sbatch \
    --note "trial07 shift2 long" --folds dsL07_deepmreye_calib \
    --shift 2 --dropout 0.2 --weight-decay 1e-2 \
    --epochs 150 --patience 40 --cosine \
    --out results/subtr/trial07_scratch_shift_long.json

# all nine folds, one per array task (script supplies --fold-index and --out itself)
sbatch --array=0-8 -A AIFAC_S07_154 -p boost_usr_prod slurm/train_voxelnet.sbatch \
    --note "trial08 shift2 long 9fold" --shift 2 --dropout 0.2 \
    --epochs 150 --patience 40 --cosine

# roll everything up
python scripts/summarize_voxelnet_trials.py
```

**Conventions.** Next `trialNN_` prefix; `--out results/subtr/trialNN_<slug>.json`; a
`--note`. Screen on `dsL07` (cheap) and only promote to nine folds once a config beats the
incumbent there. Append the rationale and the epoch table to
`docs/VOXELNET_TRIALS.md` — the CSVs carry the numbers, that file carries the *why*.

> `results/` is gitignored, so the trial log and the rolled-up tables live in `docs/`. A
> record that does not survive a clone is not a record. `.gitignore` carries two explicit
> negations for the CSVs, because `*.csv` is otherwise excluded to keep corpus labels out.
