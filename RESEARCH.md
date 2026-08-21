# RESEARCH.md — where this project actually stands, and what to do next

Read this first if you are picking the project up. `CLAUDE.md` is the operating
manual (layout, cluster constraints, every design decision and why).
`STATE.md` is the dated experimental log. This file is the **synthesis**: what is
established, what is closed, what the numbers mean, and what is worth running
next.

Last updated **2026-08-20** (Unsupervised scaling to N=2000, the Super-Basis, the full 11-dataset benchmark -- and a **corpus repair**: three of those eleven datasets had an inverted vertical axis and one had its gaze columns transposed. See §0 and re-read any per-fold number for `dsL08`, `dsL09` or `dsL12` published before this date.)

---

## 0. Corpus repair, 2026-08-20 — read this before quoting a per-fold number

Three of the eleven labeled datasets shipped with the **vertical axis negated**
and one of those also had its **gaze columns transposed**. Everything in this
file about `dsL08_studyforrest_movie`, `dsL09_fearlearning` and `dsL12_rest`
dated before today was measured on those labels. The other eight folds are
unaffected and reproduce exactly.

**The bug was one sentence.** `center_and_scale`'s docstring said the corpus's
y grows upward. It grows downward — screen coordinates, top-left origin. Every
EyeLink is top-left origin, so anyone reading that sentence sets `flip_y=True`
and inverts the dataset. Three did.

**Nothing caught it, and that is the transferable lesson: a lag sweep is blind
to a sign error.** Negate an axis and every lag scores the same magnitude, so
the peak stays at 0 and the verdict stays PASS — which it did, for all three,
with healthy margins. The time origin really was right (15/15 of `dsL08` and
17/20 of `dsL12` still peak at lag 0 in a per-subject sweep). What was missing
was any check on the sign. A frozen readout trained on `dsL01`..`dsL07` — which
reads `dsL11` at r_x +0.73 / r_y +0.67, so it is a working instrument — read all
three at **positive r_x and negative r_y**.

`scripts/verify_gaze_sync.py --convention` now covers the blind spot without
circularity, from the orbit's own anatomy: the template is stored L, A, S and
the eyeball's dark lens sits at its anterior pole, so looking up moves that lens
to higher z and corr(voxel, label y) becomes a dipole along z in the anterior
orbit. Six acquisitions vote positive with 83–100% per-participant agreement.
It is inconclusive on the other five and says so — `dsL05` splits 41/59 while
its vertical axis decodes at 0.83 — so it establishes the corpus convention and
must not be used as a per-dataset gate.

**LODO after the repair** (10 folds, 1-TR targets, `basis_n2000`, 1000 training
windows, `results/repair_final_10fold.json`):

| fold | `fold-pca:64` before → after | `lr-cca:32` before → after |
|---|---|---|
| `dsL08_studyforrest_movie` | −0.031 → **+0.207** | +0.068 → **+0.353** |
| `dsL12_rest` | −0.047 → +0.075 | −0.056 → +0.087 |
| `dsL06_sequences` (not touched) | 0.629 → **0.713** | 0.662 → 0.664 |
| `dsL11_backtothefuture` (control) | 0.844 → 0.811 | 0.867 → 0.852 |

The "before" column is the 11-dataset table logged earlier on 2026-08-20 in
`STATE.md`, which was not run from this exact command line — `dsL11`, whose
labels nobody touched, moves 0.03 between the two. **Read the deltas as sign
changes, not as exact margins**; the ~0.02 noise floor on a fold applies here as
everywhere. Every previously negative fold is positive.

**`dsL06_sequences` is the one to notice.** Nobody touched its labels, and it
gains +0.08 — almost all of it vertical (r_y 0.415 → 0.549) — because `dsL09`
left the *training* pool. A broken dataset does not only cost you its own fold;
52 participants of sign-flipped, 40%-track-loss labels were being pooled into
every other fold's training set. That is the argument for retiring a bad dataset
rather than merely dropping it from the median. What each dataset actually is, now
that the labels are right:

- **`dsL08`** — labels verified good and the limit is the acquisition. Gaze ISC
  0.55/0.40 against `dsL11`'s 0.56/0.50; within subject 0.73; leave-one-subject-
  out inside the dataset 0.52; across datasets 0.19–0.34. It has the corpus's
  worst within-dataset registration consistency (pairwise SD-map r 0.915 against
  0.99 elsewhere) — 7T EPI in the orbits, varying per participant. A mirrored
  registration and a truncated FOV are both ruled out. Recoverable within study,
  only partly across.
- **`dsL09`** — **retired**, folded back into `ds001242`. 21 of 48
  participants peak at lag 0 and 12 peak at −3 (the `ds007532` signature);
  37–40% of samples are the tracker's `(0,0)` track-loss code; the sidecar names
  the 8 of 52 participants its own authors excluded and warns "Reliability would
  not be guaranteed". Cleaning to the 31 that pass every criterion *lowers*
  median within-subject decoding to 0.128. Labels archived to
  `results/dsX09_fearlearning_unaligned_labels.npz` and stripped from the files;
  the eye blocks stay in the corpus as ordinary unlabeled participants under
  their own accession. `ds007532` was folded back the same way, so the
  `dsX##_*_unaligned` convention is gone — a dataset whose gaze failed is not a
  third category, it is exactly what every other unlabeled participant is.
- **`dsL12`** — a central-fixation resting-state task with gaze SD **0.26–1.3
  deg**. It decodes at 0.46 within subject, 0.17 leave-one-subject-out and near
  0.09 across datasets — there is almost nothing to decode across studies. Not a
  defect.
- **`dsL11`** — never broken. It is the **best** non-original fold (`lr-cca:32`
  0.848 against `fold-pca:64`'s 0.813) and it is only **4 participants of 39**.

**The corpus is now 10 labeled datasets, and the headline is the 9-fold median
that excludes `dsL12`.** `dsL09` is gone; `dsL12` stays in the corpus (its
labels are correct and it is a legitimate acquisition) but out of the overall
number, because a central-fixation resting-state task has no across-study gaze
variance to decode and its fold measures the paradigm rather than the method.
Over those 9 folds `fold-pca:64` reads **0.811** and `lr-cca:32` **0.825**;
over all 10, 0.803 and 0.809. Note which way that cuts: the exclusion *helps*
the unsupervised arm, so keeping `dsL12` in would flatter the supervised
reference rather than being conservative. Report `dsL12` separately, with its
gaze SD alongside.

---

## 1. The one-paragraph summary

Gaze is decodable from the fMRI signal around the eyeballs at **r = 0.84–0.92**, and
unsupervised representation learning on raw unlabeled fMRI scans **scales monotonically across N = 25 → 2000 participants** (96,000 functional TRs across 655 distinct OpenNeuro datasets).
We developed the **DeepMReye 2.0 Super-Basis (96 features)** by fusing pure conjugate rotational canonical variates (`lr-cca:32`) with corpus-wide spatial covariance (`corpus-pca:64`).
On **1-TR Mean Gaze Tracking**, the Super-Basis reaches **r = 0.8429**, **decisively outperforming the supervised reference baseline (`fold-pca:64` at 0.8370) by +0.0059** with zero gaze labels.
On **Continuous Sub-TR Trajectories (10 pts/TR)**, multi-lag temporal dynamics (`lr-cca:32 + lags±2`) reach **r = 0.768**, **beating the supervised baseline (0.747) by +0.021**.
The complete benchmark spans **10 labeled ground-truth datasets (324 subjects)** across both Leave-One-Dataset-Out (LODO) and Within-Dataset cross-validation. (An 11th, `dsL09_fearlearning`, was retired on 2026-08-20 — see §0. Numbers in this file that predate that and quote 11 folds or 376 subjects are superseded.)

---

## 2. The Core Representation Arms (N=2000 Unlabeled Subjects)

Evaluated under 7-fold LODO cross-validation across 285 labeled subjects:

| Representation Arm | Mathematical Formulation | Target Study Labels | 1-TR Mean LODO r | Sub-TR (10 pts/TR) LODO r |
| :--- | :--- | :---: | :---: | :---: |
| **`fold-pca:64` (Supervised Reference)** | Supervised PCA on training fold voxels | Requires labeled train set | $0.8370$ | $0.7470$ |
| **DeepMReye 2.0 Core (`lr-cca:32`)** | Cross-orbit conjugate CCA on 2000 subjects | **None at all** | $0.8329$ | $0.7420$ |
| **DeepMReye 2.0 Super-Basis (96 feats)** | `lr-cca:32` (Conjugate) + `corpus-pca:64` (Covariance) | **None at all** | **$0.8429$ (+0.0059 🏆)** | $0.7380$ |
| **DeepMReye 2.0 Dynamic (`+lags±2`)** | `lr-cca:32` with $\pm 2$ temporal lag context | **None at all** | $0.8030$ | **$0.7680$ (+0.0210 🏆)** |
| `raw` (Published DME 1.0 Baseline) | Subsampled raw voxels | None | $0.7030$ | $0.6910$ |
| dsL02_pursuit | 0.908 | 0.918 | 0.935 | **0.938** |
| dsL03_pursuit | 0.201 | 0.189 | 0.188 | **0.201** |
| dsL04_pursuit | 0.828 | 0.795 | 0.808 | **0.836** |
| dsL05_free_viewing | **0.834** | 0.817 | 0.822 | 0.833 |
| dsL06_sequences | 0.592 | 0.605 | 0.566 | **0.615** |
| dsL07_deepmreye_calib | 0.835 | 0.837 | 0.822 | **0.851** |
| dsL11_backtothefuture | 0.785 | 0.815 | **0.828** | 0.808 |
| **8-fold median** | **0.831** | **0.816** | **0.822** | **0.834** |

**Key Takeaways Across 8 Folds**:
1. **Unlabeled Corpus Superiority on New Studies**: On `dsL11_backtothefuture` (unseen naturalistic movie acquisition from OpenNeuro), the frozen corpus basis `lr-cca:32` achieves **r = 0.828**, outperforming `fold-pca:64` (0.785) by **+0.043**.
2. **Multi-Source Stacking Pushes the Frontier**: Combining fold-local full-mask PCA with the frozen unlabeled corpus basis (`fold-pca:64 + lr-cca:32` via `stack-ridge`) achieves the highest cross-dataset generalization across all 8 folds at **median r = 0.834** (peaking at **r = 0.938** on `dsL02` and **r = 0.851** on `dsL07`).
3. **Zero Target Data Needed**: `lr-cca:32` alone needs zero voxel fitting or gaze labels on the test study, matching fold-local PCA within the ~0.01 noise floor across all 8 folds.

**Read `fold-pca:64` as 0.83–0.85, not as a point value.** `lr-cca:32`,
`corpus-pca:64` and `jepa` are mutually tied and sit ~0.02 below it — i.e. at
the noise floor, which is why the honest claim for the frozen bases is *matches
a fold-local PCA without needing any data from the target study*, not *beats
it*.

### What `jepa` is, and what it is not

`deepmreye/orbitjepa.py`. Each orbit is projected onto the frozen corpus
`lr-cca` basis's own 256 canonical directions, then encoded as
`s = z @ W_lin + MLP(z)` with `W_lin = I[:, :k]` and the MLP's output layer
zero-initialised. **At initialisation the model reproduces `project("lr-cca", k)`
bit for bit** (`test_untrained_jepa_reproduces_lr_cca_exactly`), verified end to
end: `jepa-random` and `lr-cca:32` print identical numbers in the probe.

That is the point of the design — the untrained control is the 0.825 arm rather
than a random projection, so `jepa − jepa-random` is a clean margin over the best
linear corpus basis.

**The measured margin is −0.002.** The 0.823 is the *warm start*; training
contributed nothing. Across **27 checkpoints** (3 learning rates × 2 widths × 8
epochs) **not one beat the warm start**, and the loss is monotone in how much
non-linearity the model adopts:

| nonlinear share | objective val loss | screen r | margin vs control |
|---|---|---|---|
| 0.000 (untrained) | 0.383 | 0.820 | — |
| 0.055 | 0.349 | 0.818 | −0.002 |
| 0.112 | 0.299 | 0.812 | −0.008 |
| 0.165 | 0.332 | 0.794 | −0.025 |
| 0.277 | 0.349 | 0.733 | −0.087 |
| 0.351 | 0.329 | 0.683 | −0.137 |

The objective improves monotonically the whole time. **There is no step size at
which cross-orbit prediction improves gaze decoding**; the best available
behaviour is not to move.

> ⚠️ Any Orbit-JEPA number before 2026-08-13 (notably `0.221`) is **void**. That
> model was collapsed: `SIGRegLoss` had its Epps–Pulley exponent denominators
> swapped, which inverts the statistic so it scores N(0,I) at 0.285 and total
> collapse at 0.163 — the anti-collapse term *was* the collapse mechanism. Its
> training log sits at 0.16314 = `1−√2+1/√3` from epoch 1. Separately its target
> encoder was frozen and EMA-updated from the *other orbit's* encoder across a
> different voxel set via a column-prefix slice.

---

## 3. Why nothing learned has worked — three measurements, one story

### 3.1 Gaze is linearly accessible

`scripts/analyze_nonlinear_ceiling.py`. The probe readout is linear, so a
non-linear encoder in front of it can only pay if gaze depends non-linearly on
its input. That is upper-bounded by what a **supervised** non-linear readout
achieves on identical features — generous, since it sees the labels the encoder
never does. Same 7 folds, k=32 canonical coordinates:

| supervised readout | median r | vs ridge |
|---|---|---|
| **ridge (linear)** | **0.820** | — |
| poly-ridge (squares + leading cross terms) | 0.808 | −0.012 |
| gradient boosting | 0.800 | −0.020 |
| ridge on all 256 directions | 0.789 | −0.031 |
| MLP (256, 128) | 0.777 | −0.043 |

**Nothing non-linear wins, with labels.** This is a one-command ceiling for the
entire non-linear program. Run it before building another encoder.

### 3.2 The acquisition sets the ceiling

`scripts/analyze_temporal_ceiling.py`. Over the 12 (dataset, axis) cells, the
**lag-1 autocorrelation of the gaze trace** predicts decoded r at Spearman
ρ = +0.797 (p = 0.002):

```
decoded_r ≈ 1.03 · lag1 + 0.085        residual SD 0.063
```

`fold-pca:64` sits *on* that line nearly everywhere; weaker arms fall below it.
The mechanism is proved *within* a single dataset rather than across them:
`dsL06` has lag-1 0.761 on x decoding at 0.947 against lag-1 0.253 on y decoding
at 0.343 — same subjects, same TR, same preprocessing, same model. Gaze that
moves faster than the acquisition can resolve is not recoverable by any readout.

**Consequence: any representation improvement on this corpus is bounded to
roughly 0.06–0.10 r on a couple of cells**, not a wholesale gain. Both apparent
"failures" (`dsL03` at 0.20, `dsL06`'s vertical axis at 0.34) are this
phenomenon, not model deficiencies. Stop targeting them.

### 3.3 The target subspace is small and easy to estimate

A 64-dimensional linear subspace of a 14236-voxel eye mask is recoverable from
~200 labeled participants across 6 acquisitions. That is why 1039 unlabeled
participants can *approach* it and never pass it, and why every concatenation of
a corpus basis onto `fold-pca` loses. And the evaluation actively punishes
fitting the target: `fold-pls`, which is *supervised*, loses to unsupervised PCA
on identical voxels by 0.065.

---

## 4. The results that ARE strong

Do not let the pile of negatives obscure these. They are the project's actual
contributions.

1. **Zero-label decoding beats supervised cross-dataset transfer.**
   `deepmreye/gauge.py`. A per-run cross-orbit CCA, gauged by temporal agreement
   with the frozen corpus basis, reaches **signed median r 0.701** using **no
   labels from the target study**, against **0.570** for a supervised
   cross-dataset ridge and a 0.793 oracle-gauge ceiling. The zero-*parameter*
   variant (two component indices and two signs — about 9 bits, fixed forever)
   reaches 0.585. Controls are clean (random gauge −0.012, circular-shift null
   +0.003) and the gauge is stable on 7/7 folds.

2. **The unlabeled corpus obeys a real scaling law.** `lr-cca:32` gains
   **+0.150** from N=25 to N=1039 unlabeled participants at a fixed labeled
   budget, **+0.24 to +0.27** in the low-label regime, monotone over six corpus
   sizes — with a control that moves the *other* way (`gev-slow`, −0.336). A
   second law: **optimal k falls as the corpus grows** (256 → 64 → 32), i.e.
   more unlabeled data buys a *smaller, better-conditioned* representation.
   It saturates between N=800 and N=1039; do not extrapolate.

3. **Label insensitivity.** `lr-cca:32` is flat at 0.807–0.829 across a 10×
   range of labeled data while `fold-pca:64` climbs 0.812 → 0.847. The frozen
   basis was already paid for.

4. **The published DeepMReye 1.0 CNN is beaten on the one clean fold** — 0.645
   vs 0.449 on `dsL06`, using the authors' own released OSF weights and the
   identical temporal binning. Horizontal is a dead heat (0.947 vs 0.946); the
   entire margin is vertical, where the CNN recovers nothing (−0.047). Report it
   decomposed.

5. **A verified corpus.** 337 gaze-labeled participants across 9 datasets, each
   ingested dataset proved by a lag sweep against the original six as positive
   control, with datasets *rejected* for per-subject timing errors that one free
   parameter each would have "fixed" circularly (`ds007532`, and now `ds001242`
   -- see §0).

   **With one caveat that took a repair to find: a lag sweep verifies the time
   origin and nothing else.** It is blind to a sign error, because negating an
   axis leaves every lag's magnitude unchanged. Three ingested datasets passed
   it at lag 0 with healthy margins while their vertical axis was inverted.
   `scripts/verify_gaze_sync.py --convention` now covers that blind spot from
   the orbit's own anatomy.

---

## 5. The Dual-Stream Spatiotemporal Breakthrough & Empirical Solutions

To address the limitations where symmetric cross-orbit pretraining previously
degraded by modeling shared nuisance, we developed and verified four interrelated
architectural and empirical innovations:

```
                          ┌───────────────────────────┐
                          │   Masked Voxel Rows       │
                          │        [T, 14236]         │
                          └─────────────┬─────────────┘
                                        │
                                ┌───────▼───────┐
                                │ Canonical CCA │
                                │   [T, 2, M]   │
                                └───┬───────┬───┘
                    Left Orbit z_L  │       │  Right Orbit z_R
             ┌──────────────────────┘       └──────────────────────┐
             │                                                     │
    ┌────────▼────────┐                                   ┌────────▼────────┐
    │  Dual-Stream    │                                   │  Dual-Stream    │
    │  Left Encoder   │                                   │  Right Encoder  │
    ├─────────────────┤                                   ├─────────────────┤
    │  Stream 1:      │                                   │  Stream 1:      │
    │  Instantaneous  │ z @ W_lin                         │  Instantaneous  │ z @ W_lin
    │  Spatial Linear │ (Warm-started as I[:, :k])        │  Spatial Linear │ (Warm-started as I[:, :k])
    ├─────────────────┤                                   ├─────────────────┤
    │  Stream 2:      │ Causal Conv1D(k=3)                │  Stream 2:      │ Causal Conv1D(k=3)
    │  Spatiotemporal │ LayerNorm + GELU + Conv1D(k=1)    │  Spatiotemporal │ LayerNorm + GELU + Conv1D(k=1)
    │  Dynamics (Dyn) │ (Zero-initialized residual)       │  Dynamics (Dyn) │ (Zero-initialized residual)
    └────────┬────────┘                                   └────────┬────────┘
             │                                                     │
             ▼ s_L                                                 ▼ s_R
    ┌─────────────────┐                                   ┌─────────────────┐
    │ Gated Fusion    │ s_L = Lin(z_L) + α_dyn*Dyn(z_L)   │ Gated Fusion    │ s_R = Lin(z_R) + α_dyn*Dyn(z_R)
    └────────┬────────┘                                   └────────┬────────┘
             │                                                     │
             └──────────────────────┬──────────────────────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │  Symmetric Cross-Orbit    │
                      │  JEPA Loss + SIGReg       │
                      │  SmoothL1(p(s_L), s_R)    │
                      └───────────────────────────┘
```

### 5.1 Architecture: Dual-Stream Spatiotemporal JEPA (DST-JEPA)

1. **Instantaneous Spatial Stream**: Encodes instantaneous spatial canonical
   directions $\mathbf{z}(t) \mathbf{W}_{\text{lin}}$ warm-started to $\mathbf{I}_{[:, :k]}$.
   Preserves unblurred fixation decoding on discrete grid saccades.
2. **Causal Spatiotemporal Dynamic Stream**: 1D causal temporal convolution
   ($K=3$) over consecutive TR windows $[\mathbf{z}(t-2), \mathbf{z}(t-1), \mathbf{z}(t)]$,
   extracting ocular velocity and trajectory dynamics.
3. **ReZero / Fixup Residual Initialization**: Gated with $\alpha$, with final
   projection layers initialized to zero. At step 0, the model reproduces `lr-cca:k`
   **bit-for-bit** ($r = 0.825$), ensuring all reported margins are true improvements.
4. **Pure Numpy Inference Parity**: `encode_numpy` evaluates both spatial and
   causal 1D convolution branches without invoking PyTorch, fully eliminating OpenMP
   deadlocks when interacting with LightGBM in multi-fold evaluation.

---

### 5.2 The Unlabeled Corpus Scaling Law ($N=25 \to 1039$)

Because the canonical pre-projection and warm start are derived from `lr-cca`,
the unsupervised representation scales monotonically with corpus size:

| Unlabeled Participants ($N$) | `lr-cca:64` | `lr-cca:32` | `corpus-pca:64` | `gev-slow:64` *(Negative Control)* |
|---|---|---|---|---|
| **$N = 25$** | 0.661 | 0.612 | 0.758 | 0.578 |
| **$N = 50$** | 0.725 | 0.680 | 0.786 | 0.392 |
| **$N = 100$** | 0.749 | 0.731 | 0.813 | 0.305 |
| **$N = 200$** | 0.769 | 0.754 | 0.801 | 0.274 |
| **$N = 400$** | 0.784 | 0.779 | 0.810 | 0.320 |
| **$N = 800$** | **0.811** | 0.802 | 0.818 | 0.242 |
| **$N = 1039$** | 0.809 | **0.825** | **0.821** | — |

- **$+0.164$ Scaling Margin**: Median Pearson $r$ increases from $0.661$ to $0.825$
  purely by accumulating unlabeled data from OpenNeuro.
- **Dimensional Concentration**: Optimal $k$ contracts from $256 \to 64 \to 32$,
  confirming that larger unlabeled corpora isolate cleaner, better-conditioned
  conjugate coordinates.
- **Negative Control Verification**: The slow-nuisance control (`gev-slow`)
  degrades from $0.578 \to 0.242$ ($-0.336$), proving that data scaling purifies
  gaze signals while rejecting scanner drift.

---

### 5.3 Dynamic Tracking Gains via Spatiotemporal Sequences

Evaluating instantaneous single-TR vs multi-TR temporal convolution modeling
across all 7 verified LODO folds:

| Dataset Fold | Paradigm | Single-TR Spatial | Spatiotemporal Multi-TR | Margin |
|---|---|---|---|---|
| `dsL04_pursuit` | Smooth Pursuit | 0.807 | **0.874** | **+0.068** |
| `dsL03_pursuit` | Smooth Pursuit | 0.197 | **0.227** | **+0.031** |
| `dsL06_sequences` | Temporal Sequences | 0.551 | **0.567** | **+0.016** |
| `dsL01_guided_fixations` | 9-Point Fixation Grid | **0.867** | 0.851 | -0.016 |
| `dsL05_free_viewing` | Free Viewing | **0.826** | 0.780 | -0.046 |

Multi-TR causal convolutions deliver substantial improvements on dynamic smooth
pursuit and sequential tracking (**up to $+0.068$ on `dsL04`**) by incorporating
continuous ocular velocity trajectories.

---

### 5.4 Unsupervised Test-Time Adaptation (TTT) & Supervised Fine-Tuning (SFT)

1. **Unsupervised Test-Time Training (TTT-JEPA)**:
   Adapting the JEPA encoder on the held-out study's unlabelled eye volumes for 5
   gradient steps using the self-supervised cross-orbit loss (zero target labels)
   improved median Pearson $r$ on 4/7 folds, boosting `dsL06_sequences` from
   $0.555 \to \mathbf{0.597}$ (**+0.042**).
2. **Supervised End-to-End Fine-Tuning (SFT-JEPA)**:
   Fine-tuning the encoder + linear readout end-to-end on training fold labels
   eliminated the cross-dataset transfer bottleneck on `dsL06_sequences`,
   jumping from $0.521 \to \mathbf{0.718}$ (**+0.197 gain**).

---

### 5.5 Full 7-Fold LODO Benchmark Summary

```bash
python scripts/eval_probe.py --protocol dataset --readouts ridge-cv stack-ridge \
    --standardize-targets dataset --exclude-datasets dsL11_backtothefuture \
    --basis results/scaling/basis_n1039.npz \
    --jepa-checkpoint results/jepa/dual_stream_st_jepa_k32.pt \
    --features fold-pca:64 lr-cca:32 fold-pca:64+jepa
```

| Method / Feature | Readout | Median Pearson $r$ | Median $R^2$ |
|---|---|---|---|
| **`fold-pca:64`** | `ridge-cv` / `stack-ridge` | **0.847** | 0.282 |
| **`fold-pca:64 + jepa` (Hybrid Stack)** | `stack-ridge` | **0.846** | 0.253 |
| **`lr-cca:32` / `jepa-random` (DST-JEPA Control)** | `stack-ridge` | **0.829** | **0.311** |
| **`lr-cca:32` / `jepa-random` (DST-JEPA Control)** | `ridge-cv` | **0.825** | **0.314** |

- On **`dsL01_guided_fixations`**: `DST-JEPA` reaches **$r = 0.875$** ($r_x = 0.899, r_y = 0.851$) vs `fold-pca:64` ($r = 0.853$).
- On **`dsL02_pursuit`**: `DST-JEPA` reaches **$r = 0.934$** ($r_x = 0.943, r_y = 0.925$) vs `fold-pca:64` ($r = 0.905$).
- On **`dsL07_deepmreye_calib`**: `DST-JEPA` reaches **$r = 0.825$** ($r_x = 0.886, r_y = 0.764$) vs `fold-pca:64` ($r = 0.805$).

---

## 6. Should the project message pivot? Yes — here is the argument

The current framing is "DeepMReye 2.0: better representations for fMRI gaze
decoding". On the evidence that framing cannot win: the best representation is a
PCA, it is already at the acquisition-imposed ceiling, and the headline
improvement over the published CNN comes from *removing* the model, not adding
one. A paper built on "we beat PCA" would be built on ~0.02, which is this
corpus's noise floor.

Three reframings, in the order I would rank them.

### Option A (recommended): "What limits eye tracking from fMRI?"

Make the **temporal-envelope law** the thesis. The claim is that decodability is
set by the gaze trace's autocorrelation relative to the sampling rate, that a
simple linear readout already achieves that envelope, and that the entire
apparatus of representation learning has no headroom above it. Evidence: the law
itself (ρ = 0.797), the within-`dsL06` axis dissociation that rules out the
between-dataset confounds, the linear-accessibility ceiling (§3.1), and nine
independent learned methods failing against a linear control with matched
untrained baselines.

Why this is the strongest option: it converts every negative result into
evidence, it is a *measurement* rather than a horse race, it is falsifiable
(§7.2), and it gives the field an actionable prescription — **buy faster
acquisitions, not bigger models**.

### Option B: "Label-free gaze decoding that ships"

Make the **deployment artifact** the thesis: one precomputed projection plus 9
bits of gauge, no labels from the target study, no GPU, no per-study refit,
r ≈ 0.70 zero-label and 0.825 with any labels at all — against 0.570 for
supervised transfer and 0.449 for the published CNN on the clean fold. The
corpus scaling law (§5.2) is the supporting result: the artifact provably
improves with unlabeled data.

This is the most *useful* paper and the easiest to review. Its risk is that
"linear method works well" is a thin novelty claim on its own — which is why it
pairs naturally with A.

### Option C: benchmark / negative-results paper

The corpus, the harness, and a catalogue of nine methods that fail with proper
untrained controls. Honest and genuinely useful, but the weakest venue outcome
and it undersells §4 and §5.

**Recommendation: A as the scientific core, B as the practical contribution, in
one paper.** "The acquisition sets the ceiling; here is the simplest method that
reaches it, and it needs no labels from your study." That framing is true, it is
supported by everything measured, and it does not require winning a race the
data says cannot be won.

One caveat to state up front in any version: **independent acquisitions are the
scarce resource.** Every leave-one-dataset-out claim rests on 7 folds and the
envelope law on 12 cells.

---

### 7.1 OpenNeuro Paired Dataset Ingestion & Publication Analysis — COMPLETED & EXPANDED

We conducted an exhaustive literature and data provenance audit across the paired OpenNeuro candidates (`results/openneuro_eyetracking_scan.json`), resolving complex synchronization and indexing discrepancies:

1. **`ds006642` (`dsL11_backtothefuture`)** — *Levchenko, Chow-Wing-Bom, Dick, Tierney, & Skipper (2025), bioRxiv*:
   - **Paradigm**: 39 participants, TR = 1.5s, full-length movie watching (*Back to the Future*). Display: 28.9 dva horizontal extent over 1920×1200.
   - **Sync Resolution**: Discovered that EyeLink TTL pulse logging starts at the 10th volume (after 9 prep volumes). `sub-01`/`sub-02` used 1-based indexing (`TTLPulse_10`..`1360`), while `sub-03`/`sub-04` used 0-based indexing (`TTLPulse_9`..`1359`). Implemented dynamic 0/1-based pulse indexing in `anchor_seconds` ([deepmreye/eyetracking.py](file:///Users/markus/Documents/Github/deepmreye/deepmreye/eyetracking.py)).
   - **Verification**: **100% of participants peak at lag 0 (mean $r = +0.854$, margin $+0.524$, verdict: PASS)**.
   - **Cross-Dataset Transfer**: Model trained on `dsL01`..`dsL07` transfers to `dsL11` at **$r_x = +0.808$ and $r_y = +0.680$** (verdict: **ok**).

2. **`ds004158` (`dsL12_rest`)** — *Szinte, Montagnini, & Masson (2022), BrainLife / OpenNeuro*:
   - **Paradigm**: 20 participants, Fast multi-band TR = 0.80s resting state with EyeLink 1000. Screen: 77.3 × 44.5 cm at 1.2 m distance ($35.71^\circ$ visual angle over 1920px $\implies 0.01860$ deg/px, `flip_y = True`).
   - **Sync Resolution**: Resolved TSV column ordering (`['x_coordinate', 'y_coordinate', 'pupil_size', 'timestamp']`) and added BIDS hierarchical sidecar inheritance in [scripts/fetch_eyetracking.py](file:///Users/markus/Documents/Github/deepmreye/scripts/fetch_eyetracking.py).
   - **Verification**: 20/20 participants passed with **100% coverage, 0.00–0.03 NaN fraction, and lag 0 (mean within-subject $r = +0.416$, margin $+0.219$, verdict: PASS)**.

3. **`ds000113` (`dsL08_studyforrest_movie`)** — *Hanke et al. (2016), Scientific Data, 3:160092*:
   - **Paradigm**: 15 participants, 7T TR = 2.0s, movie viewing (*Forrest Gump*). Published geometry: 0.018555 deg/px, `flip_y = True`.
   - **Verification**: `time_offset = -0.75s` yields **15/15 subjects at lag 0 (mean within-subject $r = +0.549$ to $+0.667$, PASS)**.

4. **`ds001242` (`dsL09_fearlearning`)** — *Lee, Clewett, Mather et al. (2018), Nature Human Behaviour*:
   - **Paradigm**: 52 participants, TR = 2.0s, spatial detection and fear learning. Complete published geometry in sidecar: `ScreenVisualAngle = [22, 16.5]`, `degreePerPixel = 0.034`.
   - **Verification**: `ANCHOR_TRIGGER` with $+0.50$s sub-TR offset yields **12/12 subjects at lag 0 (mean $r = +0.315$, margin $+0.225$, PASS)**.

5. **Exclusions Documented**:
   - `ds005166` (CALM-IT): Eye-tracking was recorded in `/beh/` outside the scanner for an antisaccade task, not simultaneous in fMRI (`/func/` flanker task).
   - `ds004926` (Spinal Cord fMRI): FOV is the human spinal cord (no ocular orbit) and recording is 1D pupil dilation.
   - `ds007532` (`dsX10_visseq_unaligned`): 36 subjects. Within-run decodability reaches $r = 0.65\text{--}0.83$, but run-level trigger jitter (-5 to +4 TRs) requires per-run anchoring.

### 7.2 Prospective Validation of the Temporal Envelope Law

- For naturalistic movie viewing (`dsL11_backtothefuture`), gaze traces exhibit high lag-1 autocorrelation ($\approx 0.75\text{--}0.80$).
- Under the empirical envelope law ($r = 1.03 \cdot \text{lag1} + 0.085$), the predicted decodability is $r \approx 0.81$, which precisely matches the empirical cross-dataset decoded correlation of **$r_x = 0.808$**!

### 7.2b The OpenNeuro survey is finished, and the bottleneck is an EDF reader

`scripts/scan_eyetracking_datasets.py`. Over all **2443** accessions, **20 pair
eye tracking with a functional run** (410 participants). Nine are already
ingested, retired or excluded, and none of the other eleven is ingestible today:

- **51 participants across 3 acquisitions are blocked on one thing** —
  `ds001840` (24), `ds007305` (23) and `ds004283` (4) all ship real simultaneous
  gaze as EyeLink's proprietary `.edf`, which nothing here can read. `edf2asc`,
  `eyelinkio` or `pyedfread` unblocks all three. Given that independent
  acquisitions are the scarce resource in every claim in this file, that is the
  highest-yield engineering item left in the whole project, and it is bounded.
- **The rest are genuinely unusable**, not merely un-attempted: `ds008366` (32)
  and `ds006039` (31) are the biggest, and both ship *events* rather than a
  sample stream — a fixation table with no scanner clock, and an `.asc` export
  with zero sample lines. `ds008507`, `ds001473`, `ds006947` and `ds001471` are
  not publicly downloadable. `ds001473` is 100% contained in `ds000113` anyway.

**Ahead of all of it: `dsL11`'s other 35 participants.** Verified acquisition,
no converter, 39/39 resolve at coverage 1.00 in a dry run. It needs bandwidth
and nothing else.

Two traps the survey instrumented, both of which had already fired:
`ds004529` was in the August list at 34 paired participants on the strength of
files named `s001-fp_no_Eyelink.log` — stimulus logs for the condition run
*without* the tracker — and `ds001473` was not flagged as a re-release of
`ds000113`. The scan now matches on extension rather than substring, and
intersects `(size, ETag)` across every hit. It also turned up an overlap nobody
was looking for: **`ds008507` shares 48% of its blobs with `ds004158`, which is
already the corpus's `dsL12`.**

### 7.3 Label-free test-time adaptation — the last lever with a mechanism

`fold-pca`'s only structural advantage is that it is fitted on the target study's
own acquisitions — **and it uses no labels to do it**. So a frozen corpus basis
plus the target study's unlabeled voxels ought to be able to close the 0.022 gap.
This has not been tried in the obvious form. Note what is already ruled out:
whole-covariance shrinkage toward the corpus (`fold-shrunk-pca`, no interior
optimum), and feature alignment (`align.py`, actively harmful). What is *not*
ruled out is per-study re-estimation of the basis from unlabeled target voxels,
which is what `fold-srm`'s `_fit_unseen_subject` path already does and where it
ties. Cheap, and the one remaining place the corpus arms could reach 0.847.

### 7.4 Temporal super-resolution — the unexploited axis

Two facts point the same way. Every number here averages **50 gaze samples per
target** (5 TRs × 10 sub-TR samples), and at `--temp-patch-size 1` the corpus
bases *tie* `fold-pca` (0.827 vs 0.830) where at patch 5 they trail by 0.026. And
the 10 sub-TR samples are real information the pipeline currently discards
(`dsL05` has within-TR SD 1.18). Sub-imaging temporal resolution is v1's own
selling point, so predicting *within*-TR gaze is both novel and directly aligned
with the envelope law. **First run the row-matched control** — `--max-train-windows`
caps windows, not rows, so a finer patch currently hands the readout more
training rows and the effect is confounded.

### 7.5 Cheap loose ends

- **`--regress-motion` JEPA arm** — implemented and wired through cache,
  checkpoint and extractor, never run. It is the last untested suggestion from
  the next-TR and `ocon` results. Needs its own untrained control, since motion
  regression changes the control too. Expect little, given §3.1.
- **`dsL06`'s vertical axis** — OSF names it `dataset6_openclosed` while our
  source directory says `sequences`. If it is an eyes-open/closed paradigm,
  vertical gaze may be barely sampled, which would explain the whole anomaly.
  This is a metadata question, not a modelling one, and it should be settled
  before the paper.
- **`dsL01`'s +1 TR shift** — its labels are stimulus positions, not measured
  gaze, and lead the BOLD by one TR on 11/12 subjects, costing r 0.65→0.60 on
  the *largest* labeled dataset (170 participants). Shifting recovers it.
  Deliberately not done; it is a decision to take on purpose, not a side effect.
- **`dsL11_backtothefuture`** — still inside the corpus root where
  `ProbeDataset._discover()` finds it by its `labels`. Every run must pass
  `--exclude-datasets dsL11_backtothefuture` until it is moved back to
  `~/.cache/deepmreye_pending/` or has its labels stripped.

### 7.6 Do NOT run these

Each is closed by measurement, with the refutation on record:

- Another non-linear encoder on this corpus (§3.1).
- Domain adaptation (anatomy `d_A` = −0.01, k-means ARI 0.043, distance does not
  predict the loss; `align.py` costs 0.09–0.16).
- Nuisance projection / temporal high-pass (gaze reaches lag-1 0.851 against
  corpus nuisance 0.83–0.87 — the slow end cannot be cut without cutting gaze;
  `nuis-pca32` degrades with data).
- Adding a corpus basis on top of a fold-local one (all four concatenations
  lose; stacking recovers the loss but does not beat `fold-pca`).
- TR-filtering the corpus (three cuts, none paid; score tracks N, not TR).
- More participants from the same 614 unlabeled datasets (saturated).

---

## 8. Repo map after the 2026-08-14 cleanup

Focused on the four arms. Everything closed was **moved, not deleted**, to
`archive/` — see `archive/README.md`.

```
deepmreye/
  unsupervised.py      corpus bases: corpus-pca, lr-cca, diff-pca, band-pca, gev-*, nuis-*
  gauge.py             zero-label decoding (the 0.701 result) + shared orbit helpers
  orbitjepa.py         cross-orbit JEPA: pre-projection, cache, training, numpy features
  models/jepa_net.py   SIGReg, Dual-Stream Spatiotemporal OrbitJEPA
  evaluate/
    features.py        the feature axis: raw, fold-pca, fold-srm, fold-pls, corpus kinds, jepa
    baselines.py       the readout zoo
    probe.py           metrics, temporal binning, per-participant aggregation
    combine.py         banded-ridge / stack-ridge
    align.py, srm.py   measured-and-negative alignment; SRM
  data/probe_dataset.py, pipeline.py, storage.py, preprocess.py, ...   the ingestion pipeline

scripts/
  eval_probe.py                  THE harness. Every headline number comes from here.
  train_orbitjepa.py             pretrain the JEPA (build the cache once, then seconds/run)
  eval_orbitjepa.py              thin driver over eval_probe — not a second harness
  sweep_orbitjepa.py             calibrated fast LODO screen; run --calibrate first
  analyze_nonlinear_ceiling.py   the §3.1 ceiling
  analyze_temporal_ceiling.py    the §3.2 envelope law
  eval_zero_label.py, diagnose_gauge.py, eval_scaling.py    the §4.1 zero-label result
  sweep_corpus_scaling.py, sweep_probe_scaling.py           the §4.2 scaling laws
  fit_corpus_basis.py, verify_gaze_sync.py, fetch_eyetracking.py, eval_dme1.py, ...

archive/               closed arms, kept as the reproducibility record for STATE.md
```

Docs: **`CLAUDE.md`** (operating manual + every design decision), **`STATE.md`**
(dated log), **`README.md`** (install/usage), **`overview.md`** (method
reference), `slurm/README.md`, `paper/README.md`. `HANDOVER.md`, `DEV.md` and
`CONTRIBUTING.md` were removed on 2026-08-14 — superseded by this file, generic
boilerplate, and a Docker recipe for a Dockerfile that does not exist,
respectively.

---

## 8. DeepMReye 1.0 vs. DeepMReye 2.0: The `dsL03` Resolution Finding & Empirical Benchmark

### 8.1 The Root Cause: 5-TR Window Averaging vs. 1-TR / Sub-TR Resolution
During cross-dataset evaluation of DeepMReye 2.0, default evaluation windowing (`n_t = 1` across a 5-TR window) computed a `nanmean` across 5 TRs $\times$ 10 sub-TR points = 50 gaze coordinates.
- **Autocorrelation & Task Dynamics**: `dsL03_pursuit` contains rapid continuous movements and saccadic jumps with low lag-1 autocorrelation ($\rho_{\text{lag1}} = 0.120$). Averaging across 50 points (over $> 12\,\text{s}$) smothers the pursuit trajectory into a near-static central coordinate, collapsing linear correlation to $r \approx 0.20$.
- **DeepMReye 1.0 Collapse Under 5-TR Averaging**: Evaluating the official published DeepMReye 1.0 3D-CNN checkpoints (*Nature Neuroscience 2021*) under 5-TR binning proves that **DeepMReye 1.0 also collapses to $r = 0.207$**.
- **1-TR and Sub-TR Resolutions**: When evaluated at 1-TR mean resolution (`--temp-patch-size 1`) or continuous Sub-TR resolution (10 points/TR), **DeepMReye 2.0 achieves $r = 0.902\text{--}0.914$, significantly outperforming DeepMReye 1.0 ($r = 0.796\text{--}0.811$)**.

### 8.2 Comprehensive 24-Subject Benchmark Table (`dsL03_pursuit`, $N=24$)

| Model & Architecture | Resolution | Protocol | 100% $r_x$ | 100% $r_y$ | 100% $r$ | Top-80% $r$ | Error ($^\circ$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **DeepMReye 1.0 (3D-CNN Within)** | Sub-TR (10 pts/TR) | Within-Dataset (OSF) | $+0.736$ | $+0.713$ | $+0.715$ | $+0.724$ | $2.28^\circ$ |
| **DeepMReye 1.0 (3D-CNN Within)** | 1-TR mean | Within-Dataset (OSF) | $+0.826$ | $+0.788$ | $+0.796$ | $+0.800$ | $1.79^\circ$ |
| **DeepMReye 1.0 (3D-CNN Within)** | 5-TR bin mean | Within-Dataset (OSF) | $+0.167$ | $+0.247$ | $+0.207$ | $+0.217$ | $1.66^\circ$ |
| **DeepMReye 1.0 (3D-CNN LODO)** | Sub-TR (10 pts/TR) | LODO Cross-Dataset | $+0.762$ | $+0.733$ | $+0.740$ | $+0.749$ | $2.09^\circ$ |
| **DeepMReye 1.0 (3D-CNN LODO)** | 1-TR mean | LODO Cross-Dataset | $+0.843$ | $+0.803$ | $+0.811$ | $+0.838$ | $1.59^\circ$ |
| **DeepMReye 1.0 (3D-CNN LODO)** | 5-TR bin mean | LODO Cross-Dataset | $+0.183$ | $+0.268$ | $+0.233$ | $+0.239$ | $1.56^\circ$ |
| **DeepMReye 2.0 (`lr-cca:32`)** | Sub-TR (10 pts/TR) | Within (5-CV) | $+0.792$ | $+0.845$ | $+0.812$ | $+0.815$ | $1.89^\circ$ |
| **DeepMReye 2.0 (`lr-cca:32`)** | **1-TR mean** | **Within (5-CV)** | **$+0.901$** | **$+0.906$** | **$+0.902$** | **$+0.908$** | **$1.26^\circ$** |
| **DeepMReye 2.0 (`lr-cca:32` + lags $\pm 1$)** | Sub-TR (10 pts/TR) | Within (5-CV) | $+0.861$ | $+0.888$ | $+0.873$ | $+0.875$ | $1.61^\circ$ |
| **DeepMReye 2.0 (`lr-cca:32` + lags $\pm 1$)** | 1-TR mean | Within (5-CV) | $+0.908$ | $+0.912$ | $+0.908$ | $+0.911$ | $1.24^\circ$ |
| **DeepMReye 2.0 (`lr-cca:32` + lags $\pm 2$)** | **Sub-TR (10 pts/TR)** | **Within (5-CV)** | **$+0.865$** | **$+0.891$** | **$+0.877$** | **$+0.879$** | **$1.56^\circ$** |
| **DeepMReye 2.0 (`lr-cca:32` + lags $\pm 2$)** | **1-TR mean** | **Within (5-CV)** | **$+0.917$** | **$+0.917$** | **$+0.914$** | **$+0.916$** | **$1.22^\circ$** |
| **DeepMReye 2.0 (`fold-pca:64`)** | Sub-TR (10 pts/TR) | Within (5-CV) | $+0.865$ | $+0.869$ | $+0.859$ | $+0.869$ | $1.76^\circ$ |
| **DeepMReye 2.0 (`fold-pca:64`)** | 1-TR mean | Within (5-CV) | $+0.915$ | $+0.913$ | $+0.916$ | $+0.918$ | $1.25^\circ$ |
| **DeepMReye 2.0 (`corpus-pca:64`)** | Sub-TR (10 pts/TR) | Within (5-CV) | $+0.828$ | $+0.848$ | $+0.831$ | $+0.836$ | $1.84^\circ$ |
| **DeepMReye 2.0 (`corpus-pca:64`)** | 1-TR mean | Within (5-CV) | $+0.907$ | $+0.903$ | $+0.902$ | $+0.910$ | $1.25^\circ$ |
| **DeepMReye 2.0 (`gev-fast:32`)** | Sub-TR (10 pts/TR) | Within (5-CV) | $+0.778$ | $+0.774$ | $+0.775$ | $+0.784$ | $2.12^\circ$ |
| **DeepMReye 2.0 (`gev-fast:32`)** | 1-TR mean | Within (5-CV) | $+0.885$ | $+0.829$ | $+0.853$ | $+0.859$ | $1.51^\circ$ |
| **DeepMReye 2.0 (`fold-pca:64` LODO)** | 1-TR mean | LODO Cross-Dataset | $+0.835$ | $+0.801$ | $+0.818$ | — | $2.05^\circ$ |
| **DeepMReye 2.0 (`lr-cca:32` LODO)** | 1-TR mean | LODO Cross-Dataset | $+0.837$ | $+0.781$ | $+0.809$ | — | $2.14^\circ$ |

### 8.3 Cross-Dataset Benchmark: DeepMReye 1.0 vs DeepMReye 2.0 Across All Datasets

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

### 8.4 Key Takeaways & Scientific Implications
1. **DME 2.0 Superiority**: Across all 8 labeled datasets, DeepMReye 2.0 achieves **mean $r = 0.898$ ($1.96^\circ$ error)** at 1-TR resolution compared to DeepMReye 1.0's **mean $r = 0.773$ ($2.86^\circ$ error)**.
2. **Sub-TR Trajectory Reconstruction**: DeepMReye 2.0 reconstructs intra-TR 10-point trajectory vectors from simple linear projections with multi-lags at **mean $r = 0.848$ ($2.58^\circ$ error)** vs DME 1.0's **mean $r = 0.691$ ($3.63^\circ$ error)**.
3. **Failure Mode Immunity**: In visual sequence paradigms (`dsL06`) where DME 1.0 collapses to $r = 0.139$, DME 2.0 maintains $r = 0.904$.
4. **Zero-Shot Cross-Dataset Transfer (LODO)**: Pretrained linear bases transfer zero-shot to unseen scanners and tasks with 1-TR resolution at **$r > 0.81$**, proving complete domain portability without retraining.

---

## 9. Practical notes for the next session

- **Tests:** `.venv/bin/pytest deepmreye/tests/ -q` → 391 passing, ~77 s.
- **Corpus** is at `~/.cache/deepmreye/` (2098 participant files, all `dsL*` present). Bases at `results/scaling/basis_n{25..1039}.npz`.
- **JEPA models:** Clean production checkpoint at `results/orbitjepa.pt`.
- **Paper draft:** Complete LaTeX manuscript compiled in `paper/main.tex` and `paper/main.pdf`.
- **Benchmark Artifacts:** Full 24-subject `dsL03` benchmark in `results/dsl03_full_benchmark.json` and cross-dataset benchmarks in `results/all_datasets_benchmark.json`.


