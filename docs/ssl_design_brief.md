# DeepMReye 2.0 — SSL design brief

Context for brainstorming the self-supervised objective. Everything below is
measured on the real corpus, not assumed.

## The problem

Decode eye gaze (x, y in degrees of visual angle) from the fMRI signal around the
eyeballs, with no eye tracker. Version 1 did this supervised. The bet for v2:
**unlabeled fMRI is abundant, gaze labels are scarce**, so pretrain a
representation on unlabeled eye-region fMRI and fit a light probe to gaze.

The question to settle: **which SSL objective, and how do we make it generalize
to a scanner and paradigm it has never seen.**

## The data

Input to the model is a fixed eye-region crop, `[47, 29, 18, T]` float32,
normalized (per-voxel z-score across time, per-volume z-score across space,
clipped at 5 SD). Non-eyeball voxels are exactly 0; occupancy is ~58%.

| | subjects | datasets | windows (100 TR, stride 50) |
|---|---|---|---|
| unlabeled pretraining | 1,332 | 691 | 7,941 |
| gaze-labeled probe set | 270 | 6 | ~6,600 |

The unlabeled half is currently a **2-subjects-per-dataset QA sample**. Full
extraction of every subject in the 697 approved datasets is pending and
extrapolates to ~20–28k subjects, so pretraining data grows ~15x. The labeled
half is fixed and will not grow.

Labels are `[T, 10, 2]` — 10 sub-TR gaze samples per TR. NaN marks missing gaze
and is common (100% of windows in two datasets contain at least one).

### The awkward part: TR varies enormously

Repetition time across the unlabeled corpus spans **0.04 s to 10 s** (median
2.0 s). A 100-TR window is therefore anywhere from 4 s to 1000 s of real time.
The six labeled datasets sit at 0.80–1.02 s (plus one at 1.25/1.8/2.5), i.e. in
the *fast tail* — only **16.7%** of unlabeled subjects fall in that band.

So the probe is a distribution shift in sampling rate before anything else.

Current plan (not yet built): make the temporal positional encoding continuous
in **seconds** rather than ordinal in bin index, and add a `log(TR)` conditioning
embedding to every token. Rationale: JEPA's predictor is explicitly positional —
it predicts a target representation *from its position* — so an ordinal encoding
forces one learned function of "3 bins away" to simultaneously mean 12 s and
37.5 s.

## Current architecture

- **Patchify**: 8×8×8 voxel cubes × 5 TRs → linear → 256-d tokens on an
  `N_S × N_T` grid (N_T = 20 for a 100-TR window).
- **Masking**: "double-cross" — drop a set of spatial indices and a set of
  temporal indices; a token is a target if *either* its row or column is dropped.
  Curriculum from 10% to 50% on each axis.
- **JEPA**: context encoder (6-layer ViT) + EMA target encoder + 3-layer
  predictor, SmoothL1 between predicted and target representations.
- **Probe**: freeze the encoder, mean-pool over **space only**, ridge from each
  temporal token's embedding to that bin's gaze.

## Measured baselines (this is the important part)

Leave-one-dataset-out, ridge probe, ~300 windows/split, **no pretraining yet**.
`R² vs mean` is against the training-set mean gaze; 0 = learned nothing.

| held out | arm | euclid (°) | R² vs mean | r_x | r_y |
|---|---|---|---|---|---|
| dsL01 guided fixations | voxels | 6.18 | **0.399** | 0.757 | 0.700 |
| | random encoder | 7.76 | 0.097 | 0.501 | 0.330 |
| dsL02 pursuit | voxels | 1.67 | **0.397** | 0.850 | 0.704 |
| | random encoder | 2.27 | −0.047 | 0.307 | 0.245 |
| dsL03 pursuit | voxels | 2.44 | **−1.103** | 0.137 | 0.196 |
| | random encoder | 2.24 | −0.801 | 0.058 | 0.059 |
| dsL04 pursuit | voxels | 1.98 | **0.033** | 0.495 | 0.600 |
| | random encoder | 2.20 | −0.149 | 0.304 | 0.380 |
| dsL05 free viewing | voxels | 2.68 | **0.420** | 0.726 | 0.650 |
| | random encoder | 3.40 | 0.113 | 0.356 | 0.364 |
| dsL06 sequences | voxels | 3.54 | **0.113** | 0.642 | 0.400 |
| | random encoder | 3.36 | 0.071 | 0.445 | 0.234 |

Four things fall out of this, and they should shape the objective choice:

1. **Ridge on raw downsampled voxels is a strong baseline** — R² ≈ 0.40,
   r ≈ 0.7–0.85 on three of six held-out datasets. Any SSL representation has to
   beat *this*, not the random encoder.
2. **A random ViT is a bad baseline, not a strong one.** It scores below raw
   voxels everywhere and below the mean on three folds. The nonlinear random
   projection is destroying linearly-available information. That is a warning
   about the patchify+ViT stack itself, independent of the objective.
3. **dsL04 shows good correlation with near-zero R²** (r ≈ 0.5–0.6, R² = 0.03).
   The map transfers in *shape* but not in *offset and scale*. Cross-dataset
   gaze calibration (screen size, viewing distance, tracker calibration) is a
   separate problem from representation quality, and it is probably worth
   handling explicitly rather than asking the encoder to absorb it.
4. **dsL03 fails completely** (R² = −1.10). Worth understanding before averaging
   it into any headline number.

## The open questions

**Objective.** JEPA is the current choice. Alternatives worth arguing about:

- **JEPA** — predicts in representation space, no pixel reconstruction, no
  negatives. Good fit for fMRI where voxel-level detail is mostly noise. Risk:
  representation collapse; relies on EMA + masking asymmetry to avoid it.
- **MAE** — reconstruct masked voxels. Simple, stable, but spends capacity on
  reconstructing noise, which in BOLD is most of the variance.
- **DINO / iBOT** — self-distillation with centering+sharpening. Strong for
  semantic invariances in images. Here the "semantics" are a *continuous
  geometric* variable (gaze direction), not a category — unclear that
  invariance-seeking objectives help, they may actively discard the signal.
- **Contrastive (SimCLR-style)** — needs augmentations that preserve gaze.
  What *is* a gaze-preserving augmentation of an eyeball BOLD block? Spatial
  flips change gaze sign. Temporal crops change the target. This seems hard to
  define, which is an argument against.

**The invariance question is the crux.** Most SSL objectives are built to
discard nuisance variation. Here the thing we want to decode is a small,
continuous, *geometric* signal — eyeball position — and the nuisance variation
(scanner, subject anatomy, head motion) is large. An objective that learns
invariance to "which scan is this" is what we want; one that learns invariance to
"where is the eyeball pointing" destroys the entire signal. Which objectives put
gaze on the right side of that line?

**Temporal structure.** Gaze changes on a timescale of hundreds of ms; BOLD is
sampled at 0.8–2.5 s and low-pass filtered by haemodynamics. What is the right
temporal patch size, and should the objective predict *forward in time*
(autoregressive / next-window) rather than fill in masked bins? The current
double-cross masking treats time and space symmetrically, which may be wrong —
they are not symmetric.

**Generalization across scanner.** With 691 pretraining datasets and 6 probe
datasets, the pretraining distribution is far broader than the evaluation. Should
the objective explicitly encourage scanner-invariance (e.g. adversarial on
dataset identity, or treating dataset as a domain in a domain-generalization
setup)? Or does the sheer breadth of 691 datasets handle it?

**Scale.** Pretraining data grows ~15x after full extraction (7,941 → ~120k
windows). Does that change the objective choice — do the more data-hungry
objectives become viable?

## Constraints

- Input is small (`47×29×18`) and mostly empty (58% occupancy). This is not
  ImageNet; capacity is not the bottleneck, signal-to-noise is.
- The probe is *linear* by design — it is a measurement of the representation,
  not a model. Whatever we do must leave gaze linearly decodable.
- Evaluation is leave-one-dataset-out and leave-one-paradigm-out (dsL02/03/04
  are all smooth pursuit, so holding out one alone still trains on the same
  task).
- Compute: development on a laptop (MPS), real runs on Leonardo (CINECA),
  boost_usr_prod partition.
