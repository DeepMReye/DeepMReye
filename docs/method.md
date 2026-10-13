# How DeepMReye 2.0 works

## The idea

During an fMRI run the eyeballs rotate, and the BOLD signal in and around them
changes with gaze. A decoder can read gaze from those voxels, but a decoder
trained on gaze labels learns the acquisitions it was trained on and transfers
poorly to new scanners, sequences and tasks.

DeepMReye 2.0 learns the representation without labels. Normal eye movements
are conjugate: both eyes rotate together. Write the left- and right-orbit voxel
time series as

    X_L = S W_L' + N_L        X_R = S W_R' + N_R

with `S` the gaze-driven rotation, `W` each orbit's anatomical projection and
`N` noise local to each orbit. `S` is common to both orbits; `N_L` and `N_R`
are not. Canonical correlation analysis (CCA) between the orbits finds the
directions along which they correlate most, which is where `S` lives. Fitted on
thousands of unlabelled participants, it gives a fixed basis; only a linear
readout from that basis to gaze ever sees labels.

## The pipeline

1. **Preprocess** (`deepmreye.preprocess`). The run's mean volume is registered
   to the DeepMReye template in three stages with ANTs (affine on the whole
   volume, affine inside a large eye mask, SyN inside the small eye mask), and the
   transform is applied to every volume. The eye region is cropped to a fixed
   `47 x 29 x 18` box containing the 14,236-voxel eye mask (everything outside the
   mask set to zero), each voxel is z-scored over time, each volume over space,
   and values are clipped at 5 SD.
2. **Project** (`deepmreye.basis`). Each orbit's voxels are projected onto its
   32 leading canonical directions, and the two projections are averaged:
   `z = (X_L u + X_R v) / 2`. The two orbits give two estimates of the same
   gaze, and their average is the less noisy one.
3. **Add context**. `z` at the volume before, the volume itself and the volume
   after (96 features). This raises the sub-TR median from 0.741 to 0.771; per
   TR it makes no difference.
4. **Read out** (`deepmreye.model`). A ridge regression maps the 96 features to
   the 20 gaze values of each volume (ten samples, two axes). The penalty is
   chosen by leave-one-out cross-validation over 10^-2 to 10^4 on the training
   rows, and gaze is z-scored within each training dataset first.

## The basis

**Corpus.** Every eligible participant of the published corpus: 2,060
participants from 652 OpenNeuro datasets. A participant is eligible when all
14,236 mask voxels carry signal (the eyes are fully in the field of view) and
the run has at least 32 volumes. No participant of a gaze-labelled dataset is
in it, and neither is any run of the three OpenNeuro accessions that three of
the labelled datasets were taken from. Each participant contributes 48 volumes,
in four blocks spread over the run, so that long runs do not dominate (98,880
volumes in all).

**Fit.** One streaming pass accumulates the mean and the 14,236 x 14,236
second-moment matrix of the masked voxels. Each orbit is reduced to its 256
leading principal components and whitened (adding 10^-3 of the largest
eigenvalue to every eigenvalue), so that directions are ranked by how strongly
the two orbits correlate along them rather than by how much variance they
carry. The SVD of the whitened cross-covariance gives the canonical pairs. The
reduction to 256 components matters: whitening all ~7,000 voxels of an orbit
amplifies low-variance noise directions, which CCA then pairs at spuriously high
correlation. The fit takes about three minutes on a laptop.

**Choices.** 32 pairs and 256 components were chosen on the evaluation folds;
any number of pairs from 24 to 128 and of components from 64 to 512 changes r
by at most 0.011 (below 24 pairs accuracy collapses). Whitening is what makes
the criterion work: interpolating from partial least squares (no whitening) to
CCA (full whitening) improves accuracy monotonically.

## Evaluation

**Labelled data.** Nine datasets, 337 participants, gaze in degrees of visual
angle at ten samples per volume: guided fixation, three smooth-pursuit datasets,
free viewing during visual search, a fixation study acquired with nine EPI
sequences, a calibration task and two feature films. Gaze comes from an eye
tracker in five datasets and from the fixation target in four.

**Protocol** (`deepmreye.evaluate`). Leave one dataset out: fit the readout on
eight datasets, test on every participant of the ninth. For each participant,
Pearson r between decoded and measured gaze on each axis, averaged over x and
y; each dataset is the median over its participants; the summary is the median
(and the mean) over the nine datasets. Sub-TR uses all ten samples, per TR
their mean. R-squared and the Euclidean error are computed after a gain and
offset fitted on the *other* participants of the held-out dataset, because
predictions are in standard units.

| | median over datasets | mean over datasets |
|---|---|---|
| sub-TR | **0.771** | 0.721 |
| per TR | **0.835** | 0.777 |

| dataset | n | sub-TR r | per-TR r |
|---|---|---|---|
| `dsL01_guided_fixations` | 170 | 0.771 | 0.787 |
| `dsL02_pursuit` | 9 | 0.920 | 0.930 |
| `dsL03_pursuit` | 24 | 0.776 | 0.810 |
| `dsL04_pursuit` | 34 | 0.851 | 0.863 |
| `dsL05_free_viewing` | 27 | 0.803 | 0.839 |
| `dsL06_sequences` | 6 | 0.667 | 0.731 |
| `dsL07_deepmreye_calib` | 15 | 0.743 | 0.835 |
| `dsL08_studyforrest_movie` | 14 | 0.285 | 0.351 |
| `dsL11_backtothefuture` | 37 | 0.672 | 0.845 |

The median and the mean differ because one dataset, studyforrest, is much harder
than the rest. `deepmreye evaluate --check` reproduces these numbers.

## Against supervised decoders

The supervised baselines see the same labelled participants: a ridge regression
on every eye voxel (averaged in 2 x 2 x 2 or 3 x 3 x 3 blocks, with and without
temporal context), partial least squares on the same voxels, and an MLP.

- With all eight training datasets, the best supervised decoder reaches 0.752
  sub-TR (PLS) and 0.823 per TR (ridge without context), against 0.771 and
  0.835. On the mean over datasets, ours leads the same-readout ridge by +0.033
  (95% CI 0.010 to 0.072, bootstrap over datasets and participants) and is ahead
  on 8 of 9 datasets.
- The margin grows as labelled data shrinks: +0.10 on the mean over datasets
  with a single labelled dataset, positive in every random draw.
- Against the published DeepMReye 1.0 network, with every decoder trained on
  the same six datasets and tested on three datasets the network never saw:
  r = 0.564 for ours, 0.480 to 0.504 for supervised voxel decoders, 0.483 for
  the network.
- A variance PCA of the same corpus with the same readout reaches 0.705 on the
  mean (64 components; 0.672 at 32). The corpus alone recovers part of the gain,
  and the cross-orbit criterion adds the rest, most clearly at matched rank and
  with small corpora.

Inside a single acquisition (readout fitted on the test participant's own
dataset) a supervised decoder is level or ahead. The basis's advantage is
transfer to acquisitions the readout has not seen.

## Scaling

- **Unlabelled participants.** From 100 to 2,060 participants the mean over
  datasets rises from 0.585 to 0.721, every dataset improves, and the basis
  overtakes the supervised decoder at about 400 participants. Gains diminish
  beyond about 800.
- **Labelled datasets.** Accuracy still rises at eight labelled datasets, and at
  a fixed number of labelled participants spreading them over more datasets
  helps as much as doubling them. New labelled acquisitions are the clearest
  way to improve the decoder.

## Confidence

Applying the readout to each orbit alone gives two decodings that average to the
real one. Their correlation is a confidence that needs no labels. Keeping each
dataset's 80% most confident participants raises the mean from 0.721 to 0.745;
ranking by the true accuracy (an oracle) gives 0.747. The same agreement on a
random 32-dimensional basis gives 0.739 and random selection 0.721, so most of
the ranking comes from comparing the two orbits.

It does not rank datasets (what fails to transfer, the feature-to-gaze map, is
shared by both orbits) and cannot flag single volumes. Signals common to both
orbits other than gaze (head motion, blinks, drift) also enter the shared
subspace; the readout can learn to ignore them, but they can raise the agreement.

## What did not help

Measured and dropped: non-linear and self-supervised encoders in front of the
linear readout (gaze is linearly accessible from these features; a supervised
non-linear readout is below ridge), deep CCA, per-participant and
site-invariant CCA, a mirror-symmetry prior, ICA and dictionary learning,
per-participant registration, temporal models of the decoded trace
(interpolation and Gaussian-process smoothing gain at most ~0.015), and
self-training on the decoder's own labels. The full record is `FINDINGS.md` on
the `pytorch_unsup` branch.

## Limits

- Both eyes must be in the field of view and the registration must hold. Check
  the QA image (`deepmreye decode --qa`); a warning is raised when eye-mask
  voxels carry no signal.
- Output is relative gaze; degrees need a calibration on a few participants with
  an eye tracker.
- Accuracy depends on the acquisition: the 3 T feature film studyforrest decodes
  at r = 0.29, the other eight datasets at 0.67 to 0.92.
