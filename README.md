# DeepMReye: gaze from fMRI, without an eye tracker

[![tests](https://github.com/DeepMReye/DeepMReye/actions/workflows/tests.yml/badge.svg?branch=deepmreye-unsupervised)](https://github.com/DeepMReye/DeepMReye/actions/workflows/tests.yml)

The eyeballs are inside the field of view of most fMRI runs, and their signal
moves with gaze. DeepMReye reads it out: give it a BOLD run and get ten gaze
samples per volume back, with no eye tracker and no labels from your study.

This version learns its representation **without labels**. The two eyes rotate
together, so gaze is part of the signal shared by both orbits. A canonical
correlation analysis between the left and right orbit, fitted on **2,060
unlabelled participants from 652 OpenNeuro datasets**, finds that shared signal.
A linear readout, fitted on 337 gaze-labelled participants from nine datasets,
turns it into gaze.

On datasets it has never seen, it decodes gaze at a median correlation of
**r = 0.771** at sub-TR resolution and **0.835** per volume, ahead of supervised
decoders trained on the same labelled data (0.752 and 0.823), and by +0.10 when
only one labelled dataset is available.

## Install

```bash
pip install git+https://github.com/DeepMReye/DeepMReye.git@deepmreye-unsupervised
```

Python 3.9 to 3.11, no GPU. Registration uses [ANTs](https://github.com/ANTsX/ANTsPy)
(`antspyx`, installed with it); the rest is numpy and scikit-learn.

## Quick start

```bash
deepmreye decode sub-01_task-movie_bold.nii.gz -o gaze/ --qa
```

```python
import deepmreye

result = deepmreye.decode("sub-01_task-movie_bold.nii.gz")
result["gaze"]          # [T, 10, 2]: ten samples per volume, x then y
result["confidence"]    # how well the two eyes' decodings agree
```

The first call downloads the pretrained decoder (2 MB) from
[HuggingFace](https://huggingface.co/DeepMReye/deepmreye-unsupervised); no
account is needed.
Registration takes from ten seconds to a few minutes per run, depending on its
length and resolution; decoding is instant. `decode`
writes, per run:

| file | contents |
|---|---|
| `<run>_gaze.tsv` | one row per gaze sample: `volume`, `sample` (0-9), `onset` (s), `x`, `y` |
| `<run>_decode.json` | volumes, TR, units, confidence, which decoder |
| `<run>_qa.png` | with `--qa`: the registered eyes over the eye mask. Look at it. |

## Which setup fits your data?

| you have | do | expect (sub-TR r) |
|---|---|---|
| fMRI only | [decode with the pretrained decoder](#1-decode-with-the-pretrained-decoder) | 0.77 on unseen datasets, relative units |
| eye tracking for 1 to 3 participants of your study | [keep the pretrained readout and calibrate it to degrees](#2-calibrate-to-degrees) | the same, in degrees |
| eye tracking for 4 or more participants | [train a readout on them, decode the rest](#3-train-on-some-participants-decode-the-rest) | 0.79 with 4, 0.82 with 16 |
| eye tracking for part of every run (a calibration block) | [one readout per participant](#4-one-readout-per-participant-from-a-calibration-block) | 0.82 with half a run, 0.80 with a quarter |
| labelled data from other setups | [add it to the published datasets](#5-add-your-labelled-datasets-to-the-published-ones) | a more general decoder |
| interest in the representation only | [basis features](#6-basis-features-only) | 32 coordinates per volume |
| hundreds of unlabelled participants of your own | [fit your own basis](#7-fit-the-basis-on-your-own-corpus) | worthwhile only at that scale |

Where these numbers come from is under [Results](#results). The rule behind the
table: a readout trained inside your own study beats the pretrained one once it
has about four labelled participants, but not before. With one, a readout fitted
on it does worse than the pretrained one (0.69 against 0.77), with two it ties;
below four, use your labelled participants to calibrate.

## Recipes

Every recipe that uses labels starts by turning runs into eye blocks with their
gaze. `preprocess` writes one `.h5` per run: put the runs of one acquisition
(scanner, sequence and screen) in one folder.

```bash
deepmreye preprocess sub-01_bold.nii.gz -o mystudy/sub-01.h5 --gaze sub-01_eyetracker.tsv
```

`--gaze` takes the raw eye-tracker samples (a `.tsv` or `.csv` with `time`, `x`
and `y` columns, time in seconds from the first volume's onset; `--gaze-offset`
subtracts the trigger time if it is not zero) and bins them into ten samples
per volume. Blinks and track loss should be `NaN`. If your gaze is already
binned, pass `--labels gaze.npy` instead: `[T, 10, 2]`, or `[T, 2]` for one
sample per volume. Volumes with missing gaze are skipped in training and
calibration, so you can label only part of a run.

From Python, `deepmreye.extract_eye_block(run)` returns `(block, tr)`,
`deepmreye.data.bin_gaze(times, x, y, n_volumes, tr)` the labels, and
`deepmreye.data.write_subject(path, block, labels)` writes the file.

### 1. Decode with the pretrained decoder

```bash
deepmreye decode sub-*_bold.nii.gz -o gaze/ --qa --save-block
```

`--save-block` keeps each run's eye block (`gaze/<run>.h5`), so a later decode,
calibration or training run can skip registration. `decode` also accepts these
`.h5` files directly.

### 2. Calibrate to degrees

```bash
deepmreye calibrate --data-dir mystudy/ --subjects sub-01 sub-02 -o mystudy_decoder.npz
deepmreye decode sub-*_bold.nii.gz --model mystudy_decoder.npz -o gaze/
```

This fits one gain and one offset per axis on the labelled runs and leaves the
readout alone, so the correlation does not change; the output is in the labels'
units. Calibrate on runs from the same setup as the ones you decode.

### 3. Train on some participants, decode the rest

```bash
deepmreye train --data-dir mystudy/ --subjects 'sub-0[1-6]*' -o mystudy_decoder.npz
deepmreye decode mystudy/sub-07.h5 mystudy/sub-08.h5 --model mystudy_decoder.npz -o gaze/
```

The readout is fitted on the selected participants only, on the published basis.
Trained on one folder, it outputs gaze in the units of that folder's labels
(degrees, if that is what you gave it). Adding the published
labelled datasets (`--include-published`) changes little here: their volumes
outnumber yours, and below 16 participants the result is the pretrained decoder
again.

```python
from deepmreye import Decoder
from deepmreye.model import pretrained_basis

model = Decoder.from_basis(pretrained_basis())   # the published basis, no readout yet
model.fit(blocks, labels)                         # lists, one entry per run
model.save("mystudy_decoder.npz")
```

### 4. One readout per participant from a calibration block

If every run starts with a few minutes of labelled calibration (gaze `NaN`
afterwards), fit a readout for each participant on their own block:

```bash
for sub in sub-01 sub-02 sub-03; do
  deepmreye train --data-dir mystudy/ --subjects $sub -o decoders/$sub.npz
  deepmreye decode mystudy/$sub.h5 --model decoders/$sub.npz -o gaze/
done
```

### 5. Add your labelled datasets to the published ones

More labelled *acquisitions* is what still improves the general decoder. With
labelled data from setups of your own (one folder each):

```bash
deepmreye train --data-dir labelled/ --include-published -o my_general_decoder.npz
```

Gaze is z-scored within each folder before the readout is fitted, so datasets
with different screens and units can be pooled. Check the result with the
evaluation protocol before trusting it more than the pretrained decoder (see
[Reproduce the evaluation](#reproduce-the-evaluation)).

### 6. Basis features only

```bash
deepmreye features sub-01_bold.nii.gz -o features/        # [T, 32] per run
deepmreye features sub-01_bold.nii.gz -o features/ --orbits   # [T, 2, 32], each eye
```

The 32 bilateral canonical coordinates per volume, before any readout: a compact
description of what the eyes did, for example as nuisance regressors for eye
movements in a GLM, or as the input to a readout of your own.
`Decoder.orbits(block)` and `Decoder.features(block)` (with temporal context)
give the same from Python.

### 7. Fit the basis on your own corpus

```bash
deepmreye fit-basis --data-dir corpus/ -o my_basis.npz
deepmreye train --basis my_basis.npz -o my_decoder.npz
```

The corpus is every run under `corpus/` that has no gaze, at least 32 volumes,
and all 14,236 eye-mask voxels inside the field of view. Fitting takes about
three minutes for 2,060 participants. Without `--data-dir` it is the published
corpus (77 GB, downloaded on first use). A corpus needs hundreds of
participants: accuracy rises steeply up to about 400 and slowly after 800, and
below 400 the result depends heavily on which participants happen to be in it.

## Output

- **Ten samples per volume.** Sample `j` of volume `t` is the gaze during
  `[(t + j/10) TR, (t + (j+1)/10) TR)`. Their mean is the volume's gaze.
- **x grows to the right, y grows downwards** (screen coordinates).
- **Units.** The pretrained readout was fitted on gaze z-scored within each
  training dataset, because screen size and viewing distance differ between
  studies and are not in the BOLD. Uncalibrated output is therefore in standard
  units around the participant's own gaze distribution (`units: "z"`).
  Calibration (recipe 2), or a readout trained on one folder of your own labels
  (recipes 3 and 4), gives the labels' units. A readout trained on several
  folders at once stays in standard units.

## Quality control

- **Look at the QA image** (`--qa`). Both eyeballs should sit inside the red
  mask. If the registration missed, nothing downstream will work.
- **Coverage.** A warning is raised when some eye-mask voxels carry no signal:
  the eyes are partly outside the field of view (common with tilted or short
  slabs), or the registration failed.
- **Confidence** is the correlation between the gaze decoded from the left orbit
  alone and from the right orbit alone, and needs no labels. Within a study it
  ranks runs by how well they decode: dropping each dataset's least confident
  fifth raises accuracy from 0.721 to 0.745, close to dropping the truly worst
  fifth (0.747). It is not comparable across studies, says nothing about single
  volumes, and can be raised by signals both orbits share other than gaze (head
  motion, blinks).

## How it works

Write the left- and right-orbit voxel time series as `X_L = S W_L' + N_L` and
`X_R = S W_R' + N_R`, with `S` the gaze-driven rotation of the eyes, `W` each
orbit's anatomy, and `N` noise local to each orbit. Normal eye movements are
conjugate, so `S` is common to both orbits and `N_L`, `N_R` are not. CCA finds
the directions in which the two orbits correlate most, which is where `S` lives.

1. **Preprocess.** The run's mean volume is registered to the DeepMReye template
   with ANTs (affine on the whole volume, affine inside a large eye mask, SyN
   inside the small eye mask). The eyes are cropped to a `47 x 29 x 18` box around
   the 14,236-voxel eye mask, each voxel is z-scored over time, each volume over
   space, and values are clipped at 5 SD.
2. **Basis.** Pooling 48 volumes from each of the 2,060 corpus participants
   (98,880 volumes), each orbit is reduced to its 256 leading principal
   components and whitened (with 10^-3 of the largest eigenvalue added to every
   eigenvalue). Whitening makes CCA rank directions by how strongly the orbits
   correlate along them rather than by how much variance they carry; reducing
   first keeps low-variance noise directions from being paired at spuriously high
   correlation. The SVD of the whitened cross-covariance gives the canonical
   pairs; the 32 leading ones are used.
3. **Features.** Each orbit is projected onto its canonical directions and the
   two projections are averaged, since they are two estimates of the same gaze.
   Adding the volumes before and after (96 features) raises sub-TR accuracy
   from 0.741 to 0.771.
4. **Readout.** A ridge regression maps the 96 features to the 20 gaze values of
   a volume (ten samples, two axes), with gaze z-scored per training dataset and
   the penalty chosen by leave-one-out cross-validation.

No participant of a labelled dataset is in the corpus, and neither is any run
of the OpenNeuro accessions that three of the labelled datasets were taken from.

## Results

**Protocol.** Leave one dataset out: fit the readout on eight labelled datasets
and test it on every participant of the ninth. For each participant, the Pearson
r between decoded and measured gaze on each axis, averaged over x and y; each
dataset is the median over its participants; the summary is the median and the
mean over the nine datasets. Sub-TR scores all ten samples, per volume their
mean. The labelled datasets are guided fixation, three smooth-pursuit tasks,
free viewing during visual search, a fixation study with nine EPI sequences, a
calibration task and two feature films; gaze comes from an eye tracker in five
and from the fixation target in four.

| dataset | n | sub-TR r | per-volume r |
|---|---|---|---|
| guided fixations | 170 | 0.771 | 0.787 |
| smooth pursuit 1 | 9 | 0.920 | 0.930 |
| smooth pursuit 2 | 24 | 0.776 | 0.810 |
| smooth pursuit 3 | 34 | 0.851 | 0.863 |
| free viewing | 27 | 0.803 | 0.839 |
| nine EPI sequences | 6 | 0.667 | 0.731 |
| calibration task | 15 | 0.743 | 0.835 |
| studyforrest film (3 T) | 14 | 0.285 | 0.351 |
| Back to the Future film | 37 | 0.672 | 0.845 |
| **median / mean** | | **0.771** / 0.721 | **0.835** / 0.777 |

**Against supervised decoders** trained on the same labelled participants (a
ridge on every eye voxel in 2 x 2 x 2 or 3 x 3 x 3 blocks with and without
temporal context, partial least squares, an MLP): the best reaches 0.752 sub-TR
and 0.823 per volume. On the mean over datasets ours leads the same ridge on
voxels by +0.033 (95% CI 0.010 to 0.072) and is ahead on 8 of 9 datasets. The
margin grows as labelled data shrinks: +0.10 with one labelled dataset, in every
random draw. Against the published DeepMReye 1.0 network, with every decoder
trained on its six training datasets and tested on three it never saw: r = 0.564
for ours, 0.483 for the network, 0.480 to 0.504 for supervised voxel decoders.
A variance PCA of the same corpus reaches 0.705 on the mean (64 components;
0.672 at 32), so the corpus alone recovers part of the gain and the cross-orbit
criterion the rest.

**Inside one study it is a different picture.** A readout fitted on the study's
own participants (labels of K participants, tested on the others):

| labelled participants of the new study | 0 (pretrained) | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| readout on them alone | 0.771 | 0.693 | 0.774 | 0.792 | 0.816 | 0.824 |
| readout on them plus the eight other datasets | 0.771 | 0.771 | 0.771 | 0.771 | 0.772 | 0.798 |

(median over datasets of six random draws; 8 and 16 are over the datasets that
have that many participants.) A readout fitted on the first half of a
participant's own run reaches 0.824 on the second half, on a quarter 0.799.
Inside an acquisition, a supervised decoder on the voxels is level or slightly
ahead (0.833 with half a run); the frozen basis pays off when the readout has to
transfer to an acquisition it has not seen.

**Scaling.** From 100 to 2,060 unlabelled participants the mean rises from
0.585 to 0.721, every dataset improves, and the basis overtakes the supervised
decoder at about 400 participants; gains are small beyond 800. Accuracy still
rises with the number of labelled datasets at eight, and at a fixed number of
labelled participants, spreading them over more datasets helps as much as
doubling them.

**What did not help**, measured and dropped: non-linear and self-supervised
encoders in front of the readout (gaze is linearly accessible from these
features, and a non-linear readout is below ridge), deep, per-participant and
site-invariant CCA, a mirror-symmetry prior, ICA and dictionary learning,
per-participant registration, temporal models of the decoded trace (at most
+0.015), and self-training on the decoder's own output. The full record is
`FINDINGS.md` on the `pytorch_unsup` branch.

**Limits.** Both eyes must be in the field of view and the registration must
hold. The output is relative gaze until calibrated. Accuracy depends on the
acquisition: the studyforrest film decodes at 0.29, the other eight datasets at
0.67 to 0.92.

## Reproduce the evaluation

```bash
deepmreye evaluate --check
```

downloads the nine labelled datasets (20 GB) and the published basis, checks that
the protocol reproduces the numbers above, and prints every dataset at both
resolutions. `--basis my_basis.npz` evaluates another basis the same way, and
`deepmreye.evaluate.lodo(records, feature_fn)` is the protocol from Python.

## Data

The published corpus is [DeepMReye/eyeballs](https://huggingface.co/datasets/DeepMReye/eyeballs)
on HuggingFace: the preprocessed eye blocks of 3,684 OpenNeuro participants, of
which the nine `dsL*` folders (337 participants) carry gaze. No account is
needed. `deepmreye download` fetches the labelled folders, `--all` everything;
`$DEEPMREYE_DATA` points the commands at a copy elsewhere. Each run is one HDF5
file:

```
<dataset>/<subject>.h5
    eye_block  [47, 29, 18, T] float32   normalised eye-region BOLD
    labels     [T, 10, 2]      float32   gaze, labelled datasets only
```

## Python API

| | |
|---|---|
| `deepmreye.decode(run)` | preprocess and decode one run |
| `deepmreye.extract_eye_block(run)` | NIfTI -> `(eye_block, tr)` |
| `deepmreye.load_model(path=None)` | the pretrained decoder, or a saved one |
| `Decoder.predict(block)` | gaze `[T, 10, 2]` |
| `Decoder.predict_orbits(block)` | each orbit's decoding |
| `Decoder.confidence(block)` | the label-free confidence |
| `Decoder.orbits(block)`, `Decoder.features(block)` | basis coordinates, with context |
| `Decoder.fit(blocks, labels, datasets=None)` | fit the readout |
| `Decoder.calibrate(blocks, labels)` | gain and offset to the labels' units |
| `Decoder.from_basis(path)`, `Decoder.load(path)`, `Decoder.save(path)` | |
| `deepmreye.data.bin_gaze(times, x, y, n_volumes, tr)` | eye-tracker samples -> labels |
| `deepmreye.basis.fit_corpus(data_dir, out)` | fit a basis |
| `deepmreye.evaluate.lodo(records, feature_fn)` | the evaluation protocol |

## Development

```bash
git clone -b deepmreye-unsupervised https://github.com/DeepMReye/DeepMReye.git
cd DeepMReye && pip install -e ".[test]"
pytest deepmreye/tests -q
```

Every recipe above has an end-to-end test in `deepmreye/tests/test_recipes.py`.
`DEEPMREYE_NETWORK_TESTS=1` adds a check of the published decoder on a
published participant.

## Citation

If you use DeepMReye, please cite:

> Frey M, Nau M, Doeller CF. Magnetic resonance-based eye tracking using deep
> neural networks. *Nature Neuroscience* 24, 1772-1779 (2021).

The unsupervised method is in preparation.

## License

LGPL-3.0. The corpus is derived from OpenNeuro datasets; see each source dataset
for its terms.
