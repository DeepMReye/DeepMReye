# DeepMReye 2.0

Decode eye gaze from fMRI without an eye tracker.

The eyeballs are in the field of view of most fMRI runs, and their signal moves
with gaze. DeepMReye reads it out: give it a BOLD run, get ten gaze samples per
volume back, with no eye tracker and no labels from your study.

Version 2.0 replaces the supervised network of version 1 with a **frozen linear
basis fitted on unlabelled fMRI**. Both eyes rotate together, so gaze is part of
the signal shared by the two orbits; a canonical correlation analysis between the
left and right orbit, fitted on **2,060 participants from 652 OpenNeuro
datasets**, finds that shared signal without ever seeing gaze. A ridge readout
fitted on 337 gaze-labelled participants (nine datasets) maps it to gaze.

On datasets it has never seen (leave-one-dataset-out) it decodes gaze at a
median correlation of **r = 0.771** at sub-TR resolution and **0.835** per TR.
The best supervised decoders trained on the same labelled data reach 0.752 and
0.823, and the margin grows to +0.10 when only one labelled dataset is
available. See [docs/method.md](docs/method.md).

## Install

```bash
pip install git+https://github.com/DeepMReye/DeepMReye.git@deepmreye2.0
```

Python 3.9-3.11. Registration uses [ANTs](https://github.com/ANTsX/ANTsPy)
(`antspyx`, installed as a dependency); everything else is numpy and
scikit-learn. No GPU.

## Decode a run

```bash
deepmreye decode sub-01_task-movie_bold.nii.gz -o gaze/ --qa
```

writes, per run:

| file | contents |
|---|---|
| `<run>_gaze.tsv` | one row per gaze sample: `volume`, `sample` (0-9), `onset` (s), `x`, `y` |
| `<run>_decode.json` | volumes, TR, units, the confidence, which model |
| `<run>_qa.png` | (`--qa`) the registered eyes over the eye mask: check this first |

Or from Python:

```python
import deepmreye

result = deepmreye.decode("sub-01_task-movie_bold.nii.gz")
result["gaze"]          # [T, 10, 2]: ten samples per volume, x then y
result["confidence"]    # agreement of the two eyes' decodings
```

The first call downloads the pretrained decoder (2 MB) from
[HuggingFace](https://huggingface.co/DeepMReye/deepmreye-2). Preprocessing
(registration to the template, then cropping and normalising the eyes) takes a
minute or two per run; decoding takes well under a second.

### What comes out

- **Ten samples per volume.** Sample `j` of volume `t` is gaze during
  `[(t + j/10) TR, (t + (j+1)/10) TR)`. Their mean is gaze for the volume.
- **x grows to the right, y grows downwards** (screen coordinates).
- **Units are relative.** The readout was fitted on gaze z-scored within each
  training dataset, because screen size and viewing distance differ between
  studies and are not in the BOLD. So gaze comes out in standard units centred on
  the participant's own gaze distribution. For degrees, calibrate (below).
- **Confidence** is the correlation between the gaze decoded from the left orbit
  alone and from the right orbit alone. It needs no labels. Within a study it
  ranks runs by how well they decode: discarding each dataset's least confident
  fifth raises accuracy nearly as much as discarding the truly worst fifth. It is
  not comparable across studies and says nothing about single volumes.

### Calibrate to degrees

With an eye tracker on a few participants of the study, one gain and offset per
axis maps the output to degrees:

```bash
deepmreye preprocess sub-01_bold.nii.gz -o calib/sub-01.h5 --labels sub-01_gaze.npy
deepmreye preprocess sub-02_bold.nii.gz -o calib/sub-02.h5 --labels sub-02_gaze.npy
deepmreye calibrate --data-dir calib/ -o mystudy_decoder.npz
deepmreye decode sub-*_bold.nii.gz --model mystudy_decoder.npz -o gaze/
```

Labels are `.npy` arrays of shape `[T, 10, 2]` (ten samples per volume, x then y,
NaN where the tracker lost the eye) or `[T, 2]` (one sample per volume).

## Train your own readout

If you have gaze-labelled runs from your own setup, fit the readout on them
(with or without the published labelled datasets). Lay the runs out as
`<dataset>/<subject>.h5`, one folder per acquisition:

```bash
deepmreye preprocess run.nii.gz -o labelled/mystudy/sub-01.h5 --labels gaze.npy
deepmreye train --data-dir labelled/ --pattern '*' -o my_decoder.npz
```

Gaze is z-scored within each dataset folder before the readout is fitted. The
basis stays the published one unless `--basis` names another.

```python
from deepmreye import Decoder
from deepmreye.model import pretrained_basis

model = Decoder.from_basis(pretrained_basis())           # untrained readout
model.fit(blocks, labels, datasets)                      # lists, one entry per run
model.save("my_decoder.npz")
```

## Fit the basis on your own corpus

```bash
deepmreye fit-basis --data-dir corpus/ -o my_basis.npz   # ~3 min for 2,060 participants
deepmreye train --basis my_basis.npz -o my_decoder.npz
```

The corpus is every fully covered, unlabelled participant with at least 32
volumes under `corpus/`, never a gaze-labelled one. Without `--data-dir` it is
the published corpus (77 GB, downloaded on first use). Accuracy rises with the
number of participants, with diminishing returns beyond about 800.

## Reproduce the evaluation

```bash
deepmreye evaluate --check
```

downloads the nine labelled datasets (20 GB) and the published basis, checks
that the protocol reproduces the published numbers, and prints
leave-one-dataset-out results per dataset at both resolutions.

## Data

The published corpus is the HuggingFace dataset
[DeepMReye/eyeballs](https://huggingface.co/datasets/DeepMReye/eyeballs): the
preprocessed eye blocks of 3,684 OpenNeuro participants, of which the nine
`dsL*` folders (337 participants) carry gaze. `deepmreye download` fetches the
labelled folders, `--all` everything. Each participant is one HDF5 file:

```
<dataset>/<subject>.h5
    eye_block  [47, 29, 18, T] float32   normalised eye-region BOLD
    labels     [T, 10, 2]      float32   gaze, labelled datasets only
```

Set `$DEEPMREYE_DATA` to use a copy elsewhere.

## Python API

| | |
|---|---|
| `deepmreye.decode(run)` | preprocess and decode one run |
| `deepmreye.extract_eye_block(run)` | NIfTI -> `(eye_block, tr)` |
| `deepmreye.load_model(path=None)` | the pretrained decoder, or a saved one |
| `Decoder.predict(block)` | gaze `[T, 10, 2]` |
| `Decoder.predict_orbits(block)` | each orbit's decoding |
| `Decoder.confidence(block)` | the label-free confidence |
| `Decoder.fit(blocks, labels, datasets)` | fit the readout |
| `Decoder.calibrate(blocks, labels)` | gain and offset to label units |
| `deepmreye.basis.fit_corpus(dir, out)` | fit the basis |
| `deepmreye.evaluate.lodo(recs, features)` | the evaluation protocol |

## Citation

If you use DeepMReye, please cite:

> Frey M, Nau M, Doeller CF. Magnetic resonance-based eye tracking using deep
> neural networks. *Nature Neuroscience* 24, 1772-1779 (2021).

The 2.0 method is in preparation.

## License

LGPL-3.0. The corpus on HuggingFace is derived from OpenNeuro datasets; see each
source dataset for its license.
