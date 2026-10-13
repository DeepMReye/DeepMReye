# CLAUDE.md

Orientation for an agent working on the `deepmreye2.0` branch: the user-facing
release of DeepMReye 2.0. Read **`README.md`** (how a user runs it) and
**`docs/method.md`** (what the method is and what it scores) first.

This branch is the method and nothing else. The research record -- every arm
that was tried, the controls, the closed directions, the corpus ingestion, the
QA labelling and the cluster scripts -- is on the **`pytorch_unsup`** branch
(`FINDINGS.md` there). Before proposing an improvement to the method, read that
record: almost every obvious idea (non-linear encoders, other bases, temporal
models, self-training) has been measured and lost. `paper/`, `paper_long/` and
the research notes in `docs/` are author-local and gitignored.

## Layout

```
deepmreye/
  __init__.py     decode(), extract_eye_block(), load_model(), Decoder
  __main__.py     the CLI: decode, preprocess, train, calibrate, evaluate, fit-basis, download
  preprocess.py   NIfTI -> eye block: ANTs registration, crop, normalisation
  basis.py        the corpus definition, the streaming accumulator, the lr-cca fit
  model.py        Decoder (features, readout, confidence, calibration, save/load,
                  pretrained) and fit_ridge, the one readout fit
  evaluate.py     leave-one-dataset-out, the one implementation of the score
  metrics.py      r, R2, Euclidean error, calibration, orbit agreement
  data.py         HuggingFace resolution and the per-participant HDF5 layout
  thumbnail.py    the QA image of a registration
  masks/          the template and the two eye masks
scripts/
  release_model.py  builds (and with --upload publishes) the pretrained release
docs/method.md
```

## Where things live

- **Corpus**: HuggingFace dataset `DeepMReye/eyeballs` (public; 3,684
  participants, 921 folders, 77 GB; the nine `dsL*` folders are the 337 labelled
  participants, 20 GB). `deepmreye.data.resolve` finds or downloads it.
- **Pretrained decoder and basis**: HuggingFace model repo
  `DeepMReye/deepmreye-2` (`deepmreye2.npz`, `basis.npz`). `$DEEPMREYE_MODEL` /
  `$DEEPMREYE_BASIS` point at local files instead.
- **Release**: `python scripts/release_model.py --basis <basis.npz> --out
  release/` builds the decoder, the basis, `evaluation.json` and the model card,
  and refuses to build unless the protocol reproduces the published numbers;
  `--upload` publishes the directory. Publishing is outward-facing: only when
  the user asks.

## Rules that are not negotiable

- **One implementation of the number.** `evaluate.lodo` is the score and
  `model.fit_ridge` is the readout fit; `Decoder.fit` and the evaluation both
  call `fit_ridge`, and `Decoder.features` must equal `evaluate.features` on the
  same participant (`test_model.py` pins this). A variant protocol is a
  parameter, not a second file.
- **`evaluate --check` must reproduce the published numbers** (sub-TR median
  0.7712 at +-1 TR, 0.7412 without context, on the published corpus and basis)
  before anything new is quoted. `evaluate.CALIBRATION` holds them.
- **The corpus never contains an evaluation participant.** `basis.corpus_subjects`
  excludes the `dsL*` folders and the OpenNeuro accessions three of them came
  from (`basis.EVALUATION_SOURCES`). A new labelled dataset's source accession
  goes there.
- **Fit on all the rows.** No training-row subsample.
- **No model output gates data.** The coverage warning and the confidence inform
  the user; nothing is dropped on them automatically.
- The **noise floor on a 9-dataset median is ~0.02**; differences below it are
  ties.

## Conventions a user depends on

- Gaze is `[T, 10, 2]`: ten samples per volume, sample `j` covering
  `[(t + j/10) TR, (t + (j+1)/10) TR)`; x to the right, y **downwards**.
- Uncalibrated output is in the training datasets' z-units (`Decoder.units ==
  "z"`); `calibrate` makes it label units.
- An eye block is `[47, 29, 18, T]`, normalised, non-eye voxels zero, exactly as
  `preprocess.extract_eye_block` writes it. A change to preprocessing changes the
  input distribution of a basis fitted on the published corpus: check it against
  a published participant (extract one from its OpenNeuro run and compare; the
  current pipeline gives voxel r ~0.99 and decoded-gaze r ~0.999).

## Dev

`uv` + `pyproject.toml`; use `.venv/bin/python`. Install the package into the
existing venv with `uv pip install --python .venv/bin/python --no-deps -e .` --
never a plain `uv sync` (the venv carries packages outside `pyproject.toml`, and
`[tool.uv]` overrides scipy for macOS 27). Tests: `pytest deepmreye/tests/ -q`.
