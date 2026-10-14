# CLAUDE.md

Orientation for an agent working on the `deepmreye-unsupervised` branch: the user-facing
release of the unsupervised DeepMReye. Read **`README.md`** first: it is the
whole user documentation (install, every use case, output conventions, the
method and its results), deliberately in one file.

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
  __main__.py     the CLI: decode, features, preprocess, train, calibrate, evaluate,
                  fit-basis, download
  preprocess.py   NIfTI -> eye block: ANTs registration, crop, normalisation
  basis.py        the corpus definition, the streaming accumulator, the lr-cca fit
  model.py        Decoder (features, readout, confidence, calibration, save/load,
                  pretrained) and fit_ridge, the one readout fit
  evaluate.py     leave-one-dataset-out, the one implementation of the score
  metrics.py      r, R2, Euclidean error, calibration, orbit agreement
  data.py         HuggingFace resolution, the per-participant HDF5 layout, and
                  bin_gaze (eye-tracker samples -> [T, 10, 2] labels)
  thumbnail.py    the QA image of a registration
  masks/          the template and the two eye masks
scripts/
  release_model.py  builds (and with --upload publishes) the pretrained release
```

## Where things live

- **Corpus**: HuggingFace dataset `DeepMReye/eyeballs` (public, not gated -- the
  gate was removed on 2026-10-14 so that no command needs an account; 3,684
  participants, 921 folders, 77 GB; the nine `dsL*` folders are the 337 labelled
  participants, 20 GB). `deepmreye.data.resolve` finds or downloads it.
- **Pretrained decoder and basis**: HuggingFace model repo
  `DeepMReye/deepmreye-unsupervised` (`decoder.npz`, `basis.npz`). `$DEEPMREYE_MODEL` /
  `$DEEPMREYE_BASIS` point at local files instead.
- **Release**: `python scripts/release_model.py --basis <basis.npz> --out
  release/` builds the decoder, the basis, `evaluation.json` and the model card,
  and refuses to build unless the protocol reproduces the published numbers;
  `--upload` publishes the directory. Publishing is outward-facing: only when
  the user asks.
- **Versions** are git tags (`v2.0.0`, ...), read by hatch-vcs; the decoder file
  records the version it was built with. Rebuild the release after tagging.

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
- Uncalibrated output of a readout trained on several datasets (the pretrained
  one) is in z-units (`Decoder.units == "z"`); a readout trained on one dataset
  is in that dataset's label units (`"labels"`); `calibrate` gives label units.
- README's "Which setup fits your data?" table quotes measured numbers (FINDINGS
  on `pytorch_unsup`, "A new study with a few labelled participants" and the
  schemes table). Change a recommendation only with a measurement behind it.
- An eye block is `[47, 29, 18, T]`, normalised, non-eye voxels zero, exactly as
  `preprocess.extract_eye_block` writes it. A change to preprocessing changes the
  input distribution of a basis fitted on the published corpus: check it against
  a published participant (extract one from its OpenNeuro run and compare; the
  current pipeline gives voxel r ~0.99 and decoded-gaze r ~0.999).

## Tests

`deepmreye/tests/` is what tells the next person what they broke; keep it that way.

- **Every README recipe has a test** in `test_recipes.py`, run through the real
  CLI on the synthetic world of `conftest.py` (a 48-voxel two-orbit "eye" driven
  by gaze; registration replaced by a lookup). Change a recipe in the README,
  change its test.
- `test_integration.py`: ANTs registration of a run built from the template
  (full eye coverage, ~5 s), and, opt-in (`DEEPMREYE_NETWORK_TESTS=1`, weekly in
  CI), the published decoder on a published participant.
- Unit tests pin the properties that fail silently: the orbit order of the crop,
  per-dataset z-scoring, edge-padded lags, calibration that never sees the
  scored participant, the shrunk whitening, the corpus exclusions. On
  2026-10-14 twelve planted bugs of that kind were each caught by a test; a new
  silent failure mode deserves a test that would have caught it.
- CI (`.github/workflows/tests.yml`): flake8 and the suite on Python 3.9, 3.11
  and 3.13 for every push and pull request.

## Dev

`uv` + `pyproject.toml`; use `.venv/bin/python`. Install the package into the
existing venv with `uv pip install --python .venv/bin/python --no-deps -e .` --
never a plain `uv sync` (the venv carries packages outside `pyproject.toml`, and
`[tool.uv]` overrides scipy for macOS 27). Tests: `pytest deepmreye/tests/ -q`.
