#!/usr/bin/env python3
"""Build the pretrained release: the decoder, the basis, and a model card.

    python scripts/release_model.py --basis results/basis.npz --out release/
    python scripts/release_model.py --out release/ --upload      # then publish it

Writes to ``--out``:

    decoder.npz      the decoder: the basis at 32 pairs + the readout fitted on
                     every gaze-labelled participant of the published corpus
    basis.npz        the basis at all stored pairs (for `train` and `evaluate`)
    evaluation.json  the leave-one-dataset-out result on the same basis
    README.md        the model card

The readout is fitted through :meth:`Decoder.fit_features` on features computed
by :meth:`Decoder.features` from the eye blocks themselves -- the inference
path -- not from the evaluation cache. ``--upload`` pushes the directory to the
HuggingFace model repo (``$DEEPMREYE_MODEL_REPO``, default
``DeepMReye/deepmreye-unsupervised``), creating it if needed.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import deepmreye  # noqa: E402
from deepmreye import evaluate  # noqa: E402
from deepmreye.basis import load_basis, save_basis  # noqa: E402
from deepmreye.data import LABELLED, iter_subjects, read_subject, resolve  # noqa: E402
from deepmreye.model import PRETRAINED_BASIS, PRETRAINED_FILE, PRETRAINED_REPO, Decoder  # noqa: E402

CARD = """---
license: lgpl-3.0
library_name: deepmreye
tags:
- fmri
- neuroimaging
- eye-tracking
- gaze
datasets:
- DeepMReye/eyeballs
---

# DeepMReye 2.0

Decode eye gaze from fMRI without an eye tracker. The model reads gaze from
the eyeballs in a BOLD run: a frozen linear basis fitted on **{n_corpus:,}
unlabelled participants from {n_corpus_ds} OpenNeuro datasets** by canonical
correlation between the left and right orbit, and a ridge readout fitted on
**{n_lab} gaze-labelled participants from {n_lab_ds} datasets**.

```bash
pip install git+https://github.com/DeepMReye/DeepMReye.git@deepmreye-unsupervised
deepmreye decode sub-01_task-movie_bold.nii.gz -o gaze/
```

```python
import deepmreye
result = deepmreye.decode("sub-01_task-movie_bold.nii.gz")
result["gaze"]        # [T, 10, 2]: ten gaze samples per volume, x then y
result["confidence"]  # agreement of the two eyes' decodings
```

## Files

| file | what |
|---|---|
| `{decoder}` | the decoder (basis at {k} pairs, +-{lags} TR of context, ridge readout); `Decoder.pretrained()` |
| `{basis}` | the basis at {k_full} pairs, for training your own readout and for evaluation |
| `evaluation.json` | every number of the evaluation below |

## Output

Ten gaze samples per volume (sample j covers [(t + j/10) TR, (t + (j+1)/10) TR)),
x then y, x to the right and y **downwards** (screen coordinates). Units are the
training datasets' standard deviations: gaze was z-scored per dataset before the
readout was fitted, because screen size and viewing distance differ between
studies. For degrees, calibrate on a few participants with an eye tracker
(`deepmreye calibrate`), which fits one gain and offset per axis.

## Evaluation

Leave-one-dataset-out: the readout is refitted on eight labelled datasets and
scored on every participant of the ninth. Pearson r between decoded and measured
gaze per participant (mean of x and y), median over a dataset's participants,
then over datasets.

| | median over datasets | mean over datasets |
|---|---|---|
| sub-TR (10 samples per volume) | {subtr:.3f} | {subtr_mean:.3f} |
| per TR | {tr1:.3f} | {tr1_mean:.3f} |

Per dataset (sub-TR / per TR):

{per_dataset}

The labelled datasets are the `dsL*` folders of
[DeepMReye/eyeballs](https://huggingface.co/datasets/DeepMReye/eyeballs). No
participant of them, nor any run of the OpenNeuro accessions three of them were
taken from, is in the basis's corpus.

## Confidence

The readout is linear, so applying it to each orbit alone gives two decodings
whose average is the real one. Their agreement ranks a dataset's participants by
how well they decode, without labels (`Decoder.confidence`). It does not compare
datasets, and does not flag single volumes.

## Limits

- The eyes must be inside the field of view and the registration must hold:
  check the QA image (`deepmreye decode --qa`).
- Gaze is decoded relative to each participant's own gaze distribution; absolute
  position needs calibration.
- Built {built} from deepmreye {version}.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--basis", default="results/basis.npz")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default="release")
    ap.add_argument("--cache", default=None, help="evaluation cache to reuse (see evaluate)")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--lags", type=int, default=1)
    ap.add_argument("--upload", action="store_true", help="push --out to the HF model repo")
    ap.add_argument("--repo-id", default=PRETRAINED_REPO)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=args.repo_id, repo_type="model", folder_path=str(out),
                          commit_message=f"deepmreye {deepmreye.__version__}")
        print(f"[+] https://huggingface.co/{args.repo_id}")
        return

    data_dir = resolve(args.data_dir, patterns=LABELLED)
    mask, bases, meta = load_basis(args.basis)
    basis_meta = {key: meta[key] for key in ("n_subjects", "n_trs", "datasets", "k", "fitted")
                  if key in meta}
    save_basis(out / PRETRAINED_BASIS, mask, {"lr-cca": bases["lr-cca"]}, meta)

    # Evaluation first, on the same basis: the card quotes it.
    recs = evaluate.load_or_build(data_dir, args.basis, cache=args.cache)
    if not evaluate.calibrate(recs):
        raise SystemExit("[!] the protocol does not reproduce the published numbers")
    res = evaluate.lodo(recs, evaluate.features(args.k, args.lags))
    (out / "evaluation.json").write_text(json.dumps(res, indent=1, default=float))

    model = Decoder.from_basis(args.basis, k=args.k, lags=args.lags)
    feats, labels, datasets, n = [], [], [], 0
    for dataset, _subject, path in iter_subjects(data_dir, "dsL*"):
        block, y, _attrs = read_subject(path)
        if y is None:
            continue
        feats.append(model.features(block))
        labels.append(y)
        datasets.append(dataset)
        n += 1
    model.fit_features(feats, labels, datasets)
    model.meta.update({"version": deepmreye.__version__, "basis": basis_meta,
                       "n_participants": n,
                       "evaluation": {"median_subtr": res["median_subtr"],
                                      "median_1tr": res["median_1tr"],
                                      "mean_subtr": res["summary"]["subtr"]["mean_r"],
                                      "mean_1tr": res["summary"]["1tr"]["mean_r"]},
                       "built": time.strftime("%Y-%m-%d")})
    model.save(out / PRETRAINED_FILE)
    print(f"[+] decoder fitted on {n} runs from {len(set(datasets))} datasets "
          f"(penalty {model.meta['alpha']:g}) -> {out / PRETRAINED_FILE}")

    # The release decoder must reproduce, on a labelled run, what the features
    # of the evaluation cache predict: one path, not two.
    rec = recs[0]
    block, _y, _a = read_subject(Path(data_dir) / rec["dataset"] / f"{rec['subject']}.h5")
    cached = evaluate.features(args.k, args.lags)(rec)
    direct = model.features(block)[: len(cached)]
    assert np.allclose(cached, direct, atol=1e-3), "inference features differ from evaluation"

    rows = "\n".join(f"| `{ds}` | {f['n']} | {f['subtr']['r']:.3f} | {f['1tr']['r']:.3f} |"
                     for ds, f in res["folds"].items())
    (out / "README.md").write_text(CARD.format(
        n_corpus=basis_meta.get("n_subjects", 0), n_corpus_ds=basis_meta.get("datasets", 0),
        n_lab=n, n_lab_ds=len(set(datasets)), decoder=PRETRAINED_FILE, basis=PRETRAINED_BASIS,
        k=args.k, lags=args.lags, k_full=bases["lr-cca"]["left_weights"].shape[1],
        subtr=res["median_subtr"], tr1=res["median_1tr"],
        subtr_mean=res["summary"]["subtr"]["mean_r"], tr1_mean=res["summary"]["1tr"]["mean_r"],
        per_dataset="| dataset | n | sub-TR | per TR |\n|---|---|---|---|\n" + rows,
        built=time.strftime("%Y-%m-%d"), version=deepmreye.__version__))
    print(f"[+] {out}/ ready; inspect it, then rerun with --upload to publish")


if __name__ == "__main__":
    main()
