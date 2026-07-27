#!/usr/bin/env python3
"""Publish the extracted eye blocks as a HuggingFace dataset repo.

Uploads the per-participant HDF5 tree as-is, plus ``index.parquet`` at the root
so the corpus can be browsed and filtered without downloading any volumes.

Only participants that pass validation are uploaded: :func:`build_index` is run
first and anything it flags is excluded, so a truncated or all-zero file cannot
reach the published artifact. Run with ``--dry-run`` first -- it reports exactly
what would be uploaded and why anything was excluded.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_index import run_build

CARD = """---
license: cc0-1.0
task_categories:
- other
tags:
- fmri
- neuroimaging
- eye-tracking
- gaze
- self-supervised
pretty_name: DeepMReye Eye Blocks
---

# DeepMReye Eye Blocks

Eye-region fMRI extracted from public [OpenNeuro](https://openneuro.org)
datasets, prepared for gaze decoding without an eye tracker.

Each participant is one HDF5 file, foldered by source dataset:

```
<dataset>/<subject>.h5
    eye_block  [47, 29, 18, T]  float32   normalized BOLD around both eyes
    labels     [T, 10, 2]       float32   gaze x/y, only in labeled subsets
```

`index.parquet` at the repo root has one row per participant (dataset, subject,
number of TRs, repetition time, whether labels are present, QA status), so you
can select what you need before downloading anything.

## Two subsets

- **Unlabeled** — the bulk of the corpus, for self-supervised pretraining.
- **Labeled** — participants with simultaneous eye-tracking, for fitting and
  evaluating a gaze probe. Same format; `labels` is simply present.

## Processing

Each functional run was coregistered to the DeepMReye template with ANTs
(`Affine`, `Affine`, `SyNAggro`), masked to the eyeballs with voxels outside the
mask set to 0, and cropped to a fixed `[47, 29, 18]` bounding box. Values are
z-scored per voxel across time and per volume across space, then clipped at
5 SD. Labeled and unlabeled participants went through identical processing.

Gaze labels are in degrees of visual angle, sampled 10 times per TR. `NaN`
marks TRs with no valid gaze sample — mask them rather than dropping them, or
the block and the gaze go out of alignment.

## Loading

```python
import h5py
from huggingface_hub import hf_hub_download

path = hf_hub_download("{repo_id}", "ds000001/sub-01.h5", repo_type="dataset")
with h5py.File(path) as f:
    block = f["eye_block"][..., :100]   # one 100-TR window
```

Files are chunked so reading a window does not decompress the whole run.

## Caveat

Windows are a fixed number of TRs, not a fixed duration, and repetition times
differ across datasets — so a 100-TR window is not the same amount of real time
everywhere. `repetition_time` is in the index and in each file's attributes.

## Source

Derived from OpenNeuro datasets, which carry their own licenses (typically
CC0). Please cite the original datasets alongside this collection.
"""


def main():
    parser = argparse.ArgumentParser(description="Upload extracted eye blocks to HuggingFace.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--repo-id", required=True, help="e.g. username/deepmreye-eyeblocks")
    parser.add_argument("--private", action="store_true", help="Create the repo private.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report, upload nothing.")
    parser.add_argument("--deep", action="store_true", help="Full-read validation (slow, thorough).")
    parser.add_argument("--labeled-only", action="store_true", help="Upload only labeled participants.")
    parser.add_argument("--reports", action="store_true",
                        help="Also upload the QA report HTML (~5 MB/subject, larger in "
                             "total than the eye blocks). Needed to label off-cluster.")
    parser.add_argument("--no-registry", action="store_true",
                        help="Skip datasets.h5 / labels.csv. They carry the QA labels, so "
                             "omitting them means a fresh clone cannot label.")
    parser.add_argument("--publish", action="store_true",
                        help="Publication mode: drop subjects QA marked as no-eyes. Off by "
                             "default, because most pushes are a working copy you label "
                             "from -- and you cannot review or revise a label on a subject "
                             "that was filtered out. Turn this on for the final artifact.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()

    print("Validating and indexing...")
    good, bad = run_build(data_dir, deep=args.deep)
    if not good:
        print("Nothing valid to upload.")
        return

    n_candidates = len(good)

    if args.labeled_only:
        good = [r for r in good if r.get("has_labels")]
        print(f"Restricted to {len(good)} labeled participants.")

    # QA labels: 1 = eyes, 0/2 = no eyes, -1 = not yet labeled. A no-eyes
    # subject is a perfectly valid HDF5 file, so technical validation cannot
    # catch it -- but publishing it would put data with no visible eyeballs
    # into a corpus whose entire premise is eye-region signal.
    rejected = []
    if args.publish:
        rejected = [r for r in good if r.get("qa_approved") in (0, 2)]
        good = [r for r in good if r.get("qa_approved") not in (0, 2)]
        if rejected:
            print(f"Excluding {len(rejected)} subjects that QA marked as no-eyes.")

    unlabeled_qa = sum(1 for r in good if r.get("qa_approved", -1) == -1)
    if unlabeled_qa:
        print(f"NOTE: {unlabeled_qa} of these have not been QA labeled yet.")
    if not args.publish:
        print("Working-copy mode: every subject is included, no-eyes ones too, so "
              "labels stay reviewable. Use --publish for the filtered artifact.")

    allowed = {f"{r['dataset']}/{r['subject']}.h5" for r in good}
    print(f"\n{len(allowed)} participant files would be uploaded.")
    print(f"{len(bad)} excluded by validation.")

    if args.dry_run:
        print("\n[dry run] nothing uploaded.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)

    card = data_dir / "README.md"
    card.write_text(CARD.replace("{repo_id}", args.repo_id))

    # Say what to include with a couple of globs and name only the exclusions.
    # Listing every kept subject instead would mean ~3500 patterns at corpus
    # scale, to express "all of them except three".
    patterns = ["*/*.h5"]
    if args.reports:
        patterns.append("*/*/*.html")

    excluded = ({f"{r['dataset']}/{r['subject']}" for r in bad}
                | {f"{r['dataset']}/{r['subject']}" for r in rejected})
    ignore = [f"{k}.h5" for k in sorted(excluded)]
    if args.reports:
        ignore += [f"{k}/*.html" for k in sorted(excluded)]

    patterns += ["index.parquet", "README.md"]
    if not args.no_registry:
        # datasets.h5 carries the QA labels and the report pointers, so a fresh
        # clone can label immediately instead of re-deriving the registry.
        patterns += ["datasets.h5", "labels.csv"]

    print(f"\nUploading to {args.repo_id}...")
    print(f"  include: {patterns}")
    print(f"  exclude: {len(excluded)} subjects (failed validation or QA no-eyes)")
    api.upload_folder(
        folder_path=str(data_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=patterns,
        ignore_patterns=ignore or None,
    )
    print(f"\n[+] https://huggingface.co/datasets/{args.repo_id}")
    print("    Pull it anywhere with:  python -m deepmreye qa")


if __name__ == "__main__":
    main()
