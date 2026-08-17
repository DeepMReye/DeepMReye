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
  evaluating a gaze probe. Same format; `labels` is simply present. These are
  the `dsL##_*` folders, so `dsL*/*.h5` selects them without opening a file.

`dsL01`–`dsL06` come from the DeepMReye 1.0 training sets. `dsL07` and `dsL11` were
ingested from OpenNeuro datasets that recorded eye tracking during the scan:

| folder | source | n | paradigm |
|---|---|---|---|
| `dsL07_deepmreye_calib` | ds006833 | 15 | fixation / pursuit / free viewing |
| `dsL11_backtothefuture` | ds006642 | 4 | movie (backtothefuture) |

### Gaze/BOLD alignment

Every ingested dataset was checked by decoding gaze from the eye block at a
range of TR shifts and confirming the correlation peaks at **lag 0** — the
eyeball signal is not hemodynamic, so a correctly aligned recording has no delay
to absorb an error. Each file records how its time origin was recovered
(`gaze_anchor`: a BIDS `StartTime`, a scanner-trigger column, or a sync message
in the tracker stream) and any residual offset applied (`gaze_time_offset`).

## Processing

Each functional run was coregistered to the DeepMReye template with ANTs
(`Affine`, `Affine`, `SyNAggro`), masked to the eyeballs with voxels outside the
mask set to 0, and cropped to a fixed `[47, 29, 18]` bounding box. Values are
z-scored per voxel across time and per volume across space, then clipped at
5 SD. Labeled and unlabeled participants went through identical processing.

Gaze labels are sampled 10 times per TR: sub-bin `j` of volume `t` holds the
mean gaze over `[(t + j/10)·TR, (t + (j+1)/10)·TR)`, so the bins do not overlap
and the mean of the ten is the mean gaze during that volume. `NaN` marks TRs
with no valid gaze sample — mask them rather than dropping them, or the block
and the gaze go out of alignment.

**Units differ by dataset — read `label_units` from the file attributes.**

| folder | units |
|---|---|
| `dsL01`–`dsL06` | degrees of visual angle |
| `dsL07_deepmreye_calib` | degrees of visual angle |
| `dsL11_backtothefuture` | degrees of visual angle |

The source papers were checked for each dataset. `dsL07` and `dsL11` document geometry (display size and viewing distance) to convert to degrees of visual angle.

Pearson correlation is invariant to the difference. **Training is not**: if you
fit one readout over several datasets pooled, standardise the target per dataset
first, or the largest-scale dataset dominates the loss.

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


def upload_in_batches(api, data_dir, repo_id, datasets, patterns, ignore,
                      batch_size=25, retries=4):
    """Upload a few datasets per commit, retrying each batch on network errors.

    One `upload_folder` over the whole corpus is a single commit: a transient
    failure anywhere loses the entire run. That happened at 17.8 GB (a
    ConnectionError from the xet CAS server after 30 minutes, nothing
    committed), and the corpus is set to grow more than tenfold at full
    extraction, so an all-or-nothing upload is not usable.

    Batching by dataset makes progress durable -- a completed batch stays
    committed -- and each retry is cheap because the Hub deduplicates content
    it already has, so a repeated batch re-sends almost nothing.
    """
    import time

    # Whole-repo files (index, registry, card) have no dataset prefix; they go
    # last, so the registry never advertises subjects that failed to upload.
    root_patterns = [p for p in patterns if "/" not in p]
    file_patterns = [p for p in patterns if "/" in p]

    batches = [datasets[i:i + batch_size] for i in range(0, len(datasets), batch_size)]
    failed = []

    for i, batch in enumerate(batches, 1):
        allow = [f"{ds}/{p.split('/', 1)[1]}" for ds in batch for p in file_patterns]
        label = f"batch {i}/{len(batches)} ({len(batch)} datasets)"

        for attempt in range(1, retries + 1):
            try:
                api.upload_folder(
                    folder_path=str(data_dir),
                    repo_id=repo_id,
                    repo_type="dataset",
                    allow_patterns=allow,
                    ignore_patterns=ignore or None,
                    commit_message=f"Upload {batch[0]}..{batch[-1]} ({len(batch)} datasets)",
                )
                print(f"  [+] {label}")
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"  [!] {label} attempt {attempt}/{retries} failed: "
                      f"{e.__class__.__name__}: {e}")
                if attempt == retries:
                    failed.extend(batch)
                    print(f"  [!] giving up on {label}; rerun to pick it up")
                else:
                    print(f"      retrying in {wait}s")
                    time.sleep(wait)

    if root_patterns:
        for attempt in range(1, retries + 1):
            try:
                api.upload_folder(
                    folder_path=str(data_dir),
                    repo_id=repo_id,
                    repo_type="dataset",
                    allow_patterns=root_patterns,
                    commit_message="Upload registry, index and card",
                )
                print(f"  [+] root files: {root_patterns}")
                break
            except Exception as e:
                print(f"  [!] root files attempt {attempt}/{retries} failed: {e}")
                if attempt < retries:
                    time.sleep(2 ** attempt)

    if failed:
        print(f"\n[!] {len(failed)} datasets did not upload: {', '.join(failed[:10])}"
              f"{' ...' if len(failed) > 10 else ''}")
        print("    Re-run the same command; datasets already uploaded are skipped.")
    return failed


def main():
    parser = argparse.ArgumentParser(description="Upload extracted eye blocks to HuggingFace.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--repo-id", required=True, help="e.g. username/deepmreye-eyeblocks")
    parser.add_argument("--private", action="store_true", help="Create the repo private.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report, upload nothing.")
    parser.add_argument("--deep", action="store_true", help="Full-read validation (slow, thorough).")
    parser.add_argument("--labeled-only", action="store_true", help="Upload only labeled participants.")
    parser.add_argument("--exclude-datasets", nargs="*", default=(),
                        help="Dataset folders to hold back. Use for data that is "
                             "technically valid but should not be published -- "
                             "`dsX10_visseq_unaligned` carries labels that failed "
                             "sync verification, and shipping them would export "
                             "exactly the problem it was rejected for.")
    parser.add_argument("--reports", action="store_true",
                        help="Also upload the full QA report HTML (~5 MB/subject, larger "
                             "in total than the eye blocks). Not needed to label: the "
                             "~20 KB thumbnails are uploaded either way.")
    parser.add_argument("--no-registry", action="store_true",
                        help="Skip datasets.h5 / labels.csv. They carry the QA labels, so "
                             "omitting them means a fresh clone cannot label.")
    parser.add_argument("--batch-size", type=int, default=25,
                        help="Datasets per commit. Smaller batches make progress more "
                             "durable against network failures, at the cost of more commits.")
    parser.add_argument("--retries", type=int, default=4,
                        help="Attempts per batch before giving up on it and moving on.")
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

    if args.exclude_datasets:
        drop = set(args.exclude_datasets)
        held = [r for r in good if r["dataset"] in drop]
        good = [r for r in good if r["dataset"] not in drop]
        print(f"Holding back {len(held)} participants from {len(drop)} dataset(s): "
              f"{', '.join(sorted(drop))}")

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

    # `run_build` indexed everything on disk, but the upload is a subset. An
    # index that lists participants the repo does not contain is worse than no
    # index -- anyone filtering on it gets 404s -- so it is rewritten to match
    # exactly what goes up.
    if len(good) != n_candidates:
        import pandas as pd

        pd.DataFrame(good).to_parquet(data_dir / "index.parquet", index=False)
        print(f"Rewrote index.parquet with the {len(good)} uploaded participants.")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)

    card = data_dir / "README.md"
    card.write_text(CARD.replace("{repo_id}", args.repo_id))

    # Say what to include with a couple of globs and name only the exclusions.
    # Listing every kept subject instead would mean ~3500 patterns at corpus
    # scale, to express "all of them except three".
    # Thumbnails ship by default: they are ~20 KB each and the `qa` stage reads
    # them, so a copy without them cannot be labeled. The HTML reports are ~5 MB
    # each and stay opt-in.
    patterns = ["*/*.h5", "*/*.png"]
    if args.reports:
        patterns.append("*/*/*.html")

    excluded = ({f"{r['dataset']}/{r['subject']}" for r in bad}
                | {f"{r['dataset']}/{r['subject']}" for r in rejected})
    ignore = [f"{k}.h5" for k in sorted(excluded)]
    ignore += [f"{k}.png" for k in sorted(excluded)]
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

    datasets = sorted({r["dataset"] for r in good})
    upload_in_batches(api, data_dir, args.repo_id, datasets, patterns, ignore,
                      batch_size=args.batch_size, retries=args.retries)

    print(f"\n[+] https://huggingface.co/datasets/{args.repo_id}")
    print("    Pull it anywhere with:  python -m deepmreye qa")


if __name__ == "__main__":
    main()
