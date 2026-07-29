#!/usr/bin/env python3
"""Render QA thumbnails for participants extracted before thumbnails existed.

The QA sample (~1779 subjects) was extracted when the only QA artifact was the
5 MB HTML report. Everything downstream -- the rapid audit grid, the per-dataset
contact sheets, the ``qa`` stage download -- now reads a ``<subject>.png``
instead, so those subjects need one built after the fact.

Where a report exists, the arrays are recovered from it rather than recomputed,
because recomputing means re-registering: the thumbnail shows the *raw*
whole-brain volume and eye block, and the stored HDF5 keeps only the normalized
block. The report is the only surviving record of the pre-normalization data,
which is also why this is worth doing before any report is deleted.

Where no report exists -- the gaze-labeled datasets arrived as ``.npz`` exports
and were never registered here -- the thumbnail is built from the stored block
instead, and shows two eye panels rather than three (see
:func:`deepmreye.thumbnail.from_block`). So every participant ends up with one.

    python scripts/backfill_thumbnails.py --data-dir data
    python scripts/backfill_thumbnails.py --data-dir data --workers 8
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye import thumbnail
from deepmreye.datasource import resolve
from deepmreye.pipeline import thumbnail_path
from deepmreye.storage import subject_path


def find_report(data_dir, dataset, subject):
    """The subject's report, if one was kept."""
    reports = sorted((Path(data_dir) / dataset / subject).glob("*.html"))
    return reports[0] if reports else None


def render_one(args):
    """Build and save one thumbnail. Returns ``(dataset, subject, status)``."""
    data_dir, dataset, subject, force = args
    out = thumbnail_path(data_dir, dataset, subject)
    if out.exists() and not force:
        return dataset, subject, "exists"

    report = find_report(data_dir, dataset, subject)
    if report is not None:
        try:
            image = thumbnail.from_report(report.read_text(errors="replace"))
        except Exception as e:
            return dataset, subject, f"error: {e.__class__.__name__}"
        if image is None:
            return dataset, subject, "unparsable"
        thumbnail.save(image, out)
        return dataset, subject, "ok"

    # No report: fall back to the stored block. This is how the gaze-labeled
    # participants get a thumbnail at all -- they arrived as .npz exports and
    # were never registered here, so no report ever existed for them.
    block_path = subject_path(data_dir, dataset, subject)
    if not block_path.exists():
        return dataset, subject, "no_report_or_block"

    try:
        with h5py.File(block_path, "r") as f:
            image = thumbnail.from_block(f["eye_block"][:])
    except Exception as e:
        return dataset, subject, f"error: {e.__class__.__name__}"

    thumbnail.save(image, out)
    return dataset, subject, "ok_from_block"


def find_work(data_dir, force=False):
    """Every (dataset, subject) that could have a thumbnail.

    Both a report directory and a bare participant file count, so a dataset
    with no reports at all is still covered.
    """
    work = []
    for ds_dir in sorted(p for p in Path(data_dir).iterdir() if p.is_dir()):
        # `_pending` holds worker sidecars, and a corpus in the HF cache has a
        # `.cache/huggingface` alongside it -- neither is a dataset.
        if ds_dir.name.startswith(("_", ".")):
            continue
        subjects = {p.name for p in ds_dir.iterdir() if p.is_dir()}
        subjects |= {p.stem for p in ds_dir.glob("*.h5")}
        for subject in sorted(subjects):
            work.append((data_dir, ds_dir.name, subject, force))
    return work


def run_backfill(data_dir=None, workers=4, force=False, quiet=False):
    data_dir = resolve(data_dir, download=False, quiet=True)
    work = find_work(data_dir, force=force)
    if not work:
        print(f"No report directories under {data_dir}; nothing to backfill.")
        return {}

    counts, failures = {}, []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(render_one, item) for item in work]
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Thumbnails", disable=quiet):
            dataset, subject, status = future.result()
            counts[status] = counts.get(status, 0) + 1
            if status not in ("ok", "ok_from_block", "exists"):
                failures.append((dataset, subject, status))

    rendered = counts.get("ok", 0) + counts.get("ok_from_block", 0)
    print(f"\n[+] {rendered} rendered "
          f"({counts.get('ok_from_block', 0)} from the block, no report), "
          f"{counts.get('exists', 0)} already present")
    for status, n in sorted(counts.items()):
        if status not in ("ok", "ok_from_block", "exists"):
            print(f"    {status}: {n}")
    # Named individually: a subject with no thumbnail is invisible in the grid,
    # so it has to be findable rather than just counted.
    for dataset, subject, status in failures[:20]:
        print(f"      {dataset}/{subject}: {status}")
    if len(failures) > 20:
        print(f"      ... and {len(failures) - 20} more")
    return counts


def main():
    parser = argparse.ArgumentParser(description="Render QA thumbnails from existing reports.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Re-render thumbnails that exist.")
    args = parser.parse_args()
    run_backfill(args.data_dir, workers=args.workers, force=args.force)


if __name__ == "__main__":
    main()
