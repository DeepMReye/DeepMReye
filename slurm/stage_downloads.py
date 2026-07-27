#!/usr/bin/env python3
"""Stage OpenNeuro BOLD files onto scratch, ready for offline extraction.

Compute nodes on this cluster have no outbound network, so the S3 fetch has to
happen on a login node while the expensive part -- ANTs coregistration, roughly
55 s per subject against ~2 s to download -- belongs on compute. This script is
the download half: it resolves what to fetch, pulls the ``.nii.gz`` files into a
staging directory, and writes a manifest that the SLURM array consumes.

Progress lives in the manifest itself, so an interrupted run resumes by
re-reading it rather than re-downloading what is already on disk.
"""
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.pipeline import (
    BUCKET_NAME,
    find_bold_by_subject,
    is_dataset_approved,
    list_datasets,
    make_s3_client,
)
from deepmreye.storage import is_intact, subject_path

# Per-dataset ceiling on subjects taken for full extraction. Datasets larger
# than this are trimmed to the first N, never skipped: at 100-and-skip the
# pipeline discarded 28.9k of 48.7k available subjects to exclude 7% of
# datasets. 200 keeps ~74% of the corpus while bounding any single dataset's
# share. Pass --max-subjects 0 to take everything.
MAX_SUBJECTS_PER_DATASET = 200


def _drop_cache(path):
    """Evict a just-written file from the page cache.

    Page cache counts against the login session's memory cgroup. Staging the
    corpus writes hundreds of GB, and the cache from those writes filled the
    32 GB limit and got the download process killed — repeatedly, with no
    traceback and near-zero RSS, which makes it look like a leak in the job
    rather than its own I/O. These files are never read back here (extraction
    reads them later, on a compute node), so nothing is lost by dropping them.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass
    finally:
        os.close(fd)


def _skipped_datasets(registry_path):
    """Datasets marked -99 during QA, which should never be staged again."""
    if not Path(registry_path).exists():
        return set()
    with h5py.File(registry_path, "r") as f:
        return {ds for ds in f.keys() if f[ds].attrs.get("approved", -1) == -99}


def build_manifest(data_dir, staging_dir, registry_path, max_subjects=MAX_SUBJECTS_PER_DATASET,
                   only_approved=True, sample=None, datasets=None, workers=8,
                   resolved_path=None):
    """Resolve every subject that still needs extracting into a work list.

    With ``sample=N`` this takes only the first N subjects per dataset -- the
    QA sampling pass, whose registrations produce the HTML reports you label
    from. Without it, every subject of a qualifying dataset is included.

    Resolution means one S3 listing per dataset and takes ~15 minutes over the
    full corpus, so results are appended to ``resolved_path`` as they arrive and
    reloaded on a rerun. An interrupted run then costs only the datasets it had
    not reached yet.
    """
    s3 = make_s3_client()

    if datasets is None:
        with h5py.File(registry_path, "r") as f:
            if only_approved:
                datasets = [ds for ds in f.keys() if is_dataset_approved(f[ds])]
            else:
                datasets = [ds for ds in f.keys() if f[ds].attrs.get("approved", 0) != -99]

    # Which datasets a previous run already listed. Only the names are kept:
    # holding every subject->key mapping for the whole corpus is ~42k entries
    # and grows past the login-node memory cap partway through.
    resolved = set()
    if resolved_path and Path(resolved_path).exists():
        with open(resolved_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    resolved.add(json.loads(line)["dataset"])
                except (json.JSONDecodeError, KeyError):
                    continue  # partial final line from a killed run
        print(f"Reusing {len(resolved)} dataset listings from a previous run.")

    pending = [ds for ds in datasets if ds not in resolved]
    print(f"{len(datasets)} datasets qualify; {len(pending)} still need listing.")

    def resolve(ds_name):
        # One client per thread: botocore clients are not thread safe.
        client = make_s3_client()
        try:
            return ds_name, find_bold_by_subject(client, ds_name), None
        except Exception as e:
            return ds_name, None, str(e)

    def select(ds_name, bold_by_sub):
        """Reduce a dataset's listing to the manifest entries we will keep."""
        if not bold_by_sub:
            return []
        subs = sorted(bold_by_sub.items())
        if sample is None and max_subjects and len(subs) > max_subjects:
            # Trim rather than drop. Skipping the dataset outright discarded 40%
            # of all available subjects for 7% of datasets -- the large
            # collections are exactly the richest pretraining source. The cap
            # still bounds how far any one dataset can dominate the corpus.
            print(f"  [~] {ds_name}: {len(subs)} subjects, taking first {max_subjects}")
            subs = subs[:max_subjects]
        if sample is not None:
            subs = subs[:sample]
        return [
            {
                "dataset": ds_name,
                "subject": sub_id,
                "key": key,
                "local": str(Path(staging_dir) / ds_name / f"{sub_id}.nii.gz"),
            }
            for sub_id, key in subs
            if not is_intact(subject_path(data_dir, ds_name, sub_id))
        ]

    entries = []

    # Replay cached listings, keeping only the selected entries rather than the
    # full mapping.
    if resolved_path and Path(resolved_path).exists():
        with open(resolved_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entries.extend(select(rec["dataset"], rec.get("subjects")))

    if pending:
        cache_file = open(resolved_path, "a") if resolved_path else None
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                # Submit in bounded waves. Submitting all ~2400 at once keeps
                # every pending result alive in the futures list, which is what
                # pushed this over the memory cap partway through.
                bar = tqdm(total=len(pending), desc="Resolving subjects")
                for i in range(0, len(pending), workers * 8):
                    wave = pending[i:i + workers * 8]
                    futures = [pool.submit(resolve, ds) for ds in wave]
                    for fut in as_completed(futures):
                        ds_name, bold_by_sub, err = fut.result()
                        bar.update(1)
                        if err:
                            print(f"  [!] {ds_name}: listing failed: {err}")
                            continue
                        if cache_file:
                            cache_file.write(json.dumps({"dataset": ds_name,
                                                         "subjects": bold_by_sub or {}}) + "\n")
                            cache_file.flush()
                        entries.extend(select(ds_name, bold_by_sub))
                    del futures
                bar.close()
        finally:
            if cache_file:
                cache_file.close()

    return entries


def download_all(entries, workers=4):
    """Fetch every staged file that is not already present."""
    local = threading.local()

    def fetch(entry):
        path = Path(entry["local"])
        if path.exists() and path.stat().st_size > 0:
            return entry, True, None
        # botocore clients are not thread safe; one per worker thread.
        if not hasattr(local, "s3"):
            local.s3 = make_s3_client()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        try:
            local.s3.download_file(BUCKET_NAME, entry["key"], str(tmp))
            tmp.replace(path)  # rename last, so a partial file is never picked up
            _drop_cache(path)
            return entry, True, None
        except Exception as e:
            if tmp.exists():
                tmp.unlink()
            return entry, False, str(e)

    ok, failed = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, e) for e in entries]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            entry, success, err = fut.result()
            (ok if success else failed).append(entry if success else (entry, err))
    return ok, failed


def main():
    parser = argparse.ArgumentParser(description="Stage OpenNeuro BOLD files for offline extraction.")
    parser.add_argument("--data-dir", required=True, help="Where extracted participants live.")
    parser.add_argument("--staging-dir", required=True, help="Where to place downloaded .nii.gz.")
    parser.add_argument("--registry", default=None, help="datasets.h5 (default: <data-dir>/datasets.h5).")
    parser.add_argument("--manifest", default=None, help="Manifest path (default: <staging-dir>/manifest.jsonl).")
    parser.add_argument("--workers", type=int, default=4, help="Parallel downloads.")
    parser.add_argument("--max-subjects", type=int, default=MAX_SUBJECTS_PER_DATASET)
    parser.add_argument("--all-datasets", action="store_true",
                        help="Include datasets that are not QA approved yet.")
    parser.add_argument("--resolve-only", action="store_true", help="Write the manifest, download nothing.")
    parser.add_argument("--sample", type=int, default=None,
                        help="QA pass: stage only the first N subjects per dataset.")
    parser.add_argument("--discover", type=str, default=None,
                        help="Bootstrap from OpenNeuro instead of the registry. "
                             "Number of datasets, or 'all'.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(args.staging_dir).resolve()
    staging_dir.mkdir(parents=True, exist_ok=True)
    registry = Path(args.registry) if args.registry else data_dir / "datasets.h5"
    manifest = Path(args.manifest) if args.manifest else staging_dir / "manifest.jsonl"

    datasets = None
    if args.discover:
        limit = None if args.discover.lower() == "all" else int(args.discover)
        s3 = make_s3_client()
        print("Listing OpenNeuro datasets...")
        datasets = list_datasets(s3, limit=limit)
        print(f"Discovered {len(datasets)} datasets.")
        # Seed the registry so the labeling UI has dataset groups to show. The
        # per-subject rows arrive later, via the workers' sidecar records.
        with h5py.File(registry, "a") as f:
            for ds_name in tqdm(datasets, desc="Seeding registry"):
                if ds_name not in f:
                    grp = f.create_group(ds_name)
                    grp.attrs["approved"] = -1
        skipped = _skipped_datasets(registry)
        datasets = [ds for ds in datasets if ds not in skipped]

    entries = build_manifest(data_dir, staging_dir, registry,
                             max_subjects=args.max_subjects,
                             only_approved=not args.all_datasets and not args.discover,
                             sample=args.sample,
                             datasets=datasets,
                             workers=args.workers,
                             resolved_path=staging_dir / "resolved.jsonl")
    print(f"\n{len(entries)} subjects need extraction.")
    if not entries:
        return

    if not args.resolve_only:
        ok, failed = download_all(entries, workers=args.workers)
        print(f"Downloaded {len(ok)}, failed {len(failed)}.")
        for entry, err in failed[:10]:
            print(f"  [!] {entry['dataset']}/{entry['subject']}: {err}")
        entries = ok

    with open(manifest, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    print(f"\nManifest written: {manifest} ({len(entries)} tasks)")
    print("Submit with:  ./slurm/submit_extraction.sh   (sizes the array from this manifest)")


if __name__ == "__main__":
    main()
