#!/usr/bin/env python3
"""Robust expansion of OpenNeuro corpus to up to 5 subjects per dataset with parallel workers.

Uses on-the-fly streaming task generation and process isolation with spawn context
and timeouts to ensure C++ ITK/ANTsPy crashes or unexpected NIfTI scales on single subjects
never terminate the overall run or consume unbound memory.
"""
import argparse
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import multiprocessing as mp

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from deepmreye.datasource import resolve
from deepmreye.pipeline import (
    make_s3_client,
    find_bold_by_subject,
    BUCKET_NAME,
    DEFAULT_REPORT,
)
from deepmreye.storage import is_intact


def load_approved_datasets(labels_path):
    """Return list of approved dataset names from labels.csv."""
    df = pd.read_csv(labels_path)
    ds_groups = df.groupby("dataset")["label"].apply(list)
    approved = []
    for ds, lbls in ds_groups.items():
        if all(l in [1, 3, 4] for l in lbls) and len(lbls) > 0:
            approved.append(str(ds))
    return sorted(approved)


def _natural_sort_key(s):
    """Sort strings with embedded numbers naturally (e.g. sub-2 before sub-10)."""
    import re
    parts = re.split(r"(\d+)", s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def load_failed_extractions(data_dir):
    """Load set of 'dataset/subject' keys and 'dataset/*' wildcards that have previously failed."""
    failed_path = Path(data_dir) / ".failed_extractions.json"
    if failed_path.exists():
        try:
            import json
            data = json.loads(failed_path.read_text())
            return set(data.keys())
        except Exception:
            return set()
    return set()


def record_failed_extraction(data_dir, ds_name, sub_id, err_msg):
    """Persist a failed extraction, auto-marking entire dataset as failed if protocol is incompatible."""
    failed_path = Path(data_dir) / ".failed_extractions.json"
    try:
        import json
        data = {}
        if failed_path.exists():
            data = json.loads(failed_path.read_text())
        data[f"{ds_name}/{sub_id}"] = {
            "error": str(err_msg)[:200],
            "timestamp": time.time()
        }

        # Dataset-level protocol failures: if TR is implausible, or >= 2 subjects failed, blacklist the entire dataset
        err_str = str(err_msg).lower()
        is_protocol_error = "implausible tr" in err_str or "missing tr" in err_str or "no functional bold" in err_str
        ds_failures = sum(1 for k in data.keys() if k.startswith(f"{ds_name}/") and not k.endswith("/*"))
        if is_protocol_error or ds_failures >= 2:
            data[f"{ds_name}/*"] = {
                "error": f"Dataset skipped: {str(err_msg)[:120]}",
                "timestamp": time.time()
            }

        failed_path.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def plan_dataset_extractions(s3, ds_name, data_dir, failed_set=None, target_count=5, max_file_mb=500.0, seed=42):
    """Find candidate subjects to download for a dataset, capping file size and attempts."""
    if failed_set is None:
        failed_set = set()

    # Fast skip if dataset is blacklisted or has >= 2 prior failures
    if f"{ds_name}/*" in failed_set:
        return []
    ds_failures = sum(1 for k in failed_set if k.startswith(f"{ds_name}/"))
    if ds_failures >= 2:
        return []

    ds_dir = Path(data_dir) / ds_name
    existing_subs = set()
    if ds_dir.exists():
        for p in ds_dir.glob("*.h5"):
            if is_intact(p):
                existing_subs.add(p.stem)

    needed = target_count - len(existing_subs)
    if needed <= 0:
        return []

    try:
        prefix = ds_name if ds_name.endswith("/") else f"{ds_name}/"
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix)

        bold_by_sub = {}
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                size_mb = obj.get("Size", 0) / (1024 * 1024)
                if "func/" in key and key.endswith("_bold.nii.gz"):
                    parts = key.split("/")
                    sub_id = next((p for p in parts if p.startswith("sub-")), "sub-unknown")
                    if max_file_mb and size_mb > max_file_mb:
                        continue
                    if f"{ds_name}/{sub_id}" in failed_set:
                        continue
                    bold_by_sub.setdefault(sub_id, key)
    except Exception as e:
        print(f"  [!] Failed to list S3 for {ds_name}: {e}", flush=True)
        return []

    available_new = [s for s in bold_by_sub.keys() if s not in existing_subs and f"{ds_name}/{s}" not in failed_set]
    if not available_new:
        return []

    # Sort available subjects naturally for orderly, intuitive progression (sub-2, sub-3, etc.)
    available_new = sorted(available_new, key=_natural_sort_key)
    # Cap candidates to needed + 2 attempts so uncooperative/broken datasets don't block the queue
    candidate_subs = available_new[: needed + 2]
    return [(ds_name, sub, bold_by_sub[sub]) for sub in candidate_subs]


def stream_extraction_tasks(s3, approved_datasets, data_dir, failed_set=None, target_per_dataset=5,
                            timeout_s=90, max_file_mb=500.0, seed=42):
    """Yield extraction items on-the-fly across datasets to eliminate upfront scanning delays."""
    for ds_idx, ds_name in enumerate(approved_datasets, start=1):
        candidates = plan_dataset_extractions(
            s3, ds_name, data_dir, failed_set=failed_set, target_count=target_per_dataset, max_file_mb=max_file_mb, seed=seed
        )
        for ds, sub, key in candidates:
            yield (ds, sub, key, str(data_dir), DEFAULT_REPORT, timeout_s, max_file_mb, ds_idx, len(approved_datasets))


def process_single_subject_isolated(ds_name, sub_id, file_key, data_dir, report=DEFAULT_REPORT,
                                    timeout_s=90, max_file_mb=500.0, ds_idx=0, n_ds=0):
    """Run extraction in an isolated subprocess to ensure 100% memory reclamation and C++ crash safety."""
    sub_env = os.environ.copy()
    sub_env["PYTHONWARNINGS"] = "ignore"
    sub_env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    sub_env["OMP_NUM_THREADS"] = "1"
    sub_env["OPENBLAS_NUM_THREADS"] = "1"
    sub_env["MKL_NUM_THREADS"] = "1"
    sub_env["VECLIB_MAXIMUM_THREADS"] = "1"
    sub_env["NUMEXPR_NUM_THREADS"] = "1"

    cmd = [
        sys.executable,
        "-u",
        "-W", "ignore",
        "-c",
        f"""
import sys, os
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from deepmreye.pipeline import make_s3_client, process_subject, BUCKET_NAME
from deepmreye.preprocess import get_masks

s3 = make_s3_client()
if {max_file_mb} is not None and {max_file_mb} > 0:
    try:
        head = s3.head_object(Bucket=BUCKET_NAME, Key='{file_key}')
        size_mb = head.get('ContentLength', 0) / (1024 * 1024)
        if size_mb > {max_file_mb}:
            print(f"[!] Skipping {sub_id}: file size ({{size_mb:.1f}} MB) exceeds limit ({max_file_mb} MB)", flush=True)
            sys.exit(2)
    except Exception as e:
        print(f"[!] S3 check error: {{e}}", flush=True)
        sys.exit(1)

eyemask_small, eyemask_big, dme_template, _mask_np, x_edges, y_edges, z_edges = get_masks()
masks = (eyemask_small, eyemask_big, dme_template, x_edges, y_edges, z_edges)

meta = process_subject(
    s3=s3,
    ds_grp=None,
    ds_name='{ds_name}',
    sub_id='{sub_id}',
    file_key='{file_key}',
    data_dir='{data_dir}',
    masks=masks,
    force=False,
    report='{report}'
)
if meta is None:
    sys.exit(1)
sys.exit(0)
""",
    ]
    t0 = time.time()
    try:
        res = subprocess.run(cmd, env=sub_env, capture_output=True, text=True, timeout=timeout_s)
        elapsed = time.time() - t0
        if res.returncode == 0:
            return ds_name, sub_id, True, None, elapsed, ds_idx, n_ds
        else:
            combined = (res.stdout or "") + "\n" + (res.stderr or "")
            lines = [line.strip() for line in combined.splitlines() if line.strip()]
            clean_lines = [
                l for l in lines 
                if not "warning" in l.lower() 
                and not "site-packages" in l 
                and not "mallocstacklogging" in l.lower()
                and not "registering and extracting" in l.lower()
            ]
            if res.returncode == -9:
                err_msg = "Process killed (SIGKILL / Out of Memory)"
            elif res.returncode == -11:
                err_msg = "Process crashed (Segmentation fault / C++ ITK error)"
            else:
                err_msg = clean_lines[-1] if clean_lines else (lines[-1] if lines else f"Extraction failed (exit {res.returncode})")
            return ds_name, sub_id, False, err_msg, elapsed, ds_idx, n_ds
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return ds_name, sub_id, False, f"Timeout after {timeout_s}s", elapsed, ds_idx, n_ds
    except Exception as e:
        elapsed = time.time() - t0
        return ds_name, sub_id, False, str(e), elapsed, ds_idx, n_ds


def _worker_wrapper(args):
    return process_single_subject_isolated(*args)


def main():
    p = argparse.ArgumentParser(description="Robustly expand corpus to 5 subjects per dataset with parallel workers.")
    p.add_argument("--data-dir", default=None, help="Corpus directory (defaults to ~/.cache/deepmreye)")
    p.add_argument("--target-per-dataset", type=int, default=5, help="Target total subjects per dataset")
    p.add_argument("--limit-datasets", type=int, default=None, help="Limit number of datasets to process")
    p.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes (default: 1 for safe sequential execution)")
    p.add_argument("--timeout", type=int, default=90, help="Timeout in seconds per subject extraction (default: 90)")
    p.add_argument("--max-file-mb", type=float, default=500.0, help="Maximum BOLD .nii.gz file size in MB (default: 500.0)")
    p.add_argument("--retry-failed", action="store_true", help="Retry previously failed/skipped subjects instead of ignoring them")
    p.add_argument("--seed", type=int, default=42, help="Random seed for subject sampling")
    p.add_argument("--dry-run", action="store_true", help="Only plan tasks without downloading")
    args = p.parse_args()

    data_dir = resolve(args.data_dir, download=False, quiet=True)
    labels_path = Path(data_dir) / "labels.csv"

    if not labels_path.exists():
        print(f"[!] labels.csv not found under {data_dir}", flush=True)
        sys.exit(1)

    approved_datasets = load_approved_datasets(labels_path)
    print(f"[*] Found {len(approved_datasets)} approved datasets in labels.csv", flush=True)

    failed_set = set() if args.retry_failed else load_failed_extractions(data_dir)
    if failed_set:
        print(f"[*] Loaded {len(failed_set)} previously failed/skipped subjects to bypass", flush=True)

    if args.limit_datasets:
        approved_datasets = approved_datasets[:args.limit_datasets]
        print(f"[*] Limiting to first {len(approved_datasets)} datasets", flush=True)

    s3 = make_s3_client()

    if args.dry_run:
        print(f"[*] Scanning OpenNeuro S3 for candidate subjects (target: {args.target_per_dataset}/dataset, max size: {args.max_file_mb}MB)...", flush=True)
        tasks = []
        for ds_name in tqdm(approved_datasets, desc="Scanning datasets"):
            candidates = plan_dataset_extractions(
                s3, ds_name, data_dir, failed_set=failed_set, target_count=args.target_per_dataset, max_file_mb=args.max_file_mb, seed=args.seed
            )
            tasks.extend(candidates)
        print(f"\n[*] Total extraction tasks planned: {len(tasks)} candidate subjects across {len(approved_datasets)} datasets", flush=True)
        approx_gb = len(tasks) * 22.1 / 1024
        print(f"[*] Estimated new storage: ~{approx_gb:.2f} GB", flush=True)
        print("[*] Dry run complete. Exiting.", flush=True)
        return

    n_success = 0
    n_failed = 0
    n_processed = 0
    t0 = time.time()

    print(f"\n[*] Launching on-the-fly streaming pool with {args.workers} workers (timeout: {args.timeout}s, max file: {args.max_file_mb}MB)...", flush=True)
    task_iter = stream_extraction_tasks(
        s3, approved_datasets, data_dir, failed_set=failed_set, target_per_dataset=args.target_per_dataset,
        timeout_s=args.timeout, max_file_mb=args.max_file_mb, seed=args.seed
    )
    
    ctx = mp.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futures = {}
            # Prime the pool with up to 2 * workers
            for _ in range(args.workers * 2):
                try:
                    item = next(task_iter)
                    fut = pool.submit(_worker_wrapper, item)
                    futures[fut] = item
                except StopIteration:
                    break

            while futures:
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    item = futures.pop(fut)
                    n_processed += 1
                    ds, sub, ok, err, el, ds_idx, n_ds = fut.result()
                    if ok:
                        n_success += 1
                        print(f"  [DS {ds_idx}/{n_ds} #{n_processed}] [+] {ds}/{sub} ({el:.1f}s) | Total OK: {n_success} | Failed: {n_failed}", flush=True)
                    else:
                        n_failed += 1
                        record_failed_extraction(data_dir, ds, sub, err)
                        err_hint = err[:100].replace('\n', ' ') if err else 'skipped'
                        print(f"  [DS {ds_idx}/{n_ds} #{n_processed}] [!] {ds}/{sub} ({el:.1f}s) SKIPPED: {err_hint}", flush=True)

                    # Top up pool
                    try:
                        new_item = next(task_iter)
                        new_fut = pool.submit(_worker_wrapper, new_item)
                        futures[new_fut] = new_item
                    except StopIteration:
                        pass
    except KeyboardInterrupt:
        print("\n[!] Execution interrupted by user (Ctrl+C). Exiting cleanly...", flush=True)

    t_el = time.time() - t0
    print("\n" + "=" * 80, flush=True)
    print(f"EXTRACTION SUMMARY ({t_el / 60:.2f} minutes elapsed)", flush=True)
    print(f"Successfully processed: {n_success} subjects", flush=True)
    print(f"Failed / skipped:       {n_failed} subjects", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
