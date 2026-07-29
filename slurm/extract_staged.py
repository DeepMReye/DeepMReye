#!/usr/bin/env python3
"""Coregister and extract staged BOLD files. Runs on compute nodes, offline.

Reads a slice of the manifest written by ``stage_downloads.py`` and does the
expensive half of ingestion: ANTs registration to the DeepMReye template, eye
mask extraction, normalization, and the per-participant HDF5 write. No network
access is needed or attempted.

A SLURM array task takes every ``stride``-th manifest line starting at its task
id, so the work spreads evenly without a shared queue. Each task owns its output
files outright, and skips subjects already on disk, so a timed-out or preempted
task is safe to simply resubmit.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye import registry
from deepmreye.preprocess import get_masks, normalize_img, run_participant
from deepmreye.pipeline import DEFAULT_REPORT, DEFAULT_TRANSFORMS, REPORT_MODES, thumbnail_path
from deepmreye.storage import is_intact, subject_path, write_subject
from deepmreye.validation import MissingTRError, validate_and_extract_tr


# A subject that has not finished registering by now is diverging, not slow:
# the median is ~42 s and the observed p90 is under two minutes.
CHILD_TIMEOUT_S = 1800


def load_manifest(path, task_id=0, stride=1):
    with open(path) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    return entries[task_id::stride]


def extract_one(entry, data_dir, masks, force=False, report=DEFAULT_REPORT, max_input_gb=None,
                deferred_path=None):
    """Register, extract and persist one staged subject."""
    eyemask_small, eyemask_big, dme_template, x_edges, y_edges, z_edges = masks
    ds_name, sub_id = entry["dataset"], entry["subject"]

    out_path = subject_path(data_dir, ds_name, sub_id)
    if not force and is_intact(out_path):
        return "skipped", None

    local_file = entry["local"]
    if not os.path.exists(local_file):
        return "missing", f"staged file absent: {local_file}"

    # Optional size gate, off by default. It was added on the assumption that
    # ANTs memory tracks input size; measurement disproved that (a 0.10 GB
    # volume OOMed, a 1.16 GB one did not), so leaving it on only discarded
    # large-but-perfectly-fine subjects. The RSS watchdog in _extract_in_child
    # is what actually protects the task. Kept for the rare case where you
    # knowingly want to defer the biggest inputs to a separate high-memory run.
    size_gb = os.path.getsize(local_file) / 1e9
    if max_input_gb and size_gb > max_input_gb:
        if deferred_path:
            with open(deferred_path, "a") as f:
                f.write(json.dumps({**entry, "size_gb": round(size_gb, 3)}) + "\n")
                f.flush()
        return "too_large", f"{size_gb:.2f} GB input exceeds --max-input-gb {max_input_gb}"

    try:
        tr = validate_and_extract_tr(local_file)
    except MissingTRError as e:
        return "no_tr", str(e)

    # Only the HTML report needs a per-subject directory; the thumbnail sits
    # beside the participant file. At full-extraction scale that is tens of
    # thousands of directories not created.
    report_dir = Path(data_dir) / ds_name / sub_id
    if report in ("html", "both"):
        report_dir.mkdir(parents=True, exist_ok=True)

    try:
        masked_eye, transform_stats, _ = run_participant(
            fp_func=local_file,
            dme_template=dme_template,
            eyemask_big=eyemask_big,
            eyemask_small=eyemask_small,
            x_edges=x_edges,
            y_edges=y_edges,
            z_edges=z_edges,
            replace_with=0,
            transforms=DEFAULT_TRANSFORMS,
            save_path=str(report_dir),
            as_pickle=False,
            save_overview=report in ("html", "both"),
            dataset_name=sub_id,
            save_blocks=False,  # the HDF5 write below is the only copy we keep
            thumbnail_path=(thumbnail_path(data_dir, ds_name, sub_id)
                            if report in ("png", "both") else None),
        )
    except Exception as e:
        return "failed", f"{e.__class__.__name__}: {e}"

    # Same normalization as the sampling path, so sampled and fully extracted
    # subjects are interchangeable in the published artifact.
    masked_eye = normalize_img(np.asarray(masked_eye, dtype=np.float32))

    write_subject(
        out_path,
        masked_eye,
        labels=None,
        attrs={
            "dataset": ds_name,
            "subject": sub_id,
            "repetition_time": float(tr),
            "source_key": entry["key"],
            "normalized": True,
        },
    )

    meta = {
        "func_path": entry["key"],
        "data_path": str(out_path),
        "repetition_time": float(tr),
        "n_trs": int(masked_eye.shape[-1]),
    }
    reports = list(report_dir.glob("*.html"))
    if reports:
        meta["report_html_path"] = str(reports[0])
    thumb = thumbnail_path(data_dir, ds_name, sub_id)
    if thumb.exists():
        meta["thumbnail_path"] = str(thumb)
    if transform_stats is not None:
        meta["transform_stats"] = np.asarray(transform_stats, dtype=np.float32).ravel()

    # Sidecar, never datasets.h5: the labeling UI may hold the registry open.
    registry.record(data_dir, ds_name, sub_id, meta)
    return "ok", None


def _child_target(conn, entry, data_dir, masks, mem_limit_gb, kwargs):
    """Body of the isolated worker: extract and report back.

    Deliberately no ``RLIMIT_AS``. That caps *virtual* address space while the
    OOM killer acts on *resident* memory, and threaded ANTs reserves far more
    address space than it faults in -- a healthy task measured 17.6 GB RSS
    against 18.6 GB VSZ, with allocator arenas pushing virtual higher still.
    Capping address space rejected allocations that were never a problem and
    produced 309 spurious ITK failures. The parent watches RSS instead.
    """
    try:
        conn.send(extract_one(entry, data_dir, masks, **kwargs))
    except MemoryError:
        conn.send(("oom", "MemoryError during extraction"))
    except Exception as e:
        conn.send(("failed", f"{e.__class__.__name__}: {e}"))
    finally:
        conn.close()


def _rss_gb(pid):
    """Resident memory of a process, in GB. 0 if it is gone."""
    try:
        with open(f"/proc/{pid}/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1024**3
    except (OSError, IndexError, ValueError):
        return 0.0


def _extract_in_child(entry, data_dir, masks, mem_limit_gb, **kwargs):
    """Extract one subject in a memory-capped subprocess.

    Returns the same ``(status, error)`` pair as :func:`extract_one`. If the
    child is killed outright -- by its own rlimit or by the cgroup OOM killer
    before Python can raise -- that is reported as ``oom`` for this subject and
    the caller simply moves on to the next one.
    """
    import multiprocessing as mp

    ctx = mp.get_context("fork")  # inherits the already-loaded masks
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child_target,
                       args=(child_conn, entry, data_dir, masks, mem_limit_gb, kwargs))
    proc.start()
    child_conn.close()

    # Poll for the result while watching the child's resident memory. Killing
    # it ourselves just before the cgroup would keeps the OOM killer from
    # taking down this whole array task -- and everything queued behind it.
    result = None
    killed_for_memory = False
    deadline = time.time() + CHILD_TIMEOUT_S
    try:
        while True:
            if parent_conn.poll(2.0):
                try:
                    result = parent_conn.recv()
                except EOFError:
                    result = None
                break
            if not proc.is_alive():
                break
            if mem_limit_gb and _rss_gb(proc.pid) > mem_limit_gb:
                killed_for_memory = True
                proc.kill()
                break
            if time.time() > deadline:
                proc.kill()
                proc.join()
                return "timeout", f"exceeded {CHILD_TIMEOUT_S}s"
    finally:
        parent_conn.close()

    proc.join(timeout=30)
    if proc.is_alive():
        proc.kill()
        proc.join()

    if killed_for_memory:
        return "oom", f"resident memory exceeded {mem_limit_gb} GB"
    if result is None:
        # Died before reporting: the cgroup OOM killer beat our watchdog.
        return "oom", f"child died (exit {proc.exitcode}), likely out of memory"
    return result


def main():
    parser = argparse.ArgumentParser(description="Extract staged BOLD files (offline).")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--task-id", type=int, default=None, help="Defaults to SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--stride", type=int, default=None, help="Defaults to SLURM_ARRAY_TASK_COUNT.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report", choices=REPORT_MODES, default=DEFAULT_REPORT,
                        help="QA artifact per subject. 'png' (default) writes the ~20 KB "
                             "thumbnail; 'html' the ~5 MB Plotly report; 'both' each; "
                             "'none' via --no-report. Reports cost 8 GB over the 1779 "
                             "subject QA sample and would cost >100 GB over a full "
                             "extraction, which is why png is the default.")
    parser.add_argument("--no-report", action="store_true",
                        help="Write no QA artifact at all, thumbnail included.")
    parser.add_argument("--max-input-gb", type=float, default=0,
                        help="Defer inputs larger than this without attempting them. "
                             "Default 0 (disabled): file size does not predict ANTs memory "
                             "-- a 0.10 GB volume OOMed while a 1.16 GB one was fine -- so "
                             "this only discarded large-but-fine subjects. The per-subject "
                             "RSS watchdog (--mem-limit-gb) is the real protection.")
    parser.add_argument("--mem-limit-gb", type=float, default=24.0,
                        help="Per-subject address-space cap, enforced in a child process. "
                             "Keeps one runaway ANTs registration from OOM-killing the whole "
                             "array task. Set below --mem. 0 runs in-process (no protection).")
    parser.add_argument("--cleanup", action="store_true",
                        help="Delete each staged .nii.gz once its extraction is written. "
                             "Extraction shrinks the data ~13x, so this reclaims most of "
                             "the staging footprint; re-staging any subject is cheap.")
    args = parser.parse_args()

    # --no-report is the override: it wins over whatever --report asked for.
    report_mode = "none" if args.no_report else args.report

    task_id = args.task_id if args.task_id is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    stride = args.stride if args.stride is not None else int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    entries = load_manifest(args.manifest, task_id, stride)
    print(f"[task {task_id}/{stride}] {len(entries)} subjects assigned", flush=True)

    # One file per task, so concurrent array tasks never interleave writes.
    deferred_path = Path(args.manifest).parent / f"deferred_{task_id}.jsonl"

    masks_all = get_masks()
    masks = (masks_all[0], masks_all[1], masks_all[2], masks_all[4], masks_all[5], masks_all[6])

    counts = {}
    t_start = time.time()
    for i, entry in enumerate(entries, 1):
        t0 = time.time()
        try:
            if args.mem_limit_gb:
                # Run the registration in a child process with its own address
                # space cap. ANTs SyNAggro occasionally diverges and consumes
                # tens of GB on images that look unremarkable -- a 0.10 GB
                # volume blew past 32 GB while a 1.16 GB one finished fine, so
                # neither file size nor dimensions predict it. Without this the
                # OOM killer takes the whole array task down and every subject
                # queued behind it dies too; 25 of 46 tasks were lost that way.
                status, err = _extract_in_child(
                    entry, args.data_dir, masks, args.mem_limit_gb,
                    force=args.force, report=report_mode,
                    max_input_gb=args.max_input_gb or None,
                    deferred_path=deferred_path)
            else:
                status, err = extract_one(entry, args.data_dir, masks,
                                          force=args.force, report=report_mode,
                                          max_input_gb=args.max_input_gb or None,
                                          deferred_path=deferred_path)
        except Exception:
            status, err = "crashed", traceback.format_exc(limit=3)

        counts[status] = counts.get(status, 0) + 1

        # Record anything that did not produce output, so it can be rerun on a
        # larger allocation instead of quietly missing from the corpus.
        if status in ("oom", "timeout", "failed", "crashed"):
            with open(deferred_path, "a") as f:
                f.write(json.dumps({**entry, "status": status}) + "\n")
                f.flush()

        # Only after a confirmed write: a deleted input that never extracted
        # would have to be re-downloaded from a login node.
        if args.cleanup and status in ("ok", "skipped"):
            try:
                os.remove(entry["local"])
            except OSError:
                pass

        msg = f"[task {task_id}] {i}/{len(entries)} {entry['dataset']}/{entry['subject']}: {status} ({time.time()-t0:.0f}s)"
        if err:
            msg += f" -- {err}"
        print(msg, flush=True)

    elapsed = (time.time() - t_start) / 60
    print(f"[task {task_id}] done in {elapsed:.1f} min: {counts}", flush=True)

    n_deferred = sum(counts.get(k, 0) for k in
                     ("too_large", "oom", "timeout", "failed", "crashed"))
    if n_deferred:
        print(f"[task {task_id}] {n_deferred} subjects deferred -> {deferred_path}\n"
              f"    Rerun them on a bigger allocation, e.g.:\n"
              f"    sbatch --mem=64G --array=0 --export=ALL,MANIFEST={deferred_path} "
              f"slurm/extract_array.sbatch", flush=True)


if __name__ == "__main__":
    main()
