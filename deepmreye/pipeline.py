"""Shared building blocks for the DeepMReye data pipeline.

Both the sampling step (`compile`) and the full-download step (`preprocess`)
pull functional runs from OpenNeuro's public S3 bucket, coregister and extract
the eye bounding box, and write the result into per-dataset HDF5 files plus a
central registry. That per-subject work is identical between the two steps and
lives here so the logic can only ever diverge on purpose.
"""
import os
import tempfile
from pathlib import Path

import boto3
import h5py
from botocore import UNSIGNED
from botocore.client import Config

from deepmreye.preprocess import run_participant
from deepmreye.validation import validate_and_extract_tr, MissingTRError

BUCKET_NAME = "openneuro.org"
DEFAULT_TRANSFORMS = ["Affine", "Affine", "SyNAggro"]


def make_s3_client():
    """Anonymous client for the public OpenNeuro bucket."""
    return boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))


def is_dataset_approved(ds_grp):
    """Whether a dataset qualifies for training based on manual QA labels.

    Labels (on each subject's ``approved`` attribute): 1 = eyes visible,
    0 = no eyes / bad transform, 2 = no eyes / good transform, -1 = unlabeled.
    A dataset qualifies only if it was not skipped (-99) and *every* labeled
    subject shows eyes. A single 'no eyes' subject drops the whole dataset,
    since the same scanner/experiment tends to fail the same way across
    subjects and OpenNeuro has more datasets than we need.
    """
    if ds_grp.attrs.get("approved", 0) == -99:
        return False
    labels = [ds_grp[s].attrs.get("approved", -1) for s in ds_grp.keys()]
    labels = [lbl for lbl in labels if lbl != -1]
    return len(labels) > 0 and all(lbl == 1 for lbl in labels)


def find_bold_by_subject(s3, ds_name):
    """Map each subject in a dataset to its first functional BOLD S3 key."""
    prefix = ds_name if ds_name.endswith("/") else f"{ds_name}/"
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix)

    bold_by_sub = {}
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if "func/" in key and key.endswith("_bold.nii.gz"):
                parts = key.split("/")
                sub_id = next((p for p in parts if p.startswith("sub-")), "sub-unknown")
                bold_by_sub.setdefault(sub_id, key)
    return bold_by_sub


def process_subject(s3, ds_grp, ds_name, sub_id, file_key, data_dir, masks, force=False):
    """Download, coregister, extract and persist one subject.

    Returns True if the subject was processed and stored, False if it was
    skipped (already present, download failure, missing TR, or extraction
    error). Side effects: writes the eye block + transform stats into the
    per-dataset HDF5 and sets pointer attributes on the subject's registry
    group. Never destroys existing data unless ``force`` is set.
    """
    eyemask_small, eyemask_big, dme_template, x_edges, y_edges, z_edges = masks

    # Restartability: skip subjects we already fully processed.
    if not force and sub_id in ds_grp and "func_path" in ds_grp[sub_id].attrs:
        return False

    sub_grp = ds_grp[sub_id] if sub_id in ds_grp else ds_grp.create_group(sub_id)

    ds_out_dir = Path(data_dir) / ds_name / sub_id
    ds_out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_basename = file_key.split("/")[-1]
        local_file = os.path.join(tmpdir, f"{ds_name}_{file_basename}")

        try:
            s3.download_file(BUCKET_NAME, file_key, local_file)
            sub_grp.attrs["func_path"] = file_key  # store the S3 key, not a local path
        except Exception as e:
            print(f"  [!] Failed to download {sub_id}: {e}")
            return False

        # Validate TR before the expensive registration so we skip early.
        try:
            tr = validate_and_extract_tr(local_file)
        except MissingTRError as e:
            print(f"  [!] Skipping {sub_id}: {e}")
            return False

        try:
            print(f"  Registering and extracting {sub_id}...")
            masked_eye, transform_stats, _ = run_participant(
                fp_func=local_file,
                dme_template=dme_template,
                eyemask_big=eyemask_big,
                eyemask_small=eyemask_small,
                x_edges=x_edges,
                y_edges=y_edges,
                z_edges=z_edges,
                replace_with=0,  # zero out non-eyeball voxels
                transforms=DEFAULT_TRANSFORMS,
                save_path=str(ds_out_dir),
                as_pickle=False,
                save_overview=True,
                dataset_name=sub_id,
            )
        except Exception as e:
            print(f"  [!] Failed processing {sub_id}: {e}")
            return False

    ds_h5_path = ds_out_dir.parent / f"{ds_name}.h5"
    with h5py.File(ds_h5_path, "a") as ds_h5f:
        if sub_id in ds_h5f:
            del ds_h5f[sub_id]
        ds_h5f.create_dataset(
            f"{sub_id}/eye_block", data=masked_eye, compression="gzip", compression_opts=4
        )
        # transform_stats is kept for diagnostics and the QA report; approval is manual.
        if transform_stats is not None:
            ds_h5f.create_dataset(f"{sub_id}/transform_stats", data=transform_stats)

    sub_grp.attrs["data_path"] = str(ds_h5_path)
    sub_grp.attrs["repetition_time"] = float(tr)

    reports = list(ds_out_dir.glob("*.html"))
    if reports:
        sub_grp.attrs["report_html_path"] = str(reports[0])

    print(f"  [+] Saved {sub_id} to {ds_h5_path.name}")
    return True
