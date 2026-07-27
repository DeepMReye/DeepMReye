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
import numpy as np
from botocore import UNSIGNED
from botocore.client import Config

from deepmreye.preprocess import run_participant, normalize_img
from deepmreye.storage import is_intact, subject_path, write_subject
from deepmreye.validation import validate_and_extract_tr, MissingTRError

BUCKET_NAME = "openneuro.org"
DEFAULT_TRANSFORMS = ["Affine", "Affine", "SyNAggro"]


def make_s3_client():
    """Anonymous client for the public OpenNeuro bucket."""
    return boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))


# QA label vocabulary, stored on each subject's ``approved`` attribute.
LBL_UNLABELED = -1
LBL_NO_EYES_BAD_TRANSFORM = 0
LBL_EYES = 1
LBL_NO_EYES_GOOD_TRANSFORM = 2
LBL_EYES_CUT = 3  # eyeballs visible but clipped by the bounding box
LBL_DATASET_SKIPPED = -99

# Labels that count as usable eye signal. `LBL_EYES_CUT` is included: a clipped
# eyeball still carries gaze information, and dropping those would cost whole
# datasets under the all-or-nothing rule below. It is kept as a distinct label
# (rather than folded into `LBL_EYES`) so the corpus can be filtered on it later
# if clipping turns out to hurt the probe.
APPROVED_LABELS = (LBL_EYES, LBL_EYES_CUT)


def is_dataset_approved(ds_grp):
    """Whether a dataset qualifies for training based on manual QA labels.

    Labels (on each subject's ``approved`` attribute): 1 = eyes visible,
    3 = eyes visible but cut off, 0 = no eyes / bad transform,
    2 = no eyes / good transform, -1 = unlabeled.
    A dataset qualifies only if it was not skipped (-99) and *every* labeled
    subject shows eyes (clipped or not). A single 'no eyes' subject drops the
    whole dataset, since the same scanner/experiment tends to fail the same way
    across subjects and OpenNeuro has more datasets than we need.
    """
    if ds_grp.attrs.get("approved", 0) == LBL_DATASET_SKIPPED:
        return False
    labels = [ds_grp[s].attrs.get("approved", LBL_UNLABELED) for s in ds_grp.keys()]
    labels = [lbl for lbl in labels if lbl != LBL_UNLABELED]
    return len(labels) > 0 and all(lbl in APPROVED_LABELS for lbl in labels)


def list_datasets(s3, limit=None):
    """Every dataset accession on the OpenNeuro bucket, in sorted order.

    Must paginate: a bare ``list_objects_v2`` caps at 1000 common prefixes and
    silently sets ``IsTruncated``, which quietly hid every dataset past
    ~ds005000 and made "the first 1000" an artifact of the page size rather
    than a deliberate limit.
    """
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix="ds", Delimiter="/")

    datasets = []
    for page in pages:
        for prefix in page.get("CommonPrefixes", []):
            datasets.append(prefix["Prefix"].strip("/"))
            if limit is not None and len(datasets) >= limit:
                return datasets
    return datasets


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

    Returns a dict of registry metadata if the subject was processed and
    stored, or None if it was skipped (already present, download failure,
    missing TR, or extraction error). The eye block is written to its own
    participant file; the caller decides how to record the returned metadata,
    which keeps this usable both with a live registry handle and from a
    parallel worker that must not touch the shared registry.
    """
    eyemask_small, eyemask_big, dme_template, x_edges, y_edges, z_edges = masks

    out_path = subject_path(data_dir, ds_name, sub_id)

    # Restartability: skip subjects whose extraction is already on disk.
    if not force and is_intact(out_path):
        return None

    sub_grp = None
    if ds_grp is not None:
        sub_grp = ds_grp[sub_id] if sub_id in ds_grp else ds_grp.create_group(sub_id)

    # Reports land beside the participant file, under a per-subject folder.
    ds_out_dir = Path(data_dir) / ds_name / sub_id
    ds_out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_basename = file_key.split("/")[-1]
        local_file = os.path.join(tmpdir, f"{ds_name}_{file_basename}")

        try:
            s3.download_file(BUCKET_NAME, file_key, local_file)
        except Exception as e:
            print(f"  [!] Failed to download {sub_id}: {e}")
            return None

        # Validate TR before the expensive registration so we skip early.
        try:
            tr = validate_and_extract_tr(local_file)
        except MissingTRError as e:
            print(f"  [!] Skipping {sub_id}: {e}")
            return None

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
                save_blocks=False,  # the HDF5 write below is the only copy we keep
            )
        except Exception as e:
            print(f"  [!] Failed processing {sub_id}: {e}")
            return None

    # Normalize here, at extraction, so what lands on disk matches the already
    # normalized labeled datasets exactly (z-scored per voxel and per volume,
    # outliers clipped at 5 SD). Training on raw BOLD while probing on
    # normalized data would otherwise hand the encoder and the probe two
    # different input distributions.
    masked_eye = normalize_img(np.asarray(masked_eye, dtype=np.float32))

    reports = list(ds_out_dir.glob("*.html"))

    meta = {
        "func_path": file_key,  # the S3 key, not a local path
        "data_path": str(out_path),
        "repetition_time": float(tr),
        "n_trs": int(masked_eye.shape[-1]),
    }
    if reports:
        meta["report_html_path"] = str(reports[0])

    write_subject(
        out_path,
        masked_eye,
        labels=None,
        attrs={
            "dataset": ds_name,
            "subject": sub_id,
            "repetition_time": float(tr),
            "source_key": file_key,
            "normalized": True,
        },
    )

    if transform_stats is not None:
        meta["transform_stats"] = np.asarray(transform_stats, dtype=np.float32).ravel()

    if sub_grp is not None:
        for key, value in meta.items():
            sub_grp.attrs[key] = value

    print(f"  [+] Saved {sub_id} to {out_path}")
    return meta
