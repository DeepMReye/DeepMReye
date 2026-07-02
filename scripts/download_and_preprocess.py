#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import h5py
from tqdm import tqdm

# Add deepmreye to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.preprocess import get_masks
from deepmreye.pipeline import (
    make_s3_client,
    is_dataset_approved,
    find_bold_by_subject,
    process_subject,
)
import deepmreye.config as cfg

MAX_SUBJECTS_PER_DATASET = 100


def run_preprocess(data_dir, force=False):
    """Download and preprocess every subject of each manually approved dataset."""
    data_dir = Path(data_dir).resolve()
    h5_path = data_dir / "datasets.h5"

    if not h5_path.exists():
        print(f"HDF5 registry not found at {h5_path}. Run the 'compile' step first.")
        return

    s3 = make_s3_client()

    print("Loading DeepMReye masks for preprocessing...")
    eyemask_small, eyemask_big, dme_template, mask_np, x_edges, y_edges, z_edges = get_masks()
    masks = (eyemask_small, eyemask_big, dme_template, x_edges, y_edges, z_edges)

    with h5py.File(h5_path, "r") as h5f:
        approved_datasets = [ds for ds in h5f.keys() if is_dataset_approved(h5f[ds])]

    if not approved_datasets:
        print("No manually approved datasets found in the registry.")
        print("Run the 'qa' step to label datasets first.")
        return

    print(f"Found {len(approved_datasets)} approved datasets to process fully.")

    with h5py.File(h5_path, "a") as h5f:
        print("\nStarting OpenNeuro full subject extraction...")
        for ds_name in tqdm(approved_datasets, desc="Approved datasets"):
            grp = h5f[ds_name]
            existing_subs = list(grp.keys())

            print(f"  Querying S3 for all subjects in {ds_name}...")
            bold_by_sub = find_bold_by_subject(s3, ds_name)

            if force:
                subs_to_process = list(bold_by_sub.keys())
            else:
                subs_to_process = [s for s in bold_by_sub if s not in existing_subs]
            print(f"  Found {len(subs_to_process)} subjects to process.")

            if len(subs_to_process) > MAX_SUBJECTS_PER_DATASET:
                print(f"  Skipping {ds_name}: more than {MAX_SUBJECTS_PER_DATASET} subjects.")
                continue

            for sub_id in tqdm(subs_to_process, desc=f"Subjects in {ds_name}", leave=False):
                process_subject(
                    s3, grp, ds_name, sub_id, bold_by_sub[sub_id], data_dir, masks, force=force
                )


def main():
    config = cfg.DeepMReyeConfig()
    parser = argparse.ArgumentParser(description="Download and preprocess approved datasets.")
    parser.add_argument("--data-dir", type=str, default=config.data_dir, help="Central data storage directory.")
    parser.add_argument("--force", action="store_true", help="Reprocess all subjects, overwriting existing extractions.")
    args = parser.parse_args()
    run_preprocess(args.data_dir, force=args.force)


if __name__ == "__main__":
    main()
