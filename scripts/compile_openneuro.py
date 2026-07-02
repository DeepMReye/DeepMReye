#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import h5py
import requests
from tqdm import tqdm

# Add deepmreye to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.preprocess import get_masks
from deepmreye.pipeline import make_s3_client, find_bold_by_subject, process_subject, BUCKET_NAME
import deepmreye.config as cfg

GRAPHQL_URL = "https://openneuro.org/crn/graphql"
SUBJECTS_PER_DATASET = 2  # sample size per dataset for the manual QA step


def fetch_graphql_metadata(ds_name):
    """Fetch rich dataset metadata from the OpenNeuro GraphQL API (best effort)."""
    query = {
        "query": f"""
        {{
          dataset(id: "{ds_name}") {{
            latestSnapshot {{
              description {{ Name Authors DatasetDOI }}
              summary {{ subjects tasks modalities dataProcessed totalFiles }}
            }}
          }}
        }}
        """
    }
    try:
        r = requests.post(GRAPHQL_URL, json=query, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [!] GraphQL fetch failed for {ds_name}: {e}")
        return {}


def fetch_dataset_description(s3, ds_name):
    """Read the top-level dataset_description.json for a dataset."""
    key = f"{ds_name}/dataset_description.json"
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        return obj["Body"].read().decode("utf-8")
    except Exception:
        return "{}"


def run_compile(data_dir, limit=5):
    """Sample a few subjects per OpenNeuro dataset into the QA registry."""
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "datasets.h5"

    s3 = make_s3_client()

    print("Loading DeepMReye masks...")
    eyemask_small, eyemask_big, dme_template, mask_np, x_edges, y_edges, z_edges = get_masks()
    masks = (eyemask_small, eyemask_big, dme_template, x_edges, y_edges, z_edges)

    print(f"Querying {BUCKET_NAME} for datasets...")
    result = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="ds", Delimiter="/")
    datasets = [p["Prefix"].strip("/") for p in result.get("CommonPrefixes", [])]
    ds_list = datasets if limit is None else datasets[:limit]

    print(f"Targeting HDF5 storage at: {out_path}")
    with h5py.File(out_path, "a") as h5f:
        print("\nStarting OpenNeuro dataset processing...")
        for ds_name in tqdm(ds_list, desc="Datasets processed"):
            bold_by_sub = find_bold_by_subject(s3, ds_name)
            if not bold_by_sub:
                print(f"  [!] No func/bold.nii.gz found for {ds_name}")
                continue

            subs_to_process = list(bold_by_sub.keys())[:SUBJECTS_PER_DATASET]
            print(f"  Found {len(bold_by_sub)} subjects. Sampling {subs_to_process}")

            if ds_name not in h5f:
                grp = h5f.create_group(ds_name)
                grp.attrs["dataset_description"] = fetch_dataset_description(s3, ds_name)
                grp.attrs["graphql_metadata"] = json.dumps(fetch_graphql_metadata(ds_name))
                grp.attrs["approved"] = -1  # -1 unlabeled, 0/2 no eyes, 1 eyes, -99 skipped
            else:
                grp = h5f[ds_name]
                if "graphql_metadata" not in grp.attrs:
                    grp.attrs["graphql_metadata"] = json.dumps(fetch_graphql_metadata(ds_name))

            for sub_id in subs_to_process:
                process_subject(s3, grp, ds_name, sub_id, bold_by_sub[sub_id], data_dir, masks)


def main():
    config = cfg.DeepMReyeConfig()
    parser = argparse.ArgumentParser(description="Compile OpenNeuro dataset samples to HDF5.")
    parser.add_argument("--limit", type=str, default="5", help="Number of datasets to sample. Use 'None' for all.")
    parser.add_argument("--data-dir", type=str, default=config.data_dir, help="Central data storage directory.")
    args = parser.parse_args()

    limit = None if args.limit.lower() == "none" else int(args.limit)
    run_compile(args.data_dir, limit=limit)


if __name__ == "__main__":
    main()
