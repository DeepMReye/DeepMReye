#!/usr/bin/env python3
"""Sample a few subjects per OpenNeuro dataset so they can be QA labeled.

Runs on a login node: compute nodes have no outbound network, so anything that
touches S3 has to live here. Workers write their extractions to per-participant
files and their metadata to sidecar JSONL, never to ``datasets.h5`` directly, so
the labeling UI can stay open on the registry while this is still running.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import requests
from tqdm import tqdm

# Add deepmreye to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.preprocess import get_masks
from deepmreye.pipeline import (
    make_s3_client,
    list_datasets,
    find_bold_by_subject,
    process_subject,
    BUCKET_NAME,
)
from deepmreye import registry
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


def _process_one_dataset(ds_name, data_dir, masks, registry_path, force=False):
    """Sample and extract this dataset's QA subjects. Runs in a worker thread."""
    # Each thread needs its own client: botocore clients are not thread safe.
    s3 = make_s3_client()

    try:
        bold_by_sub = find_bold_by_subject(s3, ds_name)
    except Exception as e:
        return ds_name, 0, f"listing failed: {e}"

    if not bold_by_sub:
        return ds_name, 0, "no func/*_bold.nii.gz"

    subs = list(bold_by_sub)[:SUBJECTS_PER_DATASET]

    n_done = 0
    for sub_id in subs:
        try:
            meta = process_subject(
                s3, None, ds_name, sub_id, bold_by_sub[sub_id], data_dir, masks, force=force
            )
        except Exception as e:
            print(f"  [!] {ds_name}/{sub_id} failed: {e}")
            continue
        if meta is not None:
            registry.record(data_dir, ds_name, sub_id, meta)
            n_done += 1

    return ds_name, n_done, None


def run_compile(data_dir, limit=5, workers=4, force=False):
    """Sample a few subjects per OpenNeuro dataset into the QA registry."""
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "datasets.h5"

    s3 = make_s3_client()

    print("Loading DeepMReye masks...")
    eyemask_small, eyemask_big, dme_template, mask_np, x_edges, y_edges, z_edges = get_masks()
    masks = (eyemask_small, eyemask_big, dme_template, x_edges, y_edges, z_edges)

    print(f"Querying {BUCKET_NAME} for datasets...")
    ds_list = list_datasets(s3, limit=limit)
    print(f"Found {len(ds_list)} datasets to sample.")

    # Register dataset-level metadata up front, single threaded, while nothing
    # else holds the registry. Workers then only ever write sidecar files.
    print("Registering dataset metadata...")
    todo = []
    with h5py.File(out_path, "a") as h5f:
        for ds_name in tqdm(ds_list, desc="Registering"):
            if ds_name not in h5f:
                grp = h5f.create_group(ds_name)
                grp.attrs["dataset_description"] = fetch_dataset_description(s3, ds_name)
                grp.attrs["graphql_metadata"] = json.dumps(fetch_graphql_metadata(ds_name))
                grp.attrs["approved"] = -1  # -1 unlabeled, 0/2 no eyes, 1 eyes, -99 skipped
                todo.append(ds_name)
            else:
                grp = h5f[ds_name]
                if grp.attrs.get("approved", -1) == -99:
                    continue  # explicitly skipped during QA
                # Re-sample only datasets that have no extracted subject yet.
                if force or not any("data_path" in grp[s].attrs for s in grp.keys()):
                    todo.append(ds_name)

    print(f"{len(todo)} datasets need sampling ({len(ds_list) - len(todo)} already done/skipped).")
    if not todo:
        return

    print(f"\nSampling with {workers} parallel workers...")
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_one_dataset, ds, data_dir, masks, out_path, force): ds
            for ds in todo
        }
        with tqdm(total=len(futures), desc="Datasets") as bar:
            for fut in as_completed(futures):
                ds_name, n_done, err = fut.result()
                if err:
                    failures.append((ds_name, err))
                bar.set_postfix_str(f"{ds_name}: {n_done} subs")
                bar.update(1)

    print(f"\nMerging worker records into {out_path.name}...")
    applied = registry.merge_pending(data_dir, out_path)
    print(f"Merged {applied} subject records.")

    if failures:
        print(f"\n{len(failures)} datasets could not be sampled:")
        for ds_name, err in failures[:20]:
            print(f"  {ds_name}: {err}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")


def main():
    config = cfg.DeepMReyeConfig()
    parser = argparse.ArgumentParser(description="Compile OpenNeuro dataset samples to HDF5.")
    parser.add_argument("--limit", type=str, default="5", help="Number of datasets to sample. Use 'None' for all.")
    parser.add_argument("--data-dir", type=str, default=config.data_dir, help="Central data storage directory.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download/registration workers.")
    parser.add_argument("--force", action="store_true", help="Re-extract subjects that are already on disk.")
    args = parser.parse_args()

    limit = None if args.limit.lower() == "none" else int(args.limit)
    run_compile(args.data_dir, limit=limit, workers=args.workers, force=args.force)


if __name__ == "__main__":
    main()
