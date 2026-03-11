import os
import io
import json
import boto3
import requests
import h5py
import argparse
import tempfile
from pathlib import Path
from botocore import UNSIGNED
from botocore.client import Config
import sys
import numpy as np
from tqdm import tqdm

# Add deepmreye to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.preprocess import run_participant, get_masks

def main():
    parser = argparse.ArgumentParser(description="Compile OpenNeuro dataset to HDF5.")
    parser.add_argument("--limit", type=str, default="5", help="Number of datasets to process. Set to 'None' to download all.")
    import deepmreye.config as cfg
    config = cfg.DeepMReyeConfig()
    
    parser.add_argument("--data-dir", type=str, default=config.data_dir, help="Path to centralized data storage directory.")
    args = parser.parse_args()
    
    if args.limit.lower() == 'none':
        limit = None
    else:
        limit = int(args.limit)
        
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "datasets.h5"
    
    # Public OpenNeuro S3 access
    s3 = boto3.client('s3', region_name='us-east-1', config=Config(signature_version=UNSIGNED))
    bucket_name = 'openneuro.org'
    
    print("Loading DeepMReye masks...")
    eyemask_small, eyemask_big, dme_template, mask_np, x_edges, y_edges, z_edges = get_masks()
    
    print(f"Querying {bucket_name} for datasets...")
    result = s3.list_objects_v2(Bucket=bucket_name, Prefix='ds', Delimiter='/')
    datasets = [p['Prefix'] for p in result.get('CommonPrefixes', [])]
    
    print(f"Targeting HDF5 storage at: {out_path}")
    with h5py.File(out_path, 'a') as h5f:
        if limit is None: # None means run all
            ds_list = datasets
        else:
            ds_list = datasets[:limit]
            
        print("\nStarting OpenNeuro Dataset Processing...")
        for ds in tqdm(ds_list, desc="Datasets Processed"):
            ds_name = ds.strip('/')
            
            # Use pagination
            paginator = s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=ds)
            
            bold_files_by_sub = {}
            desc_content = "{}"
            
            # ---------------------------------------------------------
            # NEW: GraphQL API for rich Metadata
            # ---------------------------------------------------------
            print(f"  [API] Fetching GraphQL metadata for {ds_name}...")
            graphql_url = "https://openneuro.org/crn/graphql"
            graphql_query = {
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
                r = requests.post(graphql_url, json=graphql_query, timeout=10)
                r.raise_for_status()
                gql_data = r.json()
            except Exception as e:
                print(f"  [!] GraphQL fetch failed: {e}")
                gql_data = {}
            # ---------------------------------------------------------

            for page in pages:
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if key.endswith('dataset_description.json') and key.count('/') == 1:
                        try:
                            desc_obj = s3.get_object(Bucket=bucket_name, Key=key)
                            desc_content = desc_obj['Body'].read().decode('utf-8')
                        except:
                            pass
                        
                    # Target functional data
                    if 'func/' in key and key.endswith('_bold.nii.gz'):
                        # get sub
                        parts = key.split('/')
                        sub_id = next((p for p in parts if p.startswith('sub-')), "sub-unknown")
                        if sub_id not in bold_files_by_sub:
                            bold_files_by_sub[sub_id] = key
            
            if not bold_files_by_sub:
                print(f"  [!] No func/bold.nii.gz found for {ds_name}")
                continue
                
            # Take only up to 2 subjects
            subs_to_process = list(bold_files_by_sub.keys())[:2]
            print(f"  Found {len(bold_files_by_sub)} subjects. Sampling {subs_to_process}")
            
            # Create HDF5 group explicitly without failing
            if ds_name not in h5f:
                grp = h5f.create_group(ds_name)
                grp.attrs['dataset_description'] = desc_content
                grp.attrs['graphql_metadata'] = json.dumps(gql_data)
                grp.attrs['approved'] = -1 # -1 = Unlabeled, 0 = Rejected, 1 = Approved
            else:
                grp = h5f[ds_name]
                if 'graphql_metadata' not in grp.attrs:
                    grp.attrs['graphql_metadata'] = json.dumps(gql_data)
            
            for sub_id in subs_to_process:
                # Restartability check: If we already fully processed this sub, skip!
                if sub_id in grp and 'func_path' in grp[sub_id].attrs:
                    continue
                    
                file_key = bold_files_by_sub[sub_id]
                print(f"  Downloading {sub_id} Functional: {file_key}")
                
                if sub_id not in grp:
                    sub_grp = grp.create_group(sub_id)
                else:
                    sub_grp = grp[sub_id]
                
                # Ensure the dataset folder exists centrally
                ds_out_dir = data_dir / ds_name / sub_id
                ds_out_dir.mkdir(parents=True, exist_ok=True)
                
                with tempfile.TemporaryDirectory() as tmpdir:
                    file_basename = file_key.split('/')[-1]
                    local_file = os.path.join(tmpdir, f"{ds_name}_{file_basename}")
                    
                    try:
                        s3.download_file(bucket_name, file_key, local_file)
                        # Save the S3 KEY pointer instead of a local path!
                        sub_grp.attrs['func_path'] = file_key
                    except Exception as e:
                        print(f"  [!] Failed to download: {e}")
                        continue
                    
                    try:
                        print("  Registering and extracting...")
                        # Run extraction directly returning to the final dataset directory
                        masked_eye, transform_stats, original_input = run_participant(
                            fp_func=local_file,
                            dme_template=dme_template,
                            eyemask_big=eyemask_big,
                            eyemask_small=eyemask_small,
                            x_edges=x_edges,
                            y_edges=y_edges,
                            z_edges=z_edges,
                            replace_with=0,
                            transforms=["Affine", "Affine", "SyNAggro"],
                            save_path=str(ds_out_dir),  # Saves HTML here directly
                            as_pickle=False,
                            save_overview=True,
                            dataset_name=sub_id
                        )
                        
                        # Pack the 4D numpy array natively into the dataset-level HDF5
                        ds_h5_path = ds_out_dir.parent / f"{ds_name}.h5"
                        with h5py.File(ds_h5_path, 'a') as ds_h5f:
                            if sub_id in ds_h5f:
                                del ds_h5f[sub_id]
                            # Save eye_block dataset
                            ds_h5f.create_dataset(
                                f"{sub_id}/eye_block", 
                                data=masked_eye, 
                                compression="gzip", 
                                compression_opts=4
                            )
                            # Save the affine transformation statistics (numpy array)
                            if transform_stats is not None:
                                ds_h5f.create_dataset(
                                    f"{sub_id}/transform_stats", 
                                    data=transform_stats
                                )
                        sub_grp.attrs['data_path'] = str(ds_h5_path)
                            
                        # Find the HTML report
                        reports = list(ds_out_dir.glob("*.html"))
                        if reports:
                            sub_grp.attrs['report_html_path'] = str(reports[0])
                            
                        print(f"  [+] Saved {sub_id} paths and meta to HDF5.")
                    except Exception as e:
                        print(f"  [!] Failed processing {sub_id}: {e}")
                        
if __name__ == "__main__":
    main()
