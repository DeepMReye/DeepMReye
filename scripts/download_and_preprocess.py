#!/usr/bin/env python3
import os
import boto3
import h5py
import tempfile
import argparse
from pathlib import Path
from botocore import UNSIGNED
from botocore.client import Config
import sys
import numpy as np
from tqdm import tqdm
import joblib

# Add deepmreye to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.preprocess import run_participant, get_masks
import deepmreye.config as cfg

def main():
    parser = argparse.ArgumentParser(description="Download and Preprocess Approved Datasets.")
    config = cfg.DeepMReyeConfig()
    parser.add_argument("--data-dir", type=str, default=config.data_dir, help="Path to centralized data storage directory.")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all subjects in approved datasets")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir).resolve()
    h5_path = data_dir / "datasets.h5"
    
    if not h5_path.exists():
        print(f"HDF5 registry not found at {h5_path}. Run compile_openneuro.py first.")
        return
        
    s3 = boto3.client('s3', region_name='us-east-1', config=Config(signature_version=UNSIGNED))
    bucket_name = 'openneuro.org'
    
    print("Loading DeepMReye masks for preprocessing...")
    eyemask_small, eyemask_big, dme_template, mask_np, x_edges, y_edges, z_edges = get_masks()
    
    # Optional Machine Learning Fallback
    clf = None
    clf_path = data_dir / "transform_classifier.joblib"
    if clf_path.exists():
        try:
            clf = joblib.load(str(clf_path))
            print(f"Loaded QA ML Classifier: {clf_path}")
        except Exception as e:
            print(f"Failed to load ML Classifier: {e}")
            
    approved_datasets = []
    
    # Identify Approved Datasets to process (All Explicit QA subjects must be approved = 1)
    with h5py.File(h5_path, 'r') as h5f:
        for ds in h5f.keys():
            if h5f[ds].attrs.get('approved', 0) == -99:
                continue
                
            explicit_labels = []
            for s in h5f[ds].keys():
                lbl = h5f[ds][s].attrs.get('approved', -1)
                if lbl != -1:
                    explicit_labels.append(lbl)
                    
            if len(explicit_labels) > 0 and all(lbl == 1 for lbl in explicit_labels):
                approved_datasets.append(ds)
                
    if not approved_datasets:
        print("No datasets marked as 'approved=1' found in the registry.")
        print("Use the label_datasets.py GUI to approve datasets first.")
        return
        
    print(f"Found {len(approved_datasets)} Approved Datasets to process fully.")
    
    with h5py.File(h5_path, 'a') as h5f:
        print("\nStarting OpenNeuro Full Subject Extraction...")
        for ds_name in tqdm(approved_datasets, desc="Approved Datasets"):
            grp = h5f[ds_name]
            # Get existing subjects (the 2 we already processed)
            existing_subs = list(grp.keys())
            
            # Query S3 for all subjects in this dataset
            paginator = s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=f'ds/{ds_name}' if not ds_name.startswith('ds') else ds_name)
            
            bold_files_by_sub = {}
            
            print(f"  Querying S3 for all remaining files in {ds_name}...")
            for page in pages:
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    
                    if 'func/' in key and key.endswith('_bold.nii.gz'):
                        parts = key.split('/')
                        sub_id = next((p for p in parts if p.startswith('sub-')), "sub-unknown")
                        if sub_id not in bold_files_by_sub:
                            bold_files_by_sub[sub_id] = key
                            
            # Process remaining subjects
            if args.force:
                subs_to_process = list(bold_files_by_sub.keys())
            else:
                subs_to_process = [s for s in bold_files_by_sub.keys() if s not in existing_subs]
            print(f"  Found {len(subs_to_process)} subjects to process.")

            # If theres more than 100 subjects we dont process this dataset, we just skip it
            if len(subs_to_process) > 100:
                print(f"  Skipping {ds_name} with more than 100 subjects to process.")
                continue
            
            for sub_id in tqdm(subs_to_process, desc=f"Subjects in {ds_name}", leave=False):
                # Restartability check: If we already fully processed this sub, skip!
                if not args.force and sub_id in grp and 'func_path' in grp[sub_id].attrs:
                    continue
                    
                file_key = bold_files_by_sub[sub_id]
                
                if sub_id not in grp:
                    sub_grp = grp.create_group(sub_id)
                else:
                    sub_grp = grp[sub_id]
                    
                ds_out_dir = data_dir / ds_name / sub_id
                ds_out_dir.mkdir(parents=True, exist_ok=True)
                
                with tempfile.TemporaryDirectory() as tmpdir:
                    file_basename = file_key.split('/')[-1]
                    local_file = os.path.join(tmpdir, f"{ds_name}_{file_basename}")
                    
                    try:
                        s3.download_file(bucket_name, file_key, local_file)
                        # Save the S3 Key vs Local File Path
                        sub_grp.attrs['func_path'] = file_key
                    except Exception as e:
                        print(f"      [!] Failed to download: {e}")
                        continue
                        
                    try:
                        masked_eye, transform_stats, original_input = run_participant(
                            fp_func=local_file,
                            dme_template=dme_template,
                            eyemask_big=eyemask_big,
                            eyemask_small=eyemask_small,
                            x_edges=x_edges,
                            y_edges=y_edges,
                            z_edges=z_edges,
                            replace_with=0, # Crucial: zero out non-eyeball
                            transforms=["Affine", "Affine", "SyNAggro"],
                            save_path=str(ds_out_dir), # Save HTML report for future QA if needed
                            as_pickle=False,
                            save_overview=True, # Generate the report while we have full NIfTI in memory
                            dataset_name=sub_id
                        )
                        
                        # Store extracted block directly as contiguous HDF5 array
                        ds_h5_path = ds_out_dir.parent / f"{ds_name}.h5"
                        with h5py.File(ds_h5_path, 'a') as ds_h5f:
                            if sub_id in ds_h5f:
                                del ds_h5f[sub_id] # overwrite
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
                                
                                # Evaluate transformation certainty against ML tree
                                if clf is not None:
                                    try:
                                        proba_array = clf.predict_proba([transform_stats.flatten()])[0]
                                        classes = list(clf.classes_)
                                        if 1 in classes:
                                            idx = classes.index(1)
                                            prob = proba_array[idx]
                                            sub_grp.attrs['transform_probability'] = float(prob)
                                            # Also save directly into the dataset's localized HDF5 file
                                            ds_h5f.create_dataset(f"{sub_id}/transform_probability", data=prob)
                                    except Exception as e:
                                        print(f"      [!] Failed to extract probability: {e}")
                                        
                        sub_grp.attrs['data_path'] = str(ds_h5_path)
                                
                        # Save the HTML report path to HDF5
                        reports = list(ds_out_dir.glob("*.html"))
                        if reports:
                            sub_grp.attrs['report_html_path'] = str(reports[0])
                                
                    except Exception as e:
                        print(f"      [!] Failed processing {sub_id}: {e}")

if __name__ == "__main__":
    main()
