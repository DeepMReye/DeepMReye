#!/usr/bin/env python3
import os
import h5py
import numpy as np
import argparse
from pathlib import Path
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import joblib

# Add deepmreye to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import deepmreye.config as cfg

def main():
    parser = argparse.ArgumentParser(description="Train Transform Quality Classifier")
    config = cfg.DeepMReyeConfig()
    parser.add_argument("--data-dir", type=str, default=config.data_dir, help="Path to centralized data storage directory.")
    parser.add_argument("--out-model", type=str, default="transform_classifier.joblib", help="Output model path")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir).resolve()
    h5_path = data_dir / "datasets.h5"
    
    if not h5_path.exists():
        print(f"HDF5 registry not found at {h5_path}.")
        return
        
    X = []
    y = []
    
    print("Loading labeled subjects backwards from datasets.h5...")
    with h5py.File(h5_path, 'r') as h5f:
        for ds in h5f.keys():
            # Skip overall rejected/skipped datasets
            if h5f[ds].attrs.get('approved') == -99:
                continue
                
            for sub in h5f[ds].keys():
                # Check for subject label
                label = h5f[ds][sub].attrs.get('approved', -1)
                if label == -1:
                    continue # Unlabeled
                    
                # Map 0 (Bad transform) and 2 (Good transform, no eyes) both to 0 (No)
                if label in [0, 2]:
                    label = 0
                    
                # We need the transform_stats which is stored inside the dataset-level .h5 file !
                data_path = h5f[ds][sub].attrs.get('data_path')
                if data_path and os.path.exists(data_path):
                    try:
                        with h5py.File(data_path, 'r') as ds_h5:
                            if f"{sub}/transform_stats" in ds_h5:
                                stats = ds_h5[f"{sub}/transform_stats"][:]
                                # Flatten just in case it's multidimensional
                                X.append(stats.flatten())
                                y.append(label)
                    except Exception as e:
                        pass
                        
    if not X:
        print("No labeled data with transform_stats found. Label some datasets first!")
        return
        
    X = np.array(X)
    y = np.array(y)
    
    print(f"Found {len(X)} labeled subjects with extracted affine matrices.")
    print("Label distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        if u == 1:
            lbl_str = "Approved (1)"
        elif u == 0:
            lbl_str = "Rejected / No Eyes (0)"
        else:
            lbl_str = f"Unknown ({u})"
        print(f"  {lbl_str}: {c}")
        
    if len(unique) < 2:
        print("\nNeed at least 2 distinct classes to train a meaningful classifier!")
        return
        
    print("\nTraining Decision Tree Classifier on Affine Parameters...")
    clf = DecisionTreeClassifier(random_state=42, class_weight="balanced", max_depth=3)
    
    min_class_count = min(counts)
    if min_class_count < 2:
        print("Not enough samples in minority class to compute CV score. Training on all data directly.")
        clf.fit(X, y)
    else:
        cv_folds = min(5, min_class_count)
        cv_scores = cross_val_score(clf, X, y, cv=cv_folds)
        print(f"{cv_folds}-Fold CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        clf.fit(X, y)
    
    out_path = data_dir / args.out_model
    joblib.dump(clf, out_path)
    print(f"\n[+] Decision layer persisted efficiently to {out_path}")

if __name__ == "__main__":
    main()
