#!/usr/bin/env python3
import os
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re

def main():
    labeled_dir = Path("labeled_data")
    if not labeled_dir.exists():
        print(f"Error: {labeled_dir} does not exist.")
        return

    # Iterate over dataset directories
    for ds_dir in labeled_dir.iterdir():
        if not ds_dir.is_dir() or ds_dir.name.startswith("."):
            continue

        dataset_name = ds_dir.name
        h5_path = ds_dir / f"{dataset_name}.h5"
        print(f"[*] Processing dataset: {dataset_name}")
        
        # Find all .npz subject files
        npz_files = list(ds_dir.glob("*.npz"))
        if not npz_files:
            continue

        # Create or overwrite the HDF5 file
        with h5py.File(h5_path, 'w') as h5f:
            ds_group = h5f.create_group(dataset_name)

            for npz_path in tqdm(npz_files, desc=f"Converting {dataset_name}"):
                sub_name = npz_path.stem
                
                try:
                    with np.load(npz_path) as data:
                        # Extract all data indices
                        indices = sorted([int(re.search(r'_(\d+)$', k).group(1)) for k in data.files if k.startswith('data_')])
                        
                        if not indices:
                            continue
                            
                        # Reconstruct 4D arrays:
                        # Stack data_{idx} along new time dimension (T) -> [X, Y, Z, T]
                        # Assuming each data_{idx} is [X, Y, Z]
                        data_arrays = [data[f"data_{i}"] for i in indices]
                        # Output shape: [X, Y, Z, T]
                        eye_block = np.stack(data_arrays, axis=-1)
                        
                        # Stack labels -> [T, 10, 2]
                        label_arrays = [data[f"label_{i}"] for i in indices]
                        labels = np.stack(label_arrays, axis=0)

                        sub_group = ds_group.create_group(sub_name)
                        
                        # Save chunks
                        sub_group.create_dataset(
                            'eye_block',
                            data=eye_block,
                            compression="gzip",
                            compression_opts=4
                        )
                        
                        sub_group.create_dataset(
                            'labels',
                            data=labels,
                            compression="gzip",
                            compression_opts=4
                        )
                except Exception as e:
                    print(f"Error processing {npz_path}: {e}")

        print(f"[+] Saved {h5_path} with {len(npz_files)} subjects.")

if __name__ == "__main__":
    main()
