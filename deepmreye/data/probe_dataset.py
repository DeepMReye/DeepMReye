import os
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import logging
import h5py

class ProbeDataset(Dataset):
    """
    Streams explicit explicit `(X, Y, Z, T)` windows off the standardized unified `.h5` 
    HDF5 arrays inside `labeled_data/`. 
    """
    def __init__(self, labeled_data_dir, split="train", split_ratio=0.8, window_size=10, transforms=None):
        self.labeled_data_dir = Path(labeled_data_dir).resolve()
        self.split = split
        self.split_ratio = split_ratio
        self.window_size = window_size
        self.transforms = transforms
        
        self.samples = []
        self._build_index()
        
    def _build_index(self):
        if not self.labeled_data_dir.exists():
            raise FileNotFoundError(f"Labeled data directory not found at {self.labeled_data_dir}")
            
        logging.info(f"Scanning labeled data directory for '{self.split}' split...")
        
        all_h5_files = sorted(list(self.labeled_data_dir.rglob("*.h5")))
        if not all_h5_files:
            logging.warning("No .h5 files found in labeled_data_dir.")
            return

        all_subjects = []
        # Discover all available subjects mapping to their H5 files
        for h5_path in all_h5_files:
            try:
                with h5py.File(h5_path, 'r') as h5f:
                    for ds_key in h5f.keys():
                        for sub_key in h5f[ds_key].keys():
                            if 'eye_block' in h5f[ds_key][sub_key]:
                                all_subjects.append((str(h5_path), ds_key, sub_key))
            except Exception as e:
                logging.warning(f"Failed to scan {h5_path}: {e}")
                
        # Deterministic Train/Test Split
        np.random.seed(42)
        shuffled_subjects = np.random.permutation(all_subjects)
        
        split_idx = int(len(shuffled_subjects) * self.split_ratio)
        if self.split == "train":
            target_subjects = shuffled_subjects[:split_idx]
        else:
            target_subjects = shuffled_subjects[split_idx:]
            
        # Register window indices
        for (h5_path, ds_key, sub_key) in target_subjects:
            try:
                with h5py.File(h5_path, 'r') as h5f:
                    data_shape = h5f[ds_key][sub_key]['eye_block'].shape
                    # axis=-1 is time
                    T = data_shape[-1]
                    
                    if T < self.window_size:
                        continue
                        
                    # Calculate overlapping or sequential stride
                    stride = self.window_size // 2
                    for start_idx in range(0, T - self.window_size + 1, stride):
                        self.samples.append({
                            'h5_path': h5_path,
                            'ds_key': ds_key,
                            'sub_key': sub_key,
                            'start': start_idx,
                            'end': start_idx + self.window_size
                        })
            except Exception as e:
                pass
                
        logging.info(f"Loaded {len(self.samples)} labeled windows for {self.split}.")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        start = sample['start']
        end = sample['end']
        
        with h5py.File(sample['h5_path'], 'r') as h5f:
            grp = h5f[sample['ds_key']][sample['sub_key']]
            
            # Extract volume: [X, Y, Z, T_window]
            x_arr = grp['eye_block'][..., start:end]
            
            # Extract labels: [T_window, 10, 2]
            y_arr = grp['labels'][start:end]
            
            x_tensor = torch.from_numpy(x_arr).float()
            y_tensor = torch.from_numpy(y_arr).float()
            
            if self.transforms:
                x_tensor = self.transforms(x_tensor)
                
            return x_tensor, y_tensor, sample['ds_key']
