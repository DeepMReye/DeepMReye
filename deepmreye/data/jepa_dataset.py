import os
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import logging

from deepmreye.pipeline import is_dataset_approved

class JEPADataset(Dataset):
    """
    Streams 4D eye blocks directly from dataset-level HDF5 files.
    Filters subjects by explicit probability thresholds, and dynamically
    extracts overlapping sequence windows of `window_size`.
    """
    def __init__(self, data_dir, registry_path="datasets.h5", window_size=100, transforms=None):
        self.data_dir = Path(data_dir).resolve()
        self.registry_path = self.data_dir / registry_path
        self.window_size = window_size
        self.transforms = transforms
        
        self.sequences = [] # List of dicts pointing to file/subject and start frame
        self._build_index()
        
    def _build_index(self):
        """Scans the HDF5 registry and finds all valid sequences."""
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Registry not found at {self.registry_path}")
            
        logging.info("Scanning HDF5 registry for subjects in manually approved datasets...")

        self.total_datasets = 0
        self.total_subjects = 0
        self.valid_subjects = 0
        self.total_windows = 0

        with h5py.File(self.registry_path, 'r') as h5_reg:
            ds_keys = list(h5_reg.keys())
            self.total_datasets = len(ds_keys)

            for ds_name in ds_keys:
                if not is_dataset_approved(h5_reg[ds_name]):
                    continue

                sub_keys = list(h5_reg[ds_name].keys())
                self.total_subjects += len(sub_keys)

                for sub_id in sub_keys:
                    sub_grp = h5_reg[ds_name][sub_id]
                    data_path = sub_grp.attrs.get('data_path', None)

                    if data_path is not None and os.path.exists(data_path):
                        # Verify data length without loading full array
                        try:
                            with h5py.File(data_path, 'r') as ds_h5:
                                if f"{sub_id}/eye_block" in ds_h5:
                                    shape = ds_h5[f"{sub_id}/eye_block"].shape
                                    # shape is expected to be [X, Y, Z, T] based on usual preprocess
                                    time_len = shape[-1] 
                                    
                                    # We can extract multiple non-overlapping or overlapping windows
                                    # For simplicity, let's extract overlapping windows with a stride of 50 TRs
                                    stride = self.window_size // 2
                                    # Fetch TR, default to None if missing for backward compat
                                    tr = sub_grp.attrs.get('repetition_time')
                                    if tr is None:
                                        logging.warning(f"Missing repetition_time for {ds_name}/{sub_id}")
                                        
                                    for start_idx in range(0, time_len - self.window_size + 1, stride):
                                        self.sequences.append({
                                            'file_path': data_path,
                                            'dataset': ds_name,
                                            'subject': sub_id,
                                            'start_idx': start_idx,
                                            'repetition_time': tr
                                        })
                                        self.total_windows += 1
                                    self.valid_subjects += 1
                        except Exception as e:
                            logging.warning(f"Failed to read shape for {ds_name}/{sub_id}: {e}")
                            
        logging.info(f"Found {self.valid_subjects} valid high-quality subjects out of {self.total_subjects}.")
        logging.info(f"Extracted {self.total_windows} sequences of length {self.window_size}.")
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        data_path = seq_info['file_path']
        sub_id = seq_info['subject']
        start = seq_info['start_idx']
        end = start + self.window_size
        
        # Load just the specific window from the HDF5 file
        with h5py.File(data_path, 'r') as f:
            # eye_block is [X, Y, Z, T]
            # We slice the last dimension
            block = f[f"{sub_id}/eye_block"][..., start:end]
            
        # Standardize strictly matching original pipeline [num_voxels, T] inside patcher
        # For now, we return the 4D tensor [X, Y, Z, T]
        tensor_block = torch.from_numpy(block).float()
        
        if self.transforms:
            tensor_block = self.transforms(tensor_block)
            
        return tensor_block

if __name__ == "__main__":
    # Test dataloader setup
    logging.basicConfig(level=logging.INFO)
    ds = JEPADataset(data_dir=Path(__file__).resolve().parent.parent.parent / "data")
    if len(ds) > 0:
        sample = ds[0]
        print(f"Sample shape: {sample.shape}")
