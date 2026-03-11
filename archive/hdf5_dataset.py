import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class HDF5StreamDataset(Dataset):
    """
    Streams spatial eye blocks natively from the HDF5 datastore.
    Iterates dynamically over only datasets marked as approved by the human GUI labeler.
    """
    def __init__(self, h5_path, transform=None):
        super().__init__()
        self.h5_path = h5_path
        self.transform = transform
        self.samples = [] # List of tuples: (dataset_id, subject_id)
        
        # Build registry index
        with h5py.File(h5_path, "r") as f:
            for ds in f.keys():
                grp = f[ds]
                if 'approved' in grp.attrs:
                    approved = grp.attrs['approved']
                    if approved == 1:
                        for sub in grp.keys():
                            if 'data_path' in grp[sub].attrs:
                                self.samples.append((ds, sub))
                                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ds, sub = self.samples[idx]
        
        with h5py.File(self.h5_path, "r") as f:
            data_path = f[ds][sub].attrs['data_path']
            # Simulated load: in reality, this uses Nibabel/ANTs to stream spatial blocks dynamically
            # For scaffolding, return a dummy random block
            # Shape convention: (Channels, Depth, Height, Width)
            dummy_block = np.random.randn(1, 16, 16, 16).astype(np.float32)
            
        tensor_block = torch.from_numpy(dummy_block)
        
        if self.transform:
            tensor_block = self.transform(tensor_block)
            
        return tensor_block
