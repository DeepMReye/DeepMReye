import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pprint import pprint

from . import util


class DeepMReyeDataset(Dataset):
    """
    PyTorch Dataset wrapper for DeepMReye .npz files.
    """
    def __init__(self, file_list, nonan_indices, start_tr=None, end_tr=None, inner_timesteps=None, 
                 augment_list=None, training=False, mixed_batches=True, batch_size=8):
        self.file_list = file_list
        # We need to compute the total number of valid samples to implement __len__
        self.nonan_indices = nonan_indices
        self.start_tr = start_tr
        self.end_tr = end_tr
        self.inner_timesteps = inner_timesteps
        self.augment_list = augment_list if augment_list is not None else [0, 0, 0]
        self.training = training
        self.mixed_batches = mixed_batches
        self.batch_size = batch_size
        
        # Determine the total number of samples across all files
        self.samples = []
        for file_idx, f in enumerate(self.file_list):
            valid_indices = self.nonan_indices[file_idx]
            # Instead of opening all files constantly, __getitem__ will hit the disk, but we precompute the valid index mapping
            # This is slow if files are huge, but mmap="r" is used.
            for idx in valid_indices:
                self.samples.append((file_idx, idx))
                
        # For mixed batches, PyTorch DataLoader handles shuffling and batching by default
        # But the original code specifically sampled files then valid indices. We'll stick to a flat list
        # of all valid samples, which DataLoader will shuffle uniformly.

    def __len__(self):
        # If we use strict custom batching logic from original, __len__ might just define "epochs" artificially.
        # But a proper dataset exposes all samples:
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.samples[idx]
        file_path = self.file_list[file_idx]
        
        # Warning: np.load with mmap_mode="r" inside __getitem__ can be slow if done per item.
        # However, keeping them open in a list might exceed file handle limits. 
        # For now, we open, read, close.
        data = np.load(file_path, mmap_mode="r")
        
        X = data[f"data_{str(sample_idx)}"]
        y = data[f"label_{str(sample_idx)}"]
        
        # Downsample to number of inner timesteps (subTR)
        if self.inner_timesteps is not None:
             y = y[np.linspace(0, y.shape[0] - 1, self.inner_timesteps, dtype=int), :]

        # Add channel dimension
        X = X[..., np.newaxis]
        
        # Augment
        if self.training and any(self.augment_list):
            # util.augment_input expects batched input (B, ...), we pass a batch of 1
            X_aug = util.augment_input(np.expand_dims(X, 0), rotation=self.augment_list[0], shift=self.augment_list[1], zoom=self.augment_list[2])
            X = X_aug[0]
            
        # Convert to PyTorch tensors
        # Conv3d expects (Channels, Depth, Height, Width)
        # Original is (X, Y, Z, Channels=1). Let's permute to (Channels, X, Y, Z)
        X_tensor = torch.from_numpy(X).float().permute(3, 0, 1, 2)
        y_tensor = torch.from_numpy(y).float()
        
        return X_tensor, y_tensor


def create_dataloaders(
    full_training_list,
    full_testing_list,
    batch_size=8,
    withinsubject_split=None,
    augment_list=None,
    mixed_batches=True,
    inner_timesteps=None,
    num_workers=0
):
    if augment_list is None:
        augment_list = [0, 0, 0]
        
    training_dataset = create_dataset(
        full_training_list,
        training=True,
        inner_timesteps=inner_timesteps,
        augment_list=augment_list,
        mixed_batches=mixed_batches,
        withinsubject_split=withinsubject_split,
        batch_size=batch_size
    )
    
    testing_dataset = create_dataset(
        full_testing_list,
        training=False,
        inner_timesteps=inner_timesteps,
        mixed_batches=mixed_batches,
        withinsubject_split=withinsubject_split,
        batch_size=batch_size
    )

    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers) if training_dataset else None
    testing_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) if testing_dataset else None

    # For each testing subject create a single DataLoader
    single_testing_loaders, single_testing_names = get_single_dataloaders(
        full_testing_list,
        batch_size,
        string_cut=7,
        training=False,
        mixed_batches=mixed_batches,
        withinsubject_split=withinsubject_split,
        inner_timesteps=inner_timesteps,
        num_workers=num_workers
    )
    single_training_loaders, single_training_names = get_single_dataloaders(
        full_training_list,
        batch_size,
        string_cut=7,
        training=True,
        augment_list=augment_list,
        mixed_batches=mixed_batches,
        withinsubject_split=withinsubject_split,
        inner_timesteps=inner_timesteps,
        num_workers=num_workers
    )

    training_subjects_string = [os.path.splitext(os.path.basename(p))[0] for p in full_training_list]
    test_subjects_string = [os.path.splitext(os.path.basename(p))[0] for p in full_testing_list]

    print(
        f"{util.color.BOLD}Training set ({os.path.dirname(full_training_list[0]) if full_training_list else ''}) "
        f"contains {len(full_training_list)} subjects: {util.color.END}"
    )
    pprint(training_subjects_string, compact=True)
    print(
        f"{util.color.BOLD}Test set ({os.path.dirname(full_testing_list[0]) if full_testing_list else ''}) "
        f"contains {len(full_testing_list)} subjects: {util.color.END}"
    )
    pprint(test_subjects_string, compact=True)
    
    return (
        training_loader,
        testing_loader,
        single_testing_loaders,
        single_testing_names,
        single_training_loaders,
        single_training_names,
    )


def create_holdout_dataloaders(datasets, train_split=0.6, **args):
    full_training_list, full_testing_list = list(), list()
    for fn_data in datasets:
        fn_data_str = str(fn_data)
        this_file_list = [os.path.join(fn_data_str, p) for p in os.listdir(fn_data_str)]
        np.random.shuffle(this_file_list)
        train_test_split = int(train_split * len(this_file_list))
        this_training_list = this_file_list[0:train_test_split]
        this_testing_list = this_file_list[train_test_split::]
        full_training_list.extend(this_training_list)
        full_testing_list.extend(this_testing_list)
    (
        training_loader,
        testing_loader,
        single_testing_loaders,
        single_testing_names,
        single_training_loaders,
        single_training_names,
    ) = create_dataloaders(full_training_list, full_testing_list, **args)

    return (
        training_loader,
        testing_loader,
        single_testing_loaders,
        single_testing_names,
        single_training_loaders,
        single_training_names,
        full_testing_list,
        full_training_list,
    )


def create_cv_dataloaders(dataset, num_cvs=5, **args):
    this_file_list = [os.path.join(dataset, p) for p in os.listdir(dataset)]
    np.random.shuffle(this_file_list)
    cv_split = np.array_split(this_file_list, num_cvs)
    cv_return = []
    for idx, cvs in enumerate(cv_split):
        full_testing_list = cvs.tolist()
        full_training_list = np.concatenate([x for i, x in enumerate(cv_split) if i != idx]).tolist()

        (
            training_loader,
            testing_loader,
            single_testing_loaders,
            single_testing_names,
            single_all_loaders,
            single_all_names,
        ) = create_dataloaders(full_training_list, full_testing_list, **args)
        cv_return.append(
            (
                training_loader,
                testing_loader,
                single_testing_loaders,
                single_testing_names,
                single_all_loaders,
                single_all_names,
                full_testing_list,
                full_training_list,
            )
        )

    return cv_return


def create_leaveoneout_dataloaders(datasets, training_subset=None, **args):
    loo_return = []
    for idx, dataset in enumerate(datasets):
        dataset_str = str(dataset)
        full_testing_list = [os.path.join(dataset_str, p) for p in os.listdir(dataset_str)]
        training_datasets = [str(x) for i, x in enumerate(datasets) if i != idx]
        full_training_list = [os.path.join(tds, p) for tds in training_datasets for p in os.listdir(tds)]

        if training_subset is not None:
            size_before = len(full_training_list)
            full_training_list = [
                tds
                for tds in full_training_list
                if f"{os.path.basename(os.path.dirname(tds))}/{os.path.basename(tds)}" in training_subset
            ]
            print(f"Using subset ({len(full_training_list)} / {size_before}) for training {dataset}")

        (
            training_loader,
            testing_loader,
            single_testing_loaders,
            single_testing_names,
            single_all_loaders,
            single_all_names,
        ) = create_dataloaders(full_training_list, full_testing_list, **args)
        loo_return.append(
            (
                training_loader,
                testing_loader,
                single_testing_loaders,
                single_testing_names,
                single_all_loaders,
                single_all_names,
                full_testing_list,
                full_training_list,
            )
        )

    return loo_return


def get_single_dataloaders(fn_list, batch_size, string_cut=4, num_workers=0, **args):
    loaders, names = list(), list()
    for subject in fn_list:
        ds = create_dataset([subject], batch_size, **args)
        if ds is not None and len(ds) > 0:
            loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers))
        else:
            loaders.append(None)
        this_name = os.path.basename(subject)[:-4]
        if len(this_name) > string_cut:
            this_name = this_name[-string_cut:]
        names.append(this_name)
    return loaders, names


def create_dataset(
    file_list,
    batch_size,
    training=False,
    mixed_batches=True,
    withinsubject_split=None,
    augment_list=None,
    inner_timesteps=None,
):
    if not file_list:
        return None
    if augment_list is None:
        augment_list = [0, 0, 0]
    
    all_nonan_indices = get_nonan_indices(file_list)
    start_tr, end_tr = get_start_end_tr(withinsubject_split, training)
    
    new_filelist, new_nonan = [], []
    for x in range(len(file_list)):
        if all_nonan_indices[x].size == 0:
            continue
            
        data = np.load(file_list[x], mmap_mode="r")
        try:
            _ = data["identifier_0"]
            divisor = 3
        except:
            divisor = 2
        num_trs = len(data) // divisor
        
        start_idx, end_idx = get_tr_indices(num_trs, start_tr, end_tr)
        valid_indices = all_nonan_indices[x]
        valid_indices = valid_indices[(valid_indices >= start_idx) & (valid_indices <= end_idx)]
        
        if valid_indices.size > 0:
            new_filelist.append(file_list[x])
            new_nonan.append(valid_indices)
            
    if not new_filelist:
        return None
        
    return DeepMReyeDataset(
        new_filelist, new_nonan, start_tr, end_tr, inner_timesteps, 
        augment_list, training, mixed_batches, batch_size
    )


def get_tr_indices(num_trs, start_tr, end_tr):
    start_tr = 0 if start_tr is None else int(start_tr * num_trs)
    end_tr = num_trs if end_tr is None else int(end_tr * num_trs)
    return (start_tr, end_tr)


def get_start_end_tr(withinsubject_split, training):
    if withinsubject_split:
        if training:
            start_tr = withinsubject_split[0]
            end_tr = withinsubject_split[1]
        else:
            if withinsubject_split[0] == 0:
                start_tr = withinsubject_split[1]
                end_tr = None
            else:
                start_tr = None
                end_tr = withinsubject_split[0]
    else:
        start_tr = None
        end_tr = None
    return (start_tr, end_tr)


def get_nonan_indices(file_list):
    nonan_indices = []
    for fn_subject in file_list:
        data = np.load(fn_subject, mmap_mode="r")
        try:
            _ = data["identifier_0"]
            divisor = 3
        except:
            divisor = 2
        num_trs = len(data) // divisor
        y = np.array([data[f"label_{str(sample_index)}"] for sample_index in range(num_trs)])
        nonan_index = ~np.any(np.isnan(y), axis=(1, 2))
        nonan_indices.append(np.where(nonan_index)[0])
    return nonan_indices

# Keeping for compatibility if needed, but updated to PyTorch shapes
def get_all_subject_data(fn_subject):
    data = np.load(fn_subject, mmap_mode="r")
    try:
        _ = data["identifier_0"]
        divisor = 3
    except:
        divisor = 2
    num_trs = len(data) // divisor
    subject_data_X, subject_data_y = [], []
    for sample_index in range(num_trs):
        subject_data_X.append(data[f"data_{str(sample_index)}"])
        subject_data_y.append(data[f"label_{str(sample_index)}"])
    
    # original shape X: (batch, X, Y, Z)
    X = np.array(subject_data_X)[..., np.newaxis]
    y = np.array(subject_data_y)
    
    # Return as tensors
    X_tensor = torch.from_numpy(X).float().permute(0, 4, 1, 2, 3) # (Batch, Channels, X,Y,Z)
    y_tensor = torch.from_numpy(y).float()
    
    return X_tensor, y_tensor
