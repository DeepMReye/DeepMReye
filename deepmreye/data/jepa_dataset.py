import logging
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

from deepmreye.pipeline import is_dataset_approved
from deepmreye.storage import subject_path


class JEPADataset(Dataset):
    """Windows of unlabeled eye blocks for self-supervised pretraining.

    Reads one HDF5 file per participant (``<data_dir>/<dataset>/<subject>.h5``)
    and indexes overlapping windows of ``window_size`` TRs. Only subjects
    belonging to a manually approved dataset are included; approval is the
    eyes / no-eyes QA labeling, there is no automatic quality gate.

    The index stores offsets only, and each ``__getitem__`` reads just its own
    window, so memory stays flat no matter how much data is on disk.
    """

    def __init__(self, data_dir, registry_path="datasets.h5", window_size=100, transforms=None):
        self.data_dir = Path(data_dir).resolve()
        self.registry_path = self.data_dir / registry_path
        self.window_size = window_size
        self.transforms = transforms

        self.sequences = []
        self._build_index()

    def _build_index(self):
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Registry not found at {self.registry_path}")

        logging.info("Scanning HDF5 registry for subjects in manually approved datasets...")

        self.total_datasets = 0
        self.total_subjects = 0
        self.valid_subjects = 0
        self.total_windows = 0

        stride = self.window_size // 2

        with h5py.File(self.registry_path, "r") as h5_reg:
            ds_keys = list(h5_reg.keys())
            self.total_datasets = len(ds_keys)

            for ds_name in ds_keys:
                if not is_dataset_approved(h5_reg[ds_name]):
                    continue

                sub_keys = list(h5_reg[ds_name].keys())
                self.total_subjects += len(sub_keys)

                for sub_id in sub_keys:
                    sub_grp = h5_reg[ds_name][sub_id]
                    path = subject_path(self.data_dir, ds_name, sub_id)
                    if not path.exists():
                        continue

                    tr = sub_grp.attrs.get("repetition_time")
                    if tr is None:
                        logging.warning(f"Missing repetition_time for {ds_name}/{sub_id}")

                    try:
                        with h5py.File(path, "r") as f:
                            time_len = f["eye_block"].shape[-1]
                    except Exception as e:
                        # Truncated or half-written files are skipped rather
                        # than killing a training run that is already going.
                        logging.warning(f"Failed to read {path}: {e}")
                        continue

                    if time_len < self.window_size:
                        continue

                    for start_idx in range(0, time_len - self.window_size + 1, stride):
                        self.sequences.append(
                            {
                                "file_path": str(path),
                                "dataset": ds_name,
                                "subject": sub_id,
                                "start_idx": start_idx,
                                "repetition_time": tr,
                            }
                        )
                        self.total_windows += 1
                    self.valid_subjects += 1

        logging.info(f"Found {self.valid_subjects} valid subjects out of {self.total_subjects}.")
        logging.info(f"Extracted {self.total_windows} sequences of length {self.window_size}.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        start = seq["start_idx"]
        end = start + self.window_size

        with h5py.File(seq["file_path"], "r") as f:
            block = f["eye_block"][..., start:end]

        tensor_block = torch.from_numpy(block).float()
        if self.transforms:
            tensor_block = self.transforms(tensor_block)
        return tensor_block
