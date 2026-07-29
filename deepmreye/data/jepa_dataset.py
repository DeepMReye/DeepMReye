import logging
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

from deepmreye.pipeline import is_dataset_approved, is_labeled_dataset
from deepmreye.storage import subject_path
from deepmreye.validation import is_plausible_tr


class JEPADataset(Dataset):
    """Windows of unlabeled eye blocks for self-supervised pretraining.

    Reads one HDF5 file per participant (``<data_dir>/<dataset>/<subject>.h5``)
    and indexes overlapping windows of ``window_size`` TRs. Only subjects
    belonging to a manually approved dataset are included; approval is the
    eyes / no-eyes QA labeling, there is no automatic quality gate.

    The gaze-labeled datasets (``dsL*``) are excluded by default. They are the
    probe's evaluation set, and they are also enormously over-represented: their
    runs are 2,000-4,200 TRs against a ~270 TR median elsewhere, so 6 datasets
    would contribute 45% of all pretraining windows against 691 others. Keeping
    them out also makes leave-one-dataset-out mean "a scanner the encoder never
    saw", rather than "a scanner it saw unlabeled".

    Each item is ``(block [X, Y, Z, W], tr)``. The TR is returned because the
    model conditions its temporal encoding on it -- a window of 100 TRs is 80 s
    in one dataset and 250 s in another, and that difference is only learnable
    if it is an input.

    The index stores offsets only, and each ``__getitem__`` reads just its own
    window, so memory stays flat no matter how much data is on disk.
    """

    def __init__(self, data_dir, registry_path="datasets.h5", window_size=100, transforms=None,
                 exclude_labeled=True, require_tr=True):
        self.data_dir = Path(data_dir).resolve()
        self.registry_path = self.data_dir / registry_path
        self.window_size = window_size
        self.transforms = transforms
        self.exclude_labeled = exclude_labeled
        self.require_tr = require_tr

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
        self.skipped = {"labeled": 0, "no_file": 0, "bad_tr": 0, "too_short": 0, "unreadable": 0}

        stride = self.window_size // 2

        with h5py.File(self.registry_path, "r") as h5_reg:
            ds_keys = list(h5_reg.keys())
            self.total_datasets = len(ds_keys)

            for ds_name in ds_keys:
                ds_grp = h5_reg[ds_name]
                if not is_dataset_approved(ds_grp):
                    continue
                if self.exclude_labeled and is_labeled_dataset(ds_grp, ds_name):
                    self.skipped["labeled"] += len(ds_grp.keys())
                    continue

                sub_keys = list(ds_grp.keys())
                self.total_subjects += len(sub_keys)

                for sub_id in sub_keys:
                    sub_grp = ds_grp[sub_id]
                    path = subject_path(self.data_dir, ds_name, sub_id)
                    if not path.exists():
                        self.skipped["no_file"] += 1
                        continue

                    tr = sub_grp.attrs.get("repetition_time")
                    if self.require_tr and not is_plausible_tr(tr):
                        # TR conditions the temporal encoding, so a missing or
                        # nonsense one cannot be defaulted -- it would place the
                        # window at the wrong timescale.
                        self.skipped["bad_tr"] += 1
                        continue

                    # Prefer the registry's n_trs over opening the file. At full
                    # extraction this index covers tens of thousands of subjects,
                    # and one HDF5 open apiece is the difference between a few
                    # seconds and several minutes before training starts.
                    time_len = sub_grp.attrs.get("n_trs")
                    if time_len is None:
                        try:
                            with h5py.File(path, "r") as f:
                                time_len = f["eye_block"].shape[-1]
                        except Exception as e:
                            # Truncated or half-written files are skipped rather
                            # than killing a training run that is already going.
                            logging.warning(f"Failed to read {path}: {e}")
                            self.skipped["unreadable"] += 1
                            continue
                    time_len = int(time_len)

                    if time_len < self.window_size:
                        self.skipped["too_short"] += 1
                        continue

                    for start_idx in range(0, time_len - self.window_size + 1, stride):
                        self.sequences.append(
                            {
                                "file_path": str(path),
                                "dataset": ds_name,
                                "subject": sub_id,
                                "start_idx": start_idx,
                                "repetition_time": float(tr) if tr is not None else 0.0,
                            }
                        )
                        self.total_windows += 1
                    self.valid_subjects += 1

        logging.info(f"Found {self.valid_subjects} valid subjects out of {self.total_subjects}.")
        logging.info(f"Extracted {self.total_windows} sequences of length {self.window_size}.")
        if any(self.skipped.values()):
            logging.info(f"Skipped: {self.skipped}")

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
        return tensor_block, torch.tensor(seq["repetition_time"], dtype=torch.float32)
