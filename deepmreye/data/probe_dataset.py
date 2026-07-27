import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from deepmreye.storage import iter_subjects


class ProbeDataset(Dataset):
    """Windows of *labeled* eye blocks for the supervised gaze probe.

    Same on-disk format as :class:`JEPADataset` -- one HDF5 file per
    participant under ``<dir>/<dataset>/<subject>.h5`` -- except these files
    also carry a ``labels`` dataset of shape ``[T, 10, 2]`` (10 sub-TR gaze
    samples, x and y). Subjects without labels are skipped.

    Yields ``(block [X, Y, Z, W], labels [W, 10, 2], dataset_name)``. Labels
    keep their NaNs; the evaluation loop masks them out, so dropping them here
    would silently shift the time alignment between block and gaze.
    """

    def __init__(
        self,
        labeled_data_dir,
        split="train",
        split_ratio=0.8,
        window_size=100,
        transforms=None,
        split_by="subject",
        seed=42,
    ):
        self.labeled_data_dir = Path(labeled_data_dir).resolve()
        self.split = split
        self.split_ratio = split_ratio
        self.window_size = window_size
        self.transforms = transforms
        self.split_by = split_by
        self.seed = seed

        self.samples = []
        self._build_index()

    def _discover(self):
        """Every participant file that actually carries gaze labels."""
        found = []
        for ds_name, sub_id, path in iter_subjects(self.labeled_data_dir):
            try:
                with h5py.File(path, "r") as f:
                    if "eye_block" in f and "labels" in f:
                        found.append((str(path), ds_name, sub_id, f["eye_block"].shape[-1]))
            except Exception as e:
                logging.warning(f"Failed to scan {path}: {e}")
        return found

    def _build_index(self):
        if not self.labeled_data_dir.exists():
            raise FileNotFoundError(f"Labeled data directory not found at {self.labeled_data_dir}")

        logging.info(f"Scanning labeled data for '{self.split}' split...")
        all_subjects = self._discover()
        if not all_subjects:
            logging.warning("No labeled participant files found.")
            return

        rng = np.random.default_rng(self.seed)

        if self.split_by == "dataset":
            # Hold out whole datasets: the honest test of whether the
            # representation transfers to an unseen scanner and paradigm.
            datasets = sorted({ds for _, ds, _, _ in all_subjects})
            order = rng.permutation(len(datasets))
            cut = int(len(datasets) * self.split_ratio)
            keep = {datasets[i] for i in (order[:cut] if self.split == "train" else order[cut:])}
            target = [s for s in all_subjects if s[1] in keep]
        else:
            # Split subjects *within* each dataset. Shuffling the pooled list
            # instead would put most subjects of a small dataset on one side by
            # chance, and could leave a split with no data at all.
            target = []
            by_ds = {}
            for s in all_subjects:
                by_ds.setdefault(s[1], []).append(s)
            for ds_name in sorted(by_ds):
                subs = sorted(by_ds[ds_name], key=lambda s: s[2])
                order = rng.permutation(len(subs))
                cut = int(round(len(subs) * self.split_ratio))
                # Guarantee both splits are non-empty when a dataset has >= 2
                # subjects, so a small dataset can still be evaluated.
                if len(subs) >= 2:
                    cut = min(max(cut, 1), len(subs) - 1)
                picked = order[:cut] if self.split == "train" else order[cut:]
                target.extend(subs[i] for i in picked)

        for path, ds_name, sub_id, n_trs in target:
            if n_trs < self.window_size:
                continue
            stride = self.window_size // 2
            for start in range(0, n_trs - self.window_size + 1, stride):
                self.samples.append(
                    {
                        "path": path,
                        "dataset": ds_name,
                        "subject": sub_id,
                        "start": start,
                    }
                )

        logging.info(f"Loaded {len(self.samples)} labeled windows for {self.split}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        start = sample["start"]
        end = start + self.window_size

        with h5py.File(sample["path"], "r") as f:
            x_arr = f["eye_block"][..., start:end]
            y_arr = f["labels"][start:end]

        x_tensor = torch.from_numpy(x_arr).float()
        y_tensor = torch.from_numpy(y_arr).float()

        if self.transforms:
            x_tensor = self.transforms(x_tensor)

        return x_tensor, y_tensor, sample["dataset"]
