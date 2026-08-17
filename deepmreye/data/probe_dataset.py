import logging
from collections import namedtuple
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from deepmreye.storage import iter_subjects
from deepmreye.validation import is_plausible_tr

Subject = namedtuple("Subject", "path dataset subject n_trs tr")

# The six gaze-labeled datasets are three paradigms, not six. dsL02/03/04 are
# all smooth pursuit, so holding out one of them while training on the other two
# is not a test of transfer to an unseen paradigm -- it is leave-one-acquisition
# -out. These groups give the stricter number.
PARADIGM_GROUPS = {
    "fixation": ["dsL01_guided_fixations"],
    "pursuit": ["dsL02_pursuit", "dsL03_pursuit", "dsL04_pursuit"],
    "free_viewing": ["dsL05_free_viewing"],
    "sequences": ["dsL06_sequences"],
}


def dataset_folds(datasets):
    """Leave-one-dataset-out folds: ``[(name, {holdout}), ...]``."""
    return [(ds, {ds}) for ds in sorted(datasets)]


def paradigm_folds(datasets):
    """Leave-one-paradigm-out folds, skipping paradigms absent from ``datasets``."""
    present = set(datasets)
    folds = []
    for name, members in PARADIGM_GROUPS.items():
        holdout = present & set(members)
        if holdout and present - holdout:
            folds.append((name, holdout))
    return folds


class ProbeDataset(Dataset):
    """Windows of *labeled* eye blocks for the supervised gaze probe.

    Same on-disk format as :class:`JEPADataset` -- one HDF5 file per
    participant under ``<dir>/<dataset>/<subject>.h5`` -- except these files
    also carry a ``labels`` dataset of shape ``[T, 10, 2]`` (10 sub-TR gaze
    samples, x and y). Subjects without labels are skipped.

    Yields ``(block [X, Y, Z, W], labels [W, 10, 2], dataset_name, subject_id,
    tr)``. Labels keep their NaNs; the evaluation masks them out, so dropping
    them here would silently shift the time alignment between block and gaze.
    The TR comes along because the encoder conditions on it and because window
    duration differs between datasets. The subject id comes along because it is
    the right unit to aggregate metrics over -- pooling every window of every
    participant into one correlation mixes between-subject variance into a
    number that is meant to describe within-subject decoding.

    Splitting is one of four, in increasing strictness:

    - ``split_by="time"``    -- within subject, early timepoints train and late
      ones test. The same participant, scanner and paradigm; the easiest
      setting, and the one that says whether the method works at all.
    - ``split_by="subject"`` -- held-out participants, same scanner and paradigm.
    - ``split_by="dataset"`` -- a random 80/20 over datasets.
    - ``holdout={...}`` -- a named fold, which is what leave-one-dataset-out and
      leave-one-paradigm-out use (see :func:`dataset_folds`,
      :func:`paradigm_folds`).
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
        holdout=None,
        gap=0,
        datasets=None,
    ):
        self.labeled_data_dir = Path(labeled_data_dir).resolve()
        self.split = split
        self.split_ratio = split_ratio
        self.window_size = window_size
        self.transforms = transforms
        self.split_by = split_by
        self.seed = seed
        # Extra TRs discarded either side of a within-subject time split. Zero
        # by default: train and test windows already share no timepoint, and the
        # labeled runs are short enough (dsL01 is 270 TRs) that a gap of one
        # window would leave several datasets with no test split at all.
        self.gap = gap
        # Explicit held-out datasets. Naming the fold beats a random 80/20 over
        # datasets: with six labeled sets, a random split is one arbitrary draw,
        # while leave-one-dataset-out is six folds every subject appears in
        # exactly once as test.
        self.holdout = set(holdout) if holdout else None
        # Restrict the corpus *before* any split logic runs. Combined with
        # `holdout` this expresses train-on-one/test-on-one: pass
        # datasets={S, T}, holdout={T}, and the existing leave-one-out branch
        # yields exactly S for train and T for test. That is the protocol the
        # published single-dataset DeepMReye checkpoints were trained under, so
        # it is the only way to compare against them like for like.
        self.datasets = set(datasets) if datasets else None

        self.samples = []
        self._build_index()

    def _discover(self):
        """Every participant file that actually carries gaze labels."""
        from concurrent.futures import ThreadPoolExecutor

        subs = list(iter_subjects(self.labeled_data_dir))

        def inspect(item):
            ds_name, sub_id, path = item
            try:
                with h5py.File(path, "r") as f:
                    if "eye_block" not in f or "labels" not in f:
                        return None
                    tr = f.attrs.get("repetition_time")
                    if not is_plausible_tr(tr):
                        return None
                    return Subject(str(path), ds_name, sub_id,
                                   f["eye_block"].shape[-1], float(tr))
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=8) as pool:
            res = list(pool.map(inspect, subs))
        return [s for s in res if s is not None]

    def _build_index(self):
        if not self.labeled_data_dir.exists():
            raise FileNotFoundError(f"Labeled data directory not found at {self.labeled_data_dir}")

        logging.info(f"Scanning labeled data for '{self.split}' split...")
        all_subjects = self._discover()
        if self.datasets is not None:
            all_subjects = [s for s in all_subjects if s.dataset in self.datasets]
        if not all_subjects:
            logging.warning("No labeled participant files found.")
            return

        rng = np.random.default_rng(self.seed)

        if self.split_by == "time" and self.holdout is None:
            # Within subject: every participant appears in both splits, cut
            # along its own timeline. No timepoint is shared between the two --
            # windows overlap by half a window, so a naive index-wise split
            # would put the *same TRs* on both sides and report near-perfect
            # scores. We have no run boundaries stored, so a temporal cut is the
            # finest honest division available; train and test stay temporally
            # adjacent, which `gap` can widen at the cost of usable windows.
            target = all_subjects
        elif self.holdout is not None:
            # Leave-one-dataset-out (or leave-one-paradigm-out) fold.
            keep = ({s.dataset for s in all_subjects} - self.holdout
                    if self.split == "train" else self.holdout)
            target = [s for s in all_subjects if s.dataset in keep]
        elif self.split_by == "dataset":
            # Hold out whole datasets: the honest test of whether the
            # representation transfers to an unseen scanner and paradigm.
            datasets = sorted({s.dataset for s in all_subjects})
            order = rng.permutation(len(datasets))
            cut = int(len(datasets) * self.split_ratio)
            keep = {datasets[i] for i in (order[:cut] if self.split == "train" else order[cut:])}
            target = [s for s in all_subjects if s.dataset in keep]
        else:
            # Split subjects *within* each dataset. Shuffling the pooled list
            # instead would put most subjects of a small dataset on one side by
            # chance, and could leave a split with no data at all.
            target = []
            by_ds = {}
            for s in all_subjects:
                by_ds.setdefault(s.dataset, []).append(s)
            for ds_name in sorted(by_ds):
                subs = sorted(by_ds[ds_name], key=lambda s: s.subject)
                order = rng.permutation(len(subs))
                cut = int(round(len(subs) * self.split_ratio))
                # Guarantee both splits are non-empty when a dataset has >= 2
                # subjects, so a small dataset can still be evaluated.
                if len(subs) >= 2:
                    cut = min(max(cut, 1), len(subs) - 1)
                picked = order[:cut] if self.split == "train" else order[cut:]
                target.extend(subs[i] for i in picked)

        for sub in target:
            if sub.n_trs < self.window_size:
                continue
            stride = self.window_size // 2
            w = self.window_size
            by_time = self.split_by == "time" and self.holdout is None
            # The cut cannot sit later than the *last window start*, or the test
            # side gets nothing. Two things conspire here and only clamping to
            # the stride grid handles both: the final w TRs host no new window,
            # and starts land on multiples of `stride`. dsL01 runs are 270 TRs,
            # so at split_ratio 0.8 the cut wants to be 216, the last legal
            # start is 170, and the last start *on the grid* is 150 -- a cut
            # anywhere above 150 silently gives that dataset, 170 of the 270
            # labeled subjects, zero test windows. Clamping yields a smaller
            # test fraction on short runs rather than an empty one.
            last_start = ((sub.n_trs - w) // stride) * stride
            cut = min(int(sub.n_trs * self.split_ratio), last_start) if by_time else 0
            for start in range(0, sub.n_trs - w + 1, stride):
                if by_time:
                    # Train windows must *end* at or before the cut; test
                    # windows must start at or after it. That is what keeps the
                    # two sides disjoint in time rather than merely in index.
                    if self.split == "train" and start + w > cut - self.gap:
                        continue
                    if self.split == "test" and start < cut + self.gap:
                        continue
                self.samples.append(
                    {
                        "path": sub.path,
                        "dataset": sub.dataset,
                        "subject": sub.subject,
                        "start": start,
                        "tr": sub.tr,
                    }
                )

        n_ds = len({s["dataset"] for s in self.samples})
        logging.info(f"Loaded {len(self.samples)} labeled windows from {n_ds} datasets "
                     f"for '{self.split}'.")

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

        return (x_tensor, y_tensor, sample["dataset"], sample["subject"],
                torch.tensor(sample["tr"], dtype=torch.float32))
