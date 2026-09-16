#!/usr/bin/env python3
"""Every gaze-labelled participant's masked voxels, once, as a flat memmap.

The basis experiments downstream all have the same shape -- a `[14236, k]`
linear map applied to the same 337 participants -- and the only expensive part
of trying one is re-reading 21 GB of HDF5. Writing the masked rows out once
turns each new basis into a single matmul over a memmap, so a shrinkage sweep
costs what a shrinkage sweep should cost.

Float32 rather than float16: these are raw BOLD intensities in the hundreds, and
the covariances fitted from them are differences of large numbers, which is
exactly where half precision stops being free.

    python scripts/build_labeled_voxels.py --out <dir>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmreye import probe  # noqa: E402
from deepmreye.datasource import resolve  # noqa: E402
from deepmreye.unsupervised import corpus_mask  # noqa: E402


def labeled_files(root):
    for d in sorted(Path(root).glob("dsL*")):
        for p in sorted(d.glob("*.h5")):
            yield d.name, p


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    root = resolve(a.data_dir, download=False, quiet=True)
    mask = corpus_mask(root)
    flat = mask.reshape(-1)
    n_vox = int(flat.sum())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # Pass 1: shapes, so the memmap can be sized before anything is written.
    plan, total = [], 0
    for ds, path in labeled_files(root):
        with h5py.File(path, "r") as f:
            if "labels" not in f:
                continue
            t = f["eye_block"].shape[-1]
            lab = f["labels"][:]
        if t < probe.MIN_TRS or not np.isfinite(lab).any():
            continue
        n = min(t, len(lab))
        plan.append({"dataset": ds, "subject": path.stem, "path": str(path),
                     "start": total, "stop": total + n})
        total += n
    print(f"[*] {len(plan)} participants, {total} TRs, {n_vox} voxels "
          f"({total * n_vox * 4 / 1e9:.1f} GB)", flush=True)

    mm = np.lib.format.open_memmap(out / "voxels.npy", mode="w+",
                                   dtype=np.float32, shape=(total, n_vox))
    labels, t0 = {}, time.time()
    for i, rec in enumerate(plan):
        with h5py.File(rec["path"], "r") as f:
            block = f["eye_block"][:]
            lab = f["labels"][:]
        n = rec["stop"] - rec["start"]
        t = block.shape[-1]
        mm[rec["start"]:rec["stop"]] = block.reshape(-1, t).T[:n, flat]
        labels[f"{rec['dataset']}/{rec['subject']}"] = lab[:n].astype(np.float32)
        if i % 25 == 0:
            print(f"  [{i + 1}/{len(plan)}] {time.time() - t0:.0f}s", flush=True)
    mm.flush()
    np.savez(out / "labels.npz", **labels)
    np.save(out / "mask.npy", mask)
    (out / "index.json").write_text(json.dumps(plan, indent=1))
    print(f"[+] {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
