#!/usr/bin/env python3
"""Bring the labeled gaze datasets into the same layout as the unlabeled ones.

The labeled data arrived in two earlier shapes -- per-subject ``.npz`` files, and
a single ``<dataset>.h5`` nesting every subject under a dataset group. Both are
rewritten here to one file per participant::

    <out_dir>/<dataset>/<subject>.h5     eye_block [X, Y, Z, T] + labels [T, 10, 2]

so labeled and unlabeled participants are byte-format identical and the probe
and JEPA loaders share one code path.

The blocks are already normalized (z-scored, clipped at 5 SD) and use the same
eye mask as the pipeline, so the voxel data is copied through unchanged; only
the container, chunking and metadata are rebuilt. Labels keep their NaNs --
those mark TRs with no valid gaze sample and the evaluation masks them.
"""
import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.storage import is_intact, subject_path, write_subject

EXPECTED_SPATIAL = (47, 29, 18)

# Subject names carry the acquisition TR, e.g. S4_0004_TR1250_2MM -> 1.25 s.
_TR_RE = re.compile(r"TR(\d+)")


def tr_from_name(name):
    """Best-effort TR in seconds parsed from a subject name."""
    m = _TR_RE.search(name)
    if not m:
        return None
    value = float(m.group(1))
    # Values are milliseconds in practice; a bare "2" would mean seconds.
    return value / 1000.0 if value > 100 else value


def _write(out_dir, ds_name, sub_name, eye_block, labels, source):
    if eye_block.shape[:3] != EXPECTED_SPATIAL:
        print(f"  [!] {ds_name}/{sub_name}: unexpected spatial shape {eye_block.shape[:3]}, "
              f"expected {EXPECTED_SPATIAL} -- skipping")
        return False

    if labels.shape[0] != eye_block.shape[-1]:
        print(f"  [!] {ds_name}/{sub_name}: {labels.shape[0]} labels vs "
              f"{eye_block.shape[-1]} TRs -- skipping")
        return False

    attrs = {
        "dataset": ds_name,
        "subject": sub_name,
        "normalized": True,
        "source_file": str(source),
        "label_units": "degrees_visual_angle",
    }
    tr = tr_from_name(sub_name)
    if tr is not None:
        attrs["repetition_time"] = tr

    write_subject(subject_path(out_dir, ds_name, sub_name), eye_block, labels=labels, attrs=attrs)
    return True


def convert_h5(path, out_dir, force=False):
    """Split a nested ``<dataset>.h5`` into per-participant files."""
    ds_name = path.stem
    n = 0
    with h5py.File(path, "r") as f:
        # Tolerate both /<dataset>/<subject>/... and a flat /<subject>/...
        groups = [f[ds_name]] if ds_name in f and isinstance(f[ds_name], h5py.Group) else [f]
        for grp in groups:
            for sub_name in tqdm(list(grp.keys()), desc=f"  {ds_name}", leave=False):
                sub = grp[sub_name]
                if not isinstance(sub, h5py.Group) or "eye_block" not in sub:
                    continue

                out_path = subject_path(out_dir, ds_name, sub_name)
                if not force and is_intact(out_path):
                    continue

                if "labels" not in sub:
                    print(f"  [!] {ds_name}/{sub_name}: no labels -- skipping")
                    continue

                if _write(out_dir, ds_name, sub_name,
                          sub["eye_block"][:], sub["labels"][:], path):
                    n += 1
    return n


def convert_npz(ds_dir, out_dir, force=False):
    """Rebuild per-participant files from the original ``.npz`` exports."""
    ds_name = ds_dir.name
    n = 0
    for npz_path in tqdm(sorted(ds_dir.glob("*.npz")), desc=f"  {ds_name}", leave=False):
        sub_name = npz_path.stem
        out_path = subject_path(out_dir, ds_name, sub_name)
        if not force and is_intact(out_path):
            continue

        try:
            with np.load(npz_path) as data:
                idx = sorted(
                    int(re.search(r"_(\d+)$", k).group(1))
                    for k in data.files
                    if k.startswith("data_")
                )
                if not idx:
                    continue
                eye_block = np.stack([data[f"data_{i}"] for i in idx], axis=-1)
                labels = np.stack([data[f"label_{i}"] for i in idx], axis=0)
        except Exception as e:
            print(f"  [!] {npz_path.name}: {e}")
            continue

        if _write(out_dir, ds_name, sub_name, eye_block, labels, npz_path):
            n += 1
    return n


def run_convert(labeled_dir, out_dir, force=False):
    labeled_dir = Path(labeled_dir).resolve()
    out_dir = Path(out_dir).resolve()
    if not labeled_dir.exists():
        print(f"Error: {labeled_dir} does not exist.")
        return

    total = 0
    for h5_path in sorted(labeled_dir.glob("*.h5")):
        print(f"[*] {h5_path.name}")
        try:
            # Files still uploading fail only on open; report them rather than
            # letting the dataset silently go missing from the artifact.
            with h5py.File(h5_path, "r"):
                pass
        except Exception as e:
            print(f"  [!] {h5_path.name} is truncated or unreadable ({e.__class__.__name__}) "
                  f"-- skipping. Re-run once the upload finishes.")
            continue
        total += convert_h5(h5_path, out_dir, force=force)

    for ds_dir in sorted(p for p in labeled_dir.iterdir() if p.is_dir()):
        if not any(ds_dir.glob("*.npz")):
            continue
        print(f"[*] {ds_dir.name} (npz)")
        total += convert_npz(ds_dir, out_dir, force=force)

    print(f"\n[+] Wrote {total} participant files to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert labeled data to the unified layout.")
    parser.add_argument("--labeled-dir", required=True, help="Directory of labeled .h5 / .npz data.")
    parser.add_argument("--out-dir", required=True, help="Output root for <dataset>/<subject>.h5.")
    parser.add_argument("--force", action="store_true", help="Rewrite files that already exist.")
    args = parser.parse_args()
    run_convert(args.labeled_dir, args.out_dir, force=args.force)


if __name__ == "__main__":
    main()
