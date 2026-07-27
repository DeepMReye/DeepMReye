#!/usr/bin/env python3
"""Build the parquet index over every extracted participant, and validate them.

The index is what makes the published artifact browsable without opening a
single HDF5 file: one row per participant, so you can filter to labeled
subjects, pick a TR range, or spot short runs with a dataframe query. Parquet
because it is tabular metadata -- the volumes stay in HDF5, which is the format
that supports reading one time window without decompressing the whole run.

Validation runs in the same pass, since both need to open every file anyway.
Anything that fails here is excluded from the index rather than shipped broken.
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.storage import iter_subjects

EXPECTED_SPATIAL = (47, 29, 18)
CLIP_LIMIT = 5.0


def inspect(path, dataset, subject, deep=False):
    """One index row, or an ``error`` field describing why it is unusable."""
    row = {"dataset": dataset, "subject": subject,
           "path": str(Path(path).relative_to(Path(path).parents[1]))}
    try:
        with h5py.File(path, "r") as f:
            if "eye_block" not in f:
                row["error"] = "no eye_block"
                return row

            eb = f["eye_block"]
            row.update(
                n_trs=int(eb.shape[-1]),
                shape_x=int(eb.shape[0]), shape_y=int(eb.shape[1]), shape_z=int(eb.shape[2]),
                dtype=str(eb.dtype),
                has_labels="labels" in f,
                file_mb=round(Path(path).stat().st_size / 1e6, 2),
            )
            for key in ("repetition_time", "normalized", "format_version", "source_key"):
                if key in f.attrs:
                    value = f.attrs[key]
                    row[key] = value.item() if isinstance(value, np.generic) else value

            if eb.shape[:3] != EXPECTED_SPATIAL:
                row["error"] = f"spatial shape {eb.shape[:3]} != {EXPECTED_SPATIAL}"
                return row

            if "labels" in f and f["labels"].shape[0] != eb.shape[-1]:
                row["error"] = (f"labels {f['labels'].shape[0]} != TRs {eb.shape[-1]}")
                return row

            if deep:
                # Full read: catches interior corruption that opening misses.
                data = eb[:]
                finite = np.isfinite(data)
                if not finite.all():
                    row["error"] = "non-finite values in eye_block"
                    return row
                row["vmin"] = float(data.min())
                row["vmax"] = float(data.max())
                row["nonzero_frac"] = float((data != 0).mean())
                if row["nonzero_frac"] == 0.0:
                    row["error"] = "all-zero eye_block"
                    return row
                if f.attrs.get("normalized", False) and max(abs(row["vmin"]), row["vmax"]) > CLIP_LIMIT + 1e-3:
                    row["error"] = f"values exceed clip limit ({row['vmin']:.2f}, {row['vmax']:.2f})"
                    return row
                if "labels" in f:
                    labels = f["labels"][:]
                    row["label_nan_frac"] = float(np.isnan(labels).mean())
                    if np.isnan(labels).all():
                        row["error"] = "all-NaN labels"
                        return row
    except Exception as e:
        row["error"] = f"{e.__class__.__name__}: {e}"
    return row


def run_build(data_dir, out_path=None, deep=False, registry_path=None):
    data_dir = Path(data_dir).resolve()
    out_path = Path(out_path) if out_path else data_dir / "index.parquet"

    subjects = list(iter_subjects(data_dir))
    if not subjects:
        print(f"No participant files found under {data_dir}")
        return

    rows = [inspect(p, ds, sub, deep=deep)
            for ds, sub, p in tqdm(subjects, desc="Indexing")]

    # Fold in QA labels so the index alone answers "which data may I train on".
    registry_path = Path(registry_path) if registry_path else data_dir / "datasets.h5"
    if registry_path.exists():
        with h5py.File(registry_path, "r") as f:
            for row in rows:
                grp = f.get(f"{row['dataset']}/{row['subject']}")
                if grp is not None:
                    row["qa_approved"] = int(grp.attrs.get("approved", -1))

    bad = [r for r in rows if "error" in r]
    good = [r for r in rows if "error" not in r]

    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed; writing JSON instead of parquet.")
        out_path = out_path.with_suffix(".json")
        out_path.write_text(json.dumps(good, indent=2))
    else:
        df = pd.DataFrame(good)
        df.to_parquet(out_path, index=False)

    print(f"\n[+] Indexed {len(good)} participants -> {out_path}")
    if good:
        total_trs = sum(r.get("n_trs", 0) for r in good)
        n_labeled = sum(1 for r in good if r.get("has_labels"))
        n_ds = len({r["dataset"] for r in good})
        size = sum(r.get("file_mb", 0) for r in good)
        print(f"    {n_ds} datasets, {n_labeled} labeled, {total_trs:,} TRs, {size/1024:.1f} GB")

    if bad:
        print(f"\n[!] {len(bad)} participants failed validation and were excluded:")
        for r in bad[:20]:
            print(f"    {r['dataset']}/{r['subject']}: {r['error']}")
        if len(bad) > 20:
            print(f"    ... and {len(bad) - 20} more")
    return good, bad


def main():
    parser = argparse.ArgumentParser(description="Build and validate the participant index.")
    parser.add_argument("--data-dir", required=True, help="Root holding <dataset>/<subject>.h5.")
    parser.add_argument("--out", default=None, help="Output parquet path.")
    parser.add_argument("--registry", default=None, help="datasets.h5 to read QA labels from.")
    parser.add_argument("--deep", action="store_true",
                        help="Read every voxel: catches interior corruption, much slower.")
    args = parser.parse_args()
    run_build(args.data_dir, args.out, deep=args.deep, registry_path=args.registry)


if __name__ == "__main__":
    main()
