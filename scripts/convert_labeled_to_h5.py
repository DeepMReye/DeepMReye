#!/usr/bin/env python3
"""Bring the labeled gaze datasets into the same layout as the unlabeled ones.

The labeled data arrived in two earlier shapes -- per-subject ``.npz`` files, and
a single ``<dataset>.h5`` nesting every subject under a dataset group. Both are
rewritten here to one file per participant::

    <out_dir>/dsL<nn>_<name>/<subject>.h5   eye_block [X, Y, Z, T] + labels [T, 10, 2]

so labeled and unlabeled participants are byte-format identical and the probe
and JEPA loaders share one code path. The ``dsL`` prefix (see
``DATASET_ALIASES``) is the only thing that distinguishes them by path, which
makes ``dsL*/*.h5`` the glob for the labeled subset.

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
from deepmreye.labels import append_label_events
from deepmreye.pipeline import LBL_EYES
from deepmreye.storage import is_intact, subject_path, write_subject

EXPECTED_SPATIAL = (47, 29, 18)

# The source directories are named after how the data arrived; the corpus names
# them for what it is. `dsL` keeps them in the same namespace as the OpenNeuro
# accessions while marking them as the labeled subset -- so `dsL*/*.h5` is the
# glob that fetches exactly the gaze-labeled participants, and nothing about a
# folder name has to be looked up to know which half of the corpus it is.
DATASET_ALIASES = {
    "dataset1_guided_fixations": "dsL01_guided_fixations",
    "dataset2_pursuit": "dsL02_pursuit",
    "dataset3_pursuit": "dsL03_pursuit",
    "dataset4_pursuit": "dsL04_pursuit",
    "dataset5_free_viewing": "dsL05_free_viewing",
    "dataset6_sequences": "dsL06_sequences",
}

# Acquisition TR in seconds. None of the labeled data carries a repetition time
# anywhere -- not in the .npz, not in the nested .h5, not in the subject names
# except dataset 6's -- so it is recorded here from the acquisition protocols.
# Without this the control set is the one part of the corpus with no TR, which
# is exactly the metadata the fixed-TR-window limitation needs to be resolved.
DATASET_TR = {
    "dataset1_guided_fixations": 0.800,
    "dataset2_pursuit": 0.870,
    "dataset3_pursuit": 1.020,
    "dataset4_pursuit": 1.000,
    "dataset5_free_viewing": 1.000,
    # dataset 6 varies per subject and is parsed from the name below.
}

# Subject names carry the acquisition TR, e.g. S4_0004_TR1250_2MM -> 1.25 s.
_TR_RE = re.compile(r"TR(\d+)")


def corpus_name(source_name):
    """Corpus dataset name for a source directory or nested-h5 stem."""
    return DATASET_ALIASES.get(source_name, source_name)


def tr_from_name(name):
    """Best-effort TR in seconds parsed from a subject name."""
    m = _TR_RE.search(name)
    if not m:
        return None
    value = float(m.group(1))
    # Values are milliseconds in practice; a bare "2" would mean seconds.
    return value / 1000.0 if value > 100 else value


def tr_for(source_name, sub_name):
    """TR in seconds: the subject name if it encodes one, else the protocol."""
    tr = tr_from_name(sub_name)
    return tr if tr is not None else DATASET_TR.get(source_name)


def _write(out_dir, ds_name, sub_name, eye_block, labels, source, tr=None):
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
    if tr is not None:
        attrs["repetition_time"] = float(tr)

    write_subject(subject_path(out_dir, ds_name, sub_name), eye_block, labels=labels, attrs=attrs)
    return True


def convert_h5(path, out_dir, force=False):
    """Split a nested ``<dataset>.h5`` into per-participant files."""
    source_name = path.stem
    ds_name = corpus_name(source_name)
    written = []
    with h5py.File(path, "r") as f:
        # Tolerate both /<dataset>/<subject>/... and a flat /<subject>/...
        groups = [f[source_name]] if source_name in f and isinstance(f[source_name], h5py.Group) else [f]
        for grp in groups:
            for sub_name in tqdm(list(grp.keys()), desc=f"  {ds_name}", leave=False):
                sub = grp[sub_name]
                if not isinstance(sub, h5py.Group) or "eye_block" not in sub:
                    continue

                out_path = subject_path(out_dir, ds_name, sub_name)
                if not force and is_intact(out_path):
                    written.append(sub_name)
                    continue

                if "labels" not in sub:
                    print(f"  [!] {ds_name}/{sub_name}: no labels -- skipping")
                    continue

                if _write(out_dir, ds_name, sub_name, sub["eye_block"][:], sub["labels"][:],
                          path, tr=tr_for(source_name, sub_name)):
                    written.append(sub_name)
    return ds_name, written


def convert_npz(ds_dir, out_dir, force=False):
    """Rebuild per-participant files from the original ``.npz`` exports."""
    source_name = ds_dir.name
    ds_name = corpus_name(source_name)
    written = []
    for npz_path in tqdm(sorted(ds_dir.glob("*.npz")), desc=f"  {ds_name}", leave=False):
        sub_name = npz_path.stem
        out_path = subject_path(out_dir, ds_name, sub_name)
        if not force and is_intact(out_path):
            written.append(sub_name)
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

        if _write(out_dir, ds_name, sub_name, eye_block, labels, npz_path,
                  tr=tr_for(source_name, sub_name)):
            written.append(sub_name)
    return ds_name, written


def register(out_dir, converted):
    """Record the labeled participants in the registry, approved.

    The labeled datasets were previously folders on disk and nothing else, so
    they carried no QA status: ``is_dataset_approved`` could not see them and
    they showed up in the index with a null label. They are entered here as
    ``LBL_EYES``, which is a statement about the data rather than a judgement --
    gaze was recorded simultaneously, so the eyeballs are in frame by
    construction. Mirrored into ``labels.csv`` like any other label so a rebuilt
    registry keeps them.
    """
    registry = Path(out_dir) / "datasets.h5"
    events = []
    with h5py.File(registry, "a") as f:
        for ds_name, subjects in converted.items():
            grp = f.require_group(ds_name)
            grp.attrs["labeled"] = True
            for sub_name in subjects:
                sub_grp = grp.require_group(sub_name)
                path = subject_path(out_dir, ds_name, sub_name)
                sub_grp.attrs["approved"] = LBL_EYES
                sub_grp.attrs["is_manual"] = True
                sub_grp.attrs["has_labels"] = True
                sub_grp.attrs["data_path"] = str(path)
                with h5py.File(path, "r") as sf:
                    sub_grp.attrs["n_trs"] = int(sf.attrs["n_trs"])
                    if "repetition_time" in sf.attrs:
                        sub_grp.attrs["repetition_time"] = float(sf.attrs["repetition_time"])
                events.append((ds_name, "subject", sub_name, LBL_EYES))

    if events:
        append_label_events(Path(out_dir) / "labels.csv", events)
    return len(events)


def run_convert(labeled_dir, out_dir, force=False, skip_registry=False):
    labeled_dir = Path(labeled_dir).resolve()
    out_dir = Path(out_dir).resolve()
    if not labeled_dir.exists():
        print(f"Error: {labeled_dir} does not exist.")
        return

    # dataset -> subjects present after this run, for the registry pass below.
    converted = {}

    def record(ds_name, subjects):
        converted.setdefault(ds_name, []).extend(subjects)

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
        record(*convert_h5(h5_path, out_dir, force=force))

    for ds_dir in sorted(p for p in labeled_dir.iterdir() if p.is_dir()):
        if not any(ds_dir.glob("*.npz")):
            continue
        print(f"[*] {ds_dir.name} (npz) -> {corpus_name(ds_dir.name)}")
        record(*convert_npz(ds_dir, out_dir, force=force))

    total = sum(len(v) for v in converted.values())
    print(f"\n[+] {total} participant files in {out_dir}")
    for ds_name, subjects in sorted(converted.items()):
        print(f"    {ds_name:<28} {len(subjects):>4}")

    if not skip_registry and total:
        registry = Path(out_dir) / "datasets.h5"
        if registry.exists():
            print(f"[+] Registered {register(out_dir, converted)} subjects in {registry.name}")
        else:
            print(f"[!] No registry at {registry} -- skipping registration.")


def main():
    parser = argparse.ArgumentParser(description="Convert labeled data to the unified layout.")
    parser.add_argument("--labeled-dir", required=True, help="Directory of labeled .h5 / .npz data.")
    parser.add_argument("--out-dir", required=True, help="Output root for <dataset>/<subject>.h5.")
    parser.add_argument("--force", action="store_true", help="Rewrite files that already exist.")
    parser.add_argument("--skip-registry", action="store_true",
                        help="Do not enter the participants into datasets.h5.")
    args = parser.parse_args()
    run_convert(args.labeled_dir, args.out_dir, force=args.force, skip_registry=args.skip_registry)


if __name__ == "__main__":
    main()
