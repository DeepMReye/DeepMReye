#!/usr/bin/env python3
"""Retire a gaze-labeled dataset: archive its labels, then fold it back into
its OpenNeuro accession as ordinary unlabeled corpus data.

A labeled dataset nobody can trust is worse than one that is absent, and this
project has retired two on that principle (`ds007532`, `ds001242`). What is left
after the gaze is discarded is not a special category -- **it is exactly what
every other unlabeled participant in the corpus is**, and it should be named the
same way: `ds######`, the real accession, which is the provenance. The earlier
`dsX##_<name>_unaligned` convention invented a third kind of folder that nothing
in the pipeline understands and that reads as a status rather than a source.

Two things this gets right that doing it by hand did not:

- **A name does not retire a dataset.** ``ProbeDataset._discover()`` walks every
  directory under the corpus root and accepts anything carrying a ``labels``
  dataset, so renaming ``dsL10_visseq`` out of the ``dsL*`` prefix left it
  running as its own fold. Removing the labels is what retires it.
- **The accession folder usually already exists.** The QA sample extracted a
  couple of participants of the same accession long before the eye-tracking
  ingest extracted all of them under a `dsL##` name, so the corpus has been
  carrying those participants *twice* -- counted twice in every unlabeled basis
  fit. Folding back deduplicates them. The two copies are not bit-identical
  (ANTs is not reproducible run to run; measured r 0.83-0.99 on the same input),
  so the QA-sampled file is kept -- a human looked at its thumbnail -- and the
  eye-tracking provenance attrs are merged onto it.

The labels are archived to ``results/<dataset>_labels.npz`` before anything is
touched, so the decision is reversible if the timing is ever recovered.

    python scripts/retire_labeled_dataset.py --dataset dsL09_fearlearning \\
        --into ds001242 --reason "per-subject trigger jitter"
    # add --apply once the dry run reads right
"""
import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.datasource import resolve  # noqa: E402

# Provenance worth keeping on the surviving file: it says where the gaze came
# from and why it was discarded, which is the whole audit trail.
CARRY = ("eyetracking_key", "gaze_anchor", "gaze_anchor_detail", "gaze_coverage",
         "gaze_nan_fraction", "gaze_time_offset", "label_units")


def infer_accession(src):
    """The OpenNeuro accession these participants came from, from their attrs."""
    for p in sorted(src.glob("*.h5")):
        with h5py.File(p, "r") as f:
            for key in ("source_key", "eyetracking_key"):
                v = f.attrs.get(key)
                if v:
                    m = re.match(r"(ds\d{6})/", str(v))
                    if m:
                        return m.group(1)
    return None


def retire(data_dir, dataset, into, reason, archive_dir, apply):
    data_dir = Path(data_dir)
    src = data_dir / dataset
    if not src.is_dir():
        raise SystemExit(f"[!] {src} does not exist")
    paths = sorted(src.glob("*.h5"))
    if not paths:
        raise SystemExit(f"[!] {src} holds no participant files")

    into = into or infer_accession(src)
    if not into:
        raise SystemExit("[!] could not infer the accession; pass --into ds######")
    if not re.fullmatch(r"ds\d{6}", into):
        raise SystemExit(f"[!] --into must be an OpenNeuro accession, got {into!r}")
    dst = data_dir / into

    labels = {}
    for p in paths:
        with h5py.File(p, "r") as f:
            if "labels" in f:
                labels[p.stem] = f["labels"][...]
    existing = {p.stem for p in dst.glob("*.h5")} if dst.is_dir() else set()
    dupes = sorted({p.stem for p in paths} & existing)
    fresh = sorted({p.stem for p in paths} - existing)

    print(f"[*] {dataset}: {len(paths)} participants, {len(labels)} carrying labels")
    print(f"[*] fold into {into}"
          + (f" (already holds {len(existing)})" if existing else " (new folder)"))
    print(f"    {len(fresh)} moved in, {len(dupes)} already there"
          + (f": {', '.join(dupes)}" if dupes else ""))
    print(f"[*] reason: {reason}")

    archive = Path(archive_dir) / f"{dataset}_labels.npz"
    if not apply:
        print(f"\n[dry run] would archive {len(labels)} label arrays to {archive}")
        print(f"[dry run] would strip `labels`, move {len(fresh)} files into "
              f"{into}/, and drop {len(dupes)} duplicates after merging their "
              "eye-tracking attrs onto the copy already there")
        print("[dry run] eye blocks are kept; re-run with --apply")
        return 0

    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists() and len(labels) < len(np.load(archive).files):
        # Re-running on an already-stripped folder would replace a good archive
        # with an empty one, which is the one irreversible thing here.
        print(f"[*] {archive} already holds "
              f"{len(np.load(archive).files)} arrays; not overwriting with "
              f"{len(labels)}")
    else:
        np.savez_compressed(archive, **labels)
    meta = archive.with_suffix(".json")
    prev = json.loads(meta.read_text()) if meta.exists() else {}
    meta.write_text(json.dumps({**prev, **
        {"retired_from": dataset, "folded_into": into, "reason": reason,
         "n_participants": len(paths), "n_labels_archived": len(labels),
         "duplicates_dropped": dupes}}, indent=1))
    print(f"[+] archive at {archive}")

    dst.mkdir(parents=True, exist_ok=True)
    moved = dropped = 0
    for p in paths:
        with h5py.File(p, "r+") as f:
            if "labels" in f:
                del f["labels"]
            f.attrs["has_labels"] = False
            f.attrs["dataset"] = into
            f.attrs["retired_from"] = dataset
            f.attrs["retired_reason"] = reason
            carried = {k: f.attrs[k] for k in CARRY if k in f.attrs}
        if p.stem in existing:
            # Keep the QA-sampled copy -- a human looked at its thumbnail, and
            # its `approved` label in the registry is keyed to it -- but move
            # the gaze provenance across before dropping this one.
            with h5py.File(dst / p.name, "r+") as g:
                for k, v in carried.items():
                    g.attrs[k] = v
                g.attrs["retired_from"] = dataset
                g.attrs["retired_reason"] = reason
            p.unlink()
            dropped += 1
        else:
            p.rename(dst / p.name)
            png = p.with_suffix(".png")
            if png.exists():
                png.rename(dst / png.name)
            moved += 1
    print(f"[+] moved {moved} participants into {into}/, "
          f"dropped {dropped} duplicates (provenance merged)")

    leftovers = sorted(x.name for x in src.iterdir()) if src.is_dir() else []
    if not leftovers:
        src.rmdir()
        print(f"[+] removed empty {src.name}/")
    else:
        print(f"[*] {src.name}/ still holds {len(leftovers)} other files, left alone")

    reg = data_dir / "datasets.h5"
    if reg.exists():
        try:
            with h5py.File(reg, "r+") as f:
                if dataset in f:
                    del f[dataset]
                    print(f"[+] registry: dropped the {dataset} group "
                          f"({into} keeps its own QA labels)")
        except OSError as e:
            print(f"[!] registry not updated ({e}); drop the group by hand")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--dataset", required=True)
    p.add_argument("--into", default=None,
                   help="Accession to fold into (ds######). Inferred from the "
                        "participants' own source keys when omitted.")
    p.add_argument("--reason", required=True)
    p.add_argument("--archive-dir", default="results")
    p.add_argument("--apply", action="store_true", help="Without this it is a dry run.")
    a = p.parse_args()
    data_dir = Path(a.data_dir or resolve(None, download=False, quiet=True))
    print(f"[*] corpus {data_dir}")
    return retire(data_dir, a.dataset, a.into, a.reason, a.archive_dir, a.apply)


if __name__ == "__main__":
    sys.exit(main())
