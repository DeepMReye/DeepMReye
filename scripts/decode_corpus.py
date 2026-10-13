"""Decode gaze for every participant in the corpus with the frozen basis.

The readout is :func:`deepmreye.probe.fit_readout` on all nine gaze-labelled
datasets (the shipped arm: ``lr-cca:32`` on ``results/basis.npz``, +-1 TR, ridge), applied
to every participant's whole run -- the 2,060 that fitted the basis, the
unlabelled ones that did not, and the labelled ones (in-sample there: flagged).
``in_basis`` is read from the basis file's own list of participants.
Each orbit is also decoded alone, and the two decodings' correlation is stored
as the participant's confidence (:func:`deepmreye.metrics.orbit_agreement`).

Output, one HDF5 file:

    gaze        [sum T, 10, 2] float32  decoded gaze, training-set z-units
    offsets     [P + 1]                 participant p is gaze[offsets[p]:offsets[p+1]]
    meta        JSON, one row per participant: dataset, subject, volumes, TR,
                task (from the BOLD file name), labelled, in_basis (one of the
                corpus), QA status of the dataset, eye-mask coverage,
                orbit agreement, decoded SD

and the same table alone as ``corpus_meta.json`` next to it.

    python scripts/decode_corpus.py --out results/decoded/corpus_gaze.h5
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deepmreye import probe  # noqa: E402
from deepmreye.datasource import resolve  # noqa: E402
from deepmreye.metrics import orbit_agreement  # noqa: E402
from deepmreye.unsupervised import (  # noqa: E402
    FULL_MASK_VOXELS, corpus_mask, load_basis, orbit_projections,
)

K, LAGS = probe.DEFAULT_K, probe.DEFAULT_LAGS


def task_of(source_key):
    m = re.search(r"task-([A-Za-z0-9]+)", str(source_key or ""))
    return m.group(1).lower() if m else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--basis", default="results/basis.npz")
    ap.add_argument("--out", default="results/decoded/corpus_gaze.h5")
    a = ap.parse_args()

    import pandas as pd

    from deepmreye.labels import qa_status

    root = Path(a.data_dir) if a.data_dir else resolve(None, download=False, quiet=True)
    mask = corpus_mask(root)
    flat = mask.reshape(-1)
    _m, bases, basis_meta = load_basis(a.basis)
    arrays = bases["lr-cca"]

    def coords(path):
        with h5py.File(path, "r") as f:
            block = f["eye_block"][:]
            labels = f["labels"][:] if "labels" in f else None
        t = block.shape[-1]
        covered = int((np.abs(block[..., :min(4, t)]).sum(-1) > 0).sum()) >= FULL_MASK_VOXELS
        zl, zr = orbit_projections(block.reshape(-1, t).T[:, flat], arrays, k=K)
        return zl, zr, labels, covered

    def feats(zl, zr):
        return (probe.make_lags(0.5 * (zl + zr), LAGS), probe.make_lags(zl, LAGS),
                probe.make_lags(zr, LAGS))

    # The readout: every labelled participant, the shipped features.
    t0 = time.time()
    labelled = []
    for ds_dir in sorted(p for p in Path(root).glob("dsL*") if p.is_dir()):
        for path in sorted(ds_dir.glob("*.h5")):
            zl, zr, labels, _covered = coords(path)
            n = min(len(zl), len(labels))
            if n < probe.MIN_TRS or not np.isfinite(labels).any():
                continue
            labelled.append({"dataset": ds_dir.name, "subject": path.stem,
                             "x": feats(zl[:n], zr[:n])[0], "labels": labels[:n]})
    predict = probe.fit_readout(labelled, lambda r: r["x"])
    print(f"[*] readout on {len(labelled)} labelled participants  [{time.time() - t0:.0f}s]",
          flush=True)

    index = pd.read_parquet(Path(root) / "index.parquet").set_index(["dataset", "subject"])
    in_basis = {tuple(s.split("/", 1)) for s in basis_meta.get("subjects", [])}
    status = qa_status(Path(root) / "datasets.h5")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(root).glob("*/*.h5"))
    paths = [p for p in paths if p.parent.name != "_pending"]
    meta, chunks, offsets = [], [], [0]
    for i, path in enumerate(paths):
        ds, sub = path.parent.name, path.stem
        try:
            zl, zr, _, covered = coords(path)
        except Exception as e:                       # truncated or partial files
            print(f"  skip {ds}/{sub}: {e}", flush=True)
            continue
        if len(zl) < 10 or not np.isfinite(zl).all():
            continue
        x, xl, xr = feats(zl, zr)
        pred, left, right = predict(x), predict(xl), predict(xr)
        row = index.loc[(ds, sub)] if (ds, sub) in index.index else None
        meta.append({
            "dataset": ds, "subject": sub, "volumes": int(len(pred)),
            "tr": float(row["repetition_time"]) if row is not None else None,
            "task": task_of(row["source_key"]) if row is not None else "",
            "labelled": ds.startswith("dsL"), "in_basis": (ds, sub) in in_basis,
            # Eye mask fully covered: the corpus eligibility test. An uncovered
            # participant decodes to a near-constant trace, whose two orbits
            # "agree" trivially -- its agreement means nothing.
            "covered": covered,
            "qa": "labelled" if ds.startswith("dsL") else status.get(ds, "not in registry"),
            "agreement": orbit_agreement(left, right),
            "sd_x": float(pred.reshape(-1, 2)[:, 0].std()),
            "sd_y": float(pred.reshape(-1, 2)[:, 1].std()),
        })
        chunks.append(pred.reshape(len(pred), 10, 2).astype(np.float32))
        offsets.append(offsets[-1] + len(pred))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(paths)}  [{time.time() - t0:.0f}s]", flush=True)

    with h5py.File(a.out, "w") as f:
        f.create_dataset("gaze", data=np.concatenate(chunks), compression="gzip",
                         compression_opts=4)
        f.create_dataset("offsets", data=np.asarray(offsets, dtype=np.int64))
        f.attrs["meta"] = json.dumps(meta)
        f.attrs["readout"] = (f"lr-cca:{K}+lags{LAGS} from {a.basis}, ridge on "
                              f"{len(labelled)} labelled participants, gaze z-scored per dataset")
    # The per-participant table on its own, small enough to snapshot with a figure.
    Path(a.out).with_name("corpus_meta.json").write_text(json.dumps(meta))
    print(f"[+] {len(meta)} participants, {offsets[-1]} volumes -> {a.out}  "
          f"[{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
