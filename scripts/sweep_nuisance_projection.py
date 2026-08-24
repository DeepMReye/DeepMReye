"""Project the nuisance out BEFORE the cross-orbit readout -- the last untested escape.

`CLAUDE.md` names this twice as the one thing not tried. `ocon` established that what the two
orbits *share* is dominated by within-run global signal, motion and drift -- all common to
both orbits, all varying within a run -- and `lr-cca` is the linear form of that same
cross-orbit constraint, so it inherits the same contamination by construction. The next-TR
entry reaches the same place from the temporal side: the predictable part of an eye block is
the nuisance.

Two strengths of the same idea, both linear, both leaving the frozen basis alone:

- **motion**: regress the orbit mean signal and its temporal derivative out of the voxel rows
  per participant (`gauge.motion_proxy`, already implemented as `orbit_projections(...,
  regress_motion=True)` and never actually measured on this metric). A stated-weak proxy for
  the 6-DOF realignment parameters the corpus does not store.
- **cpca J**: project out the leading `J` `corpus-pca` directions -- the corpus's own estimate
  of the highest-variance, slowest structure -- before the `lr-cca` projection.

`nuis-pca8`/`nuis-pca32` are NOT this experiment: they rebuild a *variance* basis after
dropping slow directions. Nothing has removed the nuisance ahead of the *cross-orbit* fit,
which is where the `ocon` result says it does its damage.

Reports sub-TR and 1-TR only, each at its own optimal lag.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deepmreye.orbitjepa import orbit_projections
from deepmreye.temporal_probe import (MIN_TRS, calibrate, cca_avg, corpus_fingerprint,
                                      load_subtr_cache, lodo_subtr, make_lags)


def build_cache(root, mask, basis, m, regress_motion=False, drop_pca=None):
    """Labeled participants as canonical coords, optionally with a nuisance removed first."""
    import h5py

    flat_mask = mask.reshape(-1)
    recs = []
    for ds_dir in sorted(q for q in Path(root).glob("dsL*") if q.is_dir()):
        for path in sorted(ds_dir.glob("*.h5")):
            try:
                with h5py.File(path, "r") as f:
                    if "labels" not in f:
                        continue
                    block, labels = f["eye_block"][:], f["labels"][:]
            except Exception:
                continue
            t = block.shape[-1]
            if t < MIN_TRS or not np.isfinite(labels).any():
                continue
            rows = block.reshape(-1, t).T[:, flat_mask].astype(np.float64)
            if drop_pca is not None:
                v, mu = drop_pca                      # [V, J] orthonormal, [V]
                c = rows - mu
                rows = mu + c - (c @ v) @ v.T
            zl, zr = orbit_projections(rows, basis, m=m, regress_motion=regress_motion)
            n = min(len(zl), len(labels))
            recs.append({"dataset": ds_dir.name, "subject": path.stem,
                         "z": np.stack([zl[:n], zr[:n]], axis=1).astype(np.float32),
                         "labels": labels[:n]})
    return recs


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--basis", default="results/scaling/basis_n2000.npz")
    p.add_argument("--cache", default="results/subtr/labeled_subtr_cache.npz")
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--drop", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    p.add_argument("--out", default="results/scaling/nuisance_projection.json")
    args = p.parse_args()

    from deepmreye.datasource import resolve
    from deepmreye.unsupervised import corpus_mask, load_basis

    data_dir = Path(args.data_dir) if args.data_dir else resolve(None, download=False, quiet=True)
    mask = corpus_mask(data_dir)
    _m, bases, _meta = load_basis(Path(args.basis))
    cca, cpca = bases["lr-cca"], bases["corpus-pca"]
    k = args.k

    base_recs = load_subtr_cache(args.cache, Path(args.basis), args.m, False)
    print(f"[*] {len(base_recs)} participants, "
          f"fingerprint {corpus_fingerprint(base_recs)[:12]}", flush=True)
    if not calibrate(base_recs):
        raise SystemExit("[!] calibration failed")
    print("[+] calibrated\n", flush=True)

    def score(recs, label, out):
        sub = lodo_subtr(recs, lambda r: make_lags(cca_avg(r, k), 1))["median_subtr"]
        one = lodo_subtr(recs, lambda r: make_lags(cca_avg(r, k), 0))["median_1tr"]
        out.append({"arm": label, "subtr": sub, "1tr": one})
        b = out[0]
        print(f"{label:<22} sub-TR {sub:.4f} {sub - b['subtr']:+.4f}    "
              f"1-TR {one:.4f} {one - b['1tr']:+.4f}", flush=True)

    rows = []
    score(base_recs, "baseline", rows)
    del base_recs

    score(build_cache(data_dir, mask, cca, args.m, regress_motion=True), "motion-regressed", rows)

    comp, mu = np.asarray(cpca["components"]), np.asarray(cpca["mean"])
    for j in args.drop:
        recs = build_cache(data_dir, mask, cca, args.m,
                           drop_pca=(comp[:, :j], mu))
        score(recs, f"drop corpus-pca {j}", rows)
        del recs

    Path(args.out).write_text(json.dumps({"k": k, "rows": rows}, indent=2))
    print(f"\n[+] {args.out}")


if __name__ == "__main__":
    main()
