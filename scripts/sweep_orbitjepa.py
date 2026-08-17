#!/usr/bin/env python3
"""Fast leave-one-dataset-out screen for Orbit-JEPA checkpoints and snapshots.

Why a second harness exists at all, and what keeps it honest
------------------------------------------------------------
The real number comes from `scripts/eval_probe.py` and nothing here replaces
it. But the question that has to be answered for this objective is *how the
probe behaves along the training trajectory*, not how it behaves at the
objective's optimum -- because the `ocon` result on this corpus is that
cross-orbit agreement improves monotonically with training while gaze decoding
peaks early and then falls. Answering that needs ~8 snapshots x several configs,
and each full harness run is minutes.

So the labeled participants are reduced **once** to the same frozen canonical
pre-projection the model trains on, and every checkpoint is then screened in
milliseconds. Because the pre-projection is linear, averaging it over a temporal
bin is identical to averaging the voxels first and projecting after, which is
what `eval_probe` does -- so the screen differs from the harness only in using
non-overlapping bins instead of sliding windows.

**It is calibrated, not assumed.** `--calibrate` screens an untrained model,
which by construction is `lr-cca:k` exactly, and prints the result next to the
harness's own 0.825 for `lr-cca:32`. If those two disagree, the screen is wrong
and its ordering should not be trusted. Every conclusion drawn here is then
re-measured in the real harness before it is reported.

Usage
-----
    python scripts/sweep_orbitjepa.py --build-cache --calibrate
    python scripts/sweep_orbitjepa.py --checkpoints results/jepa/*.pt
"""
import argparse
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from deepmreye.datasource import resolve
from deepmreye.models.jepa_net import OrbitJEPA
from deepmreye.orbitjepa import encode_numpy, load_checkpoint, orbit_projections
from deepmreye.unsupervised import load_basis

EXCLUDE = ("dsL11_backtothefuture",)   # fails verify_gaze_sync; see STATE.md
PATCH = 5                              # --temp-patch-size default
MAX_TRAIN_ROWS = 20000                 # ~= --max-train-windows 1000 at patch 5


def bin_reduce(x, patch=PATCH):
    """Average consecutive ``patch`` rows, dropping the ragged tail."""
    n = (len(x) // patch) * patch
    if n == 0:
        return x[:0]
    return np.nanmean(x[:n].reshape(n // patch, patch, -1), axis=1)


def build_labeled_cache(root, mask, basis, m=256, regress_motion=False, verbose=True):
    """Reduce every gaze-labeled participant to binned canonical coords + gaze."""
    import h5py

    recs = []
    for ds_dir in sorted(p for p in root.glob("dsL*") if p.is_dir()):
        if ds_dir.name in EXCLUDE:
            continue
        for path in sorted(ds_dir.glob("*.h5")):
            try:
                with h5py.File(path, "r") as f:
                    if "labels" not in f:
                        continue
                    block = f["eye_block"][:]
                    gaze = np.nanmean(f["labels"][:], axis=1).astype(np.float64)
            except Exception:
                continue
            if block.shape[-1] < 60 or not np.isfinite(gaze).any():
                continue

            t = block.shape[-1]
            rows = block.reshape(-1, t).T[:, mask.reshape(-1)].astype(np.float64)
            zl, zr = orbit_projections(rows, basis, m=m, regress_motion=regress_motion)
            z = np.stack([bin_reduce(zl), bin_reduce(zr)], axis=1)   # [B, 2, m]
            g = bin_reduce(gaze[: len(rows)])                        # [B, 2]
            n = min(len(z), len(g))
            recs.append({"dataset": ds_dir.name, "subject": path.stem,
                         "z": z[:n].astype(np.float32), "gaze": g[:n]})
        if verbose:
            got = sum(1 for r in recs if r["dataset"] == ds_dir.name)
            print(f"    {ds_dir.name:<26}{got:>4} participants", flush=True)
    if not recs:
        raise SystemExit("[!] no labeled participants found")
    return recs


def save_labeled_cache(path, recs, basis_path, m, regress_motion):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    flat = {f"z/{i}": r["z"] for i, r in enumerate(recs)}
    flat.update({f"g/{i}": r["gaze"] for i, r in enumerate(recs)})
    np.savez(path, n=np.array([len(recs)]), m=np.array([m]),
             regress_motion=np.array([int(regress_motion)]),
             basis=np.array(str(basis_path)),
             meta=np.array(json.dumps([{"dataset": r["dataset"],
                                        "subject": r["subject"]} for r in recs])),
             **flat)


def load_labeled_cache(path, basis_path, m, regress_motion):
    d = np.load(path, allow_pickle=False)
    got = (int(d["m"][0]), bool(d["regress_motion"][0]), str(d["basis"]))
    want = (int(m), bool(regress_motion), str(basis_path))
    if got != want:
        raise SystemExit(f"[!] labeled cache built for {got}, requested {want}; "
                         f"rebuild with --build-cache")
    meta = json.loads(str(d["meta"]))
    return [{**meta[i], "z": d[f"z/{i}"], "gaze": d[f"g/{i}"]}
            for i in range(int(d["n"][0]))]


def features_for(rec, weights, head, k=None):
    s_l = encode_numpy(weights["left"], rec["z"][:, 0].astype(np.float64))
    s_r = encode_numpy(weights["right"], rec["z"][:, 1].astype(np.float64))
    out = 0.5 * (s_l + s_r) if head == "avg" else np.concatenate([s_l, s_r], axis=1)
    return out[:, :k] if k else out


def lodo_screen(recs, weights, head, k=None, seed=0):
    """Leave-one-dataset-out ridge, per-subject r, median over folds."""
    from sklearn.linear_model import RidgeCV

    feats = {id(r): features_for(r, weights, head, k) for r in recs}
    datasets = sorted({r["dataset"] for r in recs})
    per_fold = {}

    for held in datasets:
        train = [r for r in recs if r["dataset"] != held]
        test = [r for r in recs if r["dataset"] == held]

        # Targets standardised per training dataset, exactly as
        # `--standardize-targets dataset` does: the per-fold Euclidean scale
        # spans 21 to 595 across these datasets, so a pooled fit without it
        # follows whichever dataset has the largest target variance.
        # `sorted`, not a bare set: set iteration order over strings varies with
        # PYTHONHASHSEED between processes, which changes the concatenation order
        # and therefore which rows the MAX_TRAIN_ROWS subsample keeps. That alone
        # moved this screen's untrained k=32 value between 0.811 and 0.822 across
        # two runs of identical features -- ~0.01 of avoidable noise in a
        # comparison whose whole job is to resolve differences of that size.
        xs, ys = [], []
        for ds in sorted({r["dataset"] for r in train}):
            g = np.concatenate([r["gaze"] for r in train if r["dataset"] == ds])
            x = np.concatenate([feats[id(r)] for r in train if r["dataset"] == ds])
            ok = np.isfinite(g).all(axis=1) & np.isfinite(x).all(axis=1)
            if ok.sum() < 10:
                continue
            g, x = g[ok], x[ok]
            sd = g.std(axis=0)
            sd[sd < 1e-9] = 1.0
            ys.append((g - g.mean(axis=0)) / sd)
            xs.append(x)
        if not xs:
            continue
        x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
        if len(x_tr) > MAX_TRAIN_ROWS:
            idx = np.random.default_rng(seed).choice(len(x_tr), MAX_TRAIN_ROWS, replace=False)
            x_tr, y_tr = x_tr[idx], y_tr[idx]

        model = RidgeCV(alphas=np.logspace(-3, 5, 17)).fit(x_tr, y_tr)

        per_sub = []
        for r in test:
            x, g = feats[id(r)], r["gaze"]
            ok = np.isfinite(g).all(axis=1) & np.isfinite(x).all(axis=1)
            if ok.sum() < 10:
                continue
            pred = model.predict(x[ok])
            rs = []
            for ax in (0, 1):
                if np.std(pred[:, ax]) < 1e-12 or np.std(g[ok][:, ax]) < 1e-12:
                    rs.append(np.nan)
                else:
                    rs.append(np.corrcoef(pred[:, ax], g[ok][:, ax])[0, 1])
            per_sub.append(rs)
        if per_sub:
            med = np.nanmedian(np.array(per_sub, dtype=float), axis=0)
            per_fold[held] = {"r_x": float(med[0]), "r_y": float(med[1]),
                              "mean": float(np.nanmean(med)),
                              "n_subjects": len(per_sub)}

    median = float(np.median([v["mean"] for v in per_fold.values()])) if per_fold else float("nan")
    return median, per_fold


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--basis", default="results/scaling/basis_n1039.npz")
    p.add_argument("--cache", default="results/jepa/labeled_cache.npz")
    p.add_argument("--build-cache", action="store_true")
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--regress-motion", action="store_true")
    p.add_argument("--checkpoints", nargs="*", default=())
    p.add_argument("--calibrate", action="store_true",
                   help="Screen an untrained model (== lr-cca:k) at several k and "
                        "compare against the harness values, before trusting any "
                        "ordering this script produces.")
    p.add_argument("--out", default="results/jepa/screen.json")
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    mask, bases, _ = load_basis(args.basis)
    basis = bases["lr-cca"]

    cache = Path(args.cache)
    if args.build_cache or not cache.exists():
        root = Path(resolve(args.data_dir, download=False, quiet=True))
        print(f"[*] building labeled cache from {root}")
        recs = build_labeled_cache(root, mask, basis, m=args.m,
                                   regress_motion=args.regress_motion)
        save_labeled_cache(cache, recs, args.basis, args.m, args.regress_motion)
        print(f"[*] cached {len(recs)} participants -> {cache}")
    else:
        recs = load_labeled_cache(cache, args.basis, args.m, args.regress_motion)
        print(f"[*] labeled cache: {len(recs)} participants, "
              f"{len(set(r['dataset'] for r in recs))} datasets")

    rows, out = [], {}

    if args.calibrate:
        # Harness values from STATE.md, same protocol (7 verified folds,
        # ridge-cv, --standardize-targets dataset, 1000 training windows).
        harness = {32: 0.825, 64: 0.809}
        print("\n[*] calibration -- untrained model == lr-cca:k exactly")
        print(f"    {'k':>4}{'screen':>10}{'harness':>10}{'delta':>9}")
        for k in (32, 64):
            model = OrbitJEPA(in_dim=args.m, latent_dim=k, hidden_dim=64, depth=2)
            med, _ = lodo_screen(recs, model.to_numpy_weights(), "avg")
            print(f"    {k:>4}{med:>10.3f}{harness[k]:>10.3f}{med - harness[k]:>+9.3f}")
            out[f"calibration_lr_cca_{k}"] = {"screen": med, "harness": harness[k]}

    for path in sorted({p for spec in args.checkpoints for p in glob.glob(spec)}):
        ckpt = load_checkpoint(path)
        med, per_fold = lodo_screen(recs, ckpt["weights"], ckpt["head"])
        ctrl = load_checkpoint(path, untrained=True)
        med_c, _ = lodo_screen(recs, ctrl["weights"], ctrl["head"])
        meta = ckpt.get("meta", {})
        rows.append({"name": Path(path).name, "median_r": med,
                     "control_r": med_c, "margin": med - med_c,
                     "epoch": meta.get("epoch", meta.get("best_epoch")),
                     "val_loss": meta.get("best_val_loss"),
                     "nonlinear_share": meta.get("nonlinear_share"),
                     "k": ckpt["arch"]["latent_dim"], "per_fold": per_fold})
        print(f"  {Path(path).name:<34} r={med:.3f}  control={med_c:.3f}  "
              f"margin={med - med_c:+.3f}", flush=True)

    if rows:
        print("\n" + "=" * 104)
        print(f"{'checkpoint':<34}{'ep':>4}{'k':>4}{'val loss':>10}{'nonlin':>8}"
              f"{'screen r':>10}{'control':>9}{'margin':>9}")
        print("-" * 104)
        for r in sorted(rows, key=lambda r: -r["median_r"]):
            vl = f"{r['val_loss']:.4f}" if r["val_loss"] is not None else "--"
            nl = f"{r['nonlinear_share']:.3f}" if r["nonlinear_share"] is not None else "--"
            print(f"{r['name']:<34}{str(r['epoch'] or '--'):>4}{r['k']:>4}{vl:>10}{nl:>8}"
                  f"{r['median_r']:>10.3f}{r['control_r']:>9.3f}{r['margin']:>+9.3f}")
        print("=" * 104)
        print("Screen only -- confirm anything promising with scripts/eval_orbitjepa.py.")
        out["checkpoints"] = rows

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[*] -> {args.out}")


if __name__ == "__main__":
    main()
