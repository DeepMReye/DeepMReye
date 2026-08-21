#!/usr/bin/env python3
"""High-speed Comprehensive Unsupervised Corpus Scaling Benchmark (N=25 to 1800).

Evaluates representation quality across all corpus checkpoints:
- fold-pca:64 (Supervised Reference Baseline)
- DeepMReye 2.0 lr-cca:32 (Core Instantaneous)
- DeepMReye 2.0 lr-cca:32 (+lags ±1)
- DeepMReye 2.0 lr-cca:32 (+lags ±2)
- DeepMReye 2.0 lr-cca:64
- DeepMReye 2.0 lr-cca:64 (+lags ±2)
- DeepMReye 2.0 lr-cca:128
- corpus-pca:64
- diff-pca:64, band-pca:64, gev-fast:64
- gev-slow:64 (Negative Control)

Uses fast 7-fold LODO evaluation on 285 gaze-labeled participants.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from deepmreye.datasource import resolve
from deepmreye.unsupervised import (
    corpus_mask,
    load_basis,
    project,
)

EXCLUDE = ("dsL11_backtothefuture",)
PATCH = 5
MAX_TRAIN_ROWS = 15000

CHECKPOINTS = [25, 50, 100, 200, 400, 800, 1200, 1500, 1800]


def bin_reduce(x, patch=PATCH):
    n = (len(x) // patch) * patch
    if n == 0:
        return x[:0]
    return np.nanmean(x[:n].reshape(n // patch, patch, -1), axis=1)


def make_lag_features(z, lags=0):
    if lags == 0:
        return z
    parts = []
    for lag in range(-lags, lags + 1):
        if lag < 0:
            p = np.pad(z[:lag], ((-lag, 0), (0, 0)), mode="edge")
        elif lag > 0:
            p = np.pad(z[lag:], ((0, lag), (0, 0)), mode="edge")
        else:
            p = z
        parts.append(p)
    return np.concatenate(parts, axis=1)


def get_labeled_cache(data_dir, mask, cache_path="results/scaling/eval_cache_raw.npz"):
    import os, h5py
    flat = mask.reshape(-1)
    if os.path.exists(cache_path):
        print(f"[*] Loading cached labeled evaluation data from {cache_path}...", flush=True)
        d = np.load(cache_path, allow_pickle=True)
        recs = []
        for i in range(int(d["n"])):
            recs.append({
                "dataset": str(d[f"ds/{i}"]),
                "subject": str(d[f"sub/{i}"]),
                "rows": d[f"rows/{i}"],
                "gaze": d[f"gaze/{i}"],
            })
        return recs

    print("[*] Building evaluation cache from labeled datasets...", flush=True)
    recs = []
    for ds_dir in sorted(data_dir.glob("dsL*")):
        if ds_dir.name in EXCLUDE or not ds_dir.is_dir():
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
            rows = block.reshape(-1, t).T[:, flat].astype(np.float32)
            recs.append({
                "dataset": ds_dir.name,
                "subject": path.stem,
                "rows": rows,
                "gaze": gaze[:t].astype(np.float32),
            })
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    save_dict = {"n": len(recs)}
    for i, r in enumerate(recs):
        save_dict[f"ds/{i}"] = r["dataset"]
        save_dict[f"sub/{i}"] = r["subject"]
        save_dict[f"rows/{i}"] = r["rows"]
        save_dict[f"gaze/{i}"] = r["gaze"]
    np.savez_compressed(cache_path, **save_dict)
    print(f"[*] Saved {len(recs)} labeled subjects to {cache_path}", flush=True)
    return recs


def fit_and_eval_lodo(subject_features):
    """Fast LODO evaluation over 7 folds given list of dicts with 'dataset', 'z', 'gaze'."""
    datasets = sorted({r["dataset"] for r in subject_features})
    per_fold = {}
    alphas = np.logspace(-2, 4, 13)

    for held in datasets:
        train = [r for r in subject_features if r["dataset"] != held]
        test = [r for r in subject_features if r["dataset"] == held]

        xs, ys = [], []
        for ds in sorted({r["dataset"] for r in train}):
            g = np.concatenate([r["gaze"] for r in train if r["dataset"] == ds])
            x = np.concatenate([r["z"] for r in train if r["dataset"] == ds])
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
            idx = np.random.default_rng(0).choice(len(x_tr), MAX_TRAIN_ROWS, replace=False)
            x_tr, y_tr = x_tr[idx], y_tr[idx]

        model = RidgeCV(alphas=alphas).fit(x_tr, y_tr)

        per_sub = []
        for r in test:
            x, g = r["z"], r["gaze"]
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
            per_fold[held] = {
                "r_x": float(med[0]),
                "r_y": float(med[1]),
                "mean": float(np.nanmean(med)),
            }

    med_r = float(np.median([v["mean"] for v in per_fold.values()]))
    return med_r, per_fold


def evaluate_fold_pca(labeled_recs, k=64):
    """Evaluate fold-local PCA (reference baseline)."""
    datasets = sorted({r["dataset"] for r in labeled_recs})
    per_fold = {}
    alphas = np.logspace(-2, 4, 13)

    for held in datasets:
        train = [r for r in labeled_recs if r["dataset"] != held]
        test = [r for r in labeled_recs if r["dataset"] == held]

        # Fit PCA on training fold voxels (using compact SVD / randomized PCA for speed)
        x_tr_raw = np.concatenate([r["rows"] for r in train])
        idx = np.random.default_rng(0).choice(len(x_tr_raw), min(5000, len(x_tr_raw)), replace=False)
        x_tr_sample = x_tr_raw[idx]

        mu = x_tr_sample.mean(axis=0, keepdims=True)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=k, random_state=0)
        pca.fit(x_tr_sample - mu)
        basis = pca.components_.T

        # Project and reduce
        xs, ys = [], []
        for ds in sorted({r["dataset"] for r in train}):
            ds_rows = np.concatenate([r["rows"] for r in train if r["dataset"] == ds])
            ds_gaze = np.concatenate([r["gaze"] for r in train if r["dataset"] == ds])
            proj = (ds_rows - mu) @ basis
            z = bin_reduce(proj)
            g = bin_reduce(ds_gaze[:len(ds_rows)])
            n = min(len(z), len(g))
            ok = np.isfinite(g[:n]).all(axis=1) & np.isfinite(z[:n]).all(axis=1)
            if ok.sum() < 10:
                continue
            g_ok, z_ok = g[:n][ok], z[:n][ok]
            sd = g_ok.std(axis=0)
            sd[sd < 1e-9] = 1.0
            ys.append((g_ok - g_ok.mean(axis=0)) / sd)
            xs.append(z_ok)

        x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
        if len(x_tr) > MAX_TRAIN_ROWS:
            idx = np.random.default_rng(0).choice(len(x_tr), MAX_TRAIN_ROWS, replace=False)
            x_tr, y_tr = x_tr[idx], y_tr[idx]

        model = RidgeCV(alphas=alphas).fit(x_tr, y_tr)

        per_sub = []
        for r in test:
            proj = (r["rows"] - mu) @ basis
            z = bin_reduce(proj)
            g = bin_reduce(r["gaze"][:len(r["rows"])])
            n = min(len(z), len(g))
            x_te, g_te = z[:n], g[:n]
            ok = np.isfinite(g_te).all(axis=1) & np.isfinite(x_te).all(axis=1)
            if ok.sum() < 10:
                continue
            pred = model.predict(x_te[ok])
            rs = []
            for ax in (0, 1):
                if np.std(pred[:, ax]) < 1e-12 or np.std(g_te[ok, ax]) < 1e-12:
                    rs.append(np.nan)
                else:
                    rs.append(np.corrcoef(pred[:, ax], g_te[ok, ax])[0, 1])
            per_sub.append(rs)

        if per_sub:
            med = np.nanmedian(np.array(per_sub, dtype=float), axis=0)
            per_fold[held] = {
                "r_x": float(med[0]),
                "r_y": float(med[1]),
                "mean": float(np.nanmean(med)),
            }

    med_r = float(np.median([v["mean"] for v in per_fold.values()]))
    return med_r, per_fold


def main():
    data_dir = resolve(None, download=False, quiet=True)
    mask = corpus_mask(data_dir)
    labeled_recs = get_labeled_cache(data_dir, mask)

    print("\n[*] Evaluating fold-pca:64 baseline...", flush=True)
    t_base = time.time()
    r_fold_pca, _ = evaluate_fold_pca(labeled_recs, k=64)
    print(f"[*] fold-pca:64 reference baseline = {r_fold_pca:.3f} ({time.time() - t_base:.1f}s)", flush=True)

    arms = [
        ("lr-cca", 16, 0, "DeepMReye 2.0 (lr-cca:16)"),
        ("lr-cca", 32, 0, "DeepMReye 2.0 (lr-cca:32)"),
        ("lr-cca", 64, 0, "DeepMReye 2.0 (lr-cca:64)"),
        ("lr-cca", 128, 0, "DeepMReye 2.0 (lr-cca:128)"),
        ("lr-cca", 32, 1, "DeepMReye 2.0 (lr-cca:32 + lags±1)"),
        ("lr-cca", 32, 2, "DeepMReye 2.0 (lr-cca:32 + lags±2)"),
        ("lr-cca", 64, 2, "DeepMReye 2.0 (lr-cca:64 + lags±2)"),
        ("corpus-pca", 64, 0, "corpus-pca:64"),
        ("diff-pca", 64, 0, "diff-pca:64"),
        ("band-pca", 64, 0, "band-pca:64"),
        ("gev-fast", 64, 0, "gev-fast:64"),
        ("gev-slow", 64, 0, "gev-slow:64 (control)"),
    ]

    available_checkpoints = []
    for n in CHECKPOINTS:
        if Path(f"results/scaling/basis_n{n}.npz").exists():
            available_checkpoints.append(n)

    results = {
        "checkpoints": available_checkpoints,
        "fold-pca:64": r_fold_pca,
        "curves": {label: {} for _, _, _, label in arms},
        "per_fold": {label: {} for _, _, _, label in arms},
    }

    print("\n" + "=" * 115, flush=True)
    print(f"UNSUPERVISED CORPUS SCALING: fold-pca:64 vs DeepMReye 2.0 (N = {available_checkpoints[0]} -> {available_checkpoints[-1]})", flush=True)
    print("=" * 115, flush=True)

    for n in available_checkpoints:
        basis_path = Path(f"results/scaling/basis_n{n}.npz")
        _mask, bases, meta = load_basis(basis_path)
        t0 = time.time()

        # Precompute projected features per kind for max speed
        cached_projs = {}
        for kind in ("lr-cca", "corpus-pca", "diff-pca", "band-pca", "gev-fast", "gev-slow"):
            if kind in bases:
                cached_projs[kind] = [project(kind, bases[kind], r["rows"], k=128) for r in labeled_recs]

        for kind, k, lags, label in arms:
            if kind not in cached_projs:
                continue
            projs = cached_projs[kind]
            sub_feats = []
            for i, r in enumerate(labeled_recs):
                proj = projs[i][:, :k]
                if lags > 0:
                    proj = make_lag_features(proj, lags=lags)
                z = bin_reduce(proj)
                g = bin_reduce(r["gaze"][:len(r["rows"])])
                n_rows = min(len(z), len(g))
                sub_feats.append({
                    "dataset": r["dataset"],
                    "subject": r["subject"],
                    "z": z[:n_rows],
                    "gaze": g[:n_rows],
                })

            r_val, per_fold = fit_and_eval_lodo(sub_feats)
            results["curves"][label][str(n)] = r_val
            results["per_fold"][label][str(n)] = per_fold

        t_el = time.time() - t0
        cca32 = results["curves"]["DeepMReye 2.0 (lr-cca:32)"].get(str(n), float('nan'))
        cca32_l1 = results["curves"]["DeepMReye 2.0 (lr-cca:32 + lags±1)"].get(str(n), float('nan'))
        cca32_l2 = results["curves"]["DeepMReye 2.0 (lr-cca:32 + lags±2)"].get(str(n), float('nan'))
        cca64_l2 = results["curves"]["DeepMReye 2.0 (lr-cca:64 + lags±2)"].get(str(n), float('nan'))
        pca64 = results["curves"]["corpus-pca:64"].get(str(n), float('nan'))
        slow64 = results["curves"]["gev-slow:64 (control)"].get(str(n), float('nan'))
        print(f"  N={n:<5} -> lr-cca:32={cca32:.3f} | +lags±1={cca32_l1:.3f} | +lags±2={cca32_l2:.3f} | 64+lags={cca64_l2:.3f} | pca:64={pca64:.3f} | slow={slow64:.3f} ({t_el:.1f}s)", flush=True)

    # Print Final Table
    print("\n" + "=" * 130, flush=True)
    print("FULL SCALING MATRIX: LODO PEARSON r ACROSS 7 HELD-OUT DATASET FOLDS (N=285 labeled subjects)", flush=True)
    print("=" * 130, flush=True)
    header = f"{'Representation':<38}" + "".join(f"{('N=' + str(n)):>10}" for n in available_checkpoints)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    print(f"{'fold-pca:64 (Supervised Reference)':<38}" + "".join(f"{r_fold_pca:>10.3f}" for _ in available_checkpoints), flush=True)
    print("-" * len(header), flush=True)

    for _, _, _, label in arms:
        row = f"{label:<38}"
        for n in available_checkpoints:
            val = results["curves"][label].get(str(n), None)
            row += f"{val:>10.3f}" if val is not None else f"{'--':>10}"
        print(row, flush=True)
    print("=" * 130, flush=True)

    out_path = Path("results/scaling/full_corpus_scaling_n1800.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[*] Saved benchmark results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
