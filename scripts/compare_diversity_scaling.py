#!/usr/bin/env python3
"""Ablation benchmark: 1 participant/dataset vs 2 participants/dataset scaling curves.

Compares:
1. Max Diversity (1 subject / dataset): N subjects from N distinct datasets.
2. High Depth (2 subjects / dataset): N subjects from N/2 distinct datasets.
Evaluates lr-cca:32, lr-cca:64, and corpus-pca:64 on 7-fold LODO Pearson r.
"""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import h5py
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from deepmreye.datasource import resolve
from deepmreye.unsupervised import (
    corpus_mask,
    unlabeled_subjects,
    Moments,
    _slabs,
    fit_lr_cca,
    fit_pca,
    project,
)

EXCLUDE = ("dsL11_backtothefuture",)
PATCH = 5
MAX_TRAIN_ROWS = 20000


def bin_reduce(x, patch=PATCH):
    n = (len(x) // patch) * patch
    if n == 0:
        return x[:0]
    return np.nanmean(x[:n].reshape(n // patch, patch, -1), axis=1)


def get_labeled_cache(data_dir, mask, cache_path="results/scaling/eval_cache_raw.npz"):
    flat = mask.reshape(-1)
    if os.path.exists(cache_path):
        print(f"[*] Loading cached labeled evaluation data from {cache_path}...")
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

    print("[*] Building fast evaluation cache for 285 labeled subjects...")
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
    print(f"[*] Saved {len(recs)} labeled subjects to {cache_path}")
    return recs


def evaluate_lodo(labeled_recs, basis_dict, kind, k):
    """Run LODO probe evaluation."""
    features = []
    for r in labeled_recs:
        proj = project(kind, basis_dict[kind], r["rows"], k=k)
        z = bin_reduce(proj)
        g = bin_reduce(r["gaze"][:len(r["rows"])])
        n = min(len(z), len(g))
        features.append({
            "dataset": r["dataset"],
            "subject": r["subject"],
            "z": z[:n],
            "gaze": g[:n],
        })

    datasets = sorted({r["dataset"] for r in features})
    per_fold = {}

    for held in datasets:
        train = [r for r in features if r["dataset"] != held]
        test = [r for r in features if r["dataset"] == held]

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

        model = RidgeCV(alphas=np.logspace(-3, 5, 17)).fit(x_tr, y_tr)

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


def fit_from_subjects(selected_subjects, mask, k=64, trs_per_subject=48, n_slabs=4):
    flat = mask.reshape(-1)
    moments = Moments(int(flat.sum()))
    for _ds, _sub, path, n_trs in selected_subjects:
        try:
            with h5py.File(path, "r") as f:
                block = f["eye_block"]
                for start, stop in _slabs(n_trs, trs_per_subject, n_slabs):
                    slab = block[..., start:stop]
                    moments.add(slab.reshape(-1, slab.shape[-1])[flat].T)
        except Exception:
            continue
    moments.symmetrise()
    bases = {}
    bases["lr-cca"] = fit_lr_cca(moments, mask, k=k, n_reduce=256)
    bases["corpus-pca"] = fit_pca(moments, k=k)
    return bases


def main():
    data_dir = resolve(None, download=False, quiet=True)
    mask = corpus_mask(data_dir)
    labeled_recs = get_labeled_cache(data_dir, mask)

    all_unlabeled = unlabeled_subjects(data_dir)
    ds_to_subs = defaultdict(list)
    for s in all_unlabeled:
        ds_to_subs[s[0]].append(s)

    # 1 sub/ds eligible (datasets with >= 1 sub)
    ds_list_all = sorted(ds_to_subs.keys())
    # 2 subs/ds eligible (datasets with >= 2 subs)
    ds_list_2plus = sorted([d for d, subs in ds_to_subs.items() if len(subs) >= 2])

    print(f"[*] Total unique datasets: {len(ds_list_all)}")
    print(f"[*] Datasets with >= 2 subjects: {len(ds_list_2plus)}")

    checkpoints = [25, 50, 100, 200, 400, 600]
    rng = np.random.default_rng(42)

    results = {"1_per_dataset": {}, "2_per_dataset": {}}

    print("\n" + "=" * 80)
    print("RUNNING SCALING SWEEP: 1 SUB/DATASET vs 2 SUBS/DATASET")
    print("=" * 80)

    for n in checkpoints:
        print(f"\n---> Target N = {n} participants")

        # Arm 1: 1 sub per dataset (N distinct datasets)
        if n <= len(ds_list_all):
            chosen_ds = rng.choice(ds_list_all, size=n, replace=False)
            subs_1per = [ds_to_subs[d][0] for d in chosen_ds]
            t0 = time.time()
            bases_1per = fit_from_subjects(subs_1per, mask, k=64)
            r_cca32_1, _ = evaluate_lodo(labeled_recs, bases_1per, "lr-cca", k=32)
            r_cca64_1, _ = evaluate_lodo(labeled_recs, bases_1per, "lr-cca", k=64)
            r_pca64_1, _ = evaluate_lodo(labeled_recs, bases_1per, "corpus-pca", k=64)
            t_el = time.time() - t0
            results["1_per_dataset"][str(n)] = {
                "n_datasets": n,
                "lr-cca:32": r_cca32_1,
                "lr-cca:64": r_cca64_1,
                "corpus-pca:64": r_pca64_1,
            }
            print(f"  [1 sub/ds]  N={n:<3} ({n} ds)  -> lr-cca:32 = {r_cca32_1:.3f}, lr-cca:64 = {r_cca64_1:.3f}, corpus-pca:64 = {r_pca64_1:.3f} ({t_el:.1f}s)")

        # Arm 2: 2 subs per dataset (N/2 distinct datasets)
        n_ds_needed = (n + 1) // 2
        if n_ds_needed <= len(ds_list_2plus):
            chosen_ds = rng.choice(ds_list_2plus, size=n_ds_needed, replace=False)
            subs_2per = []
            for d in chosen_ds:
                subs_2per.extend(ds_to_subs[d][:2])
            subs_2per = subs_2per[:n]  # ensure exactly N
            t0 = time.time()
            bases_2per = fit_from_subjects(subs_2per, mask, k=64)
            r_cca32_2, _ = evaluate_lodo(labeled_recs, bases_2per, "lr-cca", k=32)
            r_cca64_2, _ = evaluate_lodo(labeled_recs, bases_2per, "lr-cca", k=64)
            r_pca64_2, _ = evaluate_lodo(labeled_recs, bases_2per, "corpus-pca", k=64)
            t_el = time.time() - t0
            results["2_per_dataset"][str(n)] = {
                "n_datasets": len(chosen_ds),
                "lr-cca:32": r_cca32_2,
                "lr-cca:64": r_cca64_2,
                "corpus-pca:64": r_pca64_2,
            }
            print(f"  [2 subs/ds] N={n:<3} ({len(chosen_ds)} ds) -> lr-cca:32 = {r_cca32_2:.3f}, lr-cca:64 = {r_cca64_2:.3f}, corpus-pca:64 = {r_pca64_2:.3f} ({t_el:.1f}s)")

    out_path = Path("results/scaling/diversity_scaling_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[*] Saved benchmark results to {out_path}")


if __name__ == "__main__":
    main()
