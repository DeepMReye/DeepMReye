#!/usr/bin/env python3
"""Comprehensive Multi-Resolution Benchmark: Sub-TR (10 pts/TR), 1-TR Mean, and 5-TR Bin Mean.

Evaluates scaling across checkpoints N=25..1800 for:
1. Sub-TR (10 pts/TR resolution, continuous time)
2. 1-TR Mean (Instantaneous per-TR resolution)
3. 5-TR Bin Mean (Low-frequency block mean)

Across both:
- 7-Fold LODO Cross-Dataset (N=285 labeled subjects)
- dsL03_pursuit Continuous Gaze (N=24 participants)
"""
import json
import os
import sys
import time
from pathlib import Path
import h5py
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from deepmreye.datasource import resolve
from deepmreye.unsupervised import corpus_mask, load_basis, project

EXCLUDE = ("dsL11_backtothefuture",)
CHECKPOINTS = [25, 50, 100, 200, 400, 800, 1200, 1500, 1800, 2000]


def calc_r(p, t):
    ok = np.isfinite(p) & np.isfinite(t)
    if ok.sum() < 10 or np.std(t[ok]) < 1e-6 or np.std(p[ok]) < 1e-6:
        return np.nan
    return float(np.corrcoef(p[ok], t[ok])[0, 1])


def calc_err(p, t):
    ok = np.isfinite(p).all(axis=-1) & np.isfinite(t).all(axis=-1)
    if ok.sum() < 10:
        return np.nan
    return float(np.mean(np.linalg.norm(p[ok] - t[ok], axis=-1)))


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


def load_all_labeled_data(data_dir, mask):
    flat = mask.reshape(-1)
    datasets = sorted(data_dir.glob("dsL*"))
    all_subs = []
    print(f"[*] Loading raw labeled subject data from {len(datasets)} dataset folders...", flush=True)
    for ds_dir in datasets:
        if ds_dir.name in EXCLUDE or not ds_dir.is_dir():
            continue
        for p in sorted(ds_dir.glob("*.h5")):
            try:
                with h5py.File(p, "r") as f:
                    if "labels" not in f:
                        continue
                    block = f["eye_block"][:] # [47, 29, 18, T]
                    labels = f["labels"][:]   # [T, 10, 2]
            except Exception:
                continue
            T = block.shape[-1]
            if T < 30 or not np.isfinite(labels).any():
                continue
            vox = block.reshape(-1, T).T[:, flat].astype(np.float32)
            all_subs.append({
                "dataset": ds_dir.name,
                "subject": p.stem,
                "vox": vox,
                "labels": labels[:T].astype(np.float32),
            })
    print(f"[*] Loaded {len(all_subs)} total valid labeled subjects.", flush=True)
    return all_subs


def evaluate_lodo_multi_res(subs, bases_dict, basis_name, k, lags):
    """Evaluates LODO predictions at Sub-TR (10 pts/TR), 1-TR mean, and 5-TR bin mean."""
    # Pre-extract features for each subject
    feats = []
    for s in subs:
        if basis_name == "fold-pca":
            # will be computed inside fold
            z = None
        else:
            z = project(basis_name, bases_dict[basis_name], s["vox"], k=k)
            z = make_lag_features(z, lags=lags)
        feats.append({
            "dataset": s["dataset"],
            "subject": s["subject"],
            "vox": s["vox"],
            "z": z,
            "labels": s["labels"],
        })

    datasets = sorted({s["dataset"] for s in feats})
    per_fold_subtr = []
    per_fold_1tr = []
    per_fold_5tr = []

    for held in datasets:
        train = [s for s in feats if s["dataset"] != held]
        test = [s for s in feats if s["dataset"] == held]

        # Handle fold-pca dynamically if needed
        if basis_name == "fold-pca":
            from sklearn.decomposition import PCA
            x_tr_raw = np.concatenate([s["vox"] for s in train])
            idx = np.random.default_rng(0).choice(len(x_tr_raw), min(5000, len(x_tr_raw)), replace=False)
            pca = PCA(n_components=k, random_state=0).fit(x_tr_raw[idx])
            for s in train + test:
                s["z"] = make_lag_features(pca.transform(s["vox"]), lags=lags)

        # Build training set predicting [T, 20] (the 10 sub-TR points per TR)
        xs, ys = [], []
        for ds in sorted({s["dataset"] for s in train}):
            ds_subs = [s for s in train if s["dataset"] == ds]
            ds_z = np.concatenate([s["z"] for s in ds_subs])
            ds_lab = np.concatenate([s["labels"].reshape(-1, 20) for s in ds_subs])
            ok = np.isfinite(ds_lab).all(axis=1) & np.isfinite(ds_z).all(axis=1)
            if ok.sum() < 10:
                continue
            z_ok, lab_ok = ds_z[ok], ds_lab[ok]
            sd = lab_ok.std(axis=0)
            sd[sd < 1e-9] = 1.0
            ys.append((lab_ok - lab_ok.mean(axis=0)) / sd)
            xs.append(z_ok)

        if not xs:
            continue
        x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
        if len(x_tr) > 15000:
            idx = np.random.default_rng(0).choice(len(x_tr), 15000, replace=False)
            x_tr, y_tr = x_tr[idx], y_tr[idx]

        model = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(x_tr, y_tr)

        # Evaluate on held-out test subjects
        sub_scores_subtr = []
        sub_scores_1tr = []
        sub_scores_5tr = []

        for s in test:
            x_te = s["z"]
            lab_te = s["labels"] # [T, 10, 2]
            T = len(x_te)
            y_te_flat = lab_te.reshape(T, 20)
            ok = np.isfinite(y_te_flat).all(axis=1) & np.isfinite(x_te).all(axis=1)
            if ok.sum() < 10:
                continue
            pred_20 = model.predict(x_te) # [T, 20]
            pred_subtr = pred_20.reshape(T, 10, 2) # [T, 10, 2]

            # 1. Sub-TR (10 pts/TR)
            p_flat = pred_subtr[ok].reshape(-1, 2)
            t_flat = lab_te[ok].reshape(-1, 2)
            rx_sub = calc_r(p_flat[:, 0], t_flat[:, 0])
            ry_sub = calc_r(p_flat[:, 1], t_flat[:, 1])
            sub_scores_subtr.append((rx_sub + ry_sub) / 2.0)

            # 2. 1-TR mean
            p_1tr = np.nanmean(pred_subtr[ok], axis=1) # [T, 2]
            t_1tr = np.nanmean(lab_te[ok], axis=1)    # [T, 2]
            rx_1 = calc_r(p_1tr[:, 0], t_1tr[:, 0])
            ry_1 = calc_r(p_1tr[:, 1], t_1tr[:, 1])
            sub_scores_1tr.append((rx_1 + ry_1) / 2.0)

            # 3. 5-TR bin mean
            n_bins = ok.sum() // 5
            if n_bins >= 2:
                p_5tr = np.nanmean(pred_subtr[ok][:n_bins*5].reshape(n_bins, 50, 2), axis=1)
                t_5tr = np.nanmean(lab_te[ok][:n_bins*5].reshape(n_bins, 50, 2), axis=1)
                rx_5 = calc_r(p_5tr[:, 0], t_5tr[:, 0])
                ry_5 = calc_r(p_5tr[:, 1], t_5tr[:, 1])
                sub_scores_5tr.append((rx_5 + ry_5) / 2.0)

        if sub_scores_subtr:
            per_fold_subtr.append(np.nanmedian(sub_scores_subtr))
            per_fold_1tr.append(np.nanmedian(sub_scores_1tr))
            if sub_scores_5tr:
                per_fold_5tr.append(np.nanmedian(sub_scores_5tr))

    return {
        "subtr": float(np.median(per_fold_subtr)),
        "1tr": float(np.median(per_fold_1tr)),
        "5tr": float(np.median(per_fold_5tr)),
    }


def main():
    data_dir = resolve(None, download=False, quiet=True)
    mask = corpus_mask(data_dir)
    subs = load_all_labeled_data(data_dir, mask)

    print("\n[*] Evaluating fold-pca:64 baseline across resolutions...", flush=True)
    res_fold_pca = evaluate_lodo_multi_res(subs, {}, "fold-pca", k=64, lags=0)
    res_fold_pca_lags = evaluate_lodo_multi_res(subs, {}, "fold-pca", k=64, lags=2)
    print(f"[*] fold-pca:64 (instantaneous) -> Sub-TR: {res_fold_pca['subtr']:.3f} | 1-TR: {res_fold_pca['1tr']:.3f} | 5-TR: {res_fold_pca['5tr']:.3f}", flush=True)
    print(f"[*] fold-pca:64 (+lags±2)      -> Sub-TR: {res_fold_pca_lags['subtr']:.3f} | 1-TR: {res_fold_pca_lags['1tr']:.3f} | 5-TR: {res_fold_pca_lags['5tr']:.3f}", flush=True)

    available_checkpoints = [n for n in CHECKPOINTS if Path(f"results/scaling/basis_n{n}.npz").exists()]

    models = [
        ("DeepMReye 2.0 (lr-cca:32)", "lr-cca", 32, 0),
        ("DeepMReye 2.0 (lr-cca:32 + lags±1)", "lr-cca", 32, 1),
        ("DeepMReye 2.0 (lr-cca:32 + lags±2)", "lr-cca", 32, 2),
        ("DeepMReye 2.0 (lr-cca:64 + lags±2)", "lr-cca", 64, 2),
    ]

    all_results = {
        "checkpoints": available_checkpoints,
        "fold-pca:64": res_fold_pca,
        "fold-pca:64+lags": res_fold_pca_lags,
        "curves": {name: {"subtr": {}, "1tr": {}, "5tr": {}} for name, _, _, _ in models}
    }

    print("\n" + "=" * 115, flush=True)
    print("MULTI-RESOLUTION LODO BENCHMARK: SUB-TR (10 pts/TR) vs 1-TR MEAN vs 5-TR BIN MEAN")
    print("=" * 115, flush=True)

    for n in available_checkpoints:
        basis_path = Path(f"results/scaling/basis_n{n}.npz")
        _mask, bases, meta = load_basis(basis_path)
        t0 = time.time()
        print(f"\n---> Evaluating Checkpoint N = {n:<5} ({meta.get('datasets', '?')} datasets, {meta.get('n_trs', '?')} TRs)...", flush=True)

        for name, basis_name, k, lags in models:
            res = evaluate_lodo_multi_res(subs, bases, basis_name, k=k, lags=lags)
            all_results["curves"][name]["subtr"][str(n)] = res["subtr"]
            all_results["curves"][name]["1tr"][str(n)] = res["1tr"]
            all_results["curves"][name]["5tr"][str(n)] = res["5tr"]
            print(f"     {name:<36} -> Sub-TR: {res['subtr']:.3f} | 1-TR: {res['1tr']:.3f} | 5-TR: {res['5tr']:.3f}", flush=True)

    # Print Summary Tables for Sub-TR and 1-TR
    for res_key, res_title in [
        ("subtr", "SUB-TR RESOLUTION (10 POINTS / TR) - LODO PEARSON r"),
        ("1tr", "1-TR MEAN RESOLUTION (PER-TR INSTANTANEOUS) - LODO PEARSON r"),
        ("5tr", "5-TR BIN MEAN RESOLUTION (BLOCK MEAN) - LODO PEARSON r"),
    ]:
        print("\n" + "=" * 125, flush=True)
        print(res_title, flush=True)
        print("=" * 125, flush=True)
        header = f"{'Representation':<38}" + "".join(f"{('N=' + str(n)):>10}" for n in available_checkpoints)
        print(header, flush=True)
        print("-" * len(header), flush=True)
        base_val = res_fold_pca[res_key]
        print(f"{'fold-pca:64 (Supervised Reference)':<38}" + "".join(f"{base_val:>10.3f}" for _ in available_checkpoints), flush=True)
        print("-" * len(header), flush=True)
        for name, _, _, _ in models:
            row = f"{name:<38}"
            for n in available_checkpoints:
                v = all_results["curves"][name][res_key].get(str(n), None)
                row += f"{v:>10.3f}" if v is not None else f"{'--':>10}"
            print(row, flush=True)
        print("=" * 125, flush=True)

    out_file = Path("results/scaling/multi_resolution_scaling_n1800.json")
    out_file.write_text(json.dumps(all_results, indent=2))
    print(f"\n[*] Saved multi-resolution benchmark to {out_file}", flush=True)


if __name__ == "__main__":
    main()
