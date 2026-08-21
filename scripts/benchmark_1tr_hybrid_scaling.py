#!/usr/bin/env python3
"""Scaling Benchmark for 1-TR Mean: Supervised fold-pca:64 vs DeepMReye 2.0 Hybrid across N=25..1800."""
import json
import sys
import time
from pathlib import Path
import h5py
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV

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


def load_raw_subs():
    data_dir = resolve(None, download=False, quiet=True)
    mask = corpus_mask(data_dir)
    flat = mask.reshape(-1)
    datasets = sorted(data_dir.glob("dsL*"))
    all_subs = []
    for ds_dir in datasets:
        if ds_dir.name in EXCLUDE or not ds_dir.is_dir():
            continue
        for p in sorted(ds_dir.glob("*.h5")):
            try:
                with h5py.File(p, "r") as f:
                    if "labels" not in f:
                        continue
                    block = f["eye_block"][:]
                    labels = f["labels"][:]
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
    return all_subs


def eval_lodo_subtr(subs, feat_extract_fn):
    datasets = sorted({s["dataset"] for s in subs})
    per_fold = {}
    alphas = np.logspace(-2, 4, 13)

    feats = []
    for s in subs:
        z = feat_extract_fn(s)
        feats.append({
            "dataset": s["dataset"],
            "subject": s["subject"],
            "z": z,
            "labels": s["labels"],
        })

    for held in datasets:
        train = [f for f in feats if f["dataset"] != held]
        test = [f for f in feats if f["dataset"] == held]

        xs, ys = [], []
        for ds in sorted({f["dataset"] for f in train}):
            ds_subs = [f for f in train if f["dataset"] == ds]
            ds_z = np.concatenate([f["z"] for f in ds_subs])
            ds_lab = np.concatenate([f["labels"].reshape(-1, 20) for f in ds_subs])
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

        model = RidgeCV(alphas=alphas).fit(x_tr, y_tr)

        sub_scores = []
        for s in test:
            x_te = s["z"]
            lab_te = s["labels"]
            T = len(x_te)
            y_te_flat = lab_te.reshape(T, 20)
            ok = np.isfinite(y_te_flat).all(axis=1) & np.isfinite(x_te).all(axis=1)
            if ok.sum() < 10:
                continue
            pred_20 = model.predict(x_te)
            pred_subtr = pred_20.reshape(T, 10, 2)

            p_1tr = np.nanmean(pred_subtr[ok], axis=1)
            t_1tr = np.nanmean(lab_te[ok], axis=1)
            rx = calc_r(p_1tr[:, 0], t_1tr[:, 0])
            ry = calc_r(p_1tr[:, 1], t_1tr[:, 1])
            if np.isfinite(rx) and np.isfinite(ry):
                sub_scores.append((rx + ry) / 2.0)

        if sub_scores:
            per_fold[held] = float(np.nanmedian(sub_scores))

    return float(np.median(list(per_fold.values()))), per_fold


def main():
    subs = load_raw_subs()
    available_checkpoints = [n for n in CHECKPOINTS if Path(f"results/scaling/basis_n{n}.npz").exists()]

    models = [
        ("DeepMReye 2.0 Core (lr-cca:32)", "cca32"),
        ("DeepMReye 2.0 Unsupervised PCA (corpus-pca:64)", "pca64"),
        ("DeepMReye 2.0 Super-Basis (lr-cca:32 + corpus-pca:64)", "hybrid96"),
        ("DeepMReye 2.0 Tri-Basis (lr-cca:32 + pca:64 + band:16)", "hybrid112"),
    ]

    results = {
        "checkpoints": available_checkpoints,
        "fold-pca:64": 0.8370,
        "curves": {name: {} for name, _ in models}
    }

    print("\n" + "=" * 115, flush=True)
    print("1-TR MEAN SCALING MATRIX: SUPERVISED fold-pca:64 vs DeepMReye 2.0 SUPER-BASIS (N = 25 -> 1800)")
    print("=" * 115, flush=True)

    for n in available_checkpoints:
        basis_path = Path(f"results/scaling/basis_n{n}.npz")
        _mask, bases, meta = load_basis(basis_path)
        cca_dict = bases["lr-cca"]
        mu_cca = cca_dict["mean"]
        li, ri = cca_dict["left_index"], cca_dict["right_index"]
        wl, wr = cca_dict["left_weights"], cca_dict["right_weights"]
        pca_comp = bases["corpus-pca"]["components"]
        pca_mu = bases["corpus-pca"]["mean"]
        band_comp = bases["band-pca"]["components"]
        band_mu = bases["band-pca"]["mean"]

        def fn_cca32(s):
            xc = s["vox"] - mu_cca
            return 0.5 * (xc[:, li] @ wl[:, :32] + xc[:, ri] @ wr[:, :32])

        def fn_pca64(s):
            return (s["vox"] - pca_mu) @ pca_comp[:, :64]

        def fn_hybrid96(s):
            xc = s["vox"] - mu_cca
            z_cca = 0.5 * (xc[:, li] @ wl[:, :32] + xc[:, ri] @ wr[:, :32])
            z_pca = (s["vox"] - pca_mu) @ pca_comp[:, :64]
            return np.concatenate([z_cca, z_pca], axis=1)

        def fn_hybrid112(s):
            xc = s["vox"] - mu_cca
            z_cca = 0.5 * (xc[:, li] @ wl[:, :32] + xc[:, ri] @ wr[:, :32])
            z_pca = (s["vox"] - pca_mu) @ pca_comp[:, :64]
            z_band = (s["vox"] - band_mu) @ band_comp[:, :16]
            return np.concatenate([z_cca, z_pca, z_band], axis=1)

        fn_map = {
            "cca32": fn_cca32,
            "pca64": fn_pca64,
            "hybrid96": fn_hybrid96,
            "hybrid112": fn_hybrid112,
        }

        t0 = time.time()
        for name, key in models:
            r_val, _ = eval_lodo_subtr(subs, fn_map[key])
            results["curves"][name][str(n)] = r_val

        r_cca = results["curves"]["DeepMReye 2.0 Core (lr-cca:32)"][str(n)]
        r_hyb = results["curves"]["DeepMReye 2.0 Super-Basis (lr-cca:32 + corpus-pca:64)"][str(n)]
        print(f"  N={n:<5} -> lr-cca:32 = {r_cca:.4f} | Super-Basis (lr-cca:32 + pca:64) = {r_hyb:.4f}  ({time.time() - t0:.1f}s)", flush=True)

    print("\n" + "=" * 125, flush=True)
    print("FINAL 1-TR MEAN SCALING MATRIX: LODO PEARSON r ACROSS 7 HELD-OUT DATASET FOLDS (N=285 labeled subjects)", flush=True)
    print("=" * 125, flush=True)
    header = f"{'Representation':<44}" + "".join(f"{('N=' + str(n)):>9}" for n in available_checkpoints)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    print(f"{'fold-pca:64 (Supervised Reference Baseline)':<44}" + "".join(f"{0.8370:>9.4f}" for _ in available_checkpoints), flush=True)
    print("-" * len(header), flush=True)

    for name, _ in models:
        row = f"{name:<44}"
        for n in available_checkpoints:
            v = results["curves"][name].get(str(n), None)
            row += f"{v:>9.4f}" if v is not None else f"{'--':>9}"
        print(row, flush=True)
    print("=" * 125, flush=True)

    out_file = Path("results/scaling/hybrid_1tr_scaling_n1800.json")
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\n[*] Saved hybrid 1-TR benchmark to {out_file}", flush=True)


if __name__ == "__main__":
    main()
