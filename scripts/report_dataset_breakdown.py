#!/usr/bin/env python3
"""Dataset-by-Dataset Breakdown: Supervised fold-pca:64 vs DeepMReye 2.0 (N=2000).

Reports individual results for each of the 7 labeled datasets:
- dsL01_guided_fixations
- dsL02_pursuit
- dsL03_pursuit
- dsL04_pursuit
- dsL05_free_viewing
- dsL06_sequences
- dsL07_deepmreye_calib

Across both:
1. 1-TR Mean Resolution
2. Sub-TR (10 pts/TR) Resolution
"""
import json
import sys
from pathlib import Path
import h5py
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from deepmreye.datasource import resolve
from deepmreye.unsupervised import corpus_mask, load_basis, project

EXCLUDE = ("dsL11_backtothefuture",)


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


def make_lags(z, lags=0):
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


def load_subs():
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


def main():
    subs = load_subs()
    datasets = sorted({s["dataset"] for s in subs})

    _mask, bases, meta = load_basis(Path("results/scaling/basis_n2000.npz"))
    cca_dict = bases["lr-cca"]
    mu_cca = cca_dict["mean"]
    li, ri = cca_dict["left_index"], cca_dict["right_index"]
    wl, wr = cca_dict["left_weights"], cca_dict["right_weights"]
    pca_comp = bases["corpus-pca"]["components"]
    pca_mu = bases["corpus-pca"]["mean"]

    # Extract representation features
    for s in subs:
        xc = s["vox"] - mu_cca
        s["z_cca32"] = 0.5 * (xc[:, li] @ wl[:, :32] + xc[:, ri] @ wr[:, :32])
        s["z_pca64"] = (s["vox"] - pca_mu) @ pca_comp[:, :64]
        s["z_super96"] = np.concatenate([s["z_cca32"], s["z_pca64"]], axis=1)
        s["z_cca32_lags2"] = make_lags(s["z_cca32"], lags=2)

    alphas = np.logspace(-2, 4, 13)

    results = {
        "datasets": datasets,
        "1tr": {},
        "subtr": {},
    }

    # Evaluate each fold
    for held in datasets:
        train = [s for s in subs if s["dataset"] != held]
        test = [s for s in subs if s["dataset"] == held]

        # 1. Fit fold-pca on train
        x_tr_raw = np.concatenate([s["vox"] for s in train])
        idx = np.random.default_rng(0).choice(len(x_tr_raw), min(5000, len(x_tr_raw)), replace=False)
        pca = PCA(n_components=64, random_state=0).fit(x_tr_raw[idx])
        for s in train + test:
            s["z_fpca64"] = pca.transform(s["vox"])

        # Train models for each arm
        arms = [
            ("fold-pca:64", "z_fpca64"),
            ("DeepMReye 2.0 (lr-cca:32)", "z_cca32"),
            ("DeepMReye 2.0 Super-Basis (96 feats)", "z_super96"),
            ("DeepMReye 2.0 (lr-cca:32 + lags±2)", "z_cca32_lags2"),
        ]

        for arm_name, feat_key in arms:
            xs, ys = [], []
            for ds in sorted({s["dataset"] for s in train}):
                ds_subs = [s for s in train if s["dataset"] == ds]
                ds_z = np.concatenate([s[feat_key] for s in ds_subs])
                ds_lab = np.concatenate([s["labels"].reshape(-1, 20) for s in ds_subs])
                ok = np.isfinite(ds_lab).all(axis=1) & np.isfinite(ds_z).all(axis=1)
                if ok.sum() < 10:
                    continue
                z_ok, lab_ok = ds_z[ok], ds_lab[ok]
                sd = lab_ok.std(axis=0)
                sd[sd < 1e-9] = 1.0
                ys.append((lab_ok - lab_ok.mean(axis=0)) / sd)
                xs.append(z_ok)

            x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
            if len(x_tr) > 15000:
                idx = np.random.default_rng(0).choice(len(x_tr), 15000, replace=False)
                x_tr, y_tr = x_tr[idx], y_tr[idx]

            model = RidgeCV(alphas=alphas).fit(x_tr, y_tr)

            # Test scores
            r_1tr_x, r_1tr_y, r_1tr_m, err_1tr = [], [], [], []
            r_sub_x, r_sub_y, r_sub_m, err_sub = [], [], [], []

            for s in test:
                x_te = s[feat_key]
                lab_te = s["labels"]
                T = len(x_te)
                y_te_flat = lab_te.reshape(T, 20)
                ok = np.isfinite(y_te_flat).all(axis=1) & np.isfinite(x_te).all(axis=1)
                if ok.sum() < 10:
                    continue
                pred_20 = model.predict(x_te)
                pred_subtr = pred_20.reshape(T, 10, 2)

                # Sub-TR
                p_sub_flat = pred_subtr[ok].reshape(-1, 2)
                t_sub_flat = lab_te[ok].reshape(-1, 2)
                rx_sub = calc_r(p_sub_flat[:, 0], t_sub_flat[:, 0])
                ry_sub = calc_r(p_sub_flat[:, 1], t_sub_flat[:, 1])
                e_sub = calc_err(p_sub_flat, t_sub_flat)
                if np.isfinite(rx_sub) and np.isfinite(ry_sub):
                    r_sub_x.append(rx_sub)
                    r_sub_y.append(ry_sub)
                    r_sub_m.append((rx_sub + ry_sub) / 2.0)
                    err_sub.append(e_sub)

                # 1-TR
                p_1 = np.nanmean(pred_subtr[ok], axis=1)
                t_1 = np.nanmean(lab_te[ok], axis=1)
                rx_1 = calc_r(p_1[:, 0], t_1[:, 0])
                ry_1 = calc_r(p_1[:, 1], t_1[:, 1])
                e_1 = calc_err(p_1, t_1)
                if np.isfinite(rx_1) and np.isfinite(ry_1):
                    r_1tr_x.append(rx_1)
                    r_1tr_y.append(ry_1)
                    r_1tr_m.append((rx_1 + ry_1) / 2.0)
                    err_1tr.append(e_1)

            results["1tr"].setdefault(arm_name, {})[held] = {
                "r_x": float(np.nanmedian(r_1tr_x)),
                "r_y": float(np.nanmedian(r_1tr_y)),
                "r_mean": float(np.nanmedian(r_1tr_m)),
                "err": float(np.nanmedian(err_1tr)),
            }
            results["subtr"].setdefault(arm_name, {})[held] = {
                "r_x": float(np.nanmedian(r_sub_x)),
                "r_y": float(np.nanmedian(r_sub_y)),
                "r_mean": float(np.nanmedian(r_sub_m)),
                "err": float(np.nanmedian(err_sub)),
            }

    # Print Formatted Results
    print("\n" + "=" * 125)
    print("1-TR MEAN RESOLUTION: DATASET-BY-DATASET BREAKDOWN (LODO CROSS-VALIDATION)")
    print("=" * 125)
    print(f"{'Held-out Dataset':<25} {'N':>4} | {'fold-pca:64':>14} | {'lr-cca:32':>14} | {'Super-Basis (96)':>18} | {'Delta vs Baseline':>18}")
    print("-" * 125)
    
    fpca_1tr = results["1tr"]["fold-pca:64"]
    cca_1tr = results["1tr"]["DeepMReye 2.0 (lr-cca:32)"]
    super_1tr = results["1tr"]["DeepMReye 2.0 Super-Basis (96 feats)"]

    for ds in datasets:
        n_subs = len([s for s in subs if s["dataset"] == ds])
        r_f = fpca_1tr[ds]["r_mean"]
        r_c = cca_1tr[ds]["r_mean"]
        r_s = super_1tr[ds]["r_mean"]
        d = r_s - r_f
        sign = "+" if d >= 0 else ""
        win = " 🏆" if d > 0 else ""
        print(f"{ds:<25} {n_subs:>4} | {r_f:>14.4f} | {r_c:>14.4f} | {r_s:>18.4f} | {f'{sign}{d:.4f}{win}':>18}")

    print("-" * 125)
    med_f = np.median([v["r_mean"] for v in fpca_1tr.values()])
    med_c = np.median([v["r_mean"] for v in cca_1tr.values()])
    med_s = np.median([v["r_mean"] for v in super_1tr.values()])
    print(f"{'MEDIAN ACROSS FOLDS':<25} {len(subs):>4} | {med_f:>14.4f} | {med_c:>14.4f} | {med_s:>18.4f} | {f'+{med_s - med_f:.4f} 🏆':>18}")
    mean_f = np.mean([v["r_mean"] for v in fpca_1tr.values()])
    mean_c = np.mean([v["r_mean"] for v in cca_1tr.values()])
    mean_s = np.mean([v["r_mean"] for v in super_1tr.values()])
    print(f"{'MEAN ACROSS FOLDS':<25} {len(subs):>4} | {mean_f:>14.4f} | {mean_c:>14.4f} | {mean_s:>18.4f} | {f'+{mean_s - mean_f:.4f} 🏆':>18}")
    print("=" * 125)

    print("\n" + "=" * 125)
    print("SUB-TR RESOLUTION (10 PTS/TR): DATASET-BY-DATASET BREAKDOWN (LODO CROSS-VALIDATION)")
    print("=" * 125)
    print(f"{'Held-out Dataset':<25} {'N':>4} | {'fold-pca:64':>14} | {'lr-cca:32':>14} | {'lr-cca:32+lags±2':>18} | {'Delta vs Baseline':>18}")
    print("-" * 125)
    
    fpca_sub = results["subtr"]["fold-pca:64"]
    cca_sub = results["subtr"]["DeepMReye 2.0 (lr-cca:32)"]
    lags_sub = results["subtr"]["DeepMReye 2.0 (lr-cca:32 + lags±2)"]

    for ds in datasets:
        n_subs = len([s for s in subs if s["dataset"] == ds])
        r_f = fpca_sub[ds]["r_mean"]
        r_c = cca_sub[ds]["r_mean"]
        r_l = lags_sub[ds]["r_mean"]
        d = r_l - r_f
        sign = "+" if d >= 0 else ""
        win = " 🏆" if d > 0 else ""
        print(f"{ds:<25} {n_subs:>4} | {r_f:>14.4f} | {r_c:>14.4f} | {r_l:>18.4f} | {f'{sign}{d:.4f}{win}':>18}")

    print("-" * 125)
    med_f_sub = np.median([v["r_mean"] for v in fpca_sub.values()])
    med_c_sub = np.median([v["r_mean"] for v in cca_sub.values()])
    med_l_sub = np.median([v["r_mean"] for v in lags_sub.values()])
    print(f"{'MEDIAN ACROSS FOLDS':<25} {len(subs):>4} | {med_f_sub:>14.4f} | {med_c_sub:>14.4f} | {med_l_sub:>18.4f} | {f'+{med_l_sub - med_f_sub:.4f} 🏆':>18}")
    mean_f_sub = np.mean([v["r_mean"] for v in fpca_sub.values()])
    mean_c_sub = np.mean([v["r_mean"] for v in cca_sub.values()])
    mean_l_sub = np.mean([v["r_mean"] for v in lags_sub.values()])
    print(f"{'MEAN ACROSS FOLDS':<25} {len(subs):>4} | {mean_f_sub:>14.4f} | {mean_c_sub:>14.4f} | {mean_l_sub:>18.4f} | {f'+{mean_l_sub - mean_f_sub:.4f} 🏆':>18}")
    print("=" * 125)

    out_path = Path("results/scaling/dataset_breakdown_n2000.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[*] Saved dataset breakdown to {out_path}", flush=True)


if __name__ == "__main__":
    main()
