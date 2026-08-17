#!/usr/bin/env python3
"""Sweep Temporal Window Lags per Dataset for DeepMReye 2.0.

Evaluates performance as a function of temporal window lag L in [0, 1, 2, 3, 4, 5]
(window size 2L+1 in [1, 3, 5, 7, 9, 11] TRs) across all labeled datasets.
Generates publication-ready figures showing optimal lag per dataset.
"""
import os
import sys
import json
import warnings
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = Path.home() / ".cache/deepmreye"
BASIS_PATH = REPO / "results/scaling/basis_n1039.npz"
OUT_JSON = REPO / "results/temporal_lag_sweep.json"
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_PNG = FIG_DIR / "temporal_window_lags_by_dataset.png"
FIG_PDF = FIG_DIR / "temporal_window_lags_by_dataset.pdf"

DATASET_CONFIGS = [
    {"id": "dsL01_guided_fixations", "name": "dsL01 (Fixations)", "type": "Fixation Grid", "max_subs": 20},
    {"id": "dsL02_pursuit", "name": "dsL02 (Pursuit)", "type": "Smooth Pursuit", "max_subs": 9},
    {"id": "dsL03_pursuit", "name": "dsL03 (Pursuit)", "type": "Smooth Pursuit", "max_subs": 24},
    {"id": "dsL04_pursuit", "name": "dsL04 (Pursuit)", "type": "Smooth Pursuit", "max_subs": 15},
    {"id": "dsL05_free_viewing", "name": "dsL05 (Free Viewing)", "type": "Free Viewing", "max_subs": 15},
    {"id": "dsL06_sequences", "name": "dsL06 (Sequences)", "type": "Visual Sequences", "max_subs": 6},
    {"id": "dsL07_deepmreye_calib", "name": "dsL07 (Calibration)", "type": "Gaze Calibration", "max_subs": 15},
    {"id": "dsL11_backtothefuture", "name": "dsL11 (Movie)", "type": "Movie Watching", "max_subs": 4},
]

LAGS = [0, 1, 2, 3, 4, 5]

def load_basis(path):
    b = np.load(path)
    mask = b["mask"]
    basis = {
        "mean": b["lr-cca/mean"],
        "left_index": b["lr-cca/left_index"],
        "right_index": b["lr-cca/right_index"],
        "left_weights": b["lr-cca/left_weights"],
        "right_weights": b["lr-cca/right_weights"],
    }
    return mask, basis

def project_lr_cca(basis_dict, x, k=32):
    mu = basis_dict["mean"]
    li, ri = basis_dict["left_index"], basis_dict["right_index"]
    wl, wr = basis_dict["left_weights"][:, :k], basis_dict["right_weights"][:, :k]
    xc = x - mu
    return 0.5 * (xc[:, li] @ wl + xc[:, ri] @ wr)

def calc_r(p, t):
    ok = np.isfinite(p) & np.isfinite(t)
    if ok.sum() < 20 or np.std(t[ok]) < 1e-6 or np.std(p[ok]) < 1e-6:
        return np.nan
    return float(np.corrcoef(p[ok], t[ok])[0, 1])

def calc_error(p, t):
    ok = np.isfinite(p).all(axis=-1) & np.isfinite(t).all(axis=-1)
    if ok.sum() < 20:
        return np.nan
    return float(np.mean(np.linalg.norm(p[ok] - t[ok], axis=-1)))

def make_lag_features(z, lags=2):
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

def evaluate_subtr_and_1tr(preds_subtr, labels):
    n = min(len(preds_subtr), len(labels))
    p_subtr = preds_subtr[:n]
    t_subtr = labels[:n]
    
    p_flat = p_subtr.reshape(-1, 2)
    t_flat = t_subtr.reshape(-1, 2)
    rx_sub = calc_r(p_flat[:, 0], t_flat[:, 0])
    ry_sub = calc_r(p_flat[:, 1], t_flat[:, 1])
    err_sub = calc_error(p_flat, t_flat)
    r_sub = 0.5 * (rx_sub + ry_sub) if np.isfinite(rx_sub) and np.isfinite(ry_sub) else np.nan
    
    p_tr = np.nanmean(p_subtr, axis=1)
    t_tr = np.nanmean(t_subtr, axis=1)
    rx_tr = calc_r(p_tr[:, 0], t_tr[:, 0])
    ry_tr = calc_r(p_tr[:, 1], t_tr[:, 1])
    err_tr = calc_error(p_tr, t_tr)
    r_tr = 0.5 * (rx_tr + ry_tr) if np.isfinite(rx_tr) and np.isfinite(ry_tr) else np.nan
    
    return {
        "subtr": (rx_sub, ry_sub, r_sub, err_sub),
        "1tr": (rx_tr, ry_tr, r_tr, err_tr),
    }

def run_lag_sweep():
    mask, basis = load_basis(BASIS_PATH)
    all_sweep_results = []
    
    print("[*] Starting Temporal Window Lag Sweep for DeepMReye 2.0...", flush=True)
    
    for cfg in DATASET_CONFIGS:
        ds_id = cfg["id"]
        ds_name = cfg["name"]
        ds_dir = DATA_ROOT / ds_id
        if not ds_dir.exists():
            continue
            
        files = sorted(ds_dir.glob("*.h5"))[:cfg["max_subs"]]
        print(f"[*] Processing {ds_name} ({len(files)} subjects)...", flush=True)
        
        data = []
        for p in files:
            with h5py.File(p, "r") as f:
                block = f["eye_block"][:]
                labels = f["labels"][:]
                data.append({"sub": p.stem, "block": block, "labels": labels})
                
        dataset_lag_res = {"dataset_id": ds_id, "dataset_name": ds_name, "type": cfg["type"], "lags": {}}
        
        for lag in LAGS:
            scores_1tr = []
            scores_subtr = []
            
            for d in data:
                block = d["block"]
                labels = d["labels"]
                T = block.shape[-1]
                vox = block[mask].T
                
                z = project_lr_cca(basis, vox, k=32)
                X = make_lag_features(z, lags=lag)
                
                y_sub = labels.reshape(T, 20)
                valid_sub = ~np.isnan(y_sub).any(axis=1)
                X_sv, y_sv = X[valid_sub], y_sub[valid_sub]
                
                if len(X_sv) >= 20:
                    kf = KFold(n_splits=5, shuffle=False)
                    preds_sub = np.zeros_like(y_sv)
                    for tr_idx, te_idx in kf.split(X_sv):
                        reg = Ridge(alpha=10.0)
                        reg.fit(X_sv[tr_idx], y_sv[tr_idx])
                        preds_sub[te_idx] = reg.predict(X_sv[te_idx])
                    
                    preds_full = np.full((T, 10, 2), np.nan)
                    preds_full[valid_sub] = preds_sub.reshape(-1, 10, 2)
                    res = evaluate_subtr_and_1tr(preds_full, labels)
                    scores_subtr.append(res["subtr"])
                    scores_1tr.append(res["1tr"])
                    
            r_1tr = float(np.nanmedian([s[2] for s in scores_1tr]))
            err_1tr = float(np.nanmedian([s[3] for s in scores_1tr]))
            r_sub = float(np.nanmedian([s[2] for s in scores_subtr]))
            err_sub = float(np.nanmedian([s[3] for s in scores_subtr]))
            
            dataset_lag_res["lags"][str(lag)] = {
                "window_size_tr": 2 * lag + 1,
                "1tr_r": r_1tr,
                "1tr_err": err_1tr,
                "subtr_r": r_sub,
                "subtr_err": err_sub,
                "all_sub_1tr_r": [float(s[2]) for s in scores_1tr if np.isfinite(s[2])],
                "all_sub_subtr_r": [float(s[2]) for s in scores_subtr if np.isfinite(s[2])],
            }
            print(f"    Lag +- {lag} ({2*lag+1} TRs): 1-TR r = {r_1tr:+.3f} ({err_1tr:.2f} deg) | Sub-TR r = {r_sub:+.3f} ({err_sub:.2f} deg)", flush=True)
            
        all_sweep_results.append(dataset_lag_res)
        
    with open(OUT_JSON, "w") as f:
        json.dump(all_sweep_results, f, indent=2)
    print(f"\n[+] Saved sweep data to {OUT_JSON}", flush=True)
    
    # Plotting
    plot_results(all_sweep_results)

def plot_results(results):
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("ggplot")
        
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "figure.titlesize": 16,
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
    
    # Panel A: 1-TR Mean Correlation vs Temporal Lag
    ax1 = axes[0, 0]
    for i, res in enumerate(results):
        lags = [int(k) for k in res["lags"].keys()]
        r_vals = [res["lags"][str(k)]["1tr_r"] for k in lags]
        best_idx = int(np.argmax(r_vals))
        c = colors[i % len(colors)]
        ax1.plot(lags, r_vals, marker="o", markersize=6, linewidth=2.2, label=f"{res['dataset_name']} (max: $\pm{lags[best_idx]}$)", color=c)
        ax1.scatter([lags[best_idx]], [r_vals[best_idx]], s=130, color=c, edgecolors="black", linewidths=1.5, zorder=5)
    ax1.set_title("A. 1-TR Mean Gaze Correlation ($r$) vs Temporal Window Lag", fontweight="bold", pad=10)
    ax1.set_xlabel("Temporal Window Lag $\pm L$ TRs (Window Size $= 2L+1$ TRs)")
    ax1.set_ylabel("Pearson Correlation ($r$)")
    ax1.set_xticks(LAGS)
    ax1.set_xticklabels([f"$\pm{l}$\n({2*l+1} TRs)" if l > 0 else "0\n(1 TR)" for l in LAGS])
    ax1.set_ylim(0.70, 0.98)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.92)
    
    # Panel B: Sub-TR Continuous Trajectory Correlation vs Temporal Lag
    ax2 = axes[0, 1]
    for i, res in enumerate(results):
        lags = [int(k) for k in res["lags"].keys()]
        r_vals = [res["lags"][str(k)]["subtr_r"] for k in lags]
        best_idx = int(np.argmax(r_vals))
        c = colors[i % len(colors)]
        ax2.plot(lags, r_vals, marker="s", markersize=6, linewidth=2.2, label=f"{res['dataset_name']} (max: $\pm{lags[best_idx]}$)", color=c)
        ax2.scatter([lags[best_idx]], [r_vals[best_idx]], s=130, color=c, edgecolors="black", linewidths=1.5, zorder=5)
    ax2.set_title("B. Sub-TR Continuous Trajectory ($r$) vs Temporal Window Lag", fontweight="bold", pad=10)
    ax2.set_xlabel("Temporal Window Lag $\pm L$ TRs (Window Size $= 2L+1$ TRs)")
    ax2.set_ylabel("Pearson Correlation ($r$)")
    ax2.set_xticks(LAGS)
    ax2.set_xticklabels([f"$\pm{l}$\n({2*l+1} TRs)" if l > 0 else "0\n(1 TR)" for l in LAGS])
    ax2.set_ylim(0.68, 0.98)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.92)

    # Panel C: Decoding Error (deg) vs Temporal Lag
    ax3 = axes[1, 0]
    for i, res in enumerate(results):
        lags = [int(k) for k in res["lags"].keys()]
        err_vals = [res["lags"][str(k)]["1tr_err"] for k in lags]
        best_idx = int(np.argmin(err_vals))
        c = colors[i % len(colors)]
        ax3.plot(lags, err_vals, marker="^", markersize=6, linewidth=2.2, label=f"{res['dataset_name']} (min: $\pm{lags[best_idx]}$)", color=c)
        ax3.scatter([lags[best_idx]], [err_vals[best_idx]], s=130, color=c, edgecolors="black", linewidths=1.5, zorder=5)
    ax3.set_title("C. 1-TR Mean Euclidean Error ($^\circ$) vs Temporal Window Lag", fontweight="bold", pad=10)
    ax3.set_xlabel("Temporal Window Lag $\pm L$ TRs (Window Size $= 2L+1$ TRs)")
    ax3.set_ylabel("Euclidean Error (Degrees of Visual Angle)")
    ax3.set_xticks(LAGS)
    ax3.set_xticklabels([f"$\pm{l}$\n({2*l+1} TRs)" if l > 0 else "0\n(1 TR)" for l in LAGS])
    ax3.set_ylim(0.0, 5.0)
    ax3.legend(loc="upper right", frameon=True, framealpha=0.92)

    # Panel D: Grand Mean Across Datasets
    ax4 = axes[1, 1]
    mean_1tr_r = [float(np.mean([res["lags"][str(lag)]["1tr_r"] for res in results])) for lag in LAGS]
    mean_subtr_r = [float(np.mean([res["lags"][str(lag)]["subtr_r"] for res in results])) for lag in LAGS]
    mean_1tr_err = [float(np.mean([res["lags"][str(lag)]["1tr_err"] for res in results])) for lag in LAGS]
        
    ax4_err = ax4.twinx()
    
    line1 = ax4.plot(LAGS, mean_1tr_r, marker="o", markersize=8, color="#1f77b4", linewidth=3.0, label="Grand Mean 1-TR Correlation ($r$)")
    line2 = ax4.plot(LAGS, mean_subtr_r, marker="s", markersize=8, color="#2ca02c", linewidth=3.0, label="Grand Mean Sub-TR Correlation ($r$)")
    line3 = ax4_err.plot(LAGS, mean_1tr_err, marker="v", markersize=8, color="#d62728", linewidth=2.5, linestyle="--", label="Grand Mean Error ($^\circ$) [Right Axis]")
    
    best_lag_1tr = LAGS[int(np.argmax(mean_1tr_r))]
    best_lag_sub = LAGS[int(np.argmax(mean_subtr_r))]
    ax4.scatter([best_lag_1tr], [max(mean_1tr_r)], s=180, color="#1f77b4", edgecolors="black", linewidths=2, zorder=6)
    ax4.scatter([best_lag_sub], [max(mean_subtr_r)], s=180, color="#2ca02c", edgecolors="black", linewidths=2, zorder=6)
    
    ax4.set_title(f"D. Grand Mean Across All Datasets (Optimal Window: $\pm{best_lag_1tr}$ TRs = {2*best_lag_1tr+1} TRs)", fontweight="bold", pad=10)
    ax4.set_xlabel("Temporal Window Lag $\pm L$ TRs (Window Size $= 2L+1$ TRs)")
    ax4.set_ylabel("Mean Pearson Correlation ($r$)")
    ax4_err.set_ylabel("Mean Error (Degrees of Visual Angle)", color="#d62728")
    ax4_err.tick_params(axis="y", labelcolor="#d62728")
    ax4.set_xticks(LAGS)
    ax4.set_xticklabels([f"$\pm{l}$\n({2*l+1} TRs)" if l > 0 else "0\n(1 TR)" for l in LAGS])
    ax4.set_ylim(0.78, 0.92)
    ax4_err.set_ylim(1.8, 2.5)
    
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc="lower right", frameon=True, framealpha=0.92)
    
    plt.suptitle("DeepMReye 2.0: Impact of Temporal Window Lags on Gaze Decoding Accuracy Across 8 Datasets", y=0.99, fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    plt.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(FIG_PDF, bbox_inches="tight")
    print(f"[+] Saved figure to {FIG_PNG} and {FIG_PDF}", flush=True)

if __name__ == "__main__":
    run_lag_sweep()
