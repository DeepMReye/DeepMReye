#!/usr/bin/env python3
"""Analyze relationship between Temporal Window Lags, TR Length, and Absolute Time.

Investigates whether the optimal temporal window is governed by:
1. Discrete TR units (scanner acquisition steps)
2. Absolute continuous time in seconds (biophysical dynamics / BOLD response time)
3. Task gaze autocorrelation decay time tau.
"""
import json
import warnings
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = Path.home() / ".cache/deepmreye"
SWEEP_JSON = REPO / "results/temporal_lag_sweep.json"
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_PNG = FIG_DIR / "temporal_window_absolute_time.png"
FIG_PDF = FIG_DIR / "temporal_window_absolute_time.pdf"

DATASET_CONFIGS = [
    {"id": "dsL01_guided_fixations", "name": "dsL01 (Fixations)", "type": "Fixation", "tr": 0.80, "max_subs": 20},
    {"id": "dsL02_pursuit", "name": "dsL02 (Pursuit)", "type": "Smooth Pursuit", "tr": 0.87, "max_subs": 9},
    {"id": "dsL03_pursuit", "name": "dsL03 (Pursuit)", "type": "Smooth Pursuit", "tr": 1.02, "max_subs": 24},
    {"id": "dsL04_pursuit", "name": "dsL04 (Pursuit)", "type": "Smooth Pursuit", "tr": 1.00, "max_subs": 15},
    {"id": "dsL05_free_viewing", "name": "dsL05 (Free Viewing)", "type": "Free Viewing", "tr": 1.00, "max_subs": 15},
    {"id": "dsL06_sequences", "name": "dsL06 (Sequences)", "type": "Visual Sequences", "tr": 1.80, "max_subs": 6},
    {"id": "dsL07_deepmreye_calib", "name": "dsL07 (Calibration)", "type": "Gaze Calibration", "tr": 1.20, "max_subs": 15},
    {"id": "dsL11_backtothefuture", "name": "dsL11 (Movie)", "type": "Movie Watching", "tr": 1.50, "max_subs": 4},
]

def compute_autocorrelations():
    autocorr_results = {}
    time_lags_sec = np.linspace(0, 15, 60) # 0 to 15 seconds
    
    for cfg in DATASET_CONFIGS:
        ds_dir = DATA_ROOT / cfg["id"]
        if not ds_dir.exists():
            continue
        files = sorted(ds_dir.glob("*.h5"))[:cfg["max_subs"]]
        tr = cfg["tr"]
        
        all_ac_x = []
        all_ac_y = []
        
        for p in files:
            with h5py.File(p, "r") as f:
                labels = f["labels"][:] # [T, 10, 2]
                gaze = labels.reshape(-1, 2)
                dt_sub = tr / 10.0 # sampling interval of sub-TR gaze
                
                # Compute autocorrelation for each coordinate
                for coord_idx in [0, 1]:
                    x = gaze[:, coord_idx]
                    valid = np.isfinite(x)
                    if valid.sum() < 200:
                        continue
                    x_clean = np.interp(np.arange(len(x)), np.where(valid)[0], x[valid])
                    x_c = x_clean - np.mean(x_clean)
                    var = np.var(x_c)
                    if var < 1e-6:
                        continue
                        
                    ac = []
                    for t_sec in time_lags_sec:
                        idx_lag = int(round(t_sec / dt_sub))
                        if idx_lag == 0:
                            ac.append(1.0)
                        elif idx_lag < len(x_c) - 10:
                            c = np.corrcoef(x_c[:-idx_lag], x_c[idx_lag:])[0, 1]
                            ac.append(float(c) if np.isfinite(c) else 0.0)
                        else:
                            ac.append(0.0)
                    if coord_idx == 0:
                        all_ac_x.append(ac)
                    else:
                        all_ac_y.append(ac)
                        
        mean_ac = 0.5 * (np.mean(all_ac_x, axis=0) + np.mean(all_ac_y, axis=0))
        
        # Calculate half-life (time where autocorr drops to 0.5)
        half_life = 0.0
        for i, val in enumerate(mean_ac):
            if val <= 0.5:
                half_life = time_lags_sec[i]
                break
        if half_life == 0.0:
            half_life = 15.0
            
        autocorr_results[cfg["id"]] = {
            "name": cfg["name"],
            "type": cfg["type"],
            "tr": tr,
            "time_sec": time_lags_sec.tolist(),
            "autocorr": mean_ac.tolist(),
            "half_life_sec": float(half_life),
        }
        
    return autocorr_results

def generate_figure(sweep_data, autocorr_data):
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
    
    # Panel A: Task Gaze Autocorrelation Decay in Absolute Time (Seconds)
    ax1 = axes[0, 0]
    for i, cfg in enumerate(DATASET_CONFIGS):
        ds_id = cfg["id"]
        if ds_id in autocorr_data:
            ac = autocorr_data[ds_id]
            c = colors[i % len(colors)]
            ax1.plot(ac["time_sec"], ac["autocorr"], linewidth=2.2, label=f"{ac['name']} ($t_{{1/2}} = {ac['half_life_sec']:.1f}\,$s)", color=c)
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Autocorrelation $t_{1/2}$ ($r = 0.5$)")
    ax1.set_title("A. Gaze Trajectory Autocorrelation vs. Absolute Time (Seconds)", fontweight="bold", pad=10)
    ax1.set_xlabel("Time Lag $\Delta t$ (Seconds)")
    ax1.set_ylabel("Gaze Position Autocorrelation $\\rho(\\Delta t)$")
    ax1.set_xlim(0, 15)
    ax1.set_ylim(-0.1, 1.05)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.92)

    # Panel B: 1-TR Decoding Accuracy vs. Absolute Window Size in Seconds
    ax2 = axes[0, 1]
    optimal_sec = []
    autocorr_halflife = []
    ds_labels = []
    
    for i, res in enumerate(sweep_data):
        ds_id = res["dataset_id"]
        tr = [c["tr"] for c in DATASET_CONFIGS if c["id"] == ds_id][0]
        lags = [int(k) for k in res["lags"].keys()]
        w_sec = [(2 * l + 1) * tr for l in lags]
        r_vals = [res["lags"][str(k)]["1tr_r"] for k in lags]
        best_idx = int(np.argmax(r_vals))
        
        c = colors[i % len(colors)]
        ax2.plot(w_sec, r_vals, marker="o", markersize=6, linewidth=2.2, label=f"{res['dataset_name']} (TR={tr:.2f}s, opt={w_sec[best_idx]:.1f}s)", color=c)
        ax2.scatter([w_sec[best_idx]], [r_vals[best_idx]], s=130, color=c, edgecolors="black", linewidths=1.5, zorder=5)
        
        optimal_sec.append(w_sec[best_idx])
        autocorr_halflife.append(autocorr_data[ds_id]["half_life_sec"])
        ds_labels.append(res["dataset_name"])
        
    ax2.set_title("B. 1-TR Gaze Correlation ($r$) vs. Absolute Window Duration (Seconds)", fontweight="bold", pad=10)
    ax2.set_xlabel("Total Temporal Window Duration $(2L+1) \\times \\mathrm{TR}$ (Seconds)")
    ax2.set_ylabel("Pearson Correlation ($r$)")
    ax2.set_xlim(0, 18)
    ax2.set_ylim(0.70, 0.98)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.92)

    # Panel C: Sub-TR Trajectory Correlation vs. Absolute Window Duration
    ax3 = axes[1, 0]
    for i, res in enumerate(sweep_data):
        ds_id = res["dataset_id"]
        tr = [c["tr"] for c in DATASET_CONFIGS if c["id"] == ds_id][0]
        lags = [int(k) for k in res["lags"].keys()]
        w_sec = [(2 * l + 1) * tr for l in lags]
        r_vals = [res["lags"][str(k)]["subtr_r"] for k in lags]
        best_idx = int(np.argmax(r_vals))
        
        c = colors[i % len(colors)]
        ax3.plot(w_sec, r_vals, marker="s", markersize=6, linewidth=2.2, label=f"{res['dataset_name']} (opt={w_sec[best_idx]:.1f}s)", color=c)
        ax3.scatter([w_sec[best_idx]], [r_vals[best_idx]], s=130, color=c, edgecolors="black", linewidths=1.5, zorder=5)
        
    ax3.set_title("C. Sub-TR Continuous Trajectory ($r$) vs. Absolute Window Duration (Seconds)", fontweight="bold", pad=10)
    ax3.set_xlabel("Total Temporal Window Duration $(2L+1) \\times \\mathrm{TR}$ (Seconds)")
    ax3.set_ylabel("Pearson Correlation ($r$)")
    ax3.set_xlim(0, 18)
    ax3.set_ylim(0.68, 0.98)
    ax3.legend(loc="lower right", frameon=True, framealpha=0.92)

    # Panel D: Optimal Temporal Window vs. Task Autocorrelation Decay
    ax4 = axes[1, 1]
    for i in range(len(optimal_sec)):
        c = colors[i % len(colors)]
        ax4.scatter(autocorr_halflife[i], optimal_sec[i], s=180, color=c, edgecolors="black", linewidths=1.5, zorder=5)
        ax4.annotate(ds_labels[i], (autocorr_halflife[i] + 0.3, optimal_sec[i] - 0.2), fontsize=9.5, fontweight="bold")
        
    # Fit regression line between Autocorrelation Half-Life and Optimal Window Duration
    m, b = np.polyfit(autocorr_halflife, optimal_sec, 1)
    x_line = np.linspace(0, 16, 50)
    ax4.plot(x_line, m * x_line + b, color="#333333", linestyle="--", linewidth=2.0, label=f"Linear Fit ($r = {np.corrcoef(autocorr_halflife, optimal_sec)[0, 1]:.2f}$)")
    
    # Highlight the 2-to-5s biophysical window band
    ax4.axhspan(2.0, 5.5, color="#1f77b4", alpha=0.10, label="Standard fMRI HRF / Fixation Window (2–5.5 s)")
    
    ax4.set_title("D. Optimal Window Duration vs. Task Gaze Autocorrelation $t_{1/2}$", fontweight="bold", pad=10)
    ax4.set_xlabel("Task Gaze Autocorrelation Half-Life $t_{1/2}$ (Seconds)")
    ax4.set_ylabel("Optimal Temporal Window Duration (Seconds)")
    ax4.set_xlim(0, 16)
    ax4.set_ylim(0, 13)
    ax4.legend(loc="upper left", frameon=True, framealpha=0.92)

    plt.suptitle("DeepMReye 2.0: Biophysical & Task Autocorrelation Basis of Optimal Temporal Window Width", y=0.99, fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    plt.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(FIG_PDF, bbox_inches="tight")
    print(f"[+] Saved figure to {FIG_PNG} and {FIG_PDF}", flush=True)

def main():
    with open(SWEEP_JSON) as f:
        sweep_data = json.load(f)
    print("[*] Computing ground truth gaze autocorrelation curves...", flush=True)
    autocorr_data = compute_autocorrelations()
    generate_figure(sweep_data, autocorr_data)

if __name__ == "__main__":
    main()
