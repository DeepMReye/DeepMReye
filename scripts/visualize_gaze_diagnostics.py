#!/usr/bin/env python3
"""Gaze fMRI Diagnostic Visualization Suite.

Generates 5 core visual diagnostics suggested by Opus across all 6 gaze-labeled datasets
(dsL01 through dsL06):
1. Condition-Difference Maps (Top 20% vs Bottom 20% Gaze difference dipole maps).
2. Voxel-Wise Correlation Maps (Hotspots overlaid on mean eyeball anatomy).
3. Lag & Cross-Correlation Profiles (Lags -5 to +5 TRs per dataset).
4. 2D PC Projections & Domain Shift Scatter Plots (PC1 vs PC2 by gaze_x, gaze_y, and dataset).
5. Eyeball Motion Movie (GIF animation with synchronized gaze cursor).
"""
import argparse
from pathlib import Path
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from tqdm import tqdm

from deepmreye.datasource import resolve

DATASETS = [
    "dsL01_guided_fixations",
    "dsL02_pursuit",
    "dsL03_pursuit",
    "dsL04_pursuit",
    "dsL05_free_viewing",
    "dsL06_sequences",
]

def load_dataset_data(data_dir, ds_name, max_subjects=10):
    """Load voxels [N*T, 47, 29, 18] and gaze targets [N*T, 2] for a dataset."""
    pattern = str(Path(data_dir) / ds_name / "*.h5")
    files = glob.glob(pattern)[:max_subjects]
    
    all_vols = []
    all_gaze = []
    
    for fpath in files:
        try:
            with h5py.File(fpath, "r") as f:
                if "eye_block" not in f or "labels" not in f:
                    continue
                block = f["eye_block"][()] # [47, 29, 18, T]
                labels = f["labels"][()]   # [T, 10, 2]
                
                if block.ndim != 4:
                    continue
                
                # Mean over sub-TRs to get per-TR gaze target [T, 2]
                with np.errstate(invalid="ignore"):
                    gaze = np.nanmean(labels, axis=1) # [T, 2]
                
                # Check for NaNs and drop missing timepoints
                valid_mask = ~np.isnan(gaze[:, 0]) & ~np.isnan(gaze[:, 1])
                if not np.any(valid_mask):
                    continue
                
                vols = np.moveaxis(block, -1, 0)[valid_mask] # [T_valid, 47, 29, 18]
                gaze = gaze[valid_mask]                      # [T_valid, 2]
                
                all_vols.append(vols)
                all_gaze.append(gaze)
        except Exception as e:
            print(f"[-] Warning loading {fpath}: {e}")
            continue
            
    if not all_vols:
        return None, None
        
    return np.concatenate(all_vols, axis=0), np.concatenate(all_gaze, axis=0)

def plot_condition_difference_maps(dataset_data, output_dir):
    """1. Condition-Difference Dipole Maps (Top 20% vs Bottom 20% Gaze_x and Gaze_y)."""
    print("[*] Generating 1. Condition-Difference Dipole Maps...")
    fig, axes = plt.subplots(len(DATASETS), 4, figsize=(16, 3 * len(DATASETS)))
    plt.suptitle("Condition-Difference Dipole Maps ($\Delta$ Top 20% - Bottom 20% Gaze)", fontsize=16, y=0.995)

    for i, ds in enumerate(DATASETS):
        vols, gaze = dataset_data.get(ds, (None, None))
        if vols is None:
            continue

        gaze_x, gaze_y = gaze[:, 0], gaze[:, 1]
        
        # Gaze X top/bottom quintiles
        hi_x = gaze_x > np.percentile(gaze_x, 80)
        lo_x = gaze_x < np.percentile(gaze_x, 20)
        diff_x = vols[hi_x].mean(axis=0) - vols[lo_x].mean(axis=0)

        # Gaze Y top/bottom quintiles
        hi_y = gaze_y > np.percentile(gaze_y, 80)
        lo_y = gaze_y < np.percentile(gaze_y, 20)
        diff_y = vols[hi_y].mean(axis=0) - vols[lo_y].mean(axis=0)

        # Best axial slice for eyes (z ~ 8-10)
        mean_anat = vols.mean(axis=0)
        z_slice = 9

        # Axial view (left/right eyes)
        vmax_x = np.percentile(np.abs(diff_x[:, :, z_slice]), 99) or 1.0
        vmax_y = np.percentile(np.abs(diff_y[:, :, z_slice]), 99) or 1.0

        # Subplot 1: Mean Anat
        axes[i, 0].imshow(mean_anat[:, :, z_slice].T, cmap="gray", origin="lower")
        axes[i, 0].set_title(f"{ds.split('_')[0]}\nMean Anat (Axial z={z_slice})", fontsize=10)
        axes[i, 0].axis("off")

        # Subplot 2: Diff X (Axial)
        im_x = axes[i, 1].imshow(diff_x[:, :, z_slice].T, cmap="seismic", vmin=-vmax_x, vmax=vmax_x, origin="lower")
        axes[i, 1].set_title(f"$\Delta$ Gaze X (Left vs Right)", fontsize=10)
        plt.colorbar(im_x, ax=axes[i, 1], fraction=0.046, pad=0.04)
        axes[i, 1].axis("off")

        # Subplot 3: Diff Y (Axial)
        im_y = axes[i, 2].imshow(diff_y[:, :, z_slice].T, cmap="seismic", vmin=-vmax_y, vmax=vmax_y, origin="lower")
        axes[i, 2].set_title(f"$\Delta$ Gaze Y (Up vs Down)", fontsize=10)
        plt.colorbar(im_y, ax=axes[i, 2], fraction=0.046, pad=0.04)
        axes[i, 2].axis("off")

        # Subplot 4: Sagittal View of Eye (x ~ 14)
        x_eye = 14
        vmax_sag = np.percentile(np.abs(diff_x[x_eye, :, :]), 99) or 1.0
        im_sag = axes[i, 3].imshow(diff_x[x_eye, :, :].T, cmap="seismic", vmin=-vmax_sag, vmax=vmax_sag, origin="lower")
        axes[i, 3].set_title(f"$\Delta$ Gaze X (Sagittal x={x_eye})", fontsize=10)
        plt.colorbar(im_sag, ax=axes[i, 3], fraction=0.046, pad=0.04)
        axes[i, 3].axis("off")

    plt.tight_layout()
    out_path = output_dir / "01_condition_difference_maps.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved {out_path}")

def plot_voxel_correlation_maps(dataset_data, output_dir):
    """2. Voxel-wise Pearson Correlation Hotspot Maps."""
    print("[*] Generating 2. Voxel-wise Correlation Maps...")
    fig, axes = plt.subplots(len(DATASETS), 3, figsize=(14, 3 * len(DATASETS)))
    plt.suptitle("Voxel-wise Gaze Correlation Hotspots ($|r| > 0.15$ overlaid on eyeball anatomy)", fontsize=16, y=0.995)

    for i, ds in enumerate(DATASETS):
        vols, gaze = dataset_data.get(ds, (None, None))
        if vols is None:
            continue

        n_t, nx, ny, nz = vols.shape
        vols_flat = vols.reshape(n_t, -1) # [T, V]
        
        # Standardize gaze
        gx = (gaze[:, 0] - gaze[:, 0].mean()) / (gaze[:, 0].std() + 1e-8)
        gy = (gaze[:, 1] - gaze[:, 1].mean()) / (gaze[:, 1].std() + 1e-8)
        
        # Fast batch correlation
        v_mean = vols_flat.mean(axis=0)
        v_std = vols_flat.std(axis=0) + 1e-8
        v_norm = (vols_flat - v_mean) / v_std
        
        r_x = (v_norm.T @ gx) / n_t
        r_y = (v_norm.T @ gy) / n_t
        
        map_x = r_x.reshape(nx, ny, nz)
        map_y = r_y.reshape(nx, ny, nz)
        
        z_slice = 9
        mean_anat = vols.mean(axis=0)[:, :, z_slice].T

        # Anat
        axes[i, 0].imshow(mean_anat, cmap="gray", origin="lower")
        axes[i, 0].set_title(f"{ds.split('_')[0]} Anat (z={z_slice})", fontsize=10)
        axes[i, 0].axis("off")

        # Gaze X corr overlay
        axes[i, 1].imshow(mean_anat, cmap="gray", origin="lower")
        rx_slice = map_x[:, :, z_slice].T
        im_rx = axes[i, 1].imshow(np.ma.masked_where(np.abs(rx_slice) < 0.15, rx_slice), cmap="bwr", vmin=-0.5, vmax=0.5, origin="lower")
        axes[i, 1].set_title(f"Corr(Voxel, Gaze X) [Max |r|={np.max(np.abs(r_x)):.2f}]", fontsize=10)
        plt.colorbar(im_rx, ax=axes[i, 1], fraction=0.046, pad=0.04)
        axes[i, 1].axis("off")

        # Gaze Y corr overlay
        axes[i, 2].imshow(mean_anat, cmap="gray", origin="lower")
        ry_slice = map_y[:, :, z_slice].T
        im_ry = axes[i, 2].imshow(np.ma.masked_where(np.abs(ry_slice) < 0.15, ry_slice), cmap="bwr", vmin=-0.5, vmax=0.5, origin="lower")
        axes[i, 2].set_title(f"Corr(Voxel, Gaze Y) [Max |r|={np.max(np.abs(r_y)):.2f}]", fontsize=10)
        plt.colorbar(im_ry, ax=axes[i, 2], fraction=0.046, pad=0.04)
        axes[i, 2].axis("off")

    plt.tight_layout()
    out_path = output_dir / "02_voxel_correlation_maps.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved {out_path}")

def plot_cross_correlation_lags(dataset_data, output_dir):
    """3. Cross-Correlation Lag Profiles (-5 to +5 TRs)."""
    print("[*] Generating 3. Cross-Correlation Lag Profiles...")
    lags = np.arange(-5, 6)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cross-Correlation Lag Profiles (Eyeball Voxel Signal vs Gaze at Lags -5 to +5 TRs)", fontsize=14)

    colors = plt.cm.Set1(np.linspace(0, 1, len(DATASETS)))

    for i, ds in enumerate(DATASETS):
        vols, gaze = dataset_data.get(ds, (None, None))
        if vols is None:
            continue

        n_t, nx, ny, nz = vols.shape
        vols_flat = vols.reshape(n_t, -1)
        v_norm = (vols_flat - vols_flat.mean(axis=0)) / (vols_flat.std(axis=0) + 1e-8)
        
        gx_raw = gaze[:, 0]
        gy_raw = gaze[:, 1]
        
        top_x_lags = []
        top_y_lags = []

        for lag in lags:
            if lag < 0:
                v_sub = v_norm[-lag:]
                gx_sub = gx_raw[:lag]
                gy_sub = gy_raw[:lag]
            elif lag > 0:
                v_sub = v_norm[:-lag]
                gx_sub = gx_raw[lag:]
                gy_sub = gy_raw[lag:]
            else:
                v_sub = v_norm
                gx_sub = gx_raw
                gy_sub = gy_raw

            gx_norm = (gx_sub - gx_sub.mean()) / (gx_sub.std() + 1e-8)
            gy_norm = (gy_sub - gy_sub.mean()) / (gy_sub.std() + 1e-8)

            r_x = np.abs((v_sub.T @ gx_norm) / len(gx_sub))
            r_y = np.abs((v_sub.T @ gy_norm) / len(gy_sub))

            # Mean of top 50 correlated voxels
            top_x_lags.append(np.mean(np.sort(r_x)[-50:]))
            top_y_lags.append(np.mean(np.sort(r_y)[-50:]))

        ds_label = ds.split('_')[0]
        axes[0].plot(lags, top_x_lags, "o-", label=ds_label, color=colors[i], linewidth=2)
        axes[1].plot(lags, top_y_lags, "o-", label=ds_label, color=colors[i], linewidth=2)

    axes[0].axvline(0, color="gray", linestyle="--", alpha=0.7)
    axes[0].set_xlabel("Lag (TRs)")
    axes[0].set_ylabel("Mean |r| of Top 50 Voxels")
    axes[0].set_title("Gaze X Cross-Correlation Lag Profile")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].axvline(0, color="gray", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("Lag (TRs)")
    axes[1].set_ylabel("Mean |r| of Top 50 Voxels")
    axes[1].set_title("Gaze Y Cross-Correlation Lag Profile")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    out_path = output_dir / "03_cross_correlation_lags.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved {out_path}")

def plot_2d_pc_projections(dataset_data, output_dir):
    """4. 2D PC Projections & Dataset Domain Shift Scatter Plot."""
    print("[*] Generating 4. 2D PC Projections & Domain Shift...")
    
    all_feats = []
    all_gx = []
    all_gy = []
    all_ds = []

    for ds in DATASETS:
        vols, gaze = dataset_data.get(ds, (None, None))
        if vols is None:
            continue
        
        # Subsample max 500 points per dataset for scatter clean plot
        n_pts = min(len(vols), 500)
        idx = np.random.choice(len(vols), n_pts, replace=False)
        
        all_feats.append(vols[idx].reshape(n_pts, -1))
        all_gx.append(gaze[idx, 0])
        all_gy.append(gaze[idx, 1])
        all_ds.extend([ds.split('_')[0]] * n_pts)

    if not all_feats:
        return

    X = np.vstack(all_feats)
    gx = np.concatenate(all_gx)
    gy = np.concatenate(all_gy)
    ds_arr = np.array(all_ds)

    # Standardize & Fit PCA(2)
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    pca = PCA(n_components=2, random_state=42)
    pc = pca.fit_transform(X_std)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"2D PC Space Projection (EVR: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%})", fontsize=15)

    # Subplot 1: PC1 vs PC2 colored by Gaze X
    sc1 = axes[0].scatter(pc[:, 0], pc[:, 1], c=gx, cmap="viridis", s=15, alpha=0.7)
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")
    axes[0].set_title("Colored by True Gaze X")
    plt.colorbar(sc1, ax=axes[0], label="Gaze X (deg)")

    # Subplot 2: PC1 vs PC2 colored by Gaze Y
    sc2 = axes[1].scatter(pc[:, 0], pc[:, 1], c=gy, cmap="plasma", s=15, alpha=0.7)
    axes[1].set_xlabel("PC 1")
    axes[1].set_ylabel("PC 2")
    axes[1].set_title("Colored by True Gaze Y")
    plt.colorbar(sc2, ax=axes[1], label="Gaze Y (deg)")

    # Subplot 3: PC1 vs PC2 colored by Dataset (Domain Shift)
    unique_ds = np.unique(ds_arr)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_ds)))
    for j, d in enumerate(unique_ds):
        mask = (ds_arr == d)
        axes[2].scatter(pc[mask, 0], pc[mask, 1], color=colors[j], label=d, s=15, alpha=0.7)
    axes[2].set_xlabel("PC 1")
    axes[2].set_ylabel("PC 2")
    axes[2].set_title("Colored by Dataset (Domain Shift Visualizer)")
    axes[2].legend()

    plt.tight_layout()
    out_path = output_dir / "04_pc_projection_domain_shift.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved {out_path}")

def generate_eyeball_gaze_movie(dataset_data, output_dir):
    """5. Eyeball Slice Animation with Moving Gaze Cursor."""
    print("[*] Generating 5. Eyeball Motion Movie GIF...")
    vols, gaze = dataset_data.get("dsL01_guided_fixations", (None, None))
    if vols is None or len(vols) < 50:
        return

    n_frames = 40
    clip_vols = vols[:n_frames]
    clip_gaze = gaze[:n_frames]
    z_slice = 9

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [2, 1]})

    # Top: Eyeball axial slice
    im = ax1.imshow(clip_vols[0, :, :, z_slice].T, cmap="gray", origin="lower")
    ax1.set_title("Eyeball BOLD Axial Slice (z=9)", fontsize=12)
    ax1.axis("off")

    # Bottom: 2D Gaze Trajectory
    ax2.plot(clip_gaze[:, 0], clip_gaze[:, 1], "k-", alpha=0.3, label="Gaze Path")
    cursor, = ax2.plot([], [], "ro", markersize=10, label="Current Gaze")
    ax2.set_xlim(np.min(clip_gaze[:, 0]) - 2, np.max(clip_gaze[:, 0]) + 2)
    ax2.set_ylim(np.min(clip_gaze[:, 1]) - 2, np.max(clip_gaze[:, 1]) + 2)
    ax2.set_xlabel("Gaze X (deg)")
    ax2.set_ylabel("Gaze Y (deg)")
    ax2.set_title("Synchronized Gaze Target Position")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    def update(frame):
        im.set_array(clip_vols[frame, :, :, z_slice].T)
        cursor.set_data([clip_gaze[frame, 0]], [clip_gaze[frame, 1]])
        ax1.set_title(f"Eyeball BOLD Axial Slice (TR={frame})", fontsize=12)
        return im, cursor

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=250, blit=True)
    out_path = output_dir / "05_eyeball_gaze_movie.gif"
    anim.save(out_path, writer="pillow", fps=4)
    plt.close()
    print(f"[+] Saved {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Gaze fMRI Diagnostic Visualizations")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, help="Datasets to process")
    parser.add_argument("--output-dir", type=str, default="media/visualizations", help="Output directory")
    args = parser.parse_args()

    data_dir = resolve(download=False, quiet=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading data for datasets: {args.datasets}")
    dataset_data = {}
    for ds in tqdm(args.datasets, desc="Loading datasets"):
        vols, gaze = load_dataset_data(data_dir, ds, max_subjects=10)
        if vols is not None:
            dataset_data[ds] = (vols, gaze)
            print(f"[+] Loaded {ds}: {vols.shape[0]} TRs, vols {vols.shape}")

    # Generate all 5 visual diagnostics
    plot_condition_difference_maps(dataset_data, output_dir)
    plot_voxel_correlation_maps(dataset_data, output_dir)
    plot_cross_correlation_lags(dataset_data, output_dir)
    plot_2d_pc_projections(dataset_data, output_dir)
    generate_eyeball_gaze_movie(dataset_data, output_dir)

    print("\n[+] All 5 diagnostic visualizations generated successfully in:", output_dir)

if __name__ == "__main__":
    main()
