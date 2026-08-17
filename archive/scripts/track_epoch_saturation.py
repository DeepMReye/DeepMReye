#!/usr/bin/env python3
"""Track Epoch-by-Epoch Saturation Curve of PyTorch CompositeNet with 3-Way Comparison.

Strict Zero-Shot Holdout: dsL11 is 100% excluded from PyTorch training.
Compares:
1. fold-pca:64 (Unsupervised PCA baseline)
2. composite-basis:64 (Manual PCA + PLS basis)
3. composite-net (PyTorch joint multi-task model)
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepmreye.datasource import resolve
from deepmreye.data.probe_dataset import ProbeDataset
from deepmreye.evaluate.probe import temporal_targets
from deepmreye.models.composite_net import CompositeNet, save_composite_net
from deepmreye.unsupervised import load_basis


def pool_entire_run_masked(block, per_bin, mask_flat):
    """Pool 3D+T block [X, Y, Z, W_total] into 5-TR bins -> [n_bins_total, 14236]."""
    w_total = block.shape[-1]
    n_bins_total = w_total // per_bin
    flat = block.reshape(-1, w_total)[:, :n_bins_total * per_bin]
    pooled = flat.reshape(-1, n_bins_total, per_bin).mean(axis=2).T
    return pooled[:, mask_flat].astype(np.float32)


class PrepooledRunDataset(Dataset):
    """Pre-pooled whole-run cached dataset yielding instant [20, 14236] slices."""

    def __init__(self, probe_dataset, per_bin, mask_flat):
        self.samples = probe_dataset.samples
        self.window_size = probe_dataset.window_size
        self.per_bin = per_bin
        self.n_t = self.window_size // per_bin
        self._cache = {}

        paths = sorted({s["path"] for s in self.samples})
        t0 = time.time()
        for p in paths:
            try:
                with h5py.File(p, "r") as f:
                    block = f["eye_block"][...]
                    labels = f["labels"][...]
                    pooled_run = pool_entire_run_masked(block, per_bin, mask_flat)
                    self._cache[p] = (pooled_run, labels)
            except Exception:
                pass
        print(f"[*] Pre-pooled {len(self._cache)} participant runs in {time.time() - t0:.1f}s", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        p, start = s["path"], s["start"]
        end = start + self.window_size
        pooled_run, labels = self._cache[p]

        start_bin = start // self.per_bin
        x_win = pooled_run[start_bin:start_bin + self.n_t]

        if x_win.shape[0] < self.n_t:
            pad = self.n_t - x_win.shape[0]
            x_win = np.pad(x_win, ((0, pad), (0, 0)), mode="edge")

        x_tensor = torch.from_numpy(x_win).float()
        y_tensor = torch.from_numpy(labels[start:end]).float()
        return x_tensor, y_tensor, s["dataset"], s["subject"], torch.tensor(s["tr"], dtype=torch.float32)


def run_cmd(cmd):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[!] Error: {res.stderr}")
    return res.stdout


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--basis", default="results/corpus_basis.npz")
    p.add_argument("--out-dir", default="results/epoch_saturation")
    p.add_argument("--bottleneck", type=int, default=96)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--window-size", type=int, default=100)
    p.add_argument("--temp-patch-size", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = resolve(args.data_dir, download=False, quiet=True)
    mask, bases, bmeta = load_basis(args.basis)
    mask_flat = mask.reshape(-1)
    n_voxels = int(mask_flat.sum())

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    n_t = args.window_size // args.temp_patch_size

    print(f"[*] Loaded eye mask: {n_voxels} voxels", flush=True)

    # STRICT ZERO-SHOT HOLDOUT ON dsL11 (dsL11 is 100% excluded from train_ds)
    holdout_set = {"dsL11_backtothefuture"}
    train_ds = PrepooledRunDataset(ProbeDataset(labeled_data_dir=data_dir, split="train", window_size=args.window_size, holdout=holdout_set), args.temp_patch_size, mask_flat)
    val_ds = PrepooledRunDataset(ProbeDataset(labeled_data_dir=data_dir, split="test", window_size=args.window_size, holdout=holdout_set), args.temp_patch_size, mask_flat)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"[*] Training CompositeNet (bottleneck={args.bottleneck}, alpha={args.alpha}) on pre-pooled PyTorch DataLoader ({device})...", flush=True)

    model = CompositeNet(n_voxels=n_voxels, bottleneck_dim=args.bottleneck, alpha=args.alpha).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    checkpoints_to_eval = []

    # Phase 1: Train all 20 epochs continuously on GPU (~4s total)
    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, train_gaze, train_recon, total_train_samples = 0.0, 0.0, 0.0, 0

        for x, y, _ds, _sub, _tr in train_loader:
            B, T, V = x.shape
            bx = x.view(-1, V).to(device)

            y_binned = temporal_targets(y, n_t)
            by = torch.from_numpy(y_binned.reshape(-1, 2)).float().to(device)
            bm = ~torch.isnan(by).any(dim=1)

            optimizer.zero_grad()
            loss, lgaze, lrecon = model.compute_loss(bx, by, valid_mask=bm)
            loss.backward()
            optimizer.step()

            n_b = len(bx)
            train_loss += loss.item() * n_b
            train_gaze += lgaze.item() * n_b
            train_recon += lrecon.item() * n_b
            total_train_samples += n_b

        scheduler.step()
        train_loss /= total_train_samples
        train_gaze /= total_train_samples
        train_recon /= total_train_samples

        # Validation pass
        model.eval()
        val_loss, val_gaze, val_recon, total_val_samples = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for x, y, _ds, _sub, _tr in val_loader:
                B, T, V = x.shape
                bx = x.view(-1, V).to(device)

                y_binned = temporal_targets(y, n_t)
                by = torch.from_numpy(y_binned.reshape(-1, 2)).float().to(device)
                bm = ~torch.isnan(by).any(dim=1)

                loss, lgaze, lrecon = model.compute_loss(bx, by, valid_mask=bm)

                n_b = len(bx)
                val_loss += loss.item() * n_b
                val_gaze += lgaze.item() * n_b
                val_recon += lrecon.item() * n_b
                total_val_samples += n_b

        val_loss /= max(1, total_val_samples)
        val_gaze /= max(1, total_val_samples)
        val_recon /= max(1, total_val_samples)

        print(f"Epoch {epoch:02d}/{args.max_epochs:02d} [{time.time() - t0:.2f}s] -- Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} (Gaze: {val_gaze:.4f}, Recon: {val_recon:.4f})", flush=True)

        if epoch % args.eval_every == 0 or epoch == args.max_epochs:
            ckpt_path = out_dir / f"checkpoint_epoch_{epoch:02d}.pt"
            save_composite_net(model, str(ckpt_path), metadata={"epoch": epoch, "val_loss": val_loss})
            checkpoints_to_eval.append((epoch, ckpt_path, val_loss, val_gaze, val_recon))

    # Phase 2: Evaluate saved checkpoints on dsL11 comparing CompositeNet vs PCA vs PCA+PLS
    print(f"\n[*] Evaluating {len(checkpoints_to_eval)} saved checkpoints on dsL11 (3-way comparison)...", flush=True)
    trajectory = []

    for epoch, ckpt_path, val_loss, val_gaze, val_recon in checkpoints_to_eval:
        eval_path = out_dir / f"eval_epoch_{epoch:02d}.json"
        eval_cmd = (
            f"uv run python scripts/eval_probe.py --protocol dataset "
            f"--features composite-net fold-pca:32+fold-pls:32 fold-pca:64 "
            f"--composite-checkpoint {ckpt_path} --fold-name dsL11 "
            f"--readouts ridge-cv --standardize-targets dataset --out {eval_path}"
        )
        t_eval = time.time()
        run_cmd(eval_cmd)

        rx, ry, r_mean = 0.0, 0.0, 0.0
        basis_rx, basis_ry, basis_mean = 0.0, 0.0, 0.0
        pca_rx, pca_ry, pca_mean = 0.0, 0.0, 0.0

        if eval_path.exists():
            data = json.loads(eval_path.read_text())
            dsl11 = data.get("dsL11_backtothefuture", {})
            comp = dsl11.get("composite-net/ridge-cv", {}).get("by_subject", {})
            basis = dsl11.get("composite-basis:64/ridge-cv", {}).get("by_subject", {})
            pca = dsl11.get("fold-pca:64/ridge-cv", {}).get("by_subject", {})

            rx = comp.get("pearson_r_x", 0.0)
            ry = comp.get("pearson_r_y", 0.0)
            r_mean = 0.5 * (rx + ry)

            basis_rx = basis.get("pearson_r_x", 0.0)
            basis_ry = basis.get("pearson_r_y", 0.0)
            basis_mean = 0.5 * (basis_rx + basis_ry)

            pca_rx = pca.get("pearson_r_x", 0.0)
            pca_ry = pca.get("pearson_r_y", 0.0)
            pca_mean = 0.5 * (pca_rx + pca_ry)

        print(f"Eval Epoch {epoch:02d} [{time.time() - t_eval:.1f}s]: Net={r_mean:.3f} | PCA+PLS={basis_mean:.3f} | PCA={pca_mean:.3f} (vs PCA={r_mean - pca_mean:+.3f})", flush=True)

        trajectory.append({
            "epoch": epoch,
            "val_loss": val_loss,
            "val_gaze_loss": val_gaze,
            "val_recon_loss": val_recon,
            "r_x": rx,
            "r_y": ry,
            "r_mean": r_mean,
            "basis_r_mean": basis_mean,
            "pca_r_mean": pca_mean,
            "diff_vs_pca": r_mean - pca_mean,
            "diff_vs_basis": r_mean - basis_mean
        })

    print("\n" + "=" * 90, flush=True)
    print(f"{'Epoch':<8} {'r_x':<8} {'r_y':<8} {'CompositeNet':<14} {'PCA+PLS Basis':<16} {'PCA Baseline':<14} {'vs PCA':<8}", flush=True)
    print("-" * 90, flush=True)
    for r in trajectory:
        print(f"{r['epoch']:<8d} {r['r_x']:<8.3f} {r['r_y']:<8.3f} "
              f"{r['r_mean']:<14.3f} {r['basis_r_mean']:<16.3f} {r['pca_r_mean']:<14.3f} {r['diff_vs_pca']:+8.3f}", flush=True)
    print("=" * 90, flush=True)

    (out_dir / "saturation_results_3way.json").write_text(json.dumps(trajectory, indent=2))
    print(f"\n[*] 3-way saturation trajectory saved to {out_dir / 'saturation_results_3way.json'}", flush=True)


if __name__ == "__main__":
    main()
