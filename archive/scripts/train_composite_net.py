#!/usr/bin/env python3
"""Train PyTorch CompositeNet (Joint Autoencoding + Gaze Decoding Spatial Bottleneck).

Trains an end-to-end differentiable neural network that optimizes both:
1. Gaze prediction MSE loss L_gaze (supervised PLS-like direction matching)
2. Voxel reconstruction MSE loss L_recon (unsupervised PCA-like spatial variance retention)

Example usage:
    python scripts/train_composite_net.py --epochs 20 --bottleneck 96 --alpha 0.1 --out results/composite_net.pt
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepmreye.datasource import resolve
from deepmreye.data.probe_dataset import ProbeDataset
from deepmreye.models.composite_net import CompositeNet, save_composite_net


from deepmreye.evaluate.probe import temporal_targets


def pool_time(x, n_t):
    """Average pool voxel time-series x over temporal window sub-intervals."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    b, h, w, d, t = x.shape
    x_flat = x.reshape(-1, t)
    t_pool = t // n_t
    x_pooled = x_flat[:, :n_t * t_pool].reshape(-1, n_t, t_pool).mean(axis=2)
    return x_pooled.T.reshape(n_t, b, -1).swapaxes(0, 1)


def extract_data_matrix(dataset, window_size=100, temp_patch_size=5):
    """Extract flat voxel activity matrix X [N, V] and target gaze Y [N, 2]."""
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    n_t = window_size // temp_patch_size

    voxel_rows = []
    gaze_rows = []

    for x, y, _ds, _sub, _tr in tqdm(loader, desc="Extracting voxel windows"):
        x_pooled = pool_time(x, n_t)  # [B, n_t, V]
        y_pooled = temporal_targets(y, n_t)  # [B, n_t, 2]

        B, T, V = x_pooled.shape
        voxel_rows.append(x_pooled.reshape(-1, V))
        gaze_rows.append(y_pooled.reshape(-1, 2))

    X = np.concatenate(voxel_rows, axis=0).astype(np.float32)
    Y = np.concatenate(gaze_rows, axis=0).astype(np.float32)
    return X, Y


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out", default="results/composite_net.pt")
    p.add_argument("--bottleneck", type=int, default=96, help="Spatial bottleneck dimension.")
    p.add_argument("--alpha", type=float, default=0.1, help="Reconstruction loss multiplier.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--window-size", type=int, default=100)
    p.add_argument("--temp-patch-size", type=int, default=5)
    p.add_argument("--exclude-datasets", nargs="*", default=())
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = resolve(args.data_dir, download=False, quiet=True)
    print(f"[*] data directory: {data_dir}")

    # Discover and load training dataset
    train_ds = ProbeDataset(labeled_data_dir=data_dir, split="train", window_size=args.window_size)
    print(f"[*] Discovering voxel windows from labeled dataset split...")
    X, Y = extract_data_matrix(train_ds, window_size=args.window_size, temp_patch_size=args.temp_patch_size)

    # Valid target mask (non-NaN gaze)
    valid_mask = ~np.isnan(Y).any(axis=1)
    print(f"[*] Total dataset rows: {X.shape[0]} voxels x {X.shape[1]} dims. Valid gaze rows: {valid_mask.sum()}")

    # Train / Val Split (85% train, 15% validation)
    n_samples = len(X)
    perm = np.random.permutation(n_samples)
    n_val = int(n_samples * 0.15)
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    X_train, Y_train, M_train = X[train_idx], Y[train_idx], valid_mask[train_idx]
    X_val, Y_val, M_val = X[val_idx], Y[val_idx], valid_mask[val_idx]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Training PyTorch CompositeNet model on {device} (bottleneck={args.bottleneck}, alpha={args.alpha})...")

    model = CompositeNet(n_voxels=X.shape[1], bottleneck_dim=args.bottleneck, alpha=args.alpha).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # PyTorch DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train), torch.from_numpy(M_train)),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val), torch.from_numpy(M_val)),
        batch_size=args.batch_size, shuffle=False
    )

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, train_gaze, train_recon = 0.0, 0.0, 0.0

        for bx, by, bm in train_loader:
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)
            optimizer.zero_grad()

            loss, lgaze, lrecon = model.compute_loss(bx, by, bm)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(bx)
            train_gaze += lgaze.item() * len(bx)
            train_recon += lrecon.item() * len(bx)

        scheduler.step()
        train_loss /= len(X_train)
        train_gaze /= len(X_train)
        train_recon /= len(X_train)

        # Validation
        model.eval()
        val_loss, val_gaze, val_recon = 0.0, 0.0, 0.0
        with torch.no_grad():
            for bx, by, bm in val_loader:
                bx, by, bm = bx.to(device), by.to(device), bm.to(device)
                loss, lgaze, lrecon = model.compute_loss(bx, by, bm)
                val_loss += loss.item() * len(bx)
                val_gaze += lgaze.item() * len(bx)
                val_recon += lrecon.item() * len(bx)

        val_loss /= len(X_val)
        val_gaze /= len(X_val)
        val_recon /= len(X_val)

        print(f"Epoch {epoch:02d}/{args.epochs:02d} [{time.time() - t0:.1f}s] -- "
              f"Train Loss: {train_loss:.4f} (Gaze: {train_gaze:.4f}, Recon: {train_recon:.4f}) | "
              f"Val Loss: {val_loss:.4f} (Gaze: {val_gaze:.4f}, Recon: {val_recon:.4f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            save_composite_net(model, args.out, metadata={
                "bottleneck_dim": args.bottleneck,
                "alpha": args.alpha,
                "val_loss": val_loss,
                "val_gaze_loss": val_gaze,
                "val_recon_loss": val_recon,
                "epochs": epoch
            })

    print(f"\n[*] Training complete! Best validation loss: {best_val_loss:.4f}. Saved model to {args.out}")


if __name__ == "__main__":
    main()
