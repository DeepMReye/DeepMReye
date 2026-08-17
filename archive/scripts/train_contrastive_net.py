#!/usr/bin/env python3
"""Pre-trains ContrastiveNet on the unlabeled fMRI corpus (1,005 subjects, 14,236 voxels).

Uses VICReg (Barlow Twins) temporal contrastive loss + Masked Autoencoding (MAE).
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.datasource import resolve
from deepmreye.models.contrastive_net import ContrastiveNet, save_contrastive_net
from deepmreye.unsupervised import FULL_MASK_VOXELS, corpus_mask, unlabeled_subjects


class UnlabeledMRIDataset(Dataset):

    """PyTorch Dataset that loads fMRI voxel volumes and generates temporal pairs (x_t, x_{t+dt})."""

    def __init__(self, subject_files, mask, dt_max=2):
        self.mask_flat = mask.reshape(-1)
        self.dt_max = dt_max
        self.samples = []

        logging.info(f"Loading voxel volumes across {len(subject_files)} unlabeled subjects...")
        for ds, sub, path, n_trs in tqdm(subject_files, desc="Indexing corpus"):
            try:
                with h5py.File(path, "r") as f:
                    block = f["eye_block"][()]  # [X, Y, Z, T]
                # Filter to non-zero masked voxels [T, 14236]
                vol_flat = block.transpose(3, 0, 1, 2).reshape(n_trs, -1)[:, self.mask_flat]
                if vol_flat.shape[1] != FULL_MASK_VOXELS:
                    continue
                self.samples.append(vol_flat.astype(np.float32))
            except Exception as e:
                continue

        logging.info(f"Successfully loaded {len(self.samples)} runs into memory.")

    def __len__(self):
        return sum(len(s) - self.dt_max for s in self.samples)

    def __getitem__(self, idx):
        # Find run index
        run_idx = idx % len(self.samples)
        run = self.samples[run_idx]
        n_trs = len(run)

        # Pick random timepoint t and adjacent dt
        t = np.random.randint(0, n_trs - self.dt_max)
        dt = np.random.randint(1, self.dt_max + 1)

        x1 = run[t]
        x2 = run[t + dt]

        return torch.from_numpy(x1), torch.from_numpy(x2)


def train_contrastive_net(
    data_dir: str = None,
    output_path: str = "models/contrastive_net_64.pt",
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    bottleneck_dim: int = 64,
    hidden_dim: int = 256,
    max_subjects: int = 200,
    exclude_datasets: tuple = (),
):

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # 1. Device Selection & Data Dir Resolution
    data_dir = resolve(data_dir, download=False, quiet=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device} | Data dir: {data_dir}")


    # 2. Extract eye mask
    mask = corpus_mask(data_dir)
    n_voxels = mask.sum()
    logging.info(f"Extracted canonical eye mask with {n_voxels} voxels.")

    # 3. Discover unlabeled subject files
    eligible = unlabeled_subjects(data_dir, exclude_datasets=exclude_datasets)
    if max_subjects and len(eligible) > max_subjects:
        np.random.seed(42)
        idx = np.random.choice(len(eligible), max_subjects, replace=False)
        eligible = [eligible[i] for i in idx]

    logging.info(f"Selected {len(eligible)} subjects for self-supervised pretraining.")

    # 4. Create PyTorch Dataset & DataLoader
    dataset = UnlabeledMRIDataset(eligible, mask)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 5. Initialize ContrastiveNet
    model = ContrastiveNet(
        n_voxels=n_voxels,
        bottleneck_dim=bottleneck_dim,
        hidden_dim=hidden_dim,
        mask_ratio=0.5,
        mae_coeff=0.5,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 6. Pretraining Loop
    logging.info(f"Starting ContrastiveNet self-supervised pretraining ({epochs} epochs)...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = {"total": 0.0, "sim": 0.0, "std": 0.0, "cov": 0.0, "mae": 0.0}

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for x1, x2 in pbar:
            x1, x2 = x1.to(device), x2.to(device)

            optimizer.zero_grad()
            total_loss, loss_dict = model.compute_loss(x1, x2)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in epoch_losses:
                key = f"{k}_loss" if k != "total" else "total_loss"
                epoch_losses[k] += loss_dict[key]

            pbar.set_postfix({
                "loss": f"{loss_dict['total_loss']:.3f}",
                "sim": f"{loss_dict['sim_loss']:.3f}",
                "cov": f"{loss_dict['cov_loss']:.3f}",
                "mae": f"{loss_dict['mae_loss']:.3f}",
            })

        scheduler.step()
        num_batches = len(loader)
        for k in epoch_losses:
            epoch_losses[k] /= num_batches

        logging.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Total Loss: {epoch_losses['total']:.4f} | "
            f"Sim Loss: {epoch_losses['sim']:.4f} | "
            f"Std Loss: {epoch_losses['std']:.4f} | "
            f"Cov Loss: {epoch_losses['cov']:.4f} | "
            f"MAE Loss: {epoch_losses['mae']:.4f}"
        )

    total_time = time.time() - start_time
    logging.info(f"Pretraining complete in {total_time / 60:.2f} minutes.")

    # 7. Save Checkpoint
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metadata = {
        "epochs": epochs,
        "n_subjects": len(eligible),
        "pretrain_time_sec": total_time,
        "final_total_loss": epoch_losses["total"],
    }
    save_contrastive_net(model, output_path, metadata)
    logging.info(f"Saved pre-trained ContrastiveNet model to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain ContrastiveNet on unlabeled fMRI corpus.")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing fMRI data.")

    parser.add_argument("--output-path", type=str, default="models/contrastive_net_64.pt", help="Output checkpoint path.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of pretraining epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--bottleneck-dim", type=int, default=64, help="Bottleneck dimensionality.")
    parser.add_argument("--max-subjects", type=int, default=150, help="Maximum number of subjects for fast training.")
    args = parser.parse_args()

    train_contrastive_net(
        data_dir=args.data_dir,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        bottleneck_dim=args.bottleneck_dim,
        max_subjects=args.max_subjects,
    )
