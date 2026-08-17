#!/usr/bin/env python3
"""Full 8-Fold Cross-Dataset Benchmark comparing CompositeNet vs PCA+PLS vs Pure PCA.

For each of the 8 labeled datasets:
1. Hold out dataset d completely (100% strict zero-shot, 0% double-dipping).
2. Train PyTorch CompositeNet (bottleneck=96, alpha=0.05) on the remaining 7 datasets for 10 epochs.
3. Evaluate composite-net, fold-pca:32+fold-pls:32, and fold-pca:64 side-by-side on held-out dataset d.
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

LABELED_DATASETS = [
    "dsL01_guided_fixations",
    "dsL02_pursuit",
    "dsL03_pursuit",
    "dsL04_pursuit",
    "dsL05_free_viewing",
    "dsL06_sequences",
    "dsL07_deepmreye_calib",
    "dsL11_backtothefuture",
]


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
        for p in paths:
            try:
                with h5py.File(p, "r") as f:
                    block = f["eye_block"][...]
                    labels = f["labels"][...]
                    pooled_run = pool_entire_run_masked(block, per_bin, mask_flat)
                    self._cache[p] = (pooled_run, labels)
            except Exception:
                pass

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


def train_composite_net_for_fold(data_dir, holdout_ds, mask_flat, n_voxels, device, args):
    """Train CompositeNet model on 7 datasets (excluding holdout_ds) for 10 epochs."""
    holdout_set = {holdout_ds}
    train_ds = PrepooledRunDataset(
        ProbeDataset(labeled_data_dir=data_dir, split="train", window_size=args.window_size, holdout=holdout_set),
        args.temp_patch_size,
        mask_flat
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    n_t = args.window_size // args.temp_patch_size

    model = CompositeNet(n_voxels=n_voxels, bottleneck_dim=args.bottleneck, alpha=args.alpha).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        model.train()
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

        scheduler.step()

    return model


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--basis", default="results/corpus_basis.npz")
    p.add_argument("--out-dir", default="results/full_8fold_benchmark")
    p.add_argument("--bottleneck", type=int, default=96)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=10)
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
    print(f"[*] Starting Full 8-Dataset Zero-Shot Cross-Validation ({device})...", flush=True)

    all_fold_results = {}

    for i, holdout_ds in enumerate(LABELED_DATASETS, 1):
        print(f"\n[{i}/8] --- Holdout Dataset: {holdout_ds} ---", flush=True)
        t_fold = time.time()

        # Step 1: Train CompositeNet on remaining 7 datasets
        print(f"  [*] Training CompositeNet on 7 datasets (excluding {holdout_ds})...", flush=True)
        model = train_composite_net_for_fold(data_dir, holdout_ds, mask_flat, n_voxels, device, args)

        ckpt_path = out_dir / f"checkpoint_{holdout_ds}.pt"
        save_composite_net(model, str(ckpt_path), metadata={"holdout": holdout_ds, "epochs": args.epochs})

        # Step 2: Evaluate 3 feature sets on held-out dataset
        eval_path = out_dir / f"eval_{holdout_ds}.json"
        eval_cmd = (
            f"uv run python scripts/eval_probe.py --protocol dataset "
            f"--features composite-net fold-pca:32+fold-pls:32 fold-pca:64 "
            f"--composite-checkpoint {ckpt_path} --fold-name {holdout_ds} "
            f"--readouts ridge-cv --standardize-targets dataset --out {eval_path}"
        )
        print(f"  [*] Running 3-way evaluation on {holdout_ds}...", flush=True)
        run_cmd(eval_cmd)

        if eval_path.exists():
            data = json.loads(eval_path.read_text())
            fold_data = data.get(holdout_ds, {})
            comp = fold_data.get("composite-net/ridge-cv", {}).get("by_subject", {})
            basis = fold_data.get("fold-pca:32+fold-pls:32/ridge-cv", {}).get("by_subject", {})
            pca = fold_data.get("fold-pca:64/ridge-cv", {}).get("by_subject", {})

            res = {
                "composite_net": {
                    "r_x": comp.get("pearson_r_x", 0.0),
                    "r_y": comp.get("pearson_r_y", 0.0),
                    "r_mean": 0.5 * (comp.get("pearson_r_x", 0.0) + comp.get("pearson_r_y", 0.0))
                },
                "composite_basis": {
                    "r_x": basis.get("pearson_r_x", 0.0),
                    "r_y": basis.get("pearson_r_y", 0.0),
                    "r_mean": 0.5 * (basis.get("pearson_r_x", 0.0) + basis.get("pearson_r_y", 0.0))
                },
                "pca_baseline": {
                    "r_x": pca.get("pearson_r_x", 0.0),
                    "r_y": pca.get("pearson_r_y", 0.0),
                    "r_mean": 0.5 * (pca.get("pearson_r_x", 0.0) + pca.get("pearson_r_y", 0.0))
                }
            }
            all_fold_results[holdout_ds] = res
            print(f"  [+] {holdout_ds} [{time.time() - t_fold:.1f}s]: "
                  f"Net={res['composite_net']['r_mean']:.3f} (rx={res['composite_net']['r_x']:.3f}, ry={res['composite_net']['r_y']:.3f}) | "
                  f"Basis={res['composite_basis']['r_mean']:.3f} | "
                  f"PCA={res['pca_baseline']['r_mean']:.3f}", flush=True)

    # Summary Table
    print("\n" + "=" * 105, flush=True)
    print(f"{'Held-Out Dataset':<25} {'CompositeNet (r_x / r_y / r_mean)':<35} {'PCA+PLS Basis (r_mean)':<25} {'PCA Baseline':<15}", flush=True)
    print("-" * 105, flush=True)

    net_means, basis_means, pca_means = [], [], []
    for ds in LABELED_DATASETS:
        r = all_fold_results.get(ds, {})
        net = r.get("composite_net", {})
        basis = r.get("composite_basis", {})
        pca = r.get("pca_baseline", {})

        net_m = net.get("r_mean", 0.0)
        basis_m = basis.get("r_mean", 0.0)
        pca_m = pca.get("r_mean", 0.0)

        net_means.append(net_m)
        basis_means.append(basis_m)
        pca_means.append(pca_m)

        net_str = f"{net.get('r_x',0):.3f} / {net.get('r_y',0):.3f} / {net_m:.3f}"
        print(f"{ds:<25} {net_str:<35} {basis_m:<25.3f} {pca_m:<15.3f}", flush=True)

    print("-" * 105, flush=True)
    print(f"{'MEDIAN ACROSS 8 FOLDOUTS':<25} {np.median(net_means):<35.3f} {np.median(basis_means):<25.3f} {np.median(pca_means):<15.3f}", flush=True)
    print(f"{'MEAN ACROSS 8 FOLDOUTS':<25} {np.mean(net_means):<35.3f} {np.mean(basis_means):<25.3f} {np.mean(pca_means):<15.3f}", flush=True)
    print("=" * 105, flush=True)

    (out_dir / "full_8fold_results.json").write_text(json.dumps(all_fold_results, indent=2))
    print(f"\n[*] Full 8-fold cross-validation results saved to {out_dir / 'full_8fold_results.json'}", flush=True)


if __name__ == "__main__":
    main()
