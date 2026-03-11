#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import numpy as np
import math
import wandb
from tqdm import tqdm
from sklearn.linear_model import Ridge

# Add deepmreye to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import deepmreye.config as cfg
from deepmreye.data.jepa_dataset import JEPADataset
from deepmreye.data.probe_dataset import ProbeDataset
from deepmreye.models.jepa import JEPAModel
from deepmreye.models.patcher import apply_double_cross_mask
from deepmreye.evaluate.probe import compute_probe_metrics
from sklearn.linear_model import Ridge
import collections

def evaluate_probe(model, probe_train_loader, probe_test_loader, device, args, epoch, sr, tr, mom):
    wandb_logs = {
        'mask/spatial_ratio': sr,
        'mask/temporal_ratio': tr,
        'ema/momentum': mom
    }
    
    model.eval()
    
    train_embeddings = []
    train_labels = []
    
    probe_train_pbar = tqdm(probe_train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train Probe Embeddings]", leave=False)
    with torch.no_grad():
        for batch_idx, (px, py, pds) in enumerate(probe_train_pbar):
            px, py = px.to(device), py.to(device)
            B = px.shape[0]
            
            seq, N_S, N_T = model.patcher(px)
            c_idx_full = torch.arange(N_S * N_T, device=device).unsqueeze(0).expand(B, -1)
            context_reps = model.forward_context(seq, c_idx_full, N_S, N_T)
            
            train_embeddings.append(context_reps.mean(dim=1).cpu().numpy())
            train_labels.append(py.mean(dim=(1, 2)).cpu().numpy())
            
    if len(train_embeddings) > 0:
        X_train = np.concatenate(train_embeddings, axis=0)
        Y_train = np.concatenate(train_labels, axis=0)
        
        valid_mask = ~np.isnan(Y_train).any(axis=1)
        if valid_mask.sum() > 0:
            clf = Ridge(alpha=1.0)
            clf.fit(X_train[valid_mask], Y_train[valid_mask])
        else:
            clf = None
    else:
        clf = None
        
    if clf is not None:
        test_preds = collections.defaultdict(list)
        test_labels = collections.defaultdict(list)
        
        probe_test_pbar = tqdm(probe_test_loader, desc=f"Epoch {epoch}/{args.epochs} [Eval Probe]", leave=False)
        with torch.no_grad():
            for batch_idx, (px, py, pds) in enumerate(probe_test_pbar):
                px, py = px.to(device), py.to(device)
                B = px.shape[0]
                
                seq, N_S, N_T = model.patcher(px)
                c_idx_full = torch.arange(N_S * N_T, device=device).unsqueeze(0).expand(B, -1)
                context_reps = model.forward_context(seq, c_idx_full, N_S, N_T)
                
                pred_reps = context_reps.mean(dim=1).cpu().numpy()
                py_mean = py.mean(dim=(1, 2)).cpu().numpy()
                
                pred_y = clf.predict(pred_reps)
                
                for i in range(B):
                    ds_name = pds[i]
                    test_preds[ds_name].append(pred_y[i])
                    test_labels[ds_name].append(py_mean[i])
                    
        print(f"\n--- Epoch [{epoch}/{args.epochs}] Probe Results ---")
        all_preds = []
        all_labels = []
        
        for ds_name in sorted(test_preds.keys()):
            ds_preds = np.array(test_preds[ds_name])
            ds_labels = np.array(test_labels[ds_name])
            
            all_preds.append(ds_preds)
            all_labels.append(ds_labels)
            
            metrics = compute_probe_metrics(ds_labels, ds_preds)
            print(f"  [{ds_name}] N={len(ds_preds)} | Euclidean: {metrics['euclidean_error']:.3f} | R: ({metrics['pearson_r_x']:.2f}, {metrics['pearson_r_y']:.2f})")
            
            wandb_logs[f'probe/{ds_name}/euclidean'] = metrics['euclidean_error']
            wandb_logs[f'probe/{ds_name}/pearson_x'] = metrics['pearson_r_x']
            wandb_logs[f'probe/{ds_name}/pearson_y'] = metrics['pearson_r_y']
            
        if all_preds:
            global_preds = np.concatenate(all_preds, axis=0)
            global_labels = np.concatenate(all_labels, axis=0)
            global_metrics = compute_probe_metrics(global_labels, global_preds)
            print(f"  [GLOBAL AGGREGATE] N={len(global_preds)} | Euclidean: {global_metrics['euclidean_error']:.3f} | R: ({global_metrics['pearson_r_x']:.2f}, {global_metrics['pearson_r_y']:.2f})\n")
            
            wandb_logs['probe/global/euclidean'] = global_metrics['euclidean_error']
            wandb_logs['probe/global/pearson_x'] = global_metrics['pearson_r_x']
            wandb_logs['probe/global/pearson_y'] = global_metrics['pearson_r_y']
    else:
        print(f"Epoch [{epoch}/{args.epochs}] | Probe Failed to fit Train data")
        
    return wandb_logs

def train_jepa():
    parser = argparse.ArgumentParser(description="Train PyTorch JEPA on 4D fMRI Eyes")
    config = cfg.DeepMReyeConfig()
    parser.add_argument("--data-dir", type=str, default=config.data_dir)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0, help="Number of PyTorch Dataloader workers")
    
    # Model Architecture
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--encoder-depth", type=int, default=4)
    parser.add_argument("--predictor-depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    
    # Data pipeline
    parser.add_argument("--window-size", type=int, default=25, help="TR sequences per patch window")
    parser.add_argument("--prob-threshold", type=float, default=0.7, help="Minimum transform prob")
    # Curriculum Masking (Linear Annealing Ratios)
    parser.add_argument("--s-ratio-start", type=float, default=0.1)
    parser.add_argument("--s-ratio-end", type=float, default=0.5)
    parser.add_argument("--t-ratio-start", type=float, default=0.1)
    parser.add_argument("--t-ratio-end", type=float, default=0.5)
    # Logging
    parser.add_argument("--wandb-project", type=str, default="deepmreye-jepa")
    
    # Truncated Execution Limits (for testing full pipeline)
    parser.add_argument("--limit-train-batches", type=int, default=None, help="Stop JEPA training after N batches per epoch")
    parser.add_argument("--limit-val-batches", type=int, default=None, help="Stop linear probing/evaluation after N batches")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"[*] Training JEPA on device: {device}")
    
    if not args.limit_train_batches:
        wandb.init(project=args.wandb_project, config=args)
    
    print("[*] Loading Unsupervised Dataset...")
    dataset = JEPADataset(data_dir=args.data_dir, window_size=args.window_size, prob_threshold=args.prob_threshold)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    print("[*] Loading Supervised Probe Datasets...")
    labeled_dir = Path(args.data_dir).parent / "labeled_data"
    probe_train_ds = ProbeDataset(labeled_data_dir=labeled_dir, split="train", window_size=args.window_size)
    probe_test_ds = ProbeDataset(labeled_data_dir=labeled_dir, split="test", window_size=args.window_size)
    
    if args.limit_val_batches:
        for pds in [probe_train_ds, probe_test_ds]:
            ds_to_samples = collections.defaultdict(list)
            for s in pds.samples:
                ds_to_samples[s['ds_key']].append(s)
            
            num_ds = len(ds_to_samples)
            if num_ds > 0:
                batches_per_ds = max(1, math.ceil(args.limit_val_batches / num_ds))
                samples_to_keep = batches_per_ds * args.batch_size
                
                filtered_samples = []
                for k, v in ds_to_samples.items():
                    if len(v) <= samples_to_keep:
                        filtered_samples.extend(v)
                    else:
                        ids = np.linspace(0, len(v)-1, samples_to_keep).astype(int)
                        filtered_samples.extend([v[i] for i in ids])
                pds.samples = filtered_samples
                
    probe_train_loader = DataLoader(probe_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    probe_test_loader = DataLoader(probe_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print("[*] Initializing JEPA Architecture...")
    model = JEPAModel(
        embed_dim=args.embed_dim, 
        encoder_depth=args.encoder_depth, 
        predictor_depth=args.predictor_depth, 
        num_heads=args.num_heads
    ).to(device)
    
    optimizer = torch.optim.AdamW([
        {'params': model.patcher.parameters()},
        {'params': model.context_encoder.parameters()},
        {'params': model.predictor.parameters()},
        {'params': model.pos_s.parameters()},
        {'params': model.pos_t.parameters()},
        {'params': model.mask_token}
    ], lr=args.lr, weight_decay=1e-4)
    
    criterion = nn.SmoothL1Loss()
    
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n" + "="*50)
    print(" 🧠 DeepMReye JEPA 2.0 - Training Configuration")
    print("="*50)
    print(" [Data Extraction Pipeline]")
    print(f"   ► Evaluated Datasets : {dataset.total_datasets}")
    print(f"   ► Scanned Subjects   : {dataset.total_subjects}")
    print(f"   ► Quality Threshold  : >= {args.prob_threshold}")
    print(f"   ► Approved Subjects  : {dataset.valid_subjects} ({dataset.valid_subjects/max(1, dataset.total_subjects)*100:.1f}%)")
    print(f"   ► Time Window Size   : {args.window_size} TRs")
    print(f"   ► Sequence Batches   : {dataset.total_windows:,} Extracted Windows")
    print("-" * 50)
    print(" [ViT Architecture Model]")
    print(f"   ► Embedding Dim      : {args.embed_dim}")
    print(f"   ► Encoder Depth      : {args.encoder_depth}")
    print(f"   ► Predictor Depth    : {args.predictor_depth}")
    print(f"   ► Attention Heads    : {args.num_heads}")
    print(f"   ► Total Parameters   : {model_params:,}")
    print("="*50 + "\n")
    
    # EMA Momentum schedule (standard cosine from 0.996 to 1.0)
    momentum_schedule = np.cos(np.linspace(0, np.pi, args.epochs)) * 0.5 + 0.5 # 1 to 0
    momentum_schedule = 1.0 - (0.004 * momentum_schedule) # 0.996 to 1.000
    
    print("[*] Evaluating Random Initialization (Step 0)...")
    metrics_log = evaluate_probe(model, probe_train_loader, probe_test_loader, device, args, 0, args.s_ratio_start, args.t_ratio_start, float(momentum_schedule[0]))
    if not args.limit_train_batches:
        wandb.log(metrics_log, step=0)
    
    print("[*] Beginning Curriculum Training Loop...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        # Calculate current masking ratio (Linear interpolation over epochs)
        progress = epoch / max(1, args.epochs - 1)
        sr = args.s_ratio_start + progress * (args.s_ratio_end - args.s_ratio_start)
        tr = args.t_ratio_start + progress * (args.t_ratio_end - args.t_ratio_start)
        mom = float(momentum_schedule[epoch])
        
        train_pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train JEPA]", leave=False)
        for batch_idx, x in enumerate(train_pbar):
            if args.limit_train_batches and batch_idx >= args.limit_train_batches:
                break
                
            x = x.to(device) # [B, X, Y, Z, T]
            
            # 1. Patchify input
            # seq: [B, N_S*N_T, D]
            seq, N_S, N_T = model.patcher(x)
            
            # 2. Dynamic 2D Continuous Grid Masking
            ctx_tokens, tgt_tokens, c_idx, t_idx = apply_double_cross_mask(
                seq, N_S, N_T, 
                spatial_ratio=sr, temporal_ratio=tr, 
                device=device
            )
            
            # 3. Target EMA Pass (No Grad)
            with torch.no_grad():
                target_reps = model.forward_target(tgt_tokens, c_idx, t_idx, N_S, N_T)
                
            # 4. Context Pass
            context_reps = model.forward_context(ctx_tokens, c_idx, N_S, N_T)
            
            # 5. Predictor Pass
            pred_reps = model.forward_predict(context_reps, t_idx, N_S, N_T)
            
            # 6. Loss & Backprop
            loss = criterion(pred_reps, target_reps)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 7. Step EMA Encoder
            model.update_target_encoder(momentum=mom)
            
            total_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch [{epoch+1}/{args.epochs}] | JEPA L: {avg_loss:.4f} | Mask: (s={sr:.2f}, t={tr:.2f}) | EMA: {mom:.4f}")
        metrics_log = evaluate_probe(model, probe_train_loader, probe_test_loader, device, args, epoch+1, sr, tr, mom)
        
        if not args.limit_train_batches:
            metrics_log['train/jepa_loss'] = avg_loss
            wandb.log(metrics_log, step=epoch+1)

if __name__ == "__main__":
    train_jepa()
