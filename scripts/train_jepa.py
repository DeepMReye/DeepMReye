#!/usr/bin/env python3
"""Self-supervised JEPA pretraining on unlabeled eye-region fMRI.

Trains a context encoder + EMA target encoder + predictor on windows drawn from
every manually approved dataset, with the gaze-labeled ``dsL*`` sets held out
(see :class:`~deepmreye.data.jepa_dataset.JEPADataset`).

A linear probe runs alongside as a **monitor, not a model selection
criterion**. Picking the checkpoint that maximises it would contaminate the
final evaluation: the monitor splits the labeled data by subject, so it has seen
subjects from every dataset that ``eval_probe.py --protocol dataset`` later
holds out. Checkpoints are therefore saved on a fixed schedule and by epoch, and
the reported number comes from ``eval_probe.py`` on a checkpoint chosen without
reference to it.

    python -m deepmreye train --data-dir data -- --epochs 50
    python scripts/train_jepa.py --data-dir data --epochs 50 --out runs/jepa
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.data.jepa_dataset import JEPADataset
from deepmreye.data.probe_dataset import ProbeDataset
from deepmreye.datasource import resolve
from deepmreye.evaluate.baselines import fit_readout, predict
from deepmreye.evaluate.probe import (
    aggregate_by_subject,
    flatten_valid_groups,
    pool_spatial,
    temporal_targets,
)
from deepmreye.models.jepa import JEPAModel
from deepmreye.models.patcher import apply_double_cross_mask

ARCH_KEYS = ("embed_dim", "encoder_depth", "predictor_depth", "num_heads")


def embed(model, loader, device, desc):
    """Frozen context encoder over a probe loader, keeping the temporal axis."""
    feats, targs, dsets, subs = [], [], [], []
    with torch.no_grad():
        for x, y, ds, sub, _tr in tqdm(loader, desc=desc, leave=False):
            x = x.to(device)
            seq, n_s, n_t = model.patcher(x)
            idx = torch.arange(n_s * n_t, device=device).unsqueeze(0).expand(x.shape[0], -1)
            reps = model.forward_context(seq, idx, n_s, n_t)
            feats.append(pool_spatial(reps, n_s, n_t).cpu().numpy())
            targs.append(temporal_targets(y, n_t))
            dsets.extend(ds)
            subs.extend(sub)
    if not feats:
        return None
    return (np.concatenate(feats), np.concatenate(targs),
            np.array(dsets), np.array(subs))


def evaluate_probe(model, train_loader, test_loader, device, epoch, epochs):
    """Fit a ridge on frozen features and report per-dataset gaze correlation."""
    model.eval()
    tr = embed(model, train_loader, device, f"epoch {epoch}/{epochs} [probe fit]")
    te = embed(model, test_loader, device, f"epoch {epoch}/{epochs} [probe eval]")
    if tr is None or te is None:
        print(f"epoch {epoch}: probe has no data")
        return {}

    x_tr, y_tr, _, _ = flatten_valid_groups(tr[0], tr[1], tr[2], tr[3])
    x_te, y_te, ds_te, sub_te = flatten_valid_groups(te[0], te[1], te[2], te[3])
    readout = fit_readout("ridge-cv", x_tr, y_tr)
    if readout is None or len(x_te) < 2:
        print(f"epoch {epoch}: probe could not be fitted")
        return {}

    preds = predict(readout, x_te)
    baseline = y_tr.mean(axis=0)

    logs = {}
    print(f"\n--- epoch {epoch}/{epochs} probe (per-subject medians) ---")
    for ds in sorted(np.unique(ds_te)):
        sel = ds_te == ds
        m = aggregate_by_subject(y_te[sel], preds[sel], sub_te[sel], baseline)
        print(f"  [{ds}] {m.get('n_subjects', 0)} subj | euclid "
              f"{m.get('euclidean_error', float('nan')):.3f} | r "
              f"({m.get('pearson_r_x', float('nan')):.2f}, "
              f"{m.get('pearson_r_y', float('nan')):.2f})")
        logs[f"probe/{ds}/euclidean"] = m.get("euclidean_error")
        logs[f"probe/{ds}/pearson_x"] = m.get("pearson_r_x")
        logs[f"probe/{ds}/pearson_y"] = m.get("pearson_r_y")

    overall = aggregate_by_subject(y_te, preds, sub_te, baseline)
    print(f"  [ALL] {overall.get('n_subjects', 0)} subj | euclid "
          f"{overall.get('euclidean_error', float('nan')):.3f} | r "
          f"({overall.get('pearson_r_x', float('nan')):.2f}, "
          f"{overall.get('pearson_r_y', float('nan')):.2f})\n")
    logs["probe/all/euclidean"] = overall.get("euclidean_error")
    logs["probe/all/pearson_x"] = overall.get("pearson_r_x")
    logs["probe/all/pearson_y"] = overall.get("pearson_r_y")
    return logs


def save_checkpoint(path, model, args, epoch):
    """Architecture travels with the weights.

    ``eval_probe.py`` has to rebuild the model before loading, and its defaults
    are not this script's defaults. Storing the shape here means a checkpoint
    cannot be silently loaded into the wrong architecture.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "epoch": epoch,
                "arch": {k: getattr(args, k) for k in ARCH_KEYS},
                "window_size": args.window_size}, tmp)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out", default="runs/jepa", help="Checkpoint directory.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None)

    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--encoder-depth", type=int, default=6)
    p.add_argument("--predictor-depth", type=int, default=3)
    p.add_argument("--num-heads", type=int, default=8)

    p.add_argument("--window-size", type=int, default=100)
    p.add_argument("--s-ratio-start", type=float, default=0.1)
    p.add_argument("--s-ratio-end", type=float, default=0.5)
    p.add_argument("--t-ratio-start", type=float, default=0.1)
    p.add_argument("--t-ratio-end", type=float, default=0.5)

    p.add_argument("--probe-every", type=int, default=1, help="Probe every N epochs. 0 = never.")
    p.add_argument("--probe-windows", type=int, default=400,
                   help="Cap probe windows per split; the monitor should be cheap.")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--wandb-project", default="deepmreye-jepa")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--limit-train-batches", type=int, default=None,
                   help="Stop each epoch after N batches. For smoke tests.")
    args = p.parse_args()

    data_dir = Path(resolve(args.data_dir, download=False, quiet=True))
    out_dir = Path(args.out)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu")

    use_wandb = not args.no_wandb and not args.limit_train_batches
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    print(f"[*] data {data_dir}\n[*] device {device}\n[*] checkpoints {out_dir}")

    print("[*] indexing unlabeled corpus...")
    train_set = JEPADataset(data_dir=data_dir, window_size=args.window_size)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True)

    # The probe reads the same corpus -- the gaze-labeled sets are `dsL*`
    # directories inside it, not a separate tree.
    probe_loaders = None
    if args.probe_every:
        common = dict(labeled_data_dir=data_dir, window_size=args.window_size)
        probe_sets = {s: ProbeDataset(split=s, split_by="subject", **common)
                      for s in ("train", "test")}
        for name, ds in probe_sets.items():
            if len(ds) > args.probe_windows:
                idx = np.unique(np.linspace(0, len(ds) - 1, args.probe_windows).astype(int))
                ds.samples = [ds.samples[i] for i in idx]
            print(f"[*] probe {name}: {len(ds)} windows")
        probe_loaders = {s: DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers)
                         for s, ds in probe_sets.items()}

    model = JEPAModel(embed_dim=args.embed_dim, encoder_depth=args.encoder_depth,
                      predictor_depth=args.predictor_depth,
                      num_heads=args.num_heads).to(device)
    optimizer = torch.optim.AdamW(
        [p_ for p_ in model.parameters() if p_.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.SmoothL1Loss()

    n_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print("\n" + "=" * 52)
    print(" DeepMReye JEPA 2.0")
    print("=" * 52)
    print(f"   datasets scanned   : {train_set.total_datasets}")
    print(f"   subjects scanned   : {train_set.total_subjects}")
    print(f"   subjects used      : {train_set.valid_subjects}")
    print(f"   skipped            : {dict(train_set.skipped)}")
    print(f"   windows            : {train_set.total_windows:,} of {args.window_size} TRs")
    print(f"   embed / depth      : {args.embed_dim} / {args.encoder_depth}")
    print(f"   trainable params   : {n_params:,}")
    print("=" * 52 + "\n")

    # Cosine EMA momentum, 0.996 -> 1.000.
    momentum = 1.0 - 0.004 * (np.cos(np.linspace(0, np.pi, max(2, args.epochs))) * 0.5 + 0.5)

    if probe_loaders:
        print("[*] probing random initialisation (epoch 0)...")
        logs = evaluate_probe(model, probe_loaders["train"], probe_loaders["test"],
                              device, 0, args.epochs)
        if use_wandb:
            wandb.log(logs, step=0)

    for epoch in range(args.epochs):
        model.train()
        progress = epoch / max(1, args.epochs - 1)
        s_ratio = args.s_ratio_start + progress * (args.s_ratio_end - args.s_ratio_start)
        t_ratio = args.t_ratio_start + progress * (args.t_ratio_end - args.t_ratio_start)
        mom = float(momentum[min(epoch, len(momentum) - 1)])

        total, seen = 0.0, 0
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs} [jepa]", leave=False)
        for batch_idx, (x, _tr) in enumerate(pbar):
            if args.limit_train_batches and batch_idx >= args.limit_train_batches:
                break
            # _tr is the per-window repetition time. Carried through the loader
            # but not yet consumed: the temporal encoding is still ordinal in
            # bin index rather than continuous in seconds. See overview.md.
            x = x.to(device)

            seq, n_s, n_t = model.patcher(x)
            ctx, tgt, c_idx, t_idx = apply_double_cross_mask(
                seq, n_s, n_t, spatial_ratio=s_ratio, temporal_ratio=t_ratio, device=device)

            with torch.no_grad():
                target_reps = model.forward_target(tgt, c_idx, t_idx, n_s, n_t)
            context_reps = model.forward_context(ctx, c_idx, n_s, n_t)
            pred_reps = model.forward_predict(context_reps, t_idx, n_s, n_t)

            loss = criterion(pred_reps, target_reps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_target_encoder(momentum=mom)

            total += loss.item()
            seen += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg = total / max(1, seen)
        print(f"epoch {epoch + 1}/{args.epochs} | loss {avg:.4f} | "
              f"mask s={s_ratio:.2f} t={t_ratio:.2f} | ema {mom:.4f}")

        logs = {"train/jepa_loss": avg, "mask/spatial": s_ratio,
                "mask/temporal": t_ratio, "ema/momentum": mom}
        if probe_loaders and (epoch + 1) % args.probe_every == 0:
            logs.update(evaluate_probe(model, probe_loaders["train"], probe_loaders["test"],
                                       device, epoch + 1, args.epochs))
        if use_wandb:
            wandb.log(logs, step=epoch + 1)

        save_checkpoint(out_dir / "last.pt", model, args, epoch + 1)
        if args.save_every and (epoch + 1) % args.save_every == 0:
            save_checkpoint(out_dir / f"epoch{epoch + 1:03d}.pt", model, args, epoch + 1)

    print(f"[*] done. checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
