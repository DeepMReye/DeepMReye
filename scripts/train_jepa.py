#!/usr/bin/env python3
"""Self-supervised JEPA pretraining on unlabeled eye-region fMRI.

Trains a context encoder + EMA target encoder + predictor on windows drawn from
every manually approved dataset, with the gaze-labeled ``dsL*`` sets held out
(see :class:`~deepmreye.data.jepa_dataset.JEPADataset`).

A gaze probe runs alongside, and it now computes the same thing
``scripts/eval_probe.py`` does: leave-one-dataset-out folds at full spatial
resolution (``--spatial-pool 6x4x3``) with a ``ridge-cv`` readout, plus the
held-out-subject protocol, both logged per dataset to Weights & Biases. The
older monitor mean-pooled the encoder's 72 spatial tokens, which averages away
the across-orbit contrast gaze actually lives in, and split by subject only --
between them that put the in-training curve 0.11 to 0.24 above ``eval_probe``
and, worse, ranked configs in the opposite order.

It is still a **monitor, not a model selection criterion**. The probe is fitted
on labeled data the pretraining never sees, but choosing the checkpoint that
maximises it would still tune the reported number on the test folds. Checkpoints
are saved on a fixed schedule and the headline comes from ``eval_probe.py`` on a
checkpoint chosen without reference to these curves.

    python -m deepmreye train --data-dir data -- --epochs 50
    python scripts/train_jepa.py --data-dir data --epochs 50 --out runs/jepa
"""
import argparse
import json
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
    collapse_spatial,
    flatten_valid_groups,
    median_over_subjects,
    spatial_grid,
    temporal_targets,
)
from deepmreye.models.jepa import JEPAModel
from deepmreye.models.patcher import apply_double_cross_mask

ARCH_KEYS = ("embed_dim", "encoder_depth", "predictor_depth", "num_heads", "use_tr")

PROBE_PROTOCOLS = ("dataset", "subject")

# Floor on how many windows each dataset keeps when the probe is capped. Small
# enough to stay cheap, large enough that a per-subject correlation means
# something.
PROBE_MIN_WINDOWS_PER_DATASET = 8


def cap_probe_windows(samples, cap, min_per_dataset=PROBE_MIN_WINDOWS_PER_DATASET):
    if not cap or len(samples) <= cap:
        return samples

    by_ds = {}
    for i, s in enumerate(samples):
        by_ds.setdefault(s["dataset"], []).append(i)

    quota = {ds: min(len(idx), max(min_per_dataset,
                                   int(round(cap * len(idx) / len(samples)))))
             for ds, idx in by_ds.items()}

    over = sum(quota.values()) - cap
    while over > 0:
        ds = max(quota, key=lambda d: quota[d] - min_per_dataset)
        if quota[ds] <= min_per_dataset:
            break
        take = min(over, quota[ds] - min_per_dataset)
        quota[ds] -= take
        over -= take

    keep = []
    for ds, idx in by_ds.items():
        sel = np.unique(np.linspace(0, len(idx) - 1, quota[ds]).astype(int))
        keep.extend(idx[i] for i in sel)
    return [samples[i] for i in sorted(keep)]


def embed_labeled(model, loader, device, desc, spatial_pool):
    """Frozen context encoder over *every* labeled window, keeping time.

    One pass, not one per fold. Leave-one-dataset-out re-embeds 5/6 of the
    labeled corpus for each of six folds if done the way ``eval_probe.py`` does
    it, which is six times the Lustre traffic for identical features -- the
    windows a fold trains on are exactly the windows of the other datasets. So
    the split happens downstream, on the feature rows.
    """
    feats, targs, dsets, subs = [], [], [], []
    model.eval()
    s_patch = model.patcher.s_patch
    with torch.no_grad():
        for x, y, ds, sub, tr_val in tqdm(loader, desc=desc, leave=False):
            x = x.to(device)
            tr_tensor = tr_val.to(device)
            seq, n_s, n_t = model.patcher(x)
            idx = torch.arange(n_s * n_t, device=device).unsqueeze(0).expand(x.shape[0], -1)
            reps = model.forward_context(seq, idx, n_s, n_t, tr=tr_tensor)
            grid = spatial_grid(x.shape[1:4], s_patch)
            feats.append(collapse_spatial(reps, n_s, n_t, grid, spatial_pool).cpu().numpy())
            targs.append(temporal_targets(y, n_t))
            dsets.extend(ds)
            subs.extend(sub)
    if not feats:
        return None
    # Rows are temporal bins, not windows; `flatten_valid_groups` expands the
    # per-window dataset and subject labels against the same NaN-target mask.
    return flatten_valid_groups(np.concatenate(feats), np.concatenate(targs),
                                np.array(dsets), np.array(subs))


def probe_folds(protocol, ds_rows, sub_rows, held_out_subjects):
    """``[(fold_name, test_row_mask)]``; the training side is the complement.

    - ``dataset`` -- leave one dataset out, each in turn. Six folds in which
      every labeled subject is held out exactly once, against a scanner and a
      paradigm the readout has never seen. Matches
      ``eval_probe.py --protocol dataset``.
    - ``subject`` -- one fold: the held-out participants of every dataset, same
      scanner and paradigm. Easier, and the looser number to read alongside.
    """
    if protocol == "dataset":
        return [(str(d), ds_rows == d) for d in sorted(np.unique(ds_rows))]
    if protocol == "subject":
        return [("held-out subjects", np.isin(sub_rows, list(held_out_subjects)))]
    raise ValueError(f"unknown probe protocol {protocol!r}; "
                     f"known: {', '.join(PROBE_PROTOCOLS)}")


def _mean_r(metrics):
    """The headline scalar: the mean of the x and y per-subject median r."""
    vals = [metrics.get(f"pearson_r_{a}") for a in "xy"]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def run_probe(protocol, x, y, ds_rows, sub_rows, held_out_subjects, readout):
    """Fit and score one protocol on already-extracted features.

    Returns ``(per_dataset_metrics, overall_metrics)``. Each fold's R^2 is taken
    against *its own* training-mean gaze, so a model that has only learned where
    the held-out dataset's gaze sits on average scores zero rather than well.
    """
    per_dataset, per_subject = {}, {}
    for name, test in probe_folds(protocol, ds_rows, sub_rows, held_out_subjects):
        train = ~test
        if train.sum() < 3 or test.sum() < 2:
            print(f"  [!] {protocol}/{name}: empty split -- skipped")
            continue
        model = fit_readout(readout, x[train], y[train])
        if model is None:
            print(f"  [!] {protocol}/{name}: readout could not be fitted")
            continue
        preds = predict(model, x[test])
        baseline = y[train].mean(axis=0)

        y_te, sub_te, ds_te = y[test], sub_rows[test], ds_rows[test]
        for d in np.unique(ds_te):
            sel = ds_te == d
            m = aggregate_by_subject(y_te[sel], preds[sel], sub_te[sel], baseline)
            per_dataset[str(d)] = m
            per_subject.update(m.get("per_subject", {}))

    # Every subject is held out exactly once, so the corpus-wide number is the
    # median over the union of the folds' per-subject scores -- no rescoring,
    # and each subject keeps the baseline of the fold it was held out in.
    return per_dataset, median_over_subjects(per_subject)


def evaluate_probe(model, probe, device, epoch, epochs):
    """Run every requested probe protocol on one shared feature extraction."""
    rows = embed_labeled(model, probe["loader"], device,
                         f"epoch {epoch}/{epochs} [probe embed]", probe["spatial_pool"])
    if rows is None:
        print(f"epoch {epoch}: probe has no data")
        return {}
    x, y, ds_rows, sub_rows = rows
    if not probe.get("announced"):
        gb = x.shape[0] * x.shape[1] * 8 / 1e9
        print(f"[*] probe features: {x.shape[0]:,} rows x {x.shape[1]:,} "
              f"({gb:.1f} GB per float64 copy in the readout fit)")
        probe["announced"] = True

    logs = {}
    for protocol in probe["protocols"]:
        per_dataset, overall = run_probe(protocol, x, y, ds_rows, sub_rows,
                                         probe["held_out_subjects"], probe["readout"])
        if not per_dataset:
            continue

        print(f"\n--- epoch {epoch}/{epochs} probe [{protocol}] "
              f"(per-subject medians) ---")
        for name, m in sorted(per_dataset.items()):
            print(f"  [{name}] {m.get('n_subjects', 0):>3} subj | r "
                  f"{_mean_r(m):.3f} ({m.get('pearson_r_x', float('nan')):.2f}, "
                  f"{m.get('pearson_r_y', float('nan')):.2f}) | R2 "
                  f"{m.get('r2_vs_baseline', float('nan')):>6.3f} | euclid "
                  f"{m.get('euclidean_error', float('nan')):.3f}")
            for metric, value in (("pearson_r", _mean_r(m)),
                                  ("pearson_r_x", m.get("pearson_r_x")),
                                  ("pearson_r_y", m.get("pearson_r_y")),
                                  ("r2", m.get("r2_vs_baseline")),
                                  ("euclidean", m.get("euclidean_error")),
                                  ("n_subjects", m.get("n_subjects"))):
                logs[f"probe/{protocol}/{name}/{metric}"] = value

        # Mean over datasets, matching how the STATE.md tables are read: each
        # dataset counts once regardless of how many subjects it contributes.
        # Datasets that scored nothing are skipped rather than propagated -- one
        # dataset with too few valid rows for a correlation would otherwise turn
        # the headline curve into a flat line of NaN.
        def _mean(values):
            vals = [v for v in values if v is not None and np.isfinite(v)]
            return float(np.mean(vals)) if vals else float("nan")

        r_by_dataset = [_mean_r(m) for m in per_dataset.values()]
        summary = {
            "mean_r": _mean(r_by_dataset),
            "mean_r_x": _mean([m.get("pearson_r_x") for m in per_dataset.values()]),
            "mean_r_y": _mean([m.get("pearson_r_y") for m in per_dataset.values()]),
            "mean_r2": _mean([m.get("r2_vs_baseline") for m in per_dataset.values()]),
            "mean_euclidean": _mean([m.get("euclidean_error")
                                     for m in per_dataset.values()]),
        }
        n_scored = sum(1 for v in r_by_dataset if np.isfinite(v))
        for key, value in summary.items():
            logs[f"probe/{protocol}/{key}"] = value
        for metric, value in (("pearson_r", _mean_r(overall)),
                              ("pearson_r_x", overall.get("pearson_r_x")),
                              ("pearson_r_y", overall.get("pearson_r_y")),
                              ("r2", overall.get("r2_vs_baseline")),
                              ("euclidean", overall.get("euclidean_error")),
                              ("n_subjects", overall.get("n_subjects"))):
            logs[f"probe/{protocol}/all/{metric}"] = value

        print(f"  [MEAN over {n_scored}/{len(per_dataset)} datasets] r {summary['mean_r']:.3f} "
              f"| R2 {summary['mean_r2']:>6.3f} | euclid {summary['mean_euclidean']:.3f}")
        print(f"  [ALL {overall.get('n_subjects', 0)} subjects] r {_mean_r(overall):.3f} "
              f"| R2 {overall.get('r2_vs_baseline', float('nan')):>6.3f} | euclid "
              f"{overall.get('euclidean_error', float('nan')):.3f}\n")

    # Unprefixed aliases for the first protocol, so a sweep has one headline
    # curve to sort on without knowing which protocols a run enabled.
    head = probe["protocols"][0]
    for key in ("mean_r", "mean_r_x", "mean_r_y", "mean_r2", "mean_euclidean"):
        if f"probe/{head}/{key}" in logs:
            logs[f"probe/{key}"] = logs[f"probe/{head}/{key}"]
    return logs


def build_probe(args, data_dir):
    """Index the labeled corpus once and pin down the folds the probe will use."""
    common = dict(labeled_data_dir=data_dir, window_size=args.window_size)
    full = ProbeDataset(split="train", split_by="all", **common)
    if not len(full):
        print("[!] no labeled windows found -- probe disabled")
        return None
    full.samples = cap_probe_windows(full.samples, args.probe_windows)

    by_ds = {}
    for s in full.samples:
        by_ds.setdefault(s["dataset"], set()).add(s["subject"])
    print(f"[*] probe: {len(full)} windows, {sum(len(v) for v in by_ds.values())} subjects "
          f"({', '.join(f'{k} {len(v)}' for k, v in sorted(by_ds.items()))})")
    print(f"[*] probe protocols {args.probe_protocols}  spatial-pool "
          f"{args.spatial_pool}  readout {args.probe_readout}")

    held_out_subjects = set()
    if "subject" in args.probe_protocols:
        # Take the held-out participants from ProbeDataset itself rather than
        # reimplementing its per-dataset 80/20, so this protocol cannot drift
        # away from `eval_probe.py --protocol subject`.
        held_out_subjects = {s["subject"]
                             for s in ProbeDataset(split="test", split_by="subject",
                                                   **common).samples}
        # Subjects are the unit metrics aggregate over and folds select on, so a
        # subject id shared between two datasets would silently merge them.
        seen = {}
        for ds, subs in by_ds.items():
            for sub in subs:
                if sub in seen:
                    print(f"[!] subject id {sub!r} appears in both {seen[sub]} and "
                          f"{ds}; probe splits and per-subject metrics will merge them")
                seen[sub] = ds

    return {
        "loader": DataLoader(full, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers),
        "protocols": list(args.probe_protocols),
        "held_out_subjects": held_out_subjects,
        "spatial_pool": args.spatial_pool,
        "readout": args.probe_readout,
    }


def append_metrics(path, epoch, logs):
    """Append one epoch's metrics as a JSON line, next to the checkpoints.

    wandb is offline on this cluster -- compute nodes have no outbound network,
    so a run only reaches the dashboard if someone remembers to `wandb sync` it
    afterwards. This file does not depend on that happening, or on wandb being
    installed, and it is what to read when a job is killed at the wall clock.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"epoch": epoch, **{k: (None if v is None or (isinstance(v, float)
                                                        and not np.isfinite(v))
                                  else float(v))
                              for k, v in logs.items()}}
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def save_checkpoint(path, model, args, epoch):
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
    # Masking nothing is not a valid configuration -- the predictor gets an
    # empty target sequence and the loss is undefined -- so the defaults are the
    # best schedule measured so far rather than a neutral-looking 0.0/0.0.
    # Temporal masking beat spatial at every ratio tried; t=0.6 topped the sweep
    # (mean r 0.655 against 0.638 for the best spatial run). See STATE.md.
    p.add_argument("--s-ratio", type=float, default=0.0,
                   help="Fraction of spatial patch rows masked out.")
    p.add_argument("--t-ratio", type=float, default=0.6,
                   help="Fraction of temporal patch columns masked out.")
    p.add_argument("--use-tr", action="store_true", default=True,
                   help="Use continuous TR positional embedding and log(TR) conditioning.")
    p.add_argument("--no-tr", action="store_false", dest="use_tr",
                   help="Disable TR conditioning (use ordinal bin indices).")

    p.add_argument("--probe-every", type=int, default=1, help="Probe every N epochs. 0 = never.")
    p.add_argument("--probe-windows", type=int, default=0,
                   help="Cap total probe windows. 0 = every labeled window, which "
                        "is what makes the curve comparable to eval_probe.py. Cap "
                        "it if memory is tight: the readout fits in float64, so "
                        "rows x features x 8 bytes has to fit several times over.")
    p.add_argument("--probe-protocols", nargs="+", default=["dataset", "subject"],
                   choices=list(PROBE_PROTOCOLS),
                   help="Generalization levels to score. The first one also gets "
                        "logged unprefixed as the headline curve.")
    p.add_argument("--probe-readout", default="ridge-cv",
                   help="Readout fitted on the frozen features (see "
                        "deepmreye/evaluate/baselines.py).")
    p.add_argument("--spatial-pool", default="6x4x3",
                   help="How the encoder's 6x4x3 spatial token grid collapses to "
                        "features per temporal bin: '6x4x3'/'none' keeps it, 'mean' "
                        "averages it away, GXxGYxGZ pools to a coarser grid. "
                        "Averaging costs about half the recoverable gaze "
                        "correlation, so 'mean' is for reproducing old numbers only.")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--wandb-project", default="deepmreye-jepa")
    p.add_argument("--wandb-name", default=None, help="Run name; defaults to the "
                   "checkpoint directory name.")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--limit-train-batches", type=int, default=None,
                   help="Stop each epoch after N batches. For smoke tests.")
    args = p.parse_args()

    n_t = args.window_size // 5  # temporal patch size, fixed in fMRIPatcher
    if int(72 * args.s_ratio) + int(n_t * args.t_ratio) == 0:
        # apply_double_cross_mask takes int(N * ratio), so a ratio too small to
        # drop a single row masks nothing: the target sequence comes out empty
        # and SmoothL1Loss fails on a shape mismatch several minutes into the
        # run. Say so now instead.
        p.error(f"--s-ratio {args.s_ratio} and --t-ratio {args.t_ratio} mask no "
                f"tokens at all (72 spatial x {n_t} temporal patches), so the "
                f"predictor has no target. Raise one of them.")

    data_dir = Path(resolve(args.data_dir, download=False, quiet=True))
    out_dir = Path(args.out)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu")

    use_wandb = not args.no_wandb and not args.limit_train_batches
    if use_wandb:
        import wandb
        # Compute nodes here have no outbound network, so the sbatch sets
        # WANDB_MODE=offline and the run is `wandb sync`ed from a login node
        # afterwards. Nothing below depends on which mode it is.
        wandb.init(project=args.wandb_project, config=vars(args),
                   name=args.wandb_name or out_dir.name)
        print(f"[*] wandb {wandb.run.name} (mode {wandb.run.settings.mode}, "
              f"dir {wandb.run.dir})")

    print(f"[*] data {data_dir}\n[*] device {device}\n[*] checkpoints {out_dir}")

    print("[*] indexing unlabeled corpus...")
    train_set = JEPADataset(data_dir=data_dir, window_size=args.window_size)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True)

    probe = build_probe(args, data_dir) if args.probe_every else None

    model = JEPAModel(embed_dim=args.embed_dim, encoder_depth=args.encoder_depth,
                      predictor_depth=args.predictor_depth,
                      num_heads=args.num_heads, use_tr=args.use_tr).to(device)
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
    print(f"   use TR embedding   : {args.use_tr}")
    print(f"   trainable params   : {n_params:,}")
    print("=" * 52 + "\n")

    momentum = 1.0 - 0.004 * (np.cos(np.linspace(0, np.pi, max(2, args.epochs))) * 0.5 + 0.5)

    if probe:
        print("[*] probing random initialisation (epoch 0)...")
        logs = evaluate_probe(model, probe, device, 0, args.epochs)
        append_metrics(out_dir / "metrics.jsonl", 0, logs)
        if use_wandb:
            wandb.log(logs, step=0)

    for epoch in range(args.epochs):
        model.train()
        s_ratio = args.s_ratio
        t_ratio = args.t_ratio
        mom = float(momentum[min(epoch, len(momentum) - 1)])

        total, seen = 0.0, 0
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs} [jepa]", leave=False)
        for batch_idx, (x, tr_val) in enumerate(pbar):
            if args.limit_train_batches and batch_idx >= args.limit_train_batches:
                break
            x = x.to(device)
            tr_tensor = tr_val.to(device)

            seq, n_s, n_t = model.patcher(x)
            ctx, tgt, c_idx, t_idx = apply_double_cross_mask(
                seq, n_s, n_t, spatial_ratio=s_ratio, temporal_ratio=t_ratio, device=device)

            with torch.no_grad():
                target_reps = model.forward_target(tgt, c_idx, t_idx, n_s, n_t, tr=tr_tensor)
            context_reps = model.forward_context(ctx, c_idx, n_s, n_t, tr=tr_tensor)
            pred_reps = model.forward_predict(context_reps, t_idx, n_s, n_t, tr=tr_tensor)

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

        logs = {"train/jepa_loss": avg, "train/epoch": epoch + 1,
                "mask/spatial": s_ratio, "mask/temporal": t_ratio,
                "ema/momentum": mom}
        if probe and (epoch + 1) % args.probe_every == 0:
            logs.update(evaluate_probe(model, probe, device, epoch + 1, args.epochs))
        append_metrics(out_dir / "metrics.jsonl", epoch + 1, logs)
        if use_wandb:
            wandb.log(logs, step=epoch + 1)

        save_checkpoint(out_dir / "last.pt", model, args, epoch + 1)
        if args.save_every and (epoch + 1) % args.save_every == 0:
            save_checkpoint(out_dir / f"epoch{epoch + 1:03d}.pt", model, args, epoch + 1)

    print(f"[*] done. checkpoints in {out_dir}")



if __name__ == "__main__":
    main()
