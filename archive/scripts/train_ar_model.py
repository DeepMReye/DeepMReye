#!/usr/bin/env python3
"""Pretrain a causal next-TR model on the unlabeled corpus.

Predicts TR *t+1* from TRs <= *t* over corpus-PCA coordinates; the GRU hidden
state is then a feature source for the gaze probe (``--features ar-gru``). See
``deepmreye/temporal.py`` for why the targets are whitened by default and why an
untrained model of the same architecture is the control that matters.

    python scripts/train_ar_model.py --steps 4000
    python scripts/train_ar_model.py --steps 4000 --no-whiten --out results/ar_raw.pt

The gaze-labeled datasets are excluded, so one model is valid for every
leave-one-dataset-out fold; ``--include-labeled`` folds their voxels in if you
want the domain-adapted variant.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.datasource import resolve
from deepmreye.temporal import (
    ARModel,
    build_sequences,
    device_for,
    evaluate_prediction,
    save,
    train,
)
from deepmreye.unsupervised import load_basis, unlabeled_subjects


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--basis", default="results/corpus_basis.npz")
    p.add_argument("--out", default="results/ar_gru.pt")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-subjects", type=int, default=None)
    p.add_argument("--max-trs", type=int, default=600,
                   help="TRs kept per participant, so one long run cannot dominate.")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--no-whiten", action="store_true",
                   help="Weight components by their raw variance. The leading "
                        "components then dominate the loss -- see temporal.py.")
    p.add_argument("--include-labeled", action="store_true")
    p.add_argument("--exclude-datasets", nargs="*", default=())
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data_dir = resolve(args.data_dir, download=False, quiet=True)
    mask, bases, bmeta = load_basis(args.basis)
    basis = bases["corpus-pca"]
    print(f"[*] data {data_dir}\n[*] basis {args.basis} "
          f"({bmeta['n_voxels']} voxels -> {basis['components'].shape[1]} comps)")

    subjects = unlabeled_subjects(data_dir, include_labeled=args.include_labeled,
                                  exclude_datasets=args.exclude_datasets)
    if args.max_subjects:
        idx = np.unique(np.linspace(0, len(subjects) - 1, args.max_subjects).astype(int))
        subjects = [subjects[i] for i in idx]
    print(f"[*] {len(subjects)} participants "
          f"({len({s[0] for s in subjects})} datasets)")

    t0 = time.time()
    data, offsets = build_sequences(subjects, mask, basis, args.max_trs, progress=200)
    print(f"[*] {data.shape[0]} TRs x {data.shape[1]} comps in {time.time() - t0:.0f}s")

    # Split by participant, not by timepoint: a validation crop from a run the
    # model trained on would score the run, not the objective.
    n_runs = len(offsets) - 1
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_runs)
    n_val = max(1, int(n_runs * args.val_frac))
    val_runs, tr_runs = order[:n_val], order[n_val:]

    def gather(runs):
        parts = [data[offsets[i]: offsets[i + 1]] for i in runs]
        offs = np.cumsum([0] + [len(x) for x in parts])
        return np.concatenate(parts), offs

    tr_data, tr_off = gather(tr_runs)
    val_data, val_off = gather(val_runs)
    print(f"[*] train {tr_data.shape[0]} TRs / {len(tr_runs)} runs, "
          f"val {val_data.shape[0]} TRs / {len(val_runs)} runs")

    # Whitening is the point: without it the top few components carry most of
    # the variance and the loss becomes a global-signal model.
    scale = (tr_data.std(axis=0) if not args.no_whiten
             else np.full(tr_data.shape[1], tr_data.std(), dtype=np.float32))
    scale = np.maximum(scale, 1e-6).astype(np.float32)
    print(f"[*] targets {'whitened per component' if not args.no_whiten else 'raw-variance weighted'}")

    device = device_for(args.device)
    print(f"[*] device {device}")

    import torch

    scale_t = torch.from_numpy(scale).to(device)

    # The control, scored before training so the comparison cannot drift.
    control = ARModel(data.shape[1], args.hidden, args.layers, args.seed, device)
    base = evaluate_prediction(control, val_data, val_off, scale_t,
                               args.length, args.batch, args.seed + 1)
    print(f"[*] untrained model: val R2 {base['r2']:+.4f} "
          f"(persistence {base['r2_persistence']:+.4f})")

    model = ARModel(data.shape[1], args.hidden, args.layers, args.seed, device)
    t0 = time.time()
    history, best = train(model, tr_data, tr_off, val_data, val_off, scale_t,
                          steps=args.steps, batch=args.batch, length=args.length,
                          lr=args.lr, seed=args.seed)
    final = max(history, key=lambda m: m["r2"])
    print(f"[*] trained in {time.time() - t0:.0f}s: best val R2 {best['r2']:+.4f} "
          f"(step {best['step']}) vs untrained {base['r2']:+.4f} "
          f"vs persistence {final['r2_persistence']:+.4f}")

    meta = {
        "n_subjects": len(subjects),
        "n_trs": int(data.shape[0]),
        "n_components": int(data.shape[1]),
        "hidden": args.hidden,
        "layers": args.layers,
        "steps": args.steps,
        "whitened": not args.no_whiten,
        "include_labeled": bool(args.include_labeled),
        "excluded_datasets": sorted(args.exclude_datasets),
        "val_r2": best["r2"],
        "best_step": best["step"],
        "val_r2_untrained": base["r2"],
        "val_r2_persistence": final["r2_persistence"],
        "basis": str(args.basis),
        "history": history,
        "trained": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = save(args.out, model, scale, meta)
    print(f"[*] wrote {path}")
    print(json.dumps({k: v for k, v in meta.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
