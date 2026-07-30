#!/usr/bin/env python3
"""The baseline table: how well can gaze be read out, and by what.

A probe number on its own says nothing. This crosses two axes so that it does.

**Generalization level** (``--protocol``), in increasing strictness. This is the
structure DeepMReye 1.0 reported, plus one:

- ``within``   -- same participant, early timepoints train and late ones test.
                  The easiest setting; it answers "does this work at all". Train
                  and test share no timepoint, but we have no run boundaries
                  stored, so they are temporally adjacent.
- ``subject``  -- held-out participants, same scanner and paradigm.
- ``dataset``  -- leave one dataset out, each in turn. A scanner and a
                  population the readout has never seen.
- ``paradigm`` -- leave one *paradigm* out. dsL02/03/04 are all smooth pursuit,
                  so holding out one of them alone still trains on the same
                  task; this is the honest unseen-task number.

**Feature source** (``--arms``):

- ``voxels``  -- downsampled raw voxels. No representation learning at all. If
                 the encoder does not beat this, it is not adding anything a
                 linear map could not take from the data directly.
- ``random``  -- the same architecture, untrained. A random ViT is a non-linear
                 random projection and can score surprisingly well, so the gap
                 between ``random`` and ``trained`` is the actual claim of the
                 method -- not the gap to zero.
- ``trained`` -- a JEPA checkpoint.

**Readout** (``--readouts``): see ``deepmreye/evaluate/baselines.py``. Every
readout is fitted on every arm's features, so the comparison is like for like.

Metrics are aggregated **per participant, then median across participants**.
Pooling every row of every subject into one correlation would let a model score
well by predicting only which subject it is looking at. ``--pooled`` also prints
the pooled numbers for comparison.

Pearson r is the headline here rather than R^2, deliberately: cross-dataset gaze
is mis-calibrated in gain (measured gains 0.11-2.27 against the training scale),
which destroys R^2 while leaving the correlation intact. That is a separate
problem from whether the representation carries gaze -- see
``scripts/analyze_calibration.py``.

    python scripts/eval_probe.py --protocol within
    python scripts/eval_probe.py --protocol dataset --readouts mean linear ridge-cv pca-ridge pls
    python scripts/eval_probe.py --protocol dataset --checkpoint runs/jepa/best.pt
"""
import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.data.probe_dataset import ProbeDataset, dataset_folds, paradigm_folds
from deepmreye.datasource import resolve
from deepmreye.evaluate.baselines import ALL_READOUTS, DEFAULT_READOUTS, fit_readout, predict
from deepmreye.evaluate.probe import (
    aggregate_by_subject,
    collapse_spatial,
    compute_probe_metrics,
    flatten_valid_groups,
    spatial_grid,
    temporal_targets,
)
from deepmreye.models.jepa import JEPAModel


def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(args, device, checkpoint=None):
    """Build the encoder, taking the architecture from the checkpoint if it has one.

    ``train_jepa.py`` stores the shape it trained with. Trusting this script's
    own defaults instead would either raise on a size mismatch or, worse, load
    successfully into a differently-shaped model on some future refactor.

    Seeded, because the ``random`` arm has no checkpoint to fix its weights and
    an unseeded control is a different encoder on every run.
    """
    arch = dict(embed_dim=args.embed_dim, encoder_depth=args.encoder_depth,
                predictor_depth=args.predictor_depth, num_heads=args.num_heads)
    state = None
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and state.get("arch"):
            if state["arch"] != arch:
                print(f"    [*] using checkpoint architecture {state['arch']}")
            arch = state["arch"]

    torch.manual_seed(getattr(args, "seed", 0))
    model = JEPAModel(**arch)
    if state is not None:
        model.load_state_dict(state.get("model", state))
    return model.to(device)


def build_arms(args, device):
    """One encoder per non-voxel arm, built **once** for the whole evaluation.

    This has to be built once rather than per split, and the reason is not
    efficiency. A fresh ``JEPAModel()`` per call gives the ``random`` arm
    *different weights for the train split and the test split*: the readout is
    then fitted in one random basis and asked to predict in another, so it
    scores ~0 no matter what the encoder does. That is a broken control, not a
    measurement of an untrained encoder -- and it is what produced the
    `random` = -0.003 row at full spatial resolution, which in turn was read as
    "a random spatially-resolved projection cannot do this". One shared random
    encoder scores far higher; see STATE.md.
    """
    return {arm: build_model(args, device,
                             args.checkpoint if arm == "trained" else None)
            for arm in args.arms
            if arm != "voxels" and not (arm == "trained" and not args.checkpoint)}


def encoder_features(model, loader, device, desc, spatial_pool="mean"):
    """Frozen context encoder over a loader, keeping the temporal axis.

    The encoder's spatial tokens form a 6x4x3 grid over the eye box (patch size
    8 on a 47x29x18 block); ``spatial_pool`` decides how that grid collapses to
    features per temporal bin -- see
    :func:`deepmreye.evaluate.probe.collapse_spatial`, which
    ``scripts/train_jepa.py``'s in-training probe shares, so a curve logged
    during training and a number from here mean the same thing.
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
    return _stack(feats, targs, dsets, subs)


def voxel_features(loader, n_t, stride, desc):
    """Downsampled raw voxels per temporal bin -- the no-encoder arm."""
    feats, targs, dsets, subs = [], [], [], []
    for x, y, ds, sub, _tr in tqdm(loader, desc=desc, leave=False):
        b, _, _, _, w = x.shape
        per_bin = int(np.ceil(w / n_t))
        pad = per_bin * n_t - w
        if pad:
            # Pad the same way the patcher does, so a voxel bin and a token bin
            # cover the same TRs and the two arms are comparable.
            x = torch.nn.functional.pad(x, (0, pad))
        sub_x = x[:, ::stride, ::stride, ::stride, :]
        sub_x = sub_x.reshape(b, -1, n_t, per_bin).mean(dim=3).permute(0, 2, 1)
        feats.append(sub_x.numpy())
        targs.append(temporal_targets(y, n_t))
        dsets.extend(ds)
        subs.extend(sub)
    return _stack(feats, targs, dsets, subs)


def _stack(feats, targs, dsets, subs):
    if not feats:
        return None
    return (np.concatenate(feats), np.concatenate(targs),
            np.array(dsets), np.array(subs))


def _source_fingerprint(*modules):
    """Content hash of the modules that decide what the features are.

    Without this the cache is a trap rather than a speedup: changing the model
    or the split logic leaves the key identical, so a rerun silently reports
    numbers from the old code. That has already happened once here -- a
    positional-embedding init fix and a split fix both left stale features in
    place under unchanged keys.
    """
    h = hashlib.sha1()
    for mod in modules:
        path = Path(mod.__file__)
        h.update(path.read_bytes())
    return h.hexdigest()[:12]


def cache_key(args, fold, split, arm):
    """Identity of a feature matrix, so a rerun does not redo the I/O.

    Everything that changes the features goes in: the split parameters, the
    architecture, the source of the code that produces them, and -- for a
    checkpoint -- its mtime, so a retrained model at the same path does not
    reuse the old features.
    """
    import deepmreye.data.probe_dataset as probe_dataset

    from deepmreye.models import jepa, patcher

    parts = [args.protocol, fold, split, arm, args.window_size,
             args.temp_patch_size, args.voxel_stride, args.max_windows,
             args.gap, args.embed_dim, args.encoder_depth, args.num_heads,
             _source_fingerprint(probe_dataset)]
    if arm != "voxels":
        # Pooling changes the features but nothing else in this key would show
        # it, so a pooled and an unpooled run would collide on the same cache
        # entry and silently report each other's numbers.
        parts += [args.spatial_pool, _source_fingerprint(jepa, patcher)]
    if arm == "trained" and args.checkpoint:
        parts += [args.checkpoint, Path(args.checkpoint).stat().st_mtime]
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def get_features(args, device, fold, split, arm, loader, n_t, model=None):
    cache = Path(args.feature_cache) / f"{cache_key(args, fold, split, arm)}.npz" \
        if args.feature_cache else None
    if cache and cache.exists():
        try:
            d = np.load(cache, allow_pickle=False)
            return d["f"], d["y"], d["ds"], d["sub"]
        except (zipfile.BadZipFile, EOFError, ValueError) as e:
            # A concurrent job may be part-way through writing this entry (the
            # `voxels` and `random` keys carry no checkpoint, so every job in a
            # sweep shares them). Recomputing costs minutes; failing the job
            # costs hours, and the cache is only ever a speedup.
            print(f"    [!] cache {cache.name} unreadable ({e.__class__.__name__}); "
                  f"recomputing")

    desc = f"{fold}/{arm} {split}"
    if arm == "voxels":
        out = voxel_features(loader, n_t, args.voxel_stride, desc)
    else:
        # The caller owns the encoder; both splits of a fold must go through the
        # *same* weights (see build_arms).
        out = encoder_features(model, loader, device, desc, args.spatial_pool)
    if out is None:
        return None

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        # Write then rename. Concurrent jobs share cache entries by design --
        # `voxels` and `random` keys carry no checkpoint, so every job in a
        # sweep computes the same ones -- and a reader that opens a half-written
        # file gets `BadZipFile` rather than a wait. Rename is atomic on the
        # same filesystem, so a reader sees either the old file or a complete
        # new one. The pid keeps two writers from sharing a temp name.
        tmp = cache.with_suffix(f".{os.getpid()}.tmp.npz")
        # Uncompressed, deliberately. These entries reach 3 GB at full spatial
        # resolution (18,432 features x 30k rows), and zlib on float32 activations
        # buys little while costing minutes per file -- measured, it was the
        # entire reason four 6-fold jobs hit a 6 h wall while the ridge fits they
        # fed took 15 s per fold. Scratch has the space; the time we do not have.
        np.savez(tmp, f=out[0], y=out[1], ds=out[2], sub=out[3])
        os.replace(tmp, cache)
    return out


def cap(ds, max_windows):
    """Evenly-spaced subsample, so every subject keeps a share of its windows
    rather than some subjects vanishing entirely."""
    if not max_windows or len(ds) <= max_windows:
        return ds
    idx = np.linspace(0, len(ds) - 1, max_windows).astype(int)
    return Subset(ds, np.unique(idx).tolist())


def make_splits(protocol, holdout, data_dir, args):
    common = dict(labeled_data_dir=data_dir, window_size=args.window_size)
    if protocol == "within":
        kw = dict(split_by="time", gap=args.gap)
    elif protocol == "subject":
        kw = dict(split_by="subject")
    else:
        kw = dict(holdout=holdout)
    return (ProbeDataset(split="train", **kw, **common),
            ProbeDataset(split="test", **kw, **common))


def run_fold(fold, holdout, data_dir, args, device, arm_models):
    train_ds, test_ds = make_splits(args.protocol, holdout, data_dir, args)
    if not len(train_ds) or not len(test_ds):
        print(f"  [!] {fold}: empty split (train {len(train_ds)}, test {len(test_ds)}) -- skipped")
        return {}

    train_ds, test_ds = cap(train_ds, args.max_windows), cap(test_ds, args.max_windows)
    print(f"    train {len(train_ds)} windows, test {len(test_ds)} windows")

    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers),
        "test": DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers),
    }
    n_t = args.window_size // args.temp_patch_size

    results = {}
    for arm in args.arms:
        if arm == "trained" and not args.checkpoint:
            continue
        encoder = arm_models.get(arm)
        tr = get_features(args, device, fold, "train", arm, loaders["train"], n_t, encoder)
        te = get_features(args, device, fold, "test", arm, loaders["test"], n_t, encoder)
        if tr is None or te is None:
            continue

        x_tr, y_tr, _, _ = flatten_valid_groups(tr[0], tr[1], tr[2], tr[3])
        x_te, y_te, ds_te, sub_te = flatten_valid_groups(te[0], te[1], te[2], te[3])
        if len(x_tr) < 3 or len(x_te) < 2:
            continue

        # R^2 is measured against the *training* mean gaze. Against the test
        # mean it would flatter a model that has only learned where this
        # dataset's gaze sits on average.
        baseline = y_tr.mean(axis=0)

        for readout in args.readouts:
            t0 = time.time()
            model = fit_readout(readout, x_tr, y_tr, args.n_components, args.seed)
            if model is None:
                continue
            preds = predict(model, x_te)
            results[(arm, readout)] = {
                "by_subject": aggregate_by_subject(y_te, preds, sub_te, baseline),
                "pooled": compute_probe_metrics(y_te, preds, baseline),
                "by_dataset": {
                    str(d): aggregate_by_subject(
                        y_te[ds_te == d], preds[ds_te == d], sub_te[ds_te == d], baseline)
                    for d in np.unique(ds_te)
                },
                "seconds": round(time.time() - t0, 1),
                "n_features": int(x_tr.shape[1]),
                "n_train_rows": int(len(x_tr)),
            }
    return results


def report(all_results, pooled=False):
    print("\n" + "=" * 88)
    head = f"{'fold':<20} {'arm':<8} {'readout':<10} {'subj':>5} {'euclid':>8} {'R2':>8} {'r_x':>7} {'r_y':>7}"
    print(head)
    print("-" * 88)
    for fold, arms in all_results.items():
        for (arm, readout), res in arms.items():
            m = res["pooled"] if pooled else res["by_subject"]
            print(f"{fold:<20} {arm:<8} {readout:<10} "
                  f"{m.get('n_subjects', 0):>5} "
                  f"{m.get('euclidean_error', float('nan')):>8.3f} "
                  f"{m.get('r2_vs_baseline', float('nan')):>8.3f} "
                  f"{m.get('pearson_r_x', float('nan')):>7.3f} "
                  f"{m.get('pearson_r_y', float('nan')):>7.3f}")
    print("=" * 88)
    print("Per-subject medians. R2 is against the training-mean gaze; 0 = learned nothing."
          if not pooled else "Pooled over all rows (mixes between-subject variance in).")


def to_json(all_results):
    """Serialisable form: tuple keys and numpy scalars are not JSON."""
    out = {}
    for fold, arms in all_results.items():
        out[fold] = {f"{arm}/{readout}": res for (arm, readout), res in arms.items()}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--protocol", choices=["within", "subject", "dataset", "paradigm"],
                   default="dataset")
    p.add_argument("--arms", nargs="+", default=["voxels", "random", "trained"])
    p.add_argument("--readouts", nargs="+", default=list(DEFAULT_READOUTS),
                   choices=list(ALL_READOUTS))
    p.add_argument("--checkpoint", default=None, help="Trained JEPA checkpoint.")
    p.add_argument("--window-size", type=int, default=100)
    p.add_argument("--temp-patch-size", type=int, default=5)
    p.add_argument("--voxel-stride", type=int, default=4)
    p.add_argument("--spatial-pool", default="mean",
                   help="How the encoder's 6x4x3 spatial token grid collapses to "
                        "features per temporal bin: 'mean' (1x1x1, 256 features), "
                        "'none' (6x4x3, 18432), or GXxGYxGZ to pool to a coarser "
                        "grid, e.g. 2x1x1 (512, left/right orbit) or 3x2x1 (1536). "
                        "Averaging away spatial contrast costs about half the "
                        "recoverable gaze correlation; 18k features do not finish.")
    p.add_argument("--n-components", type=int, default=32, help="For pca-ridge and pls.")
    p.add_argument("--gap", type=int, default=0, help="TRs discarded either side of a `within` split.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--encoder-depth", type=int, default=6)
    p.add_argument("--predictor-depth", type=int, default=3)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--limit-folds", type=int, default=None)
    p.add_argument("--max-windows", type=int, default=None,
                   help="Subsample each split to at most N windows, evenly spaced. "
                        "For iterating locally; leave off for a real number.")
    p.add_argument("--feature-cache", default=None,
                   help="Directory to cache extracted features in. Feature extraction "
                        "is the expensive half; the readouts are seconds.")
    p.add_argument("--out", default=None, help="Write full results as JSON here.")
    p.add_argument("--pooled", action="store_true", help="Print pooled instead of per-subject.")
    args = p.parse_args()

    data_dir = resolve(args.data_dir, download=False, quiet=True)
    device = pick_device(args.device)
    print(f"[*] data {data_dir}\n[*] device {device}\n[*] protocol {args.protocol}")
    print(f"[*] arms {args.arms}  readouts {args.readouts}")

    present = sorted({s.dataset for s in ProbeDataset(
        labeled_data_dir=data_dir, split="train", window_size=args.window_size)._discover()})
    print(f"[*] labeled datasets: {', '.join(present)}")

    if args.protocol == "within":
        folds = [("within-subject", None)]
    elif args.protocol == "subject":
        folds = [("held-out subjects", None)]
    elif args.protocol == "dataset":
        folds = dataset_folds(present)
    else:
        folds = paradigm_folds(present)

    if args.limit_folds:
        folds = folds[: args.limit_folds]

    # Built once, before the folds: every fold and both splits must see the
    # same weights, or the `random` arm compares two different random encoders.
    arm_models = build_arms(args, device)

    all_results = {}
    for name, holdout in folds:
        print(f"\n[*] fold: {name}" + (f"  (holding out {sorted(holdout)})" if holdout else ""))
        all_results[name] = run_fold(name, holdout, data_dir, args, device, arm_models)

    report(all_results, args.pooled)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(to_json(all_results), indent=2, default=float))
        print(f"[*] wrote {args.out}")


if __name__ == "__main__":
    main()
