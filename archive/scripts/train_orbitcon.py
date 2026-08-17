#!/usr/bin/env python3
"""Pretrain the cross-orbit contrastive encoder on the unlabeled corpus.

VICReg between the two orbits of the *same* TR. See ``deepmreye/orbitcon.py``
for why this pairing and not a temporal one, and for the three defenses against
the objective's degenerate solution (encode anatomy, which is also shared
between orbits).

    # the gate: is a small run anywhere near the linear arms?
    python scripts/train_orbitcon.py --max-subjects 100 --steps 2000
    python scripts/eval_probe.py --protocol dataset --features ocon ocon-random \\
        lr-cca:64 --readouts ridge-cv --standardize-targets dataset \\
        --ocon-checkpoint results/orbitcon.pt

    # the scaling curve, which decides whether scaling up is worth it
    python scripts/train_orbitcon.py --scaling 50 100 200 400

Read the log for ``agree`` against ``shuf``, not for the loss. VICReg's loss
falls in every configuration, including ones that have encoded nothing but
anatomy -- the loss cannot distinguish them. Agreement above the shuffled
control says the two orbits are aligned *at matching timepoints*, which anatomy
alone does not deliver.

The gaze-labeled datasets are excluded, so one checkpoint is valid for every
leave-one-dataset-out fold without retraining.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.crossorbit import ORBIT_SHAPE, build_orbit_cache
from deepmreye.datasource import resolve
from deepmreye.orbitcon import (
    DEFAULT_AUG,
    OrbitContrastModel,
    center_runs,
    evaluate,
    save,
    train,
    unmirror_right,
)
from deepmreye.temporal import device_for
from deepmreye.unsupervised import unlabeled_subjects


def load_cache(cache, data_dir, max_subjects, trs_per_subject, include_labeled,
               exclude_datasets):
    """Cached orbit volumes, built on first use.

    The cache stores raw orbit volumes, so it is only valid for the geometry
    that built it. ``crossorbit``'s loader learned this the hard way -- a stale
    cache survived a change to the left/right split silently -- so the orbit
    shape is recorded and checked.
    """
    cache = Path(cache)
    if cache.exists():
        blob = np.load(cache)
        data, offsets = blob["data"], blob["offsets"]
        cached_shape = tuple(blob["orbit_shape"]) if "orbit_shape" in blob else None
        if cached_shape != tuple(ORBIT_SHAPE):
            raise SystemExit(
                f"[!] cache {cache} was built for orbit shape {cached_shape}, "
                f"but the current split gives {tuple(ORBIT_SHAPE)}. Delete it "
                f"and rerun to rebuild.")
        print(f"[*] reusing cache {cache} ({data.shape[0]} TRs, "
              f"{len(offsets) - 1} runs)")
        return data, offsets

    subjects = unlabeled_subjects(data_dir, include_labeled=include_labeled,
                                 exclude_datasets=exclude_datasets)
    if max_subjects and max_subjects < len(subjects):
        # Spread over the sorted list rather than taking a prefix: the sort is by
        # path, so a prefix is a handful of datasets and would confound "more
        # participants" with "more acquisitions".
        idx = np.unique(np.linspace(0, len(subjects) - 1, max_subjects).astype(int))
        subjects = [subjects[i] for i in idx]
    print(f"[*] caching orbits for {len(subjects)} participants "
          f"({len({s[0] for s in subjects})} datasets)")
    t0 = time.time()
    data, offsets = build_orbit_cache(subjects, trs_per_subject, progress=100)
    np.savez(cache, data=data, offsets=offsets, orbit_shape=np.array(ORBIT_SHAPE))
    print(f"[*] {data.shape[0]} TRs, {data.nbytes / 1e9:.2f} GB, "
          f"{time.time() - t0:.0f}s -> {cache}")
    return data, offsets


def split_runs(data, offsets, val_frac, seed):
    """Split by **run**, never by timepoint.

    A validation TR from a run the model trained on shares that participant's
    anatomy, which is exactly the shortcut being guarded against -- holding out
    timepoints would report the shortcut as generalisation.
    """
    n_runs = len(offsets) - 1
    order = np.random.default_rng(seed).permutation(n_runs)
    n_val = max(1, int(n_runs * val_frac))

    def gather(runs):
        parts = [data[int(offsets[i]): int(offsets[i + 1])] for i in runs]
        return np.concatenate(parts), np.cumsum([0] + [len(x) for x in parts])

    return gather(order[n_val:]), gather(order[:n_val])


def subset_runs(data, offsets, n_runs, seed=0):
    """The first ``n_runs`` of a shuffled run order, for the scaling curve."""
    order = np.random.default_rng(seed).permutation(len(offsets) - 1)[:n_runs]
    parts = [data[int(offsets[i]): int(offsets[i + 1])] for i in order]
    return np.concatenate(parts), np.cumsum([0] + [len(x) for x in parts])


def run_one(tr_data, tr_off, val_data, val_off, args, out, tag=""):
    """Build a control, train a model, return both sets of metrics."""
    device = device_for(args.device)
    aug = dict(DEFAULT_AUG, shift_voxels=args.shift, dropout=args.dropout,
               noise=args.noise, gain=args.gain, bias=args.bias)

    def build():
        return OrbitContrastModel(args.embed, args.width, args.expander,
                                  seed=args.seed, device=device, head=args.head,
                                  mirror_right=args.mirror_right)

    # The untrained control, built by the identical factory. Its agreement is the
    # floor: both orbits share global signal, motion and drift, so this number is
    # never zero and the trained model has to be read against it.
    control = build()
    base = evaluate(control, val_data, val_off, args.batch, args.seed + 1,
                    aug=aug, runs_per_batch=args.runs_per_batch)
    print(f"[*] untrained{tag}: val loss {base['loss']:.3f}, "
          f"agreement {base['agreement']:+.3f} "
          f"(shuffled {base['agreement_shuffled']:+.3f}, "
          f"within-run {base['agreement_within_run']:+.3f})")

    meta = {"embed": args.embed, "width": args.width, "expander": args.expander,
            "head": args.head, "mirror_right": args.mirror_right,
            "steps": args.steps, "batch": args.batch, "lr": args.lr,
            "weight_decay": args.weight_decay,
            "runs_per_batch": args.runs_per_batch, "aug": aug,
            "n_runs": len(tr_off) - 1, "n_trs": int(tr_data.shape[0]),
            "untrained": base}
    model = build()
    t0 = time.time()
    history, best = train(model, tr_data, tr_off, val_data, val_off,
                          steps=args.steps, batch=args.batch, lr=args.lr,
                          weight_decay=args.weight_decay, seed=args.seed,
                          log_every=args.log_every, aug=aug,
                          runs_per_batch=args.runs_per_batch,
                          checkpoint_path=out, meta=meta)
    final = evaluate(model, val_data, val_off, args.batch, args.seed + 1,
                     aug=aug, runs_per_batch=args.runs_per_batch)
    meta = dict(meta, train_time_sec=time.time() - t0, best_step=best["step"],
                val_loss=best["loss"], agreement=final["agreement"],
                agreement_shuffled=final["agreement_shuffled"],
                agreement_within_run=final["agreement_within_run"],
                agreement_margin=final["agreement_margin"], partial=False)
    save(out, model, meta)
    print(f"[*] trained{tag}: val loss {best['loss']:.3f}, "
          f"agreement {final['agreement']:+.3f} "
          f"(shuffled {final['agreement_shuffled']:+.3f}, "
          f"within-run {final['agreement_within_run']:+.3f} vs untrained "
          f"{base['agreement_within_run']:+.3f}) -> {out}")
    return {"untrained": base, "trained": final, "history": history,
            "n_runs": len(tr_off) - 1, "n_trs": int(tr_data.shape[0]),
            "best_step": best["step"], "val_loss": best["loss"]}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out", default="results/orbitcon.pt")
    p.add_argument("--cache", default="results/orbit_cache.npz",
                   help="Shared with train_crossorbit.py -- same geometry.")
    p.add_argument("--embed", type=int, default=32,
                   help="Per orbit. The probe feature is twice this, so 32 "
                        "matches lr-cca:64, the arm this has to beat.")
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--expander", type=int, default=128)
    p.add_argument("--head", choices=("flat", "gap"), default="flat",
                   help="How the conv feature map becomes an embedding. `flat` "
                        "keeps its spatial layout, which is where gaze lives; "
                        "`gap` averages it away and is the ablation.")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2,
                   help="Deliberately 100x AdamW's usual default; independent "
                        "acquisitions are the scarce resource here.")
    p.add_argument("--runs-per-batch", type=int, default=4,
                   help="Runs a batch is drawn from. Small on purpose: VICReg's "
                        "variance term is a batch statistic, and a batch "
                        "spanning hundreds of participants can satisfy it by "
                        "encoding participant identity.")
    p.add_argument("--shift", type=float, default=DEFAULT_AUG["shift_voxels"])
    p.add_argument("--dropout", type=float, default=DEFAULT_AUG["dropout"])
    p.add_argument("--noise", type=float, default=DEFAULT_AUG["noise"])
    p.add_argument("--gain", type=float, default=DEFAULT_AUG["gain"])
    p.add_argument("--bias", type=float, default=DEFAULT_AUG["bias"])
    p.add_argument("--mirror-right", action="store_true",
                   help="Keep `split_orbits`' x-flip of the right orbit. OFF by "
                        "default and that is the substantive choice: the flip "
                        "inverts horizontal gaze between the two orbits, so "
                        "VICReg's invariance term penalises the very feature we "
                        "want (see deepmreye/orbitcon.unmirror_right). This flag "
                        "exists to reproduce the failure as an ablation.")
    p.add_argument("--no-center-runs", action="store_true",
                   help="Skip per-run re-centering. An ablation, not a default: "
                        "the residual static component is what an invariance "
                        "term latches onto first.")
    p.add_argument("--max-subjects", type=int, default=500)
    p.add_argument("--trs-per-subject", type=int, default=128)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--include-labeled", action="store_true")
    p.add_argument("--exclude-datasets", nargs="*", default=())
    p.add_argument("--scaling", nargs="*", type=int, default=None,
                   help="Run counts to train at, e.g. `--scaling 50 100 200 "
                        "400`. Writes one checkpoint per point plus a summary, "
                        "so 'does more data help' is measured rather than "
                        "assumed before committing to a long run.")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data_dir = resolve(args.data_dir, download=False, quiet=True)
    print(f"[*] data {data_dir}")

    data, offsets = load_cache(args.cache, data_dir, args.max_subjects,
                              args.trs_per_subject, args.include_labeled,
                              args.exclude_datasets)
    # float16 in the cache, float32 for arithmetic: centering in float16 would
    # lose more than the mean it removes.
    data = np.asarray(data, dtype=np.float32)
    # The cache is built by `crossorbit.build_orbit_cache`, which mirrors the
    # right orbit. Undo it unless the ablation asks for it.
    if not args.mirror_right:
        data = unmirror_right(data)
        print("[*] un-mirrored the right orbit (contrastive geometry)")
    if not args.no_center_runs:
        data = center_runs(data, offsets)
        print(f"[*] re-centered {len(offsets) - 1} runs per voxel")

    (tr_data, tr_off), (val_data, val_off) = split_runs(data, offsets,
                                                        args.val_frac, args.seed)
    print(f"[*] train {tr_data.shape[0]} TRs / {len(tr_off) - 1} runs, "
          f"val {val_data.shape[0]} TRs / {len(val_off) - 1} runs")
    print(f"[*] device {device_for(args.device)}; embed {args.embed} per orbit "
          f"({2 * args.embed} features), wd {args.weight_decay}, "
          f"{args.runs_per_batch} runs/batch")

    out_dir, stem = Path(args.out).parent, Path(args.out).stem
    if args.scaling:
        results = {}
        for n in args.scaling:
            n = min(n, len(tr_off) - 1)
            sub_data, sub_off = subset_runs(tr_data, tr_off, n, args.seed)
            print(f"\n[*] === scaling point: {n} runs, "
                  f"{sub_data.shape[0]} TRs ===")
            # Named off `--out`, not off the run count alone. Two configurations
            # swept into one directory otherwise write the same file: a `--shift
            # 0` run silently overwrote the full-augmentation checkpoint at the
            # same n, and nothing said so.
            out = out_dir / f"{stem}_n{n}.pt"
            results[str(n)] = run_one(sub_data, sub_off, val_data, val_off,
                                      args, out, tag=f" (n={n})")
        summary = out_dir / f"{stem}_scaling.json"
        summary.write_text(json.dumps(results, indent=2, default=float))
        print(f"\n[*] scaling summary -> {summary}")
        # within-run agreement is the column that matters: pooled agreement is
        # equally consistent with the encoder having learned only anatomy.
        print(f"{'runs':>6}  {'TRs':>7}  {'val loss':>9}  {'agree':>7}  "
              f"{'shuf':>7}  {'within-run':>11}  {'untr w-run':>11}")
        for n, r in results.items():
            u, t = r["untrained"], r["trained"]
            print(f"{n:>6}  {r['n_trs']:>7}  {r['val_loss']:>9.3f}  "
                  f"{t['agreement']:>+7.3f}  {t['agreement_shuffled']:>+7.3f}  "
                  f"{t['agreement_within_run']:>+11.3f}  "
                  f"{u['agreement_within_run']:>+11.3f}")
    else:
        res = run_one(tr_data, tr_off, val_data, val_off, args, args.out)
        (out_dir / f"{stem}_history.json").write_text(
            json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
