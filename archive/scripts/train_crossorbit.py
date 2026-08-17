#!/usr/bin/env python3
"""Pretrain a cross-orbit model on the unlabeled corpus.

Reconstructs each orbit from the *other* orbit's bottleneck plus its own
nuisance code taken from a different TR. Two bottlenecks share this script, and
sharing it is the point -- same cache, same run split, same optimizer, same
selection rule -- so the only difference between the arms is the bottleneck:

- ``--bottleneck xorb`` soft-argmax **position** (``deepmreye/crossorbit.py``).
- ``--bottleneck xrot`` a 2-DOF **rotation** of a learned canonical orbit
  (``deepmreye/orbitrot.py``). Gaze rotates the eyeball rather than translating
  it, and the measured travel of the xorb coordinate -- 0.187 voxels over a
  whole run -- is what a centroid looks like on a rotational signal.

    python scripts/train_crossorbit.py --bottleneck xrot --steps 4000
    python scripts/train_crossorbit.py --keypoints 1 --nuisance 16   # tighter

Watch ``coord contributes`` in the log, not the reconstruction R^2. If shuffling
the coordinates across the batch does not hurt reconstruction, the bottleneck is
dead and the decoder is running off the nuisance path -- the failure mode this
architecture is most prone to, and the reason selection is on that number.

The gaze-labeled datasets are excluded, so one model is valid for every
leave-one-dataset-out fold.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.crossorbit import (
    ORBIT_SHAPE,
    CrossOrbitModel,
    build_orbit_cache,
    evaluate,
    save,
    train,
)
from deepmreye.datasource import resolve
from deepmreye.orbitrot import RotationOrbitModel
from deepmreye.temporal import device_for
from deepmreye.unsupervised import unlabeled_subjects

BOTTLENECKS = ("xorb", "xrot")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out", default=None,
                   help="default: results/crossorbit.pt / results/orbitrot.pt")
    p.add_argument("--bottleneck", choices=BOTTLENECKS, default="xorb",
                   help="xorb = soft-argmax position; xrot = 2-DOF rotation")
    p.add_argument("--cache", default="results/orbit_cache.npz",
                   help="Cached orbit volumes; built on first use, reused after.")
    p.add_argument("--keypoints", type=int, default=2,
                   help="xorb only. Coordinates per orbit. Each is 3 numbers, so "
                        "K=2 is a 6-dimensional bottleneck -- keep it small.")
    p.add_argument("--angles", type=int, default=2, choices=(2, 3),
                   help="xrot only. 2 is the true dimensionality of gaze; 3 adds "
                        "torsion, which the eye has but gaze does not.")
    p.add_argument("--template-channels", type=int, default=8,
                   help="xrot only. Channels of the learned canonical orbit.")
    p.add_argument("--parts", type=int, default=1,
                   help="xrot only. Independently rotating template blocks. The "
                        "latent stays a set of rotations but widens to "
                        "2*parts per orbit -- the capacity control for whether "
                        "4 dimensions is the limit or the encoder is.")
    p.add_argument("--nuisance", type=int, default=32)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-subjects", type=int, default=500)
    p.add_argument("--trs-per-subject", type=int, default=128)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--include-labeled", action="store_true")
    p.add_argument("--exclude-datasets", nargs="*", default=())
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data_dir = resolve(args.data_dir, download=False, quiet=True)
    print(f"[*] data {data_dir}")

    cache = Path(args.cache)
    if cache.exists():
        blob = np.load(cache)
        data, offsets = blob["data"], blob["offsets"]
        # The cache stores raw orbit volumes, so it is only valid for the orbit
        # geometry that built it. Silently reusing one across a change to the
        # left/right split produces a shape error deep inside the loss -- or,
        # worse, no error at all if the sizes happen to agree.
        cached_shape = tuple(blob["orbit_shape"]) if "orbit_shape" in blob else None
        if cached_shape != tuple(ORBIT_SHAPE):
            raise SystemExit(
                f"[!] cache {cache} was built for orbit shape {cached_shape}, but "
                f"the current split gives {tuple(ORBIT_SHAPE)}. Delete it and "
                f"rerun to rebuild.")
        print(f"[*] reusing cache {cache} ({data.shape[0]} TRs, "
              f"{len(offsets) - 1} runs)")
    else:
        subjects = unlabeled_subjects(data_dir, include_labeled=args.include_labeled,
                                      exclude_datasets=args.exclude_datasets)
        if args.max_subjects:
            idx = np.unique(np.linspace(0, len(subjects) - 1,
                                        args.max_subjects).astype(int))
            subjects = [subjects[i] for i in idx]
        print(f"[*] caching orbits for {len(subjects)} participants "
              f"({len({s[0] for s in subjects})} datasets)")
        t0 = time.time()
        data, offsets = build_orbit_cache(subjects, args.trs_per_subject, progress=100)
        np.savez(cache, data=data, offsets=offsets,
                 orbit_shape=np.array(ORBIT_SHAPE))
        print(f"[*] {data.shape[0]} TRs, {data.nbytes / 1e9:.2f} GB, "
              f"{time.time() - t0:.0f}s -> {cache}")

    # Split by run: a validation timepoint from a run the model trained on would
    # share that participant's anatomy through the nuisance path.
    n_runs = len(offsets) - 1
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_runs)
    n_val = max(1, int(n_runs * args.val_frac))

    def gather(runs):
        parts = [data[offsets[i]: offsets[i + 1]] for i in runs]
        return np.concatenate(parts), np.cumsum([0] + [len(x) for x in parts])

    tr_data, tr_off = gather(order[n_val:])
    val_data, val_off = gather(order[:n_val])
    print(f"[*] train {tr_data.shape[0]} TRs / {len(order) - n_val} runs, "
          f"val {val_data.shape[0]} TRs / {n_val} runs")

    device = device_for(args.device)

    # One factory, used for both the control and the trained model: a control
    # built by a different code path is not a control.
    if args.bottleneck == "xrot":
        def build():
            return RotationOrbitModel(
                args.angles, args.nuisance, args.width, args.seed, device,
                template_channels=args.template_channels, n_parts=args.parts)
        width = (f"{args.angles} angles x {args.parts} parts = "
                 f"{args.angles * args.parts} per orbit")
    else:
        def build():
            return CrossOrbitModel(args.keypoints, args.nuisance, args.width,
                                   args.seed, device)
        width = f"{args.keypoints}x3 coordinates per orbit"

    print(f"[*] device {device}; bottleneck {args.bottleneck} ({width}), "
          f"nuisance {args.nuisance}")

    control = build()
    base = evaluate(control, val_data, val_off, args.batch, args.seed + 1)
    print(f"[*] untrained: val R2 {base['r2']:+.4f}, "
          f"coord contributes {base['coord_contribution']:+.4f}")

    out = args.out or ("results/orbitrot.pt" if args.bottleneck == "xrot"
                       else "results/crossorbit.pt")

    model = build()
    t0 = time.time()
    history, best = train(model, tr_data, tr_off, val_data, val_off,
                          steps=args.steps, batch=args.batch, lr=args.lr,
                          seed=args.seed, checkpoint_path=out,
                          meta={"bottleneck": args.bottleneck,
                                "width": args.width,
                                "angles": args.angles,
                                "parts": args.parts,
                                "keypoints": args.keypoints,
                                "n_nuisance": args.nuisance,
                                "template_channels": args.template_channels,
                                "coord_contribution_untrained":
                                    base["coord_contribution"]})
    final = max(history, key=lambda m: m["coord_contribution"])
    print(f"[*] trained in {time.time() - t0:.0f}s: "
          f"val R2 {final['r2']:+.4f}, coord contributes "
          f"{best['coord_contribution']:+.4f} (untrained "
          f"{base['coord_contribution']:+.4f})")
    if best["coord_contribution"] < 0.005:
        print("[!] the coordinate barely changes reconstruction -- the bottleneck "
              "is not being used, and the probe result will reflect the "
              "architecture, not the objective.")

    meta = {
        "n_trs": int(data.shape[0]),
        "n_runs": int(n_runs),
        "bottleneck": args.bottleneck,
        "keypoints": args.keypoints,
        "angles": args.angles,
        "template_channels": args.template_channels,
        "parts": args.parts,
        "n_nuisance": args.nuisance,
        "width": args.width,
        "steps": args.steps,
        "val_r2": final["r2"],
        "coord_contribution": best["coord_contribution"],
        "coord_contribution_untrained": base["coord_contribution"],
        "best_step": best["step"],
        "include_labeled": bool(args.include_labeled),
        "excluded_datasets": sorted(args.exclude_datasets),
        "history": history,
        "trained": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = save(out, model, meta)
    print(f"[*] wrote {path}")
    print(json.dumps({k: v for k, v in meta.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
