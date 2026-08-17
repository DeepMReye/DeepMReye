#!/usr/bin/env python3
"""Pretrain the cross-orbit Orbit-JEPA on the unlabeled corpus.

The model is a linear identity path plus a zero-initialised MLP over the frozen
canonical pre-projection, so at step 0 its features are *exactly* `lr-cca:k`
(median r 0.825 on the 7 verified folds). Training can therefore only be
credited with what it adds on top of the best linear corpus basis, and the
untrained control that `eval_probe --features jepa-random` builds is that
baseline rather than a random projection.

The labeled datasets (`dsL*`) are never pretrained on -- `build_corpus_cache`
skips any participant carrying `labels`.

Usage
-----
    # once: reduce the corpus to its canonical pre-projection (~280 MB)
    python scripts/train_orbitjepa.py --build-cache --max-files 1039

    # then train (seconds, reuses the cache)
    python scripts/train_orbitjepa.py --dim 32 --epochs 40 --out results/orbitjepa.pt

    # and evaluate against the linear arms on identical folds
    python scripts/eval_probe.py --protocol dataset --readouts ridge-cv \\
        --standardize-targets dataset --exclude-datasets dsL11_backtothefuture \\
        --jepa-checkpoint results/orbitjepa.pt \\
        --features jepa jepa-random lr-cca:32 fold-pca:64
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.datasource import resolve
from deepmreye.models.jepa_net import OrbitJEPA
from deepmreye.orbitjepa import (
    build_corpus_cache,
    load_cache,
    save_cache,
    save_checkpoint,
    train_orbit_jepa,
)
from deepmreye.unsupervised import load_basis


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--basis", default="results/scaling/basis_n1039.npz")
    p.add_argument("--cache", default="results/jepa_cache.npz")
    p.add_argument("--build-cache", action="store_true",
                   help="Rebuild the canonical pre-projection cache from the corpus.")
    p.add_argument("--max-files", type=int, default=1039,
                   help="Unlabeled participants to cache.")
    p.add_argument("--m", type=int, default=256,
                   help="Canonical directions per orbit fed to the encoder.")
    p.add_argument("--regress-motion", action="store_true",
                   help="Project out the mean-signal motion proxy per orbit first. "
                        "The one untested suggestion left by the next-TR and ocon "
                        "results; note it changes the control as well as the model.")
    p.add_argument("--dim", type=int, default=32,
                   help="Latent width k. Match it to the lr-cca:k arm being beaten.")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--head", choices=("avg", "concat"), default="avg")
    p.add_argument("--freeze-linear", action="store_true",
                   help="Pin the linear path at the lr-cca solution, so the model "
                        "is 'lr-cca plus a learned non-linear correction'.")
    p.add_argument("--temp-kernel", type=int, default=1,
                   help="Temporal kernel size for 1D spatiotemporal causal convolution (default 1 = spatial only, 3 = spatiotemporal).")
    p.add_argument("--alpha-gate", type=float, default=1.0,
                   help="Initial ReZero alpha gating scalar.")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--sigreg-weight", type=float, default=1.0)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--aug-dropout", type=float, default=0.0)
    p.add_argument("--gain", type=float, default=0.0,
                   help="Per-view gain jitter. Off by default: a gaze coordinate "
                        "IS a signed amplitude, so independent rescaling of the "
                        "two orbits corrupts the conjugate relationship.")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--snapshot-every", type=int, default=0,
                   help="Also save a checkpoint every N epochs, as `<out>.ep<N>.pt`. "
                        "The cross-orbit objective is known to improve while gaze "
                        "decoding falls (see the ocon result), so the probe has to be "
                        "measured along the trajectory, not only at the objective's best.")
    p.add_argument("--out", default="results/orbitjepa.pt")
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    mask, bases, meta = load_basis(args.basis)
    basis = bases["lr-cca"]

    cache = Path(args.cache)
    if args.build_cache or not cache.exists():
        root = Path(resolve(args.data_dir, download=False, quiet=True))
        paths = sorted(root.glob("ds0*/*.h5")) + sorted(root.glob("ds1*/*.h5"))
        print(f"[*] building cache from {len(paths)} candidate unlabeled files "
              f"(cap {args.max_files})...")
        z, run_id, runs = build_corpus_cache(paths, mask, basis, m=args.m,
                                             max_files=args.max_files,
                                             regress_motion=args.regress_motion)
        save_cache(cache, z, run_id, runs, args.basis, args.m, args.regress_motion)
        print(f"[*] cached {len(runs)} runs / {len(z)} TRs -> {cache} "
              f"({cache.stat().st_size / 1e6:.0f} MB)")
    else:
        z, run_id, n_runs = load_cache(cache, args.basis, args.m, args.regress_motion)
        runs = [None] * n_runs
        print(f"[*] cache {cache}: {n_runs} runs / {len(z)} TRs, M={args.m}")

    model = OrbitJEPA(in_dim=z.shape[-1], latent_dim=args.dim,
                      hidden_dim=args.hidden_dim, depth=args.depth,
                      dropout=args.dropout, sigreg_weight=args.sigreg_weight,
                      train_linear=not args.freeze_linear,
                      temp_kernel=args.temp_kernel, alpha_gate=args.alpha_gate)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Orbit-JEPA k={args.dim} over M={z.shape[-1]} canonical dirs, "
          f"hidden {args.hidden_dim}x{args.depth}, {n_params} trainable params")
    print(f"[*] warm start == lr-cca:{args.dim} exactly; training for "
          f"{args.epochs} epochs (lr {args.lr}, wd {args.weight_decay}, "
          f"sigreg {args.sigreg_weight})")

    out_base = Path(args.out)

    def snapshot(epoch, live_model, row):
        if args.snapshot_every and epoch % args.snapshot_every == 0:
            path = out_base.with_suffix(f".ep{epoch:03d}.pt")
            save_checkpoint(path, live_model, args.basis, args.m, args.head,
                            args.regress_motion,
                            meta={"epoch": epoch, "best_epoch": epoch,
                                  "best_val_loss": row["val_loss"],
                                  "nonlinear_share": row["nonlinear_share"],
                                  "snapshot": True, "args": vars(args)})

    model, info = train_orbit_jepa(
        model, z, run_id, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, noise=args.noise,
        dropout=args.aug_dropout, gain=args.gain, val_frac=args.val_frac,
        seed=args.seed, device=args.device, on_epoch=snapshot)

    meta_out = dict(info)
    meta_out.update({"n_runs": len(runs), "n_trs": int(len(z)),
                     "nonlinear_share": info["history"][info["best_epoch"] - 1]["nonlinear_share"],
                     "args": vars(args)})
    save_checkpoint(args.out, model, args.basis, args.m, args.head,
                    args.regress_motion, meta=meta_out)
    print(f"[*] -> {args.out}")


if __name__ == "__main__":
    main()
