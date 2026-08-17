#!/usr/bin/env python3
"""Is a cross-orbit bottleneck failing because it is dead, or because it is small?

The leave-one-dataset-out probe gives one number per arm and cannot tell those
apart. ``xrot`` scores 0.293 against ``xorb``'s 0.389, which reads like a worse
model -- but the diagnostics below say the rotation bottleneck is *healthy* and
merely narrow, while ``xorb``'s larger score is mostly its untrained
architecture. Four measurements, none of which the probe table shows:

- **within-subject decodability.** Fit the readout inside one participant, where
  anatomy is constant and only gaze varies. This is the bottleneck's own ceiling,
  separate from whether it transfers. ``xrot`` reads 0.428 from 4 numbers,
  ``xorb`` 0.635 from 24.
- **latent travel.** How far the latent actually moves across a run. ``xorb``'s
  coordinate travels 0.187 voxels -- the measurement that motivated replacing a
  centroid with a rotation, because gaze rotates the eyeball rather than
  translating it.
- **left/right agreement.** Both eyes rotate conjugately, so the two orbits must
  produce the *same* latent if the cross-orbit objective did its job. This is
  the direct check that it did, and it is unavailable from the probe.
- **saturation.** If the angles sit at their ``tanh`` cap the bottleneck is
  clipped; if they barely move it is collapsed. Either invalidates the arm.

For ``xrot`` it also renders the learned canonical orbit, which is the one place
you can see what the model thinks an eyeball looks like.

    python scripts/analyze_orbit_bottleneck.py --checkpoint results/orbitrot.pt
    python scripts/analyze_orbit_bottleneck.py --checkpoint results/crossorbit_k4.pt --kind xorb
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepmreye.crossorbit import split_orbits  # noqa: E402
from deepmreye.datasource import resolve  # noqa: E402

OUT_DIR = Path("media/visualizations")


def encode_run(model, block, torch):
    """``[X, Y, Z, T]`` -> ``(latent [T, 2*k], per-orbit latents)``."""
    left, right = split_orbits(block)
    out = []
    with torch.no_grad():
        for v in (left, right):
            v = v.transpose(3, 0, 1, 2).astype(np.float32)
            lat, _ = model.encode(torch.from_numpy(v[:, None]).to(model.device))
            out.append(lat.reshape(len(v), -1).cpu().numpy())
    return np.concatenate(out, axis=1), out


def within_subject_r(feat, gaze, train_frac=0.6, min_rows=40):
    """Ridge fitted *inside* one participant. Anatomy is constant here, so this
    isolates what the latent carries about gaze from whether it transfers."""
    if len(feat) < min_rows:
        return np.nan
    cut = int(len(feat) * train_frac)
    model = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(feat[:cut], gaze[:cut])
    pred = model.predict(feat[cut:])
    rs = [np.corrcoef(pred[:, j], gaze[cut:, j])[0, 1]
          for j in range(2) if gaze[cut:, j].std() > 1e-9]
    return float(np.nanmean(rs)) if rs else np.nan


def render_template(model, torch, path):
    """The learned canonical orbit, as max-intensity projections per axis."""
    tmpl = model.template.detach().cpu().numpy()[0]          # [C, X, Y, Z]
    parts = getattr(model, "n_parts", 1)
    per = tmpl.shape[0] // parts
    fig, axes = plt.subplots(parts, 3, figsize=(8.5, 2.7 * parts), squeeze=False)
    for p in range(parts):
        block = tmpl[p * per:(p + 1) * per]
        # Energy across channels: what this part contributes at each voxel.
        vol = np.sqrt((block ** 2).sum(0))
        for a, (axis, name) in enumerate(zip((0, 1, 2), ("x", "y", "z"))):
            ax = axes[p][a]
            ax.imshow(vol.max(axis=axis).T, cmap="magma", origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"part {p}, max proj along {name}", fontsize=8)
    fig.suptitle("The learned canonical orbit (what the rotation acts on)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", default="results/orbitrot.pt")
    p.add_argument("--kind", choices=["xrot", "xorb"], default="xrot")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--per-dataset", type=int, default=5)
    p.add_argument("--out", default=None)
    p.add_argument("--untrained", action="store_true",
                   help="Rebuild the same architecture from the checkpoint's "
                        "metadata and DISCARD the weights. Required to read "
                        "left/right agreement: both orbits sit in one volume, "
                        "so global signal, motion and drift are common to them "
                        "and a random encoder can agree with itself. Only the "
                        "excess over this control is evidence of shared gaze.")
    args = p.parse_args()

    import torch

    if args.kind == "xrot":
        from deepmreye.orbitrot import load
    else:
        from deepmreye.crossorbit import load
    model, meta = load(args.checkpoint, device="cpu")
    if args.untrained:
        if args.kind == "xrot":
            from deepmreye.orbitrot import RotationOrbitModel

            model = RotationOrbitModel(
                meta.get("angles", 2), meta["n_nuisance"], meta.get("width", 16),
                seed=1000, device="cpu",
                template_channels=meta.get("template_channels", 8),
                n_parts=meta.get("parts", 1)).eval()
        else:
            from deepmreye.crossorbit import CrossOrbitModel

            model = CrossOrbitModel(meta["keypoints"], meta["n_nuisance"],
                                    meta.get("width", 16), seed=1000,
                                    device="cpu").eval()
        print("[!] WEIGHTS DISCARDED -- untrained control")
    data_dir = resolve(args.data_dir, download=False, quiet=True)

    cap = getattr(model, "max_angle", None)
    print(f"[*] {args.checkpoint}  kind={args.kind}  "
          f"bottleneck={model.k}/orbit"
          + (f"  parts={getattr(model, 'n_parts', 1)}  cap={cap} rad" if cap else ""))
    print(f"[*] trained contribution {meta.get('coord_contribution', float('nan')):+.4f} "
          f"(untrained {meta.get('coord_contribution_untrained', float('nan')):+.4f})")

    rows = []
    for ds in sorted(d.name for d in Path(data_dir).glob("dsL*")):
        for path in sorted((Path(data_dir) / ds).glob("*.h5"))[: args.per_dataset]:
            with h5py.File(path, "r") as f:
                block, labels = f["eye_block"][...], f["labels"][...]
            gaze = np.nanmean(labels, axis=1)
            n = min(block.shape[-1], len(gaze))
            block, gaze = block[..., :n], gaze[:n]
            ok = np.isfinite(gaze).all(1)
            block, gaze = block[..., ok], gaze[ok]
            if block.shape[-1] < 60:
                continue

            feat, (lat_l, lat_r) = encode_run(model, block, torch)
            agree = [np.corrcoef(lat_l[:, k], lat_r[:, k])[0, 1]
                     for k in range(lat_l.shape[1])
                     if lat_l[:, k].std() > 1e-9 and lat_r[:, k].std() > 1e-9]
            rows.append({
                "dataset": ds, "subject": path.stem,
                "within_r": within_subject_r(feat, gaze),
                "travel": float(feat.std(0).mean()),
                "lr_agreement": float(np.nanmean(agree)) if agree else np.nan,
                "saturation": (float(np.abs(feat).max() / cap) if cap else np.nan),
            })

    def med(key):
        v = np.array([r[key] for r in rows], dtype=float)
        return float(np.nanmedian(v[np.isfinite(v)])) if np.isfinite(v).any() else np.nan

    print(f"\n[*] {len(rows)} participants\n")
    print(f"  within-subject gaze r        {med('within_r'):.3f}"
          "   <- the bottleneck's own ceiling")
    print(f"  latent travel (SD across TRs) {med('travel'):.4f}")
    print(f"  left/right agreement          {med('lr_agreement'):+.3f}"
          "   <- conjugate gaze; the cross-orbit objective's own check")
    if cap:
        print(f"  peak |angle| / cap            {med('saturation'):.3f}"
              "   <- ~1.0 means clipped, ~0 means collapsed")

    print(f"\n  {'dataset':<24}{'n':>4}{'within r':>10}{'L/R agree':>11}")
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    for ds, rs in sorted(by_ds.items()):
        w = np.nanmean([r["within_r"] for r in rs])
        a = np.nanmean([r["lr_agreement"] for r in rs])
        print(f"  {ds:<24}{len(rs):>4}{w:>10.3f}{a:>11.3f}")

    out = Path(args.out or f"results/bottleneck_{Path(args.checkpoint).stem}.json")
    out.write_text(json.dumps({"checkpoint": args.checkpoint, "kind": args.kind,
                               "k_per_orbit": int(model.k),
                               "summary": {k: med(k) for k in
                                           ("within_r", "travel", "lr_agreement",
                                            "saturation")},
                               "per_subject": rows}, indent=1))
    print(f"\n[+] wrote {out}")

    if args.kind == "xrot":
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fig = render_template(
            model, torch, OUT_DIR / f"09_template_{Path(args.checkpoint).stem}.png")
        print(f"[+] Saved {fig}")


if __name__ == "__main__":
    main()
