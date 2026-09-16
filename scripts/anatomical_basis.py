#!/usr/bin/env python3
"""A gaze basis derived from anatomy, with no corpus and nothing fitted.

Every arm in this project estimates its filters from data. But the response of a
voxel to an eye movement is not an empirical question to first order: **an
eyeball is a rigid, high-contrast sphere, so rotating it changes a voxel's
intensity by the spatial gradient of the eye image along the displacement.**

    y(v, t) ~ m(v) + grad m(v) . d(v, t),    d(v, t) = omega(t) x (v - c)

for an eye image `m`, an orbit centre `c` and a gaze rotation `omega`. That is
**three** directions per orbit -- one per rotation axis -- written down rather
than estimated, and two of the three are the horizontal and vertical gaze the
probe scores.

The reason this was not tried earlier is that `eye_block` is z-scored per voxel
across time, so the stored corpus has no `m` in it (`mu` spans +-0.04). But `m`
does not have to come from the corpus: `deepmreye/masks/dme_template.nii` is the
template every participant is *registered to*, it is a real anatomical volume,
and cropping it with `cut_mask`'s own edges lands it on exactly the 14236-voxel
grid the corpus lives on. Zero new preprocessing.

Two details that are not free choices:

- **Gradients are taken per orbit, before the crop halves are concatenated.**
  `cut_mask` builds the block as `[right | left]`, so the x axis has a seam at
  the midline and a gradient across it would be an artefact of the layout.
- **`logm` is offered beside `m`.** The stored data is unit-variance per voxel,
  so the matched filter for a gaze displacement is `grad m / sigma`, not
  `grad m`; BOLD temporal SD scales roughly with intensity, which makes
  `grad log m = grad m / m` the cheap stand-in for the sigma we no longer have.

    python scripts/anatomical_basis.py --voxels <voxdir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmreye import probe  # noqa: E402
from scripts.improve_lrcca import (  # noqa: E402
    K_STORE, concat, project, slicer, summarise,
)

LR_SPLIT_X = 24


def template_block(smooth=0.0):
    """The DeepMReye template on the eye-block grid, plus its per-orbit gradients.

    Returns `(m, grads)` where `m` is `[47, 29, 18]` and `grads` is
    `[47, 29, 18, 3]`, both already zeroed outside the eye mask.
    """
    from scipy.ndimage import gaussian_filter

    from deepmreye.preprocess import get_masks

    _es, _eb, tmpl, mask, xe, ye, ze = get_masks()
    full = tmpl.numpy().astype(np.float64)
    if smooth:
        full = gaussian_filter(full, smooth)

    halves, grads = [], []
    for lo, hi in ((xe[3], xe[2]), (xe[1], xe[0])):     # right, then left
        sub = full[lo:hi, ye[1]:ye[0], ze[1]:ze[0]]
        halves.append(sub)
        grads.append(np.stack(np.gradient(sub), axis=-1))
    m = np.concatenate(halves)
    g = np.concatenate(grads)

    keep = np.zeros(m.shape, dtype=bool)
    msk = mask.copy()
    msk[msk < 1] = 0
    for i, (lo, hi) in enumerate(((xe[3], xe[2]), (xe[1], xe[0]))):
        sub = msk[lo:hi, ye[1]:ye[0], ze[1]:ze[0]] > 0
        off = 0 if i == 0 else halves[0].shape[0]
        keep[off:off + sub.shape[0]] = sub
    m[~keep] = 0.0
    g[~keep] = 0.0
    return m, g, keep


def orbit_centres(m, keep):
    """Intensity-weighted centre of each eyeball, for the rotation lever arm.

    Weighted by `m^2` so the bright vitreous dominates: the sphere is what
    rotates, and the orbital fat around it does not.
    """
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in m.shape], indexing="ij"), -1)
    out = {}
    for side, sel in (("right", grid[..., 0] < LR_SPLIT_X),
                      ("left", grid[..., 0] >= LR_SPLIT_X)):
        w = np.where(sel & keep, m, 0.0) ** 2
        out[side] = (grid * w[..., None]).sum((0, 1, 2)) / w.sum()
    return out


def modes(m, g, keep, kind="rot", log=False):
    """The analytic response maps, as `[n_voxels, 3]` per orbit.

    `rot` is the physical model -- a rigid rotation about the orbit centre, so
    the displacement at `v` is `omega x (v - c)` and the response is
    `-grad m . (omega x (v - c))`. `trans` is the cruder rigid-translation
    model, kept because registration error is a translation and the two make
    different predictions about the spatial pattern.
    """
    gg = g / np.maximum(m, 1e-6)[..., None] if log else g
    gg = np.where(keep[..., None], gg, 0.0)
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in m.shape], indexing="ij"), -1)
    centres = orbit_centres(m, keep)

    out = {}
    for side, sel in (("right", grid[..., 0] < LR_SPLIT_X),
                      ("left", grid[..., 0] >= LR_SPLIT_X)):
        cols = []
        for axis in range(3):
            if kind == "trans":
                d = np.zeros(3)
                d[axis] = 1.0
                field = np.broadcast_to(d, grid.shape)
            else:
                omega = np.zeros(3)
                omega[axis] = 1.0
                field = np.cross(omega, grid - centres[side])
            resp = -(gg * field).sum(-1)
            cols.append(np.where(sel & keep, resp, 0.0).reshape(-1))
        out[side] = np.stack(cols, axis=1)
    return out


def to_block(per_side, flat_mask, combine="avg"):
    """Per-orbit voxel maps -> one `[14236, k]` projection on the corpus grid.

    `avg` mirrors what the cross-orbit arm hands its readout -- the two orbits
    are two measurements of one conjugate gaze -- and `concat` keeps them apart
    so the readout can weight them itself.
    """
    left = per_side["left"][flat_mask]
    right = per_side["right"][flat_mask]
    for w in (left, right):
        n = np.linalg.norm(w, axis=0)
        n[n < 1e-12] = 1.0
        w /= n
    return np.concatenate([left, right], axis=1) if combine == "concat" \
        else 0.5 * (left + right)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--voxels", required=True)
    p.add_argument("--out", default="results/lrcca_variants/anatomical.json")
    p.add_argument("--corpus", default=None,
                   help="Optional lrcca_variants moments file, to score the "
                        "concatenation with the fitted cross-orbit basis.")
    a = p.parse_args()

    mask = np.load(Path(a.voxels) / "mask.npy")
    flat = mask.reshape(-1)
    n_vox = int(flat.sum())

    blocks = []
    for smooth in (0.0, 1.0):
        m, g, keep = template_block(smooth)
        print(f"[*] smooth {smooth}: template on {int(keep.sum())} voxels, "
              f"range {m[keep].min():.0f}..{m[keep].max():.0f}", flush=True)
        for kind in ("rot", "trans"):
            for log in (False, True):
                per_side = modes(m, g, keep, kind, log)
                tag = f"{kind}{'-log' if log else ''}_s{smooth:g}"
                for combine in ("avg", "concat"):
                    w = to_block(per_side, flat, combine)
                    blocks.append((f"{tag}_{combine}", w.astype(np.float32),
                                   np.zeros(n_vox), w.shape[1]))

    recs, index = project(a.voxels, blocks)
    got = {}
    print(f"\n{'arm':<34}{'k':>4}{'sub-TR':>9}{'1-TR':>9}")
    for name, w, _mu, k in blocks:
        got[name] = summarise(probe.lodo(recs, slicer(index, name, lags=1)))
        print(f"{name:<34}{k:>4}{got[name]['subtr']:>9.4f}{got[name]['1tr']:>9.4f}",
              flush=True)
        Path(a.out).write_text(json.dumps(got, indent=1))

    # The two together: does an analytic direction carry anything the fitted
    # basis missed, or is it already inside its span?
    if a.corpus:
        from scripts.improve_lrcca import Corpus, Eigs, cca_to_block, fit_cca
        corpus = Corpus(a.corpus)
        li, ri = corpus.orbits()
        cov = corpus.c.astype(np.float64) / corpus.n - np.outer(corpus.mu, corpus.mu)
        fit = fit_cca(cov, li, ri, K_STORE, 256, 1e-3, eigs=Eigs(), key="total")
        del cov
        extra = [("lr-cca",) + cca_to_block(fit, n_vox, corpus.mu) + (K_STORE,)]
        best = max(got, key=lambda n: got[n]["subtr"])
        keepw = [b for b in blocks if b[0] == best]
        recs, index = project(a.voxels, extra + keepw)
        for name, fn in (("lr-cca:32", slicer(index, "lr-cca", k=32)),
                         (f"lr-cca:32 + {best}",
                          concat([slicer(index, "lr-cca", k=32),
                                  slicer(index, best)]))):
            got[name] = summarise(probe.lodo(recs, fn))
            print(f"{name:<34}{'':>4}{got[name]['subtr']:>9.4f}"
                  f"{got[name]['1tr']:>9.4f}", flush=True)
            Path(a.out).write_text(json.dumps(got, indent=1))
    print(f"\n[+] {a.out}")


if __name__ == "__main__":
    main()
