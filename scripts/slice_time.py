#!/usr/bin/env python3
"""The z axis of an eye block is partly a *time* axis. The basis averages over it.

The reason ten gaze labels per TR is a meaningful target at all is acquisition
physics, stated in the DeepMReye paper itself: "different imaging slices are
being acquired at a different time within each TR and thus inherently carry some
sub-TR information". A volume is not an instant. It is a sweep.

Every basis in this project pools all eighteen z slices of the eye crop into one
spatial pattern, which throws that sweep away -- and the wider slice-timing
literature is entirely about *correcting* the effect rather than decoding with
it. So this asks two questions that have not been asked here:

1. **Diagnostic.** Restricting the cross-orbit basis to a band of z slices, does
   the band's prediction line up better with *early* or *late* gaze samples
   inside the TR, and does that preference move with z? `r(band, sample)`
   answers it directly. Bands differ in how much eyeball they contain, so the
   profile is normalised per band -- the shape across samples is the evidence,
   not its level.
2. **Arm.** Do z-resolved features decode better? The prediction is sharp and
   falsifiable: they should help **sub-TR and not 1-TR**, because averaging the
   ten samples destroys exactly the information a slice sweep carries.

One limitation to state up front rather than discover later: the corpus spans
TRs from 0.8 to 3 s with different slice counts, orientations and multiband
factors, so the map from template z to acquisition time is **not the same across
datasets**. A frozen z-resolved basis read by one pooled ridge is therefore
being asked to transfer a timing convention it cannot see. If the diagnostic
fires but the arm does not, that is the reason, and the next step is
within-dataset rather than a better basis.

    python scripts/slice_time.py --voxels <voxdir>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmreye import metrics, probe  # noqa: E402
from scripts.improve_lrcca import (  # noqa: E402
    Corpus, Eigs, cca_to_block, concat, fit_cca, project, slicer, summarise,
)


def z_bands(mask, n_bands):
    """Masked-voxel indices grouped into contiguous bands of z slices."""
    zs = np.nonzero(mask.reshape(-1))[0] % mask.shape[2]
    edges = np.linspace(0, mask.shape[2], n_bands + 1).astype(int)
    return [(int(edges[b]), int(edges[b + 1]),
             np.nonzero((zs >= edges[b]) & (zs < edges[b + 1]))[0])
            for b in range(n_bands)]


def sub_tr_profile(recs, feature_fn, seed=0):
    """`r(sample)` -- per within-TR gaze sample, the decoded correlation.

    A diagnostic, not a score: it reuses `probe.lodo`'s fit exactly (same folds,
    same target standardisation, same ridge) and only reads the ten sub-TR
    samples apart instead of pooling them. Nothing downstream quotes it.
    """
    from sklearn.linear_model import RidgeCV

    feats = {id(r): np.asarray(feature_fn(r), dtype=np.float64) for r in recs}
    datasets = sorted({r["dataset"] for r in recs})
    per_sample = [[] for _ in range(10)]
    for held in datasets:
        train = [r for r in recs if r["dataset"] != held]
        xs, ys = [], []
        for ds in sorted({r["dataset"] for r in train}):
            members = [r for r in train if r["dataset"] == ds]
            x = np.concatenate([feats[id(r)] for r in members])
            y = np.concatenate([r["labels"].reshape(len(r["labels"]), 20)
                                for r in members])
            ok = np.isfinite(y).all(axis=1) & np.isfinite(x).all(axis=1)
            if ok.sum() < 10:
                continue
            sd = y[ok].std(axis=0)
            sd[sd < 1e-9] = 1.0
            xs.append(x[ok])
            ys.append((y[ok] - y[ok].mean(axis=0)) / sd)
        if not xs:
            continue
        model = RidgeCV(alphas=probe.ALPHAS).fit(np.concatenate(xs), np.concatenate(ys))
        for s in (r for r in recs if r["dataset"] == held):
            pred = model.predict(feats[id(s)])
            lab = np.asarray(s["labels"])[:len(pred)]
            pred = pred.reshape(len(pred), 10, 2)
            for i in range(10):
                p, t = pred[:, i, :].ravel(), lab[:, i, :].ravel()
                ok = np.isfinite(p) & np.isfinite(t)
                if ok.sum() > 10:
                    per_sample[i].append(metrics.pearson(p[ok], t[ok]))
    return [float(metrics.nanmedian(v)) if v else float("nan") for v in per_sample]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--voxels", required=True)
    p.add_argument("--moments", default="results/lrcca_variants/moments_n2000.npz")
    p.add_argument("--out", default="results/lrcca_variants/slice_time.json")
    p.add_argument("--bands", type=int, default=6)
    p.add_argument("--k-band", type=int, default=16)
    p.add_argument("--k", type=int, default=32)
    a = p.parse_args()

    corpus = Corpus(a.moments)
    cov = corpus.cov("total")
    li, ri = corpus.orbits()
    n_vox = int(corpus.mask.sum())
    eigs = Eigs()

    blocks = [("full",) + cca_to_block(
        fit_cca(cov, li, ri, 64, 256, 1e-3, eigs=eigs, key="total"),
        n_vox, corpus.mu) + (64,)]

    bands = z_bands(corpus.mask, a.bands)
    t0 = time.time()
    for b, (lo, hi, idx) in enumerate(bands):
        bl = np.intersect1d(li, idx)
        br = np.intersect1d(ri, idx)
        fit = fit_cca(cov, bl, br, a.k_band, min(256, len(bl) - 1, len(br) - 1),
                      1e-3, eigs=eigs, key=f"band{b}")
        blk = cca_to_block(fit, n_vox, corpus.mu, a.k_band)
        blocks.append((f"band{b}",) + blk + (a.k_band,))
        print(f"  band {b} z[{lo}:{hi}]  {len(bl)}/{len(br)} voxels  "
              f"rho1 {fit['rho'][0]:.3f}  [{time.time() - t0:5.0f}s]", flush=True)
    del cov, corpus._cache

    recs, index = project(a.voxels, blocks)
    got = {}

    print(f"\n{'arm':<30}{'sub-TR':>9}{'1-TR':>9}")
    arms = {"lr-cca:32 [incumbent]": slicer(index, "full", k=a.k, lags=1)}
    for b in range(a.bands):
        arms[f"band{b} only"] = slicer(index, f"band{b}", lags=1)
    arms["all bands concatenated"] = concat(
        [slicer(index, f"band{b}", lags=1) for b in range(a.bands)])
    both = [slicer(index, "full", k=a.k, lags=1)]
    both += [slicer(index, f"band{b}", lags=1) for b in range(a.bands)]
    arms["incumbent + all bands"] = concat(both)
    for name, fn in arms.items():
        got[name] = summarise(probe.lodo(recs, fn))
        print(f"{name:<30}{got[name]['subtr']:>9.4f}{got[name]['1tr']:>9.4f}", flush=True)
        Path(a.out).write_text(json.dumps(got, indent=1))

    print("\n[*] r(band, within-TR sample) -- normalised per band (mean 1.0)")
    print(f"{'band':<10}" + "".join(f"{i:>7}" for i in range(10)) + f"{'centroid':>10}")
    prof = {}
    for b in range(a.bands):
        r = np.array(sub_tr_profile(recs, slicer(index, f"band{b}", lags=1)))
        prof[f"band{b}"] = r.tolist()
        norm = r / np.nanmean(r)
        w = np.clip(r - np.nanmin(r), 0, None)
        centroid = float((w * np.arange(10)).sum() / max(w.sum(), 1e-9))
        row = "".join(f"{v:>7.3f}" for v in norm) + f"{centroid:>10.2f}"
        print(f"{'band ' + str(b):<10}{row}", flush=True)
    r = np.array(sub_tr_profile(recs, slicer(index, "full", k=a.k, lags=1)))
    prof["full"] = r.tolist()
    print(f"{'full':<10}" + "".join(f"{v:>7.3f}" for v in r / np.nanmean(r)))
    got["_profiles"] = prof
    Path(a.out).write_text(json.dumps(got, indent=1))
    print(f"\n[+] {a.out}")


if __name__ == "__main__":
    main()
