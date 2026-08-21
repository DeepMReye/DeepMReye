#!/usr/bin/env python3
"""Rapid Exploration of 1-TR Mean Decoding Improvements on 7-Fold LODO Benchmark.

Tests:
1. Target Directness: Training directly on 1-TR mean vs training on 20 sub-TR outputs.
2. Bilateral Split (lr-split:32): Keeping left and right orbit projections separate [z_left, z_right] (64 features).
3. Bilateral Symmetric + Anti-symmetric basis: z_sum = (z_l + z_r)/2 and z_diff = (z_l - z_r)/2.
4. Hybrid Representations: lr-cca + gev-fast, lr-cca + corpus-pca, lr-cca + nuis-pca.
5. Dynamic Velocity Features: [z_t, z_t - z_{t-1}, z_t - z_{t+1}].
6. Temporal Smoothing / Filtering.
7. Subject-level feature normalization (z-scoring features per scan).
8. RidgeCV alpha tuning per axis (x and y separately).
"""
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from deepmreye.datasource import resolve
from deepmreye.unsupervised import corpus_mask, load_basis, project

EXCLUDE = ("dsL11_backtothefuture",)


def calc_r(p, t):
    ok = np.isfinite(p) & np.isfinite(t)
    if ok.sum() < 10 or np.std(t[ok]) < 1e-6 or np.std(p[ok]) < 1e-6:
        return np.nan
    return float(np.corrcoef(p[ok], t[ok])[0, 1])


def load_cached_labeled():
    cache_path = "results/scaling/eval_cache_raw.npz"
    d = np.load(cache_path, allow_pickle=True)
    recs = []
    for i in range(int(d["n"])):
        recs.append({
            "dataset": str(d[f"ds/{i}"]),
            "subject": str(d[f"sub/{i}"]),
            "rows": d[f"rows/{i}"],
            "gaze": d[f"gaze/{i}"],
        })
    return recs


def eval_lodo_feature_extractor(recs, extract_fn):
    """Generic fast 7-fold LODO evaluator for any feature extraction function."""
    datasets = sorted({r["dataset"] for r in recs})
    per_fold = {}
    alphas = np.logspace(-2, 5, 15)

    # Pre-extract features
    feats = []
    for r in recs:
        z = extract_fn(r)
        g = r["gaze"][:len(r["rows"])]
        n = min(len(z), len(g))
        feats.append({
            "dataset": r["dataset"],
            "subject": r["subject"],
            "z": z[:n],
            "gaze": g[:n],
        })

    for held in datasets:
        train = [f for f in feats if f["dataset"] != held]
        test = [f for f in feats if f["dataset"] == held]

        xs, ys = [], []
        for ds in sorted({f["dataset"] for f in train}):
            ds_f = [f for f in train if f["dataset"] == ds]
            g = np.concatenate([f["gaze"] for f in ds_f])
            x = np.concatenate([f["z"] for f in ds_f])
            ok = np.isfinite(g).all(axis=1) & np.isfinite(x).all(axis=1)
            if ok.sum() < 10:
                continue
            g, x = g[ok], x[ok]
            sd = g.std(axis=0)
            sd[sd < 1e-9] = 1.0
            ys.append((g - g.mean(axis=0)) / sd)
            xs.append(x)
        if not xs:
            continue
        x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
        if len(x_tr) > 20000:
            idx = np.random.default_rng(0).choice(len(x_tr), 20000, replace=False)
            x_tr, y_tr = x_tr[idx], y_tr[idx]

        # Separate RidgeCV for x and y to allow independent regularization
        model_x = RidgeCV(alphas=alphas).fit(x_tr, y_tr[:, 0])
        model_y = RidgeCV(alphas=alphas).fit(x_tr, y_tr[:, 1])

        per_sub = []
        for f in test:
            x, g = f["z"], f["gaze"]
            ok = np.isfinite(g).all(axis=1) & np.isfinite(x).all(axis=1)
            if ok.sum() < 10:
                continue
            px = model_x.predict(x[ok])
            py = model_y.predict(x[ok])
            rx = calc_r(px, g[ok, 0])
            ry = calc_r(py, g[ok, 1])
            if np.isfinite(rx) and np.isfinite(ry):
                per_sub.append((rx + ry) / 2.0)
        if per_sub:
            per_fold[held] = float(np.nanmedian(per_sub))

    return float(np.median(list(per_fold.values()))), per_fold


def main():
    print("[*] Loading labeled evaluation cache...", flush=True)
    recs = load_cached_labeled()
    print(f"[*] Loaded {len(recs)} subjects.", flush=True)

    # 1. Baseline: fold-pca:64
    def extract_fold_pca(r):
        # Local PCA reference
        pass

    # Load Basis n1800
    _mask, bases, meta = load_basis(Path("results/scaling/basis_n1800.npz"))
    cca_dict = bases["lr-cca"]
    mu_cca = cca_dict["mean"]
    li, ri = cca_dict["left_index"], cca_dict["right_index"]
    wl, wr = cca_dict["left_weights"], cca_dict["right_weights"]

    experiments = []

    # Exp 1: Standard lr-cca:32 instantaneous
    def exp_cca32(r):
        xc = r["rows"] - mu_cca
        return 0.5 * (xc[:, li] @ wl[:, :32] + xc[:, ri] @ wr[:, :32])
    experiments.append(("DeepMReye 2.0 (lr-cca:32 standard)", exp_cca32))

    # Exp 2: Standard lr-cca:64 instantaneous
    def exp_cca64(r):
        xc = r["rows"] - mu_cca
        return 0.5 * (xc[:, li] @ wl[:, :64] + xc[:, ri] @ wr[:, :64])
    experiments.append(("DeepMReye 2.0 (lr-cca:64 standard)", exp_cca64))

    # Exp 3: Bilateral Split lr-split:32 (separate left and right features: 64 total)
    def exp_split32(r):
        xc = r["rows"] - mu_cca
        zl = xc[:, li] @ wl[:, :32]
        zr = xc[:, ri] @ wr[:, :32]
        return np.concatenate([zl, zr], axis=1)
    experiments.append(("DeepMReye 2.0 (lr-split:32 separate L/R)", exp_split32))

    # Exp 4: Bilateral Sum + Difference [ (zl+zr)/2, (zl-zr)/2 ] (Conjugate + Disconjugate)
    def exp_sumdiff32(r):
        xc = r["rows"] - mu_cca
        zl = xc[:, li] @ wl[:, :32]
        zr = xc[:, ri] @ wr[:, :32]
        return np.concatenate([0.5 * (zl + zr), 0.5 * (zl - zr)], axis=1)
    experiments.append(("DeepMReye 2.0 (lr-cca:32 conjugate + disconjugate)", exp_sumdiff32))

    # Exp 5: Bilateral Split lr-split:64 (separate left and right features: 128 total)
    def exp_split64(r):
        xc = r["rows"] - mu_cca
        zl = xc[:, li] @ wl[:, :64]
        zr = xc[:, ri] @ wr[:, :64]
        return np.concatenate([zl, zr], axis=1)
    experiments.append(("DeepMReye 2.0 (lr-split:64 separate L/R)", exp_split64))

    # Exp 6: lr-cca:32 + Velocity Differencing [z_t, z_t - z_{t-1}]
    def exp_cca32_vel(r):
        xc = r["rows"] - mu_cca
        z = 0.5 * (xc[:, li] @ wl[:, :32] + xc[:, ri] @ wr[:, :32])
        diff = np.diff(z, axis=0, prepend=z[:1])
        return np.concatenate([z, diff], axis=1)
    experiments.append(("DeepMReye 2.0 (lr-cca:32 + instantaneous velocity)", exp_cca32_vel))

    # Exp 7: lr-split:32 + Velocity Differencing [zl, zr, diff_zl, diff_zr]
    def exp_split32_vel(r):
        xc = r["rows"] - mu_cca
        zl = xc[:, li] @ wl[:, :32]
        zr = xc[:, ri] @ wr[:, :32]
        z = np.concatenate([zl, zr], axis=1)
        diff = np.diff(z, axis=0, prepend=z[:1])
        return np.concatenate([z, diff], axis=1)
    experiments.append(("DeepMReye 2.0 (lr-split:32 + velocity)", exp_split32_vel))

    # Exp 8: lr-cca:32 + gev-fast:16 (Bilateral conjugate + rapid saccadic ocular dynamics)
    gev_comp = bases["gev-fast"]["components"]
    gev_mu = bases["gev-fast"]["mean"]
    def exp_cca32_gev16(r):
        xc = r["rows"] - mu_cca
        z_cca = 0.5 * (xc[:, li] @ wl[:, :32] + xc[:, ri] @ wr[:, :32])
        z_gev = (r["rows"] - gev_mu) @ gev_comp[:, :16]
        return np.concatenate([z_cca, z_gev], axis=1)
    experiments.append(("DeepMReye 2.0 (lr-cca:32 + gev-fast:16 hybrid)", exp_cca32_gev16))

    # Exp 9: lr-split:32 + gev-fast:16
    def exp_split32_gev16(r):
        xc = r["rows"] - mu_cca
        zl = xc[:, li] @ wl[:, :32]
        zr = xc[:, ri] @ wr[:, :32]
        z_gev = (r["rows"] - gev_mu) @ gev_comp[:, :16]
        return np.concatenate([zl, zr, z_gev], axis=1)
    experiments.append(("DeepMReye 2.0 (lr-split:32 + gev-fast:16 hybrid)", exp_split32_gev16))

    # Exp 10: lr-cca:32 + corpus-pca:16 (Conjugate + global intensity)
    pca_comp = bases["corpus-pca"]["components"]
    pca_mu = bases["corpus-pca"]["mean"]
    def exp_cca32_pca16(r):
        xc = r["rows"] - mu_cca
        z_cca = 0.5 * (xc[:, li] @ wl[:, :32] + xc[:, ri] @ wr[:, :32])
        z_pca = (r["rows"] - pca_mu) @ pca_comp[:, :16]
        return np.concatenate([z_cca, z_pca], axis=1)
    experiments.append(("DeepMReye 2.0 (lr-cca:32 + corpus-pca:16 hybrid)", exp_cca32_pca16))

    # Exp 11: lr-split:32 + corpus-pca:16
    def exp_split32_pca16(r):
        xc = r["rows"] - mu_cca
        zl = xc[:, li] @ wl[:, :32]
        zr = xc[:, ri] @ wr[:, :32]
        z_pca = (r["rows"] - pca_mu) @ pca_comp[:, :16]
        return np.concatenate([zl, zr, z_pca], axis=1)
    experiments.append(("DeepMReye 2.0 (lr-split:32 + corpus-pca:16 hybrid)", exp_split32_pca16))

    # Exp 12: lr-split:32 + Scan-level Robust Normalization (z-score per subject)
    def exp_split32_norm(r):
        xc = r["rows"] - mu_cca
        zl = xc[:, li] @ wl[:, :32]
        zr = xc[:, ri] @ wr[:, :32]
        z = np.concatenate([zl, zr], axis=1)
        z = (z - z.mean(axis=0, keepdims=True)) / (z.std(axis=0, keepdims=True) + 1e-6)
        return z
    experiments.append(("DeepMReye 2.0 (lr-split:32 + scan-normalized)", exp_split32_norm))

    # Exp 13: lr-split:32 + corpus-pca:16 + Scan-level Normalization
    def exp_split32_pca16_norm(r):
        xc = r["rows"] - mu_cca
        zl = xc[:, li] @ wl[:, :32]
        zr = xc[:, ri] @ wr[:, :32]
        z_pca = (r["rows"] - pca_mu) @ pca_comp[:, :16]
        z = np.concatenate([zl, zr, z_pca], axis=1)
        z = (z - z.mean(axis=0, keepdims=True)) / (z.std(axis=0, keepdims=True) + 1e-6)
        return z
    experiments.append(("DeepMReye 2.0 (lr-split:32 + pca:16 + scan-norm)", exp_split32_pca16_norm))

    print("\n" + "=" * 80)
    print("1-TR MEAN LODO BENCHMARK: TESTING METHOD IMPROVEMENTS")
    print("=" * 80)

    for name, fn in experiments:
        t0 = time.time()
        r_med, per_fold = eval_lodo_feature_extractor(recs, fn)
        print(f"  {name:<55} -> r = {r_med:.4f}  ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
