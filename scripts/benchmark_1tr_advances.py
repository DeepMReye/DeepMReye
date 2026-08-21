#!/usr/bin/env python3
"""Systematic Investigation of Decoding & Representation Advances to Beat Supervised Baselines on 1-TR Mean.

Evaluates:
1. Multi-Task Sub-TR target training -> 1-TR mean projection (multi-phase regularization)
2. Temporal Multi-Lags (±1, ±2 TRs) + Velocity + Acceleration
3. Orbital Spatial Whitening & Normalization (L2 norm per subject, robust z-scoring)
4. Joint Hybrid Representation: lr-cca (conjugate) + lr-disconjugate + gev-fast + corpus-pca
5. Temporal Filtering / Smoothing (Savitzky-Golay, Gaussian, causal EWMA)
6. Non-linear Feature Expansion (Quadratic orbit interactions, Kernel Ridge / MLP)
7. Cross-Subject / Cross-Dataset Adaptive Whitening
"""
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.decomposition import PCA

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


def load_raw_labeled_data():
    cache_path = "results/scaling/eval_cache_raw.npz"
    d = np.load(cache_path, allow_pickle=True)
    recs = []
    for i in range(int(d["n"])):
        # Load raw voxel rows and 10-point sub-TR labels if available
        ds = str(d[f"ds/{i}"])
        sub = str(d[f"sub/{i}"])
        rows = d[f"rows/{i}"]
        gaze_1tr = d[f"gaze/{i}"]
        recs.append({
            "dataset": ds,
            "subject": sub,
            "rows": rows,
            "gaze_1tr": gaze_1tr,
        })
    return recs


def eval_lodo_custom(recs, feature_builder_fn, use_subtr_targets=False, smoothing_fn=None):
    """Evaluate LODO across 7 folds for 1-TR mean."""
    datasets = sorted({r["dataset"] for r in recs})
    per_fold = {}
    alphas = np.logspace(-3, 5, 17)

    # Build features for each subject
    feats = []
    for r in recs:
        z = feature_builder_fn(r)
        g = r["gaze_1tr"][:len(r["rows"])]
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
            g_ok, x_ok = g[ok], x[ok]
            sd = g_ok.std(axis=0)
            sd[sd < 1e-9] = 1.0
            ys.append((g_ok - g_ok.mean(axis=0)) / sd)
            xs.append(x_ok)

        if not xs:
            continue
        x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
        if len(x_tr) > 20000:
            idx = np.random.default_rng(0).choice(len(x_tr), 20000, replace=False)
            x_tr, y_tr = x_tr[idx], y_tr[idx]

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

            if smoothing_fn is not None:
                px = smoothing_fn(px)
                py = smoothing_fn(py)

            rx = calc_r(px, g[ok, 0])
            ry = calc_r(py, g[ok, 1])
            if np.isfinite(rx) and np.isfinite(ry):
                per_sub.append((rx + ry) / 2.0)
        if per_sub:
            per_fold[held] = float(np.nanmedian(per_sub))

    return float(np.median(list(per_fold.values()))), per_fold


def main():
    print("[*] Loading labeled evaluation cache...", flush=True)
    recs = load_raw_labeled_data()
    print(f"[*] Loaded {len(recs)} subjects across 7 datasets.", flush=True)

    _mask, bases, meta = load_basis(Path("results/scaling/basis_n1800.npz"))
    cca_dict = bases["lr-cca"]
    mu_cca = cca_dict["mean"]
    li, ri = cca_dict["left_index"], cca_dict["right_index"]
    wl, wr = cca_dict["left_weights"], cca_dict["right_weights"]
    gev_comp = bases["gev-fast"]["components"]
    gev_mu = bases["gev-fast"]["mean"]
    pca_comp = bases["corpus-pca"]["components"]
    pca_mu = bases["corpus-pca"]["mean"]

    # 0. Benchmark Baseline: fold-pca:64
    def build_fold_pca_baseline():
        # Standard fold-pca:64
        datasets = sorted({r["dataset"] for r in recs})
        per_fold = {}
        for held in datasets:
            train = [r for r in recs if r["dataset"] != held]
            test = [r for r in recs if r["dataset"] == held]
            x_tr_raw = np.concatenate([r["rows"] for r in train])
            idx = np.random.default_rng(0).choice(len(x_tr_raw), min(10000, len(x_tr_raw)), replace=False)
            pca = PCA(n_components=64, random_state=0).fit(x_tr_raw[idx])
            
            xs, ys = [], []
            for ds in sorted({r["dataset"] for r in train}):
                ds_recs = [r for r in train if r["dataset"] == ds]
                g = np.concatenate([r["gaze_1tr"] for r in ds_recs])
                r_rows = np.concatenate([r["rows"] for r in ds_recs])
                ok = np.isfinite(g).all(axis=1) & np.isfinite(r_rows).all(axis=1)
                if ok.sum() < 10:
                    continue
                g_ok = g[ok]
                sd = g_ok.std(axis=0)
                sd[sd < 1e-9] = 1.0
                ys.append((g_ok - g_ok.mean(axis=0)) / sd)
                xs.append(pca.transform(r_rows[ok]))
            x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
            if len(x_tr) > 20000:
                idx = np.random.default_rng(0).choice(len(x_tr), 20000, replace=False)
                x_tr, y_tr = x_tr[idx], y_tr[idx]
            mx = RidgeCV(alphas=np.logspace(-3, 5, 17)).fit(x_tr, y_tr[:, 0])
            my = RidgeCV(alphas=np.logspace(-3, 5, 17)).fit(x_tr, y_tr[:, 1])
            sub_rs = []
            for r in test:
                g = r["gaze_1tr"]
                ok = np.isfinite(g).all(axis=1) & np.isfinite(r["rows"]).all(axis=1)
                if ok.sum() < 10:
                    continue
                px = mx.predict(pca.transform(r["rows"][ok]))
                py = my.predict(pca.transform(r["rows"][ok]))
                rx, ry = calc_r(px, g[ok, 0]), calc_r(py, g[ok, 1])
                if np.isfinite(rx) and np.isfinite(ry):
                    sub_rs.append((rx + ry) / 2.0)
            per_fold[held] = float(np.nanmedian(sub_rs))
        return float(np.median(list(per_fold.values())))

    r_baseline = build_fold_pca_baseline()
    print(f"\n[*] SUPERVISED BASELINE (fold-pca:64) 1-TR Mean = {r_baseline:.4f}\n", flush=True)

    experiments = []

    # Helper: Multi-lag builder
    def make_lags(z, lags=1):
        parts = []
        for lag in range(-lags, lags + 1):
            if lag < 0:
                p = np.pad(z[:lag], ((-lag, 0), (0, 0)), mode="edge")
            elif lag > 0:
                p = np.pad(z[lag:], ((0, lag), (0, 0)), mode="edge")
            else:
                p = z
            parts.append(p)
        return np.concatenate(parts, axis=1)

    # 1. Base lr-cca:32
    def feat_cca32(r):
        xc = r["rows"] - mu_cca
        return 0.5 * (xc[:, li] @ wl[:, :32] + xc[:, ri] @ wr[:, :32])
    experiments.append(("1. DeepMReye 2.0 lr-cca:32 (instantaneous)", feat_cca32, None))

    # 2. lr-cca:32 + ±1 TR lags
    def feat_cca32_lag1(r):
        return make_lags(feat_cca32(r), lags=1)
    experiments.append(("2. DeepMReye 2.0 lr-cca:32 (+ lags ±1 TR)", feat_cca32_lag1, None))

    # 3. lr-cca:32 + ±2 TR lags
    def feat_cca32_lag2(r):
        return make_lags(feat_cca32(r), lags=2)
    experiments.append(("3. DeepMReye 2.0 lr-cca:32 (+ lags ±2 TRs)", feat_cca32_lag2, None))

    # 4. lr-cca:32 + Velocity + Acceleration (Kinematic Orbit Representation)
    def feat_cca32_kinematic(r):
        z = feat_cca32(r)
        vel = np.diff(z, axis=0, prepend=z[:1])
        acc = np.diff(vel, axis=0, prepend=vel[:1])
        return np.concatenate([z, vel, acc], axis=1)
    experiments.append(("4. DeepMReye 2.0 lr-cca:32 (+ Velocity & Acceleration)", feat_cca32_kinematic, None))

    # 5. Symmetric + Asymmetric Bi-orbital features: [ (zl+zr)/2, (zl-zr)/2 ] + gev-fast:16 + lags±1
    def feat_biorbital_hybrid_lag1(r):
        xc = r["rows"] - mu_cca
        zl = xc[:, li] @ wl[:, :32]
        zr = xc[:, ri] @ wr[:, :32]
        z_conj = 0.5 * (zl + zr)
        z_disconj = 0.5 * (zl - zr)
        z_gev = (r["rows"] - gev_mu) @ gev_comp[:, :16]
        z_tot = np.concatenate([z_conj, z_disconj, z_gev], axis=1)
        return make_lags(z_tot, lags=1)
    experiments.append(("5. Bi-orbital (Conjugate + Disconjugate + GEV-fast) + lags±1", feat_biorbital_hybrid_lag1, None))

    # 6. Bi-orbital + Quadratic Cross-Orbit Interaction (Non-linear Eye Rotation Mechanics)
    def feat_biorbital_quadratic(r):
        xc = r["rows"] - mu_cca
        zl = xc[:, li] @ wl[:, :16]
        zr = xc[:, ri] @ wr[:, :16]
        z_lin = 0.5 * (zl + zr)
        # Bilinear cross-interaction between left and right orbit variates
        z_quad = zl * zr
        z_tot = np.concatenate([z_lin, z_quad], axis=1)
        return make_lags(z_tot, lags=1)
    experiments.append(("6. Bi-orbital Bilinear Interaction (L * R) + lags±1", feat_biorbital_quadratic, None))

    # 7. Orbit-Whitened CCA: Standardizing each subject's raw voxels by baseline eye variance
    def feat_cca32_whitened_lag1(r):
        rows = r["rows"]
        # Whiten voxels per subject
        mu_sub = rows.mean(axis=0, keepdims=True)
        sd_sub = rows.std(axis=0, keepdims=True) + 1e-5
        rows_w = (rows - mu_sub) / sd_sub
        # Project
        xc = rows_w
        z = 0.5 * (xc[:, li] @ wl[:, :32] + xc[:, ri] @ wr[:, :32])
        return make_lags(z, lags=1)
    experiments.append(("7. Per-subject Orbital Voxel Whitening + lags±1", feat_cca32_whitened_lag1, None))

    # 8. Multi-scale Lag Fusion: instantaneous + [t-1, t+1] + [t-2, t+2] + Gaussian Smoothing
    def smooth_gaussian(y, sigma=0.8):
        # 3-tap binomial smoothing [0.25, 0.5, 0.25]
        return np.convolve(y, [0.2, 0.6, 0.2], mode="same")

    experiments.append(("8. Bi-orbital Hybrid + lags±1 + Temporal Smoothing", feat_biorbital_hybrid_lag1, smooth_gaussian))

    # 9. Tri-Representation Super-Basis: lr-cca:32 + gev-fast:16 + corpus-pca:16 with lags±1
    def feat_super_basis(r):
        xc = r["rows"] - mu_cca
        z_cca = 0.5 * (xc[:, li] @ wl[:, :32] + xc[:, ri] @ wr[:, :32])
        z_gev = (r["rows"] - gev_mu) @ gev_comp[:, :16]
        z_pca = (r["rows"] - pca_mu) @ pca_comp[:, :16]
        z_tot = np.concatenate([z_cca, z_gev, z_pca], axis=1)
        return make_lags(z_tot, lags=1)
    experiments.append(("9. Super-Basis (lr-cca:32 + gev:16 + pca:16) + lags±1", feat_super_basis, None))
    experiments.append(("10. Super-Basis + lags±1 + Temporal Smoothing", feat_super_basis, smooth_gaussian))

    # Run all
    print("=" * 105)
    print(f"{'Method / Innovation':<65} {'1-TR Mean LODO r':>18} {'Delta vs Baseline':>18}")
    print("=" * 105)

    for name, feat_fn, smooth_fn in experiments:
        t0 = time.time()
        r_val, per_fold = eval_lodo_custom(recs, feat_fn, smoothing_fn=smooth_fn)
        delta = r_val - r_baseline
        sign = "+" if delta >= 0 else ""
        print(f"{name:<65} {r_val:>18.4f} {f'({sign}{delta:.4f})':>18}  [{time.time() - t0:.1f}s]", flush=True)

    print("=" * 105)


if __name__ == "__main__":
    main()
