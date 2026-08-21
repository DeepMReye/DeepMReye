#!/usr/bin/env python3
"""Investigate Representations & Decoders under Sub-TR multi-task training for 1-TR Mean.

Evaluates:
- Dimension scaling: lr-cca:32, lr-cca:48, lr-cca:64, lr-cca:96, lr-cca:128
- Orbit-split representation: lr-split (separate L and R orbits: 2 * k features)
- Multi-basis combination: lr-cca + diff-pca + band-pca + gev-fast
- Bilateral Conjugate + Intensity modulation: lr-cca + corpus-pca
- Decoders: Multi-output RidgeCV, MLP (2-layer neural probe with GELU / LayerNorm), Huber / Robust Loss
"""
import sys
import time
from pathlib import Path
import h5py
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


def load_raw_subs():
    data_dir = resolve(None, download=False, quiet=True)
    mask = corpus_mask(data_dir)
    flat = mask.reshape(-1)
    datasets = sorted(data_dir.glob("dsL*"))
    all_subs = []
    for ds_dir in datasets:
        if ds_dir.name in EXCLUDE or not ds_dir.is_dir():
            continue
        for p in sorted(ds_dir.glob("*.h5")):
            try:
                with h5py.File(p, "r") as f:
                    if "labels" not in f:
                        continue
                    block = f["eye_block"][:] # [47, 29, 18, T]
                    labels = f["labels"][:]   # [T, 10, 2]
            except Exception:
                continue
            T = block.shape[-1]
            if T < 30 or not np.isfinite(labels).any():
                continue
            vox = block.reshape(-1, T).T[:, flat].astype(np.float32)
            all_subs.append({
                "dataset": ds_dir.name,
                "subject": p.stem,
                "vox": vox,
                "labels": labels[:T].astype(np.float32),
            })
    return all_subs


def eval_lodo_subtr_model(subs, feat_extract_fn, model_type="ridge"):
    """Evaluate 1-TR mean using multi-task Sub-TR [T, 20] training."""
    datasets = sorted({s["dataset"] for s in subs})
    per_fold = {}
    alphas = np.logspace(-2, 4, 13)

    # Extract features
    feats = []
    for s in subs:
        z = feat_extract_fn(s)
        feats.append({
            "dataset": s["dataset"],
            "subject": s["subject"],
            "z": z,
            "labels": s["labels"],
        })

    for held in datasets:
        train = [f for f in feats if f["dataset"] != held]
        test = [f for f in feats if f["dataset"] == held]

        xs, ys = [], []
        for ds in sorted({f["dataset"] for f in train}):
            ds_subs = [f for f in train if f["dataset"] == ds]
            ds_z = np.concatenate([f["z"] for f in ds_subs])
            ds_lab = np.concatenate([f["labels"].reshape(-1, 20) for f in ds_subs])
            ok = np.isfinite(ds_lab).all(axis=1) & np.isfinite(ds_z).all(axis=1)
            if ok.sum() < 10:
                continue
            z_ok, lab_ok = ds_z[ok], ds_lab[ok]
            sd = lab_ok.std(axis=0)
            sd[sd < 1e-9] = 1.0
            ys.append((lab_ok - lab_ok.mean(axis=0)) / sd)
            xs.append(z_ok)

        if not xs:
            continue
        x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
        if len(x_tr) > 15000:
            idx = np.random.default_rng(0).choice(len(x_tr), 15000, replace=False)
            x_tr, y_tr = x_tr[idx], y_tr[idx]

        if model_type == "ridge":
            model = RidgeCV(alphas=alphas).fit(x_tr, y_tr)
        elif model_type == "mlp":
            # Fast PyTorch MLP or scikit-learn MLPRegressor
            from sklearn.neural_network import MLPRegressor
            model = MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu", max_iter=20, random_state=0).fit(x_tr, y_tr)

        sub_scores = []
        for s in test:
            x_te = s["z"]
            lab_te = s["labels"]
            T = len(x_te)
            y_te_flat = lab_te.reshape(T, 20)
            ok = np.isfinite(y_te_flat).all(axis=1) & np.isfinite(x_te).all(axis=1)
            if ok.sum() < 10:
                continue
            pred_20 = model.predict(x_te) # [T, 20]
            pred_subtr = pred_20.reshape(T, 10, 2) # [T, 10, 2]

            # 1-TR mean score
            p_1tr = np.nanmean(pred_subtr[ok], axis=1) # [T, 2]
            t_1tr = np.nanmean(lab_te[ok], axis=1)    # [T, 2]
            rx = calc_r(p_1tr[:, 0], t_1tr[:, 0])
            ry = calc_r(p_1tr[:, 1], t_1tr[:, 1])
            if np.isfinite(rx) and np.isfinite(ry):
                sub_scores.append((rx + ry) / 2.0)

        if sub_scores:
            per_fold[held] = float(np.nanmedian(sub_scores))

    return float(np.median(list(per_fold.values()))), per_fold


def main():
    print("[*] Loading raw labeled subjects...", flush=True)
    subs = load_raw_subs()
    print(f"[*] Loaded {len(subs)} labeled subjects.", flush=True)

    _mask, bases, meta = load_basis(Path("results/scaling/basis_n1800.npz"))
    cca_dict = bases["lr-cca"]
    mu_cca = cca_dict["mean"]
    li, ri = cca_dict["left_index"], cca_dict["right_index"]
    wl, wr = cca_dict["left_weights"], cca_dict["right_weights"]
    gev_comp = bases["gev-fast"]["components"]
    gev_mu = bases["gev-fast"]["mean"]
    pca_comp = bases["corpus-pca"]["components"]
    pca_mu = bases["corpus-pca"]["mean"]
    diff_comp = bases["diff-pca"]["components"]
    diff_mu = bases["diff-pca"]["mean"]

    # Baseline fold-pca:64
    def feat_fold_pca_builder(s):
        pass # evaluated inside benchmark_multi_resolution_scaling.py = 0.837

    experiments = []

    # 1. lr-cca:32
    def feat_cca32(s):
        xc = s["vox"] - mu_cca
        return 0.5 * (xc[:, li] @ wl[:, :32] + xc[:, ri] @ wr[:, :32])
    experiments.append(("DeepMReye 2.0 (lr-cca:32)", feat_cca32, "ridge"))

    # 2. lr-cca:48
    def feat_cca48(s):
        xc = s["vox"] - mu_cca
        return 0.5 * (xc[:, li] @ wl[:, :48] + xc[:, ri] @ wr[:, :48])
    experiments.append(("DeepMReye 2.0 (lr-cca:48)", feat_cca48, "ridge"))

    # 3. lr-cca:64
    def feat_cca64(s):
        xc = s["vox"] - mu_cca
        return 0.5 * (xc[:, li] @ wl[:, :64] + xc[:, ri] @ wr[:, :64])
    experiments.append(("DeepMReye 2.0 (lr-cca:64)", feat_cca64, "ridge"))

    # 4. lr-cca:96
    def feat_cca96(s):
        xc = s["vox"] - mu_cca
        return 0.5 * (xc[:, li] @ wl[:, :96] + xc[:, ri] @ wr[:, :96])
    experiments.append(("DeepMReye 2.0 (lr-cca:96)", feat_cca96, "ridge"))

    # 5. lr-cca:128
    def feat_cca128(s):
        xc = s["vox"] - mu_cca
        return 0.5 * (xc[:, li] @ wl[:, :128] + xc[:, ri] @ wr[:, :128])
    experiments.append(("DeepMReye 2.0 (lr-cca:128)", feat_cca128, "ridge"))

    # 6. lr-split:32 (separate left and right orbit: 64 features)
    def feat_split32(s):
        xc = s["vox"] - mu_cca
        zl = xc[:, li] @ wl[:, :32]
        zr = xc[:, ri] @ wr[:, :32]
        return np.concatenate([zl, zr], axis=1)
    experiments.append(("DeepMReye 2.0 (lr-split:32 separate L/R = 64 features)", feat_split32, "ridge"))

    # 7. lr-split:48 (separate left and right orbit: 96 features)
    def feat_split48(s):
        xc = s["vox"] - mu_cca
        zl = xc[:, li] @ wl[:, :48]
        zr = xc[:, ri] @ wr[:, :48]
        return np.concatenate([zl, zr], axis=1)
    experiments.append(("DeepMReye 2.0 (lr-split:48 separate L/R = 96 features)", feat_split48, "ridge"))

    # 8. Multi-basis Fusion: lr-cca:32 + diff-pca:16 + gev-fast:16 (64 features)
    def feat_fusion64(s):
        xc = s["vox"] - mu_cca
        z_cca = 0.5 * (xc[:, li] @ wl[:, :32] + xc[:, ri] @ wr[:, :32])
        z_diff = (s["vox"] - diff_mu) @ diff_comp[:, :16]
        z_gev = (s["vox"] - gev_mu) @ gev_comp[:, :16]
        return np.concatenate([z_cca, z_diff, z_gev], axis=1)
    experiments.append(("DeepMReye 2.0 (lr-cca:32 + diff-pca:16 + gev-fast:16)", feat_fusion64, "ridge"))

    # 9. Multi-basis Fusion: lr-split:32 + diff-pca:16 + gev-fast:16 (96 features)
    def feat_fusion_split(s):
        xc = s["vox"] - mu_cca
        zl = xc[:, li] @ wl[:, :32]
        zr = xc[:, ri] @ wr[:, :32]
        z_diff = (s["vox"] - diff_mu) @ diff_comp[:, :16]
        z_gev = (s["vox"] - gev_mu) @ gev_comp[:, :16]
        return np.concatenate([zl, zr, z_diff, z_gev], axis=1)
    experiments.append(("DeepMReye 2.0 (lr-split:32 + diff:16 + gev:16 = 96 feats)", feat_fusion_split, "ridge"))

    # 10. Multi-basis Fusion: lr-split:32 + corpus-pca:16 + gev-fast:16
    def feat_fusion_pca(s):
        xc = s["vox"] - mu_cca
        zl = xc[:, li] @ wl[:, :32]
        zr = xc[:, ri] @ wr[:, :32]
        z_pca = (s["vox"] - pca_mu) @ pca_comp[:, :16]
        z_gev = (s["vox"] - gev_mu) @ gev_comp[:, :16]
        return np.concatenate([zl, zr, z_pca, z_gev], axis=1)
    experiments.append(("DeepMReye 2.0 (lr-split:32 + pca:16 + gev:16 = 96 feats)", feat_fusion_pca, "ridge"))

    # 11. Multi-basis Fusion with MLP Decoder
    experiments.append(("DeepMReye 2.0 (lr-split:32 + diff:16 + gev:16) + MLP Decoder", feat_fusion_split, "mlp"))

    # Baseline reference
    print("\n" + "=" * 95)
    print(f"{'Method / Configuration':<60} {'1-TR Mean LODO r':>18} {'Delta vs Baseline':>14}")
    print("=" * 95)
    print(f"{'fold-pca:64 (Supervised Reference Baseline)':<60} {0.8370:>18.4f} {'(Baseline)':>14}")
    print("-" * 95)

    for name, fn, mtype in experiments:
        t0 = time.time()
        r_val, per_fold = eval_lodo_subtr_model(subs, fn, model_type=mtype)
        delta = r_val - 0.8370
        sign = "+" if delta >= 0 else ""
        print(f"{name:<60} {r_val:>18.4f} {f'({sign}{delta:.4f})':>14}  [{time.time() - t0:.1f}s]", flush=True)

    print("=" * 95)


if __name__ == "__main__":
    main()
