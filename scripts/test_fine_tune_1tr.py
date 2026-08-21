#!/usr/bin/env python3
"""Fine-tuning Hybrid Basis Architectures to Maximize 1-TR Mean LODO Performance."""
import sys
import time
from pathlib import Path
import h5py
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV

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
                    block = f["eye_block"][:]
                    labels = f["labels"][:]
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


def make_lags(z, lags=0):
    if lags == 0:
        return z
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


def eval_lodo_subtr(subs, feat_extract_fn, lags=0):
    datasets = sorted({s["dataset"] for s in subs})
    per_fold = {}
    alphas = np.logspace(-2, 4, 13)

    feats = []
    for s in subs:
        z = feat_extract_fn(s)
        if lags > 0:
            z = make_lags(z, lags=lags)
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

        model = RidgeCV(alphas=alphas).fit(x_tr, y_tr)

        sub_scores = []
        for s in test:
            x_te = s["z"]
            lab_te = s["labels"]
            T = len(x_te)
            y_te_flat = lab_te.reshape(T, 20)
            ok = np.isfinite(y_te_flat).all(axis=1) & np.isfinite(x_te).all(axis=1)
            if ok.sum() < 10:
                continue
            pred_20 = model.predict(x_te)
            pred_subtr = pred_20.reshape(T, 10, 2)

            p_1tr = np.nanmean(pred_subtr[ok], axis=1)
            t_1tr = np.nanmean(lab_te[ok], axis=1)
            rx = calc_r(p_1tr[:, 0], t_1tr[:, 0])
            ry = calc_r(p_1tr[:, 1], t_1tr[:, 1])
            if np.isfinite(rx) and np.isfinite(ry):
                sub_scores.append((rx + ry) / 2.0)

        if sub_scores:
            per_fold[held] = float(np.nanmedian(sub_scores))

    return float(np.median(list(per_fold.values()))), per_fold


def main():
    subs = load_raw_subs()

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
    band_comp = bases["band-pca"]["components"]
    band_mu = bases["band-pca"]["mean"]

    configs = []

    # 1. cca:32 + pca:64
    def f1(s):
        xc = s["vox"] - mu_cca
        z_cca = 0.5 * (xc[:, li] @ wl[:, :32] + xc[:, ri] @ wr[:, :32])
        z_pca = (s["vox"] - pca_mu) @ pca_comp[:, :64]
        return np.concatenate([z_cca, z_pca], axis=1)
    configs.append(("DeepMReye 2.0 Hybrid (lr-cca:32 + corpus-pca:64 = 96 feats)", f1, 0))

    # 2. cca:48 + pca:64
    def f2(s):
        xc = s["vox"] - mu_cca
        z_cca = 0.5 * (xc[:, li] @ wl[:, :48] + xc[:, ri] @ wr[:, :48])
        z_pca = (s["vox"] - pca_mu) @ pca_comp[:, :64]
        return np.concatenate([z_cca, z_pca], axis=1)
    configs.append(("DeepMReye 2.0 Hybrid (lr-cca:48 + corpus-pca:64 = 112 feats)", f2, 0))

    # 3. cca:32 + pca:96
    def f3(s):
        xc = s["vox"] - mu_cca
        z_cca = 0.5 * (xc[:, li] @ wl[:, :32] + xc[:, ri] @ wr[:, :32])
        z_pca = (s["vox"] - pca_mu) @ pca_comp[:, :96]
        return np.concatenate([z_cca, z_pca], axis=1)
    configs.append(("DeepMReye 2.0 Hybrid (lr-cca:32 + corpus-pca:96 = 128 feats)", f3, 0))

    # 4. cca:48 + pca:48
    def f4(s):
        xc = s["vox"] - mu_cca
        z_cca = 0.5 * (xc[:, li] @ wl[:, :48] + xc[:, ri] @ wr[:, :48])
        z_pca = (s["vox"] - pca_mu) @ pca_comp[:, :48]
        return np.concatenate([z_cca, z_pca], axis=1)
    configs.append(("DeepMReye 2.0 Hybrid (lr-cca:48 + corpus-pca:48 = 96 feats)", f4, 0))

    # 5. cca:32 + pca:64 + band-pca:16
    def f5(s):
        xc = s["vox"] - mu_cca
        z_cca = 0.5 * (xc[:, li] @ wl[:, :32] + xc[:, ri] @ wr[:, :32])
        z_pca = (s["vox"] - pca_mu) @ pca_comp[:, :64]
        z_band = (s["vox"] - band_mu) @ band_comp[:, :16]
        return np.concatenate([z_cca, z_pca, z_band], axis=1)
    configs.append(("DeepMReye 2.0 Hybrid (lr-cca:32 + pca:64 + band:16 = 112 feats)", f5, 0))

    # 6. cca:32 + pca:64 + diff-pca:16
    def f6(s):
        xc = s["vox"] - mu_cca
        z_cca = 0.5 * (xc[:, li] @ wl[:, :32] + xc[:, ri] @ wr[:, :32])
        z_pca = (s["vox"] - pca_mu) @ pca_comp[:, :64]
        z_diff = (s["vox"] - diff_mu) @ diff_comp[:, :16]
        return np.concatenate([z_cca, z_pca, z_diff], axis=1)
    configs.append(("DeepMReye 2.0 Hybrid (lr-cca:32 + pca:64 + diff:16 = 112 feats)", f6, 0))

    # 7. cca:32 + pca:64 with temporal lags ±1
    configs.append(("DeepMReye 2.0 Hybrid (lr-cca:32 + pca:64) + lags±1", f1, 1))

    print("\n" + "=" * 110)
    print(f"{'Configuration':<72} {'1-TR Mean LODO r':>18} {'Delta vs Baseline':>14}")
    print("=" * 110)
    print(f"{'fold-pca:64 (Supervised Reference Baseline)':<72} {0.8370:>18.4f} {'(Baseline)':>14}")
    print("-" * 110)

    for name, fn, lags in configs:
        t0 = time.time()
        r_val, per_fold = eval_lodo_subtr(subs, fn, lags=lags)
        delta = r_val - 0.8370
        sign = "+" if delta >= 0 else ""
        winner = " 🏆" if delta > 0 else ""
        print(f"{name:<72} {r_val:>18.4f} {f'({sign}{delta:.4f})':>14}{winner}  [{time.time() - t0:.1f}s]", flush=True)

    print("=" * 110)


if __name__ == "__main__":
    main()
