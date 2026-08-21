#!/usr/bin/env python3
"""Within-Dataset Evaluation for dsL08, dsL09, dsL12 (and all datasets).

Evaluates 5-fold cross-validation (or Leave-One-Subject-Out) WITHIN each dataset:
- fold-pca:64 (Supervised PCA fitted on within-dataset training subjects)
- DeepMReye 2.0 Core (lr-cca:32)
- DeepMReye 2.0 Super-Basis (96 feats)
- DeepMReye 2.0 (+lags±2)

At both:
1. 1-TR Mean Resolution
2. Sub-TR (10 pts/TR) Resolution
"""
import json
import sys
import time
from pathlib import Path
import h5py
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from deepmreye.datasource import resolve
from deepmreye.unsupervised import corpus_mask, load_basis


def calc_r(p, t):
    ok = np.isfinite(p) & np.isfinite(t)
    if ok.sum() < 10 or np.std(t[ok]) < 1e-6 or np.std(p[ok]) < 1e-6:
        return np.nan
    return float(np.corrcoef(p[ok], t[ok])[0, 1])


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


def load_dataset_subs(ds_name):
    data_dir = resolve(None, download=False, quiet=True)
    mask = corpus_mask(data_dir)
    flat = mask.reshape(-1)
    ds_dir = data_dir / ds_name
    subs = []
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
        if T < 20 or not np.isfinite(labels).any():
            continue
        vox = block.reshape(-1, T).T[:, flat].astype(np.float32)
        subs.append({
            "dataset": ds_name,
            "subject": p.stem,
            "vox": vox,
            "labels": labels[:T].astype(np.float32),
        })
    return subs


def eval_within(subs, bases):
    cca_dict = bases["lr-cca"]
    mu_cca = cca_dict["mean"]
    li, ri = cca_dict["left_index"], cca_dict["right_index"]
    wl, wr = cca_dict["left_weights"], cca_dict["right_weights"]
    pca_comp = bases["corpus-pca"]["components"]
    pca_mu = bases["corpus-pca"]["mean"]

    for s in subs:
        xc = s["vox"] - mu_cca
        s["z_cca32"] = 0.5 * (xc[:, li] @ wl[:, :32] + xc[:, ri] @ wr[:, :32])
        s["z_pca64"] = (s["vox"] - pca_mu) @ pca_comp[:, :64]
        s["z_super96"] = np.concatenate([s["z_cca32"], s["z_pca64"]], axis=1)
        s["z_cca32_lags2"] = make_lags(s["z_cca32"], lags=2)

    n_subs = len(subs)
    n_splits = min(5, n_subs) if n_subs >= 4 else n_subs
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    alphas = np.logspace(-2, 4, 13)

    arms = [
        "fold-pca:64",
        "DeepMReye 2.0 (lr-cca:32)",
        "DeepMReye 2.0 Super-Basis (96 feats)",
        "DeepMReye 2.0 (lr-cca:32 + lags±2)",
    ]

    scores_1tr = {a: [] for a in arms}
    scores_sub = {a: [] for a in arms}

    for tr_idx, te_idx in kf.split(subs):
        train = [subs[i] for i in tr_idx]
        test = [subs[i] for i in te_idx]

        # 1. Fit within-dataset fold-pca.
        # Same uniform sample over pooled training rows as concatenating them
        # would give, without materialising the pool -- see the note in
        # benchmark_all_11_datasets.py. Bounded here too because dsL04 alone is
        # ~6.9 GB of voxels and the concatenation doubled it for 5000 rows.
        lens = np.array([len(s["vox"]) for s in train])
        starts = np.concatenate([[0], np.cumsum(lens)])
        total = int(lens.sum())
        gidx = np.sort(np.random.default_rng(0).choice(
            total, min(5000, total), replace=False))
        owner = np.searchsorted(starts, gidx, side="right") - 1
        x_fit = np.concatenate([
            train[j]["vox"][gidx[owner == j] - starts[j]]
            for j in range(len(train)) if np.any(owner == j)])
        pca = PCA(n_components=64, random_state=0).fit(x_fit)
        del x_fit
        for s in train + test:
            s["z_fpca64"] = pca.transform(s["vox"])

        arm_keys = [
            ("fold-pca:64", "z_fpca64"),
            ("DeepMReye 2.0 (lr-cca:32)", "z_cca32"),
            ("DeepMReye 2.0 Super-Basis (96 feats)", "z_super96"),
            ("DeepMReye 2.0 (lr-cca:32 + lags±2)", "z_cca32_lags2"),
        ]

        for arm_name, feat_key in arm_keys:
            xs = [s[feat_key] for s in train]
            ys = [s["labels"].reshape(-1, 20) for s in train]
            x_tr = np.concatenate(xs)
            y_tr = np.concatenate(ys)
            ok_tr = np.isfinite(y_tr).all(axis=1) & np.isfinite(x_tr).all(axis=1)
            if ok_tr.sum() < 10:
                continue

            model = RidgeCV(alphas=alphas).fit(x_tr[ok_tr], y_tr[ok_tr])

            for s in test:
                x_te = s[feat_key]
                lab_te = s["labels"]
                T = len(x_te)
                y_te_flat = lab_te.reshape(T, 20)
                ok = np.isfinite(y_te_flat).all(axis=1) & np.isfinite(x_te).all(axis=1)
                if ok.sum() < 10:
                    continue
                pred_20 = model.predict(x_te)
                pred_subtr = pred_20.reshape(T, 10, 2)

                # Sub-TR
                p_sub_flat = pred_subtr[ok].reshape(-1, 2)
                t_sub_flat = lab_te[ok].reshape(-1, 2)
                rx_sub = calc_r(p_sub_flat[:, 0], t_sub_flat[:, 0])
                ry_sub = calc_r(p_sub_flat[:, 1], t_sub_flat[:, 1])
                if np.isfinite(rx_sub) and np.isfinite(ry_sub):
                    scores_sub[arm_name].append((rx_sub + ry_sub) / 2.0)

                # 1-TR
                p_1 = np.nanmean(pred_subtr[ok], axis=1)
                t_1 = np.nanmean(lab_te[ok], axis=1)
                rx_1 = calc_r(p_1[:, 0], t_1[:, 0])
                ry_1 = calc_r(p_1[:, 1], t_1[:, 1])
                if np.isfinite(rx_1) and np.isfinite(ry_1):
                    scores_1tr[arm_name].append((rx_1 + ry_1) / 2.0)

    out_1tr = {a: float(np.nanmedian(scores_1tr[a])) if scores_1tr[a] else np.nan for a in arms}
    out_sub = {a: float(np.nanmedian(scores_sub[a])) if scores_sub[a] else np.nan for a in arms}
    return out_1tr, out_sub


def main():
    _mask, bases, meta = load_basis(Path("results/scaling/basis_n2000.npz"))
    
    datasets = [
        "dsL08_studyforrest_movie",
        "dsL09_fearlearning",
        "dsL12_rest",
        "dsL01_guided_fixations",
        "dsL02_pursuit",
        "dsL03_pursuit",
        "dsL04_pursuit",
        "dsL05_free_viewing",
        "dsL06_sequences",
        "dsL07_deepmreye_calib",
        "dsL11_backtothefuture",
    ]

    res_1tr = {}
    res_sub = {}

    print("\n" + "=" * 125)
    print("WITHIN-DATASET EVALUATION: 1-TR MEAN GAZE (5-FOLD CV WITHIN EACH DATASET)")
    print("=" * 125)
    print(f"{'Dataset':<30} {'N':>4} | {'fold-pca:64':>14} | {'lr-cca:32':>14} | {'Super-Basis (96)':>18} | {'Delta vs Supervised':>20}")
    print("-" * 125)

    for ds in datasets:
        subs = load_dataset_subs(ds)
        if not subs:
            continue
        out_1tr, out_sub = eval_within(subs, bases)
        res_1tr[ds] = out_1tr
        res_sub[ds] = out_sub

        r_f = out_1tr["fold-pca:64"]
        r_c = out_1tr["DeepMReye 2.0 (lr-cca:32)"]
        r_s = out_1tr["DeepMReye 2.0 Super-Basis (96 feats)"]
        d = r_s - r_f
        sign = "+" if d >= 0 else ""
        win = " 🏆" if d > 0 else ""
        print(f"{ds:<30} {len(subs):>4} | {r_f:>14.4f} | {r_c:>14.4f} | {r_s:>18.4f} | {f'{sign}{d:.4f}{win}':>20}")

    print("=" * 125)

    print("\n" + "=" * 125)
    print("WITHIN-DATASET EVALUATION: SUB-TR CONTINUOUS GAZE (10 PTS/TR, 5-FOLD CV WITHIN EACH DATASET)")
    print("=" * 125)
    print(f"{'Dataset':<30} {'N':>4} | {'fold-pca:64':>14} | {'lr-cca:32':>14} | {'lr-cca:32+lags±2':>18} | {'Delta vs Supervised':>20}")
    print("-" * 125)

    for ds in datasets:
        if ds not in res_sub:
            continue
        subs = load_dataset_subs(ds)
        out_sub = res_sub[ds]
        r_f = out_sub["fold-pca:64"]
        r_c = out_sub["DeepMReye 2.0 (lr-cca:32)"]
        r_l = out_sub["DeepMReye 2.0 (lr-cca:32 + lags±2)"]
        d = r_l - r_f
        sign = "+" if d >= 0 else ""
        win = " 🏆" if d > 0 else ""
        print(f"{ds:<30} {len(subs):>4} | {r_f:>14.4f} | {r_c:>14.4f} | {r_l:>18.4f} | {f'{sign}{d:.4f}{win}':>20}")

    print("=" * 125)

    out_file = Path("results/scaling/within_dataset_benchmark.json")
    out_file.write_text(json.dumps({"1tr": res_1tr, "subtr": res_sub}, indent=2))
    print(f"\n[*] Saved within-dataset benchmark to {out_file}", flush=True)


if __name__ == "__main__":
    main()
