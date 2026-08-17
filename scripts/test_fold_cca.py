#!/usr/bin/env python3
"""Test Fold-Local Cross-Orbit CCA (fold-cca) and Shrunk Cross-Orbit CCA (fold-shrunk-cca)
to see if combining target-study adaptation with conjugate cross-orbit constraints
beats fold-pca:64 (0.847) and lr-cca:32 (0.825).
"""
import sys
from pathlib import Path
import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from deepmreye.data.probe_dataset import ProbeDataset
from deepmreye.evaluate.probe import aggregate_by_subject, temporal_targets, flatten_valid_groups
from deepmreye.evaluate.features import pool_time
from deepmreye.unsupervised import LR_SPLIT_X, load_basis
from torch.utils.data import DataLoader, Subset

DATA_DIR = Path.home() / ".cache" / "deepmreye"
BASIS_PATH = "results/scaling/basis_n1039.npz"
WINDOW_SIZE = 20
PATCH = 5
MAX_TRAIN = 1000
BASIS_FIT = 400

VERIFIED_DATASETS = [
    "dsL01_guided_fixations",
    "dsL02_pursuit",
    "dsL03_pursuit",
    "dsL04_pursuit",
    "dsL05_free_viewing",
    "dsL06_sequences",
    "dsL07_deepmreye_calib",
]


def fit_local_cca(rows, mask, k=32, n_reduce=128, shrinkage=1e-3, seed=0):
    flat = mask.reshape(-1)
    xs = np.nonzero(flat)[0] // (mask.shape[1] * mask.shape[2])
    left = xs < LR_SPLIT_X
    li, ri = np.nonzero(left)[0], np.nonzero(~left)[0]

    mu = rows.mean(axis=0)
    xc = rows - mu
    cov_ll = (xc[:, li].T @ xc[:, li]) / len(rows)
    cov_rr = (xc[:, ri].T @ xc[:, ri]) / len(rows)
    cov_lr = (xc[:, li].T @ xc[:, ri]) / len(rows)

    def whitener(cov, n_red):
        u, s, _ = randomized_svd(cov, n_components=min(n_red, cov.shape[0] - 1), random_state=seed)
        s = s + shrinkage * float(s.max())
        return u / np.sqrt(s)

    wl = whitener(cov_ll, n_reduce)
    wr = whitener(cov_rr, n_reduce)

    m = wl.T @ cov_lr @ wr
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    k_clamped = int(min(k, u.shape[1], vt.shape[0]))

    left_weights = wl @ u[:, :k_clamped]
    right_weights = wr @ vt[:k_clamped, :].T
    return {"mean": mu, "left_weights": left_weights, "right_weights": right_weights, "li": li, "ri": ri}


def project_local_cca(basis, rows):
    xc = rows - basis["mean"]
    zl = xc[:, basis["li"]] @ basis["left_weights"]
    zr = xc[:, basis["ri"]] @ basis["right_weights"]
    return 0.5 * (zl + zr)


def cap(ds, max_windows):
    if not max_windows or len(ds) <= max_windows:
        return ds
    idx = np.linspace(0, len(ds) - 1, max_windows).astype(int)
    return Subset(ds, np.unique(idx).tolist())


def main():
    mask, bases, _ = load_basis(BASIS_PATH)
    corpus_cca = bases["lr-cca"]
    flat_mask = mask.reshape(-1)

    folds = [(ds, {ds}) for ds in VERIFIED_DATASETS]
    n_t = WINDOW_SIZE // PATCH
    results_fold_pca = {}
    results_lr_cca = {}
    results_fold_cca = {}
    results_hybrid = {}

    print(f"[*] Testing Fold-Local CCA vs Corpus CCA vs Fold-PCA across {len(folds)} verified folds...", flush=True)

    for holdout_name, holdout_set in folds:
        train_ds = ProbeDataset(labeled_data_dir=str(DATA_DIR), split="train", holdout=holdout_set, window_size=WINDOW_SIZE)
        test_ds = ProbeDataset(labeled_data_dir=str(DATA_DIR), split="test", holdout=holdout_set, window_size=WINDOW_SIZE)

        train_ds = cap(train_ds, MAX_TRAIN)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

        # 1. Fit Fold-PCA and Fold-CCA on basis_fit_windows
        fit_ds = cap(train_ds, BASIS_FIT)
        fit_loader = DataLoader(fit_ds, batch_size=32, shuffle=False)

        fit_rows = []
        for x, y, ds, sub, tr in fit_loader:
            p = pool_time(x.numpy(), n_t)
            yt = temporal_targets(y.numpy(), n_t)
            sel = p[..., flat_mask].reshape(-1, int(flat_mask.sum()))
            valid = np.isfinite(yt.reshape(-1, 2)).all(axis=1)
            if valid.any():
                fit_rows.append(sel[valid])
        fit_rows = np.concatenate(fit_rows)

        # Fit Fold-PCA
        mu_pca = fit_rows.mean(axis=0)
        _, _, vt = randomized_svd(fit_rows - mu_pca, n_components=64, n_iter=4, random_state=0)
        fold_pca_basis = {"mean": mu_pca, "components": vt.T}

        # Fit Fold-CCA
        fold_cca_basis = fit_local_cca(fit_rows, mask, k=32, n_reduce=128)

        # Extract features
        def extract(loader):
            fpca_list, lcca_list, fcca_list, y_list, ds_list, sub_list = [], [], [], [], [], []
            for x, y, ds, sub, tr in loader:
                p = pool_time(x.numpy(), n_t)
                b, nt, v = p.shape
                flat_p = p[..., flat_mask].reshape(-1, int(flat_mask.sum()))
                
                # fold-pca
                feat_fpca = ((flat_p - fold_pca_basis["mean"]) @ fold_pca_basis["components"]).reshape(b, nt, -1)
                fpca_list.append(feat_fpca)
                
                # corpus lr-cca
                zl = (flat_p - corpus_cca["mean"])[:, corpus_cca["left_index"]] @ corpus_cca["left_weights"][:, :32]
                zr = (flat_p - corpus_cca["mean"])[:, corpus_cca["right_index"]] @ corpus_cca["right_weights"][:, :32]
                lcca_list.append((0.5 * (zl + zr)).reshape(b, nt, -1))
                
                # fold-cca
                fcca_list.append(project_local_cca(fold_cca_basis, flat_p).reshape(b, nt, -1))

                y_list.append(temporal_targets(y.numpy(), n_t))
                ds_list.extend(ds)
                sub_list.extend(sub)
            return (np.concatenate(fpca_list),
                    np.concatenate(lcca_list),
                    np.concatenate(fcca_list),
                    np.concatenate(y_list),
                    np.array(ds_list),
                    np.array(sub_list))

        tr_fpca_raw, tr_lcca_raw, tr_fcca_raw, y_tr_raw, ds_tr_raw, sub_tr_raw = extract(train_loader)
        te_fpca_raw, te_lcca_raw, te_fcca_raw, y_te_raw, ds_te_raw, sub_te_raw = extract(test_loader)

        x_tr_fpca, y_tr, ds_tr, sub_tr = flatten_valid_groups(tr_fpca_raw, y_tr_raw, ds_tr_raw, sub_tr_raw)
        x_tr_lcca, _, _, _ = flatten_valid_groups(tr_lcca_raw, y_tr_raw, ds_tr_raw, sub_tr_raw)
        x_tr_fcca, _, _, _ = flatten_valid_groups(tr_fcca_raw, y_tr_raw, ds_tr_raw, sub_tr_raw)

        x_te_fpca, y_te, ds_te, sub_te = flatten_valid_groups(te_fpca_raw, y_te_raw, ds_te_raw, sub_te_raw)
        x_te_lcca, _, _, _ = flatten_valid_groups(te_lcca_raw, y_te_raw, ds_te_raw, sub_te_raw)
        x_te_fcca, _, _, _ = flatten_valid_groups(te_fcca_raw, y_te_raw, ds_te_raw, sub_te_raw)

        # Standardize targets per dataset
        y_tr_norm = y_tr.copy()
        for d in np.unique(ds_tr):
            m = ds_tr == d
            sd = y_tr_norm[m].std(axis=0)
            sd[sd < 1e-9] = 1.0
            y_tr_norm[m] = (y_tr_norm[m] - y_tr_norm[m].mean(axis=0)) / sd

        # Train models
        m_fpca = RidgeCV(alphas=np.logspace(-3, 5, 17)).fit(x_tr_fpca, y_tr_norm)
        m_lcca = RidgeCV(alphas=np.logspace(-3, 5, 17)).fit(x_tr_lcca, y_tr_norm)
        m_fcca = RidgeCV(alphas=np.logspace(-3, 5, 17)).fit(x_tr_fcca, y_tr_norm)

        # Hybrid of fold-pca + fold-cca + lr-cca
        x_tr_hyb = np.concatenate([x_tr_fpca, x_tr_fcca, x_tr_lcca], axis=-1)
        x_te_hyb = np.concatenate([x_te_fpca, x_te_fcca, x_te_lcca], axis=-1)
        m_hyb = RidgeCV(alphas=np.logspace(-3, 5, 17)).fit(x_tr_hyb, y_tr_norm)

        baseline = y_tr_norm.mean(axis=0)
        p_fpca = m_fpca.predict(x_te_fpca)
        p_lcca = m_lcca.predict(x_te_lcca)
        p_fcca = m_fcca.predict(x_te_fcca)
        p_hyb = m_hyb.predict(x_te_hyb)

        r_fpca = aggregate_by_subject(y_te, p_fpca, sub_te, baseline)
        r_lcca = aggregate_by_subject(y_te, p_lcca, sub_te, baseline)
        r_fcca = aggregate_by_subject(y_te, p_fcca, sub_te, baseline)
        r_hyb = aggregate_by_subject(y_te, p_hyb, sub_te, baseline)

        m_f = np.nanmean([r_fpca["pearson_r_x"], r_fpca["pearson_r_y"]])
        m_l = np.nanmean([r_lcca["pearson_r_x"], r_lcca["pearson_r_y"]])
        m_c = np.nanmean([r_fcca["pearson_r_x"], r_fcca["pearson_r_y"]])
        m_h = np.nanmean([r_hyb["pearson_r_x"], r_hyb["pearson_r_y"]])

        results_fold_pca[holdout_name] = m_f
        results_lr_cca[holdout_name] = m_l
        results_fold_cca[holdout_name] = m_c
        results_hybrid[holdout_name] = m_h

        print(f"  {holdout_name:<24}: fold-pca={m_f:.3f} | lr-cca={m_l:.3f} | fold-cca={m_c:.3f} | hybrid={m_h:.3f}", flush=True)

    print("\n" + "=" * 80)
    print(f"{'Method':<25}{'Median Pearson r':>20}")
    print("-" * 80)
    print(f"{'fold-pca:64':<25}{np.median(list(results_fold_pca.values())):>20.3f}")
    print(f"{'lr-cca:32':<25}{np.median(list(results_lr_cca.values())):>20.3f}")
    print(f"{'fold-cca:32':<25}{np.median(list(results_fold_cca.values())):>20.3f}")
    print(f"{'hybrid (fpca+fcca+lcca)':<25}{np.median(list(results_hybrid.values())):>20.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
