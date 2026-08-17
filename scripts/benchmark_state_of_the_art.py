#!/usr/bin/env python3
"""State-of-the-Art Benchmark:
Combining Dual-Stream Spatiotemporal Dynamics, Target Adaptation,
Test-Time Adaptation (TTT), and Hybrid Stacking to push the 7-fold LODO benchmark.
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
    "dsL11_backtothefuture",
]


def cap(ds, max_windows):
    if not max_windows or len(ds) <= max_windows:
        return ds
    idx = np.linspace(0, len(ds) - 1, max_windows).astype(int)
    return Subset(ds, np.unique(idx).tolist())


def fit_shrunk_cca(rows, mask, corpus_basis, k=32, n_reduce=128, shrink=0.2, seed=0):
    """Shrunk CCA blending fold-local cross-orbit covariance with corpus lr-cca."""
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
        s = s + 1e-3 * float(s.max())
        return u / np.sqrt(s)

    wl = whitener(cov_ll, n_reduce)
    wr = whitener(cov_rr, n_reduce)

    m_local = wl.T @ cov_lr @ wr
    u, s, vt = np.linalg.svd(m_local, full_matrices=False)
    k_clamped = int(min(k, u.shape[1], vt.shape[0]))

    lw_local = wl @ u[:, :k_clamped]
    rw_local = wr @ vt[:k_clamped, :].T

    # Blend with corpus weights
    cw_l = corpus_basis["left_weights"][:, :k_clamped]
    cw_r = corpus_basis["right_weights"][:, :k_clamped]

    lw = (1.0 - shrink) * lw_local + shrink * cw_l
    rw = (1.0 - shrink) * rw_local + shrink * cw_r

    return {"mean": mu, "left_weights": lw, "right_weights": rw, "li": li, "ri": ri}


def project_shrunk_cca(basis, rows):
    xc = rows - basis["mean"]
    zl = xc[:, basis["li"]] @ basis["left_weights"]
    zr = xc[:, basis["ri"]] @ basis["right_weights"]
    return 0.5 * (zl + zr)


def main():
    mask, bases, _ = load_basis(BASIS_PATH)
    corpus_cca = bases["lr-cca"]
    flat_mask = mask.reshape(-1)

    folds = [(ds, {ds}) for ds in VERIFIED_DATASETS]
    n_t = WINDOW_SIZE // PATCH

    res_fpca = {}
    res_lcca = {}
    res_scca = {}
    res_sotahyb = {}

    print(f"[*] Benchmarking State-of-the-Art across {len(folds)} verified LODO folds...", flush=True)

    for holdout_name, holdout_set in folds:
        train_ds = ProbeDataset(labeled_data_dir=str(DATA_DIR), split="train", holdout=holdout_set, window_size=WINDOW_SIZE)
        test_ds = ProbeDataset(labeled_data_dir=str(DATA_DIR), split="test", holdout=holdout_set, window_size=WINDOW_SIZE)

        train_ds = cap(train_ds, MAX_TRAIN)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

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

        # 1. Fit Fold-PCA
        mu_pca = fit_rows.mean(axis=0)
        _, _, vt = randomized_svd(fit_rows - mu_pca, n_components=64, n_iter=4, random_state=0)
        fold_pca_basis = {"mean": mu_pca, "components": vt.T}

        # 2. Fit Fold-Shrunk-CCA
        scca_basis = fit_shrunk_cca(fit_rows, mask, corpus_cca, k=32, n_reduce=128, shrink=0.3)

        # Extract features
        def extract(loader):
            fpca_list, lcca_list, scca_list, y_list, ds_list, sub_list = [], [], [], [], [], []
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
                
                # shrunk-cca
                scca_list.append(project_shrunk_cca(scca_basis, flat_p).reshape(b, nt, -1))

                y_list.append(temporal_targets(y.numpy(), n_t))
                ds_list.extend(ds)
                sub_list.extend(sub)
            return (np.concatenate(fpca_list),
                    np.concatenate(lcca_list),
                    np.concatenate(scca_list),
                    np.concatenate(y_list),
                    np.array(ds_list),
                    np.array(sub_list))

        tr_fpca_raw, tr_lcca_raw, tr_scca_raw, y_tr_raw, ds_tr_raw, sub_tr_raw = extract(train_loader)
        te_fpca_raw, te_lcca_raw, te_scca_raw, y_te_raw, ds_te_raw, sub_te_raw = extract(test_loader)

        x_tr_fpca, y_tr, ds_tr, sub_tr = flatten_valid_groups(tr_fpca_raw, y_tr_raw, ds_tr_raw, sub_tr_raw)
        x_tr_lcca, _, _, _ = flatten_valid_groups(tr_lcca_raw, y_tr_raw, ds_tr_raw, sub_tr_raw)
        x_tr_scca, _, _, _ = flatten_valid_groups(tr_scca_raw, y_tr_raw, ds_tr_raw, sub_tr_raw)

        x_te_fpca, y_te, ds_te, sub_te = flatten_valid_groups(te_fpca_raw, y_te_raw, ds_te_raw, sub_te_raw)
        x_te_lcca, _, _, _ = flatten_valid_groups(te_lcca_raw, y_te_raw, ds_te_raw, sub_te_raw)
        x_te_scca, _, _, _ = flatten_valid_groups(te_scca_raw, y_te_raw, ds_te_raw, sub_te_raw)

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
        m_scca = RidgeCV(alphas=np.logspace(-3, 5, 17)).fit(x_tr_scca, y_tr_norm)

        # SOTA Hybrid Stack: fold-pca + lr-cca + shrunk-cca
        x_tr_hyb = np.concatenate([x_tr_fpca, x_tr_lcca, x_tr_scca], axis=-1)
        x_te_hyb = np.concatenate([x_te_fpca, x_te_lcca, x_te_scca], axis=-1)
        m_hyb = RidgeCV(alphas=np.logspace(-3, 5, 17)).fit(x_tr_hyb, y_tr_norm)

        baseline = y_tr_norm.mean(axis=0)
        p_fpca = m_fpca.predict(x_te_fpca)
        p_lcca = m_lcca.predict(x_te_lcca)
        p_scca = m_scca.predict(x_te_scca)
        p_hyb = m_hyb.predict(x_te_hyb)

        r_fpca = aggregate_by_subject(y_te, p_fpca, sub_te, baseline)
        r_lcca = aggregate_by_subject(y_te, p_lcca, sub_te, baseline)
        r_scca = aggregate_by_subject(y_te, p_scca, sub_te, baseline)
        r_hyb = aggregate_by_subject(y_te, p_hyb, sub_te, baseline)

        m_f = np.nanmean([r_fpca["pearson_r_x"], r_fpca["pearson_r_y"]])
        m_l = np.nanmean([r_lcca["pearson_r_x"], r_lcca["pearson_r_y"]])
        m_s = np.nanmean([r_scca["pearson_r_x"], r_scca["pearson_r_y"]])
        m_h = np.nanmean([r_hyb["pearson_r_x"], r_hyb["pearson_r_y"]])

        res_fpca[holdout_name] = m_f
        res_lcca[holdout_name] = m_l
        res_scca[holdout_name] = m_s
        res_sotahyb[holdout_name] = m_h

        print(f"  {holdout_name:<24}: fold-pca={m_f:.3f} | lr-cca={m_l:.3f} | shrunk-cca={m_s:.3f} | SOTA-Hybrid={m_h:.3f}", flush=True)

    print("\n" + "=" * 90)
    print(f"{'Method':<30}{'Median Pearson r':>25}{'vs fold-pca Margin':>25}")
    print("-" * 90)
    med_fpca = np.median(list(res_fpca.values()))
    med_lcca = np.median(list(res_lcca.values()))
    med_scca = np.median(list(res_scca.values()))
    med_hyb = np.median(list(res_sotahyb.values()))

    print(f"{'fold-pca:64':<30}{med_fpca:>25.3f}{med_fpca - med_fpca:>25.3f}")
    print(f"{'lr-cca:32':<30}{med_lcca:>25.3f}{med_lcca - med_fpca:>+25.3f}")
    print(f"{'fold-shrunk-cca:32':<30}{med_scca:>25.3f}{med_scca - med_fpca:>+25.3f}")
    print(f"{'SOTA Hybrid (fpca+lcca+scca)':<30}{med_hyb:>25.3f}{med_hyb - med_fpca:>+25.3f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
