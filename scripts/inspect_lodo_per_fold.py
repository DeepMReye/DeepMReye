#!/usr/bin/env python3
"""Inspect Per-Fold 1-TR Mean Performance on all 7 LODO Folds."""
import sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
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


def load_cached():
    d = np.load("results/scaling/eval_cache_raw.npz", allow_pickle=True)
    recs = []
    for i in range(int(d["n"])):
        recs.append({
            "dataset": str(d[f"ds/{i}"]),
            "subject": str(d[f"sub/{i}"]),
            "rows": d[f"rows/{i}"],
            "gaze": d[f"gaze/{i}"],
        })
    return recs


def main():
    recs = load_cached()
    datasets = sorted({r["dataset"] for r in recs})
    print(f"[*] Found {len(datasets)} held-out datasets: {datasets}", flush=True)

    _mask, bases, meta = load_basis(Path("results/scaling/basis_n1800.npz"))
    cca_dict = bases["lr-cca"]
    mu_cca = cca_dict["mean"]
    li, ri = cca_dict["left_index"], cca_dict["right_index"]
    wl, wr = cca_dict["left_weights"], cca_dict["right_weights"]

    # 1. fold-pca:64
    fold_pca_scores = {}
    # 2. lr-cca:32
    cca32_scores = {}
    # 3. lr-cca:64
    cca64_scores = {}
    # 4. lr-cca:32 + gev-fast:16
    gev_comp = bases["gev-fast"]["components"]
    gev_mu = bases["gev-fast"]["mean"]
    hybrid_scores = {}

    for held in datasets:
        train = [r for r in recs if r["dataset"] != held]
        test = [r for r in recs if r["dataset"] == held]

        # Fit fold-pca on train
        x_tr_raw = np.concatenate([r["rows"] for r in train])
        idx = np.random.default_rng(0).choice(len(x_tr_raw), min(10000, len(x_tr_raw)), replace=False)
        pca = PCA(n_components=64, random_state=0).fit(x_tr_raw[idx])

        # Prepare training data
        ys = []
        xs_fpca = []
        xs_cca32 = []
        xs_cca64 = []
        xs_hyb = []

        for ds in sorted({r["dataset"] for r in train}):
            ds_recs = [r for r in train if r["dataset"] == ds]
            ds_rows = np.concatenate([r["rows"] for r in ds_recs])
            ds_gaze = np.concatenate([r["gaze"] for r in ds_recs])
            ok = np.isfinite(ds_gaze).all(axis=1) & np.isfinite(ds_rows).all(axis=1)
            if ok.sum() < 10:
                continue
            g_ok = ds_gaze[ok]
            sd = g_ok.std(axis=0)
            sd[sd < 1e-9] = 1.0
            ys.append((g_ok - g_ok.mean(axis=0)) / sd)

            r_ok = ds_rows[ok]
            # fold-pca
            xs_fpca.append(pca.transform(r_ok))
            # lr-cca:32
            xc = r_ok - mu_cca
            xs_cca32.append(0.5 * (xc[:, li] @ wl[:, :32] + xc[:, ri] @ wr[:, :32]))
            # lr-cca:64
            xs_cca64.append(0.5 * (xc[:, li] @ wl[:, :64] + xc[:, ri] @ wr[:, :64]))
            # hybrid
            z_cca = 0.5 * (xc[:, li] @ wl[:, :32] + xc[:, ri] @ wr[:, :32])
            z_gev = (r_ok - gev_mu) @ gev_comp[:, :16]
            xs_hyb.append(np.concatenate([z_cca, z_gev], axis=1))

        y_tr = np.concatenate(ys)
        x_tr_fpca = np.concatenate(xs_fpca)
        x_tr_cca32 = np.concatenate(xs_cca32)
        x_tr_cca64 = np.concatenate(xs_cca64)
        x_tr_hyb = np.concatenate(xs_hyb)

        if len(y_tr) > 20000:
            idx = np.random.default_rng(0).choice(len(y_tr), 20000, replace=False)
            y_tr = y_tr[idx]
            x_tr_fpca = x_tr_fpca[idx]
            x_tr_cca32 = x_tr_cca32[idx]
            x_tr_cca64 = x_tr_cca64[idx]
            x_tr_hyb = x_tr_hyb[idx]

        alphas = np.logspace(-2, 5, 15)
        m_fpca = RidgeCV(alphas=alphas).fit(x_tr_fpca, y_tr)
        m_cca32 = RidgeCV(alphas=alphas).fit(x_tr_cca32, y_tr)
        m_cca64 = RidgeCV(alphas=alphas).fit(x_tr_cca64, y_tr)
        m_hyb = RidgeCV(alphas=alphas).fit(x_tr_hyb, y_tr)

        def score_model(m, get_feat_fn):
            sub_rs = []
            for r in test:
                g = r["gaze"]
                ok = np.isfinite(g).all(axis=1) & np.isfinite(r["rows"]).all(axis=1)
                if ok.sum() < 10:
                    continue
                feat = get_feat_fn(r["rows"][ok])
                pred = m.predict(feat)
                rx = calc_r(pred[:, 0], g[ok, 0])
                ry = calc_r(pred[:, 1], g[ok, 1])
                if np.isfinite(rx) and np.isfinite(ry):
                    sub_rs.append((rx + ry) / 2.0)
            return float(np.nanmedian(sub_rs)) if sub_rs else np.nan

        fold_pca_scores[held] = score_model(m_fpca, lambda x: pca.transform(x))
        cca32_scores[held] = score_model(m_cca32, lambda x: 0.5 * ((x - mu_cca)[:, li] @ wl[:, :32] + (x - mu_cca)[:, ri] @ wr[:, :32]))
        cca64_scores[held] = score_model(m_cca64, lambda x: 0.5 * ((x - mu_cca)[:, li] @ wl[:, :64] + (x - mu_cca)[:, ri] @ wr[:, :64]))
        cca_hyb_fn = lambda x: np.concatenate([
            0.5 * ((x - mu_cca)[:, li] @ wl[:, :32] + (x - mu_cca)[:, ri] @ wr[:, :32]),
            (x - gev_mu) @ gev_comp[:, :16]
        ], axis=1)
        hybrid_scores[held] = score_model(m_hyb, cca_hyb_fn)

    print("\n" + "=" * 90)
    print(f"{'Held-out Dataset':<25} {'fold-pca:64':>12} {'lr-cca:32':>12} {'lr-cca:64':>12} {'lr-cca+gev16':>14}")
    print("=" * 90)
    for ds in datasets:
        print(f"{ds:<25} {fold_pca_scores[ds]:>12.4f} {cca32_scores[ds]:>12.4f} {cca64_scores[ds]:>12.4f} {hybrid_scores[ds]:>14.4f}")
    print("-" * 90)
    print(f"{'MEDIAN':<25} {np.median(list(fold_pca_scores.values())):>12.4f} {np.median(list(cca32_scores.values())):>12.4f} {np.median(list(cca64_scores.values())):>12.4f} {np.median(list(hybrid_scores.values())):>14.4f}")
    print(f"{'MEAN':<25} {np.mean(list(fold_pca_scores.values())):>12.4f} {np.mean(list(cca32_scores.values())):>12.4f} {np.mean(list(cca64_scores.values())):>12.4f} {np.mean(list(hybrid_scores.values())):>14.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
