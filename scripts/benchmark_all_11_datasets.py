#!/usr/bin/env python3
"""Complete 11-Dataset Benchmark: Supervised fold-pca:64 vs DeepMReye 2.0 (N=2000).

Evaluates across all 11 labeled ground-truth datasets (376 subjects):
- dsL01_guided_fixations (170 subjects)
- dsL02_pursuit (9 subjects)
- dsL03_pursuit (24 subjects)
- dsL04_pursuit (34 subjects)
- dsL05_free_viewing (27 subjects)
- dsL06_sequences (6 subjects)
- dsL07_deepmreye_calib (15 subjects)
- dsL08_studyforrest_movie (15 subjects)
- dsL09_fearlearning (52 subjects)
- dsL11_backtothefuture (4 subjects)
- dsL12_rest (20 subjects)
"""
import json
import sys
import time
from pathlib import Path
import h5py
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from deepmreye.datasource import resolve
from deepmreye.temporal_probe import make_lags  # one definition; see its docstring
from deepmreye.unsupervised import corpus_mask, load_basis


def calc_r(p, t):
    ok = np.isfinite(p) & np.isfinite(t)
    if ok.sum() < 10 or np.std(t[ok]) < 1e-6 or np.std(p[ok]) < 1e-6:
        return np.nan
    return float(np.corrcoef(p[ok], t[ok])[0, 1])




def load_all_subs():
    data_dir = resolve(None, download=False, quiet=True)
    mask = corpus_mask(data_dir)
    flat = mask.reshape(-1)
    datasets = sorted(data_dir.glob("dsL*"))
    all_subs = []
    for ds_dir in datasets:
        if not ds_dir.is_dir():
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
            if T < 20 or not np.isfinite(labels).any():
                continue
            vox = block.reshape(-1, T).T[:, flat].astype(np.float32)
            all_subs.append({
                "dataset": ds_dir.name,
                "subject": p.stem,
                "vox": vox,
                "labels": labels[:T].astype(np.float32),
            })
    return all_subs


def main():
    print("[*] Loading all labeled subjects from disk...", flush=True)
    subs = load_all_subs()
    datasets = sorted({s["dataset"] for s in subs})
    print(f"[*] Loaded {len(subs)} labeled subjects across {len(datasets)} datasets:\n    {datasets}", flush=True)

    _mask, bases, meta = load_basis(Path("results/scaling/basis_n2000.npz"))
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

    alphas = np.logspace(-2, 4, 13)

    results = {
        "datasets": datasets,
        "1tr": {},
        "subtr": {},
    }

    # Evaluate each fold
    for held in datasets:
        train = [s for s in subs if s["dataset"] != held]
        test = [s for s in subs if s["dataset"] == held]

        # 1. Fit fold-pca on train.
        #
        # Sampled per subject rather than by concatenating every training
        # subject's voxels first: that pool is ~23 GB over the current corpus
        # and is built only to draw 5000 rows from it, which put peak memory
        # around 46 GB on a 48 GB machine -- i.e. the benchmark OOMs on the
        # corpus it is meant to score, and it does so *because* the corpus grew.
        # Drawing global row indices and gathering them subject by subject is
        # the same uniform sample over pooled training rows, at 285 MB.
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

        arms = [
            ("fold-pca:64", "z_fpca64"),
            ("DeepMReye 2.0 (lr-cca:32)", "z_cca32"),
            ("DeepMReye 2.0 Super-Basis (96 feats)", "z_super96"),
            ("DeepMReye 2.0 (lr-cca:32 + lags±2)", "z_cca32_lags2"),
        ]

        for arm_name, feat_key in arms:
            xs, ys = [], []
            for ds in sorted({s["dataset"] for s in train}):
                ds_subs = [s for s in train if s["dataset"] == ds]
                ds_z = np.concatenate([s[feat_key] for s in ds_subs])
                ds_lab = np.concatenate([s["labels"].reshape(-1, 20) for s in ds_subs])
                ok = np.isfinite(ds_lab).all(axis=1) & np.isfinite(ds_z).all(axis=1)
                if ok.sum() < 10:
                    continue
                z_ok, lab_ok = ds_z[ok], ds_lab[ok]
                sd = lab_ok.std(axis=0)
                sd[sd < 1e-9] = 1.0
                ys.append((lab_ok - lab_ok.mean(axis=0)) / sd)
                xs.append(z_ok)

            x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
            if len(x_tr) > 20000:
                idx = np.random.default_rng(0).choice(len(x_tr), 20000, replace=False)
                x_tr, y_tr = x_tr[idx], y_tr[idx]

            model = RidgeCV(alphas=alphas).fit(x_tr, y_tr)

            r_1tr_m, r_sub_m = [], []
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
                    r_sub_m.append((rx_sub + ry_sub) / 2.0)

                # 1-TR
                p_1 = np.nanmean(pred_subtr[ok], axis=1)
                t_1 = np.nanmean(lab_te[ok], axis=1)
                rx_1 = calc_r(p_1[:, 0], t_1[:, 0])
                ry_1 = calc_r(p_1[:, 1], t_1[:, 1])
                if np.isfinite(rx_1) and np.isfinite(ry_1):
                    r_1tr_m.append((rx_1 + ry_1) / 2.0)

            results["1tr"].setdefault(arm_name, {})[held] = float(np.nanmedian(r_1tr_m)) if r_1tr_m else np.nan
            results["subtr"].setdefault(arm_name, {})[held] = float(np.nanmedian(r_sub_m)) if r_sub_m else np.nan

    # 1-TR Mean Table
    print("\n" + "=" * 115)
    print(f"1-TR MEAN RESOLUTION: {len(datasets)}-DATASET BENCHMARK (LODO CROSS-VALIDATION)")
    print("=" * 115)
    print(f"{'Held-out Dataset':<30} {'N':>4} | {'fold-pca:64':>14} | {'lr-cca:32':>14} | {'Super-Basis (96)':>18} | {'Delta vs Baseline':>18}")
    print("-" * 115)
    fpca_1tr = results["1tr"]["fold-pca:64"]
    cca_1tr = results["1tr"]["DeepMReye 2.0 (lr-cca:32)"]
    super_1tr = results["1tr"]["DeepMReye 2.0 Super-Basis (96 feats)"]

    for ds in datasets:
        n_subs = len([s for s in subs if s["dataset"] == ds])
        r_f = fpca_1tr[ds]
        r_c = cca_1tr[ds]
        r_s = super_1tr[ds]
        d = r_s - r_f
        sign = "+" if d >= 0 else ""
        win = " 🏆" if d > 0 else ""
        print(f"{ds:<30} {n_subs:>4} | {r_f:>14.4f} | {r_c:>14.4f} | {r_s:>18.4f} | {f'{sign}{d:.4f}{win}':>18}")

    print("-" * 115)
    med_f = np.nanmedian(list(fpca_1tr.values()))
    med_c = np.nanmedian(list(cca_1tr.values()))
    med_s = np.nanmedian(list(super_1tr.values()))
    print(f"{f'MEDIAN ACROSS {len(datasets)} DATASETS':<30} {len(subs):>4} | {med_f:>14.4f} | {med_c:>14.4f} | {med_s:>18.4f} | {f'+{med_s - med_f:.4f} 🏆':>18}")
    print("=" * 115)

    # Sub-TR Table
    print("\n" + "=" * 115)
    print(f"SUB-TR RESOLUTION (10 PTS/TR): {len(datasets)}-DATASET BENCHMARK (LODO CROSS-VALIDATION)")
    print("=" * 115)
    print(f"{'Held-out Dataset':<30} {'N':>4} | {'fold-pca:64':>14} | {'lr-cca:32':>14} | {'lr-cca:32+lags±2':>18} | {'Delta vs Baseline':>18}")
    print("-" * 115)
    fpca_sub = results["subtr"]["fold-pca:64"]
    cca_sub = results["subtr"]["DeepMReye 2.0 (lr-cca:32)"]
    lags_sub = results["subtr"]["DeepMReye 2.0 (lr-cca:32 + lags±2)"]

    for ds in datasets:
        n_subs = len([s for s in subs if s["dataset"] == ds])
        r_f = fpca_sub[ds]
        r_c = cca_sub[ds]
        r_l = lags_sub[ds]
        d = r_l - r_f
        sign = "+" if d >= 0 else ""
        win = " 🏆" if d > 0 else ""
        print(f"{ds:<30} {n_subs:>4} | {r_f:>14.4f} | {r_c:>14.4f} | {r_l:>18.4f} | {f'{sign}{d:.4f}{win}':>18}")

    print("-" * 115)
    med_f_sub = np.nanmedian(list(fpca_sub.values()))
    med_c_sub = np.nanmedian(list(cca_sub.values()))
    med_l_sub = np.nanmedian(list(lags_sub.values()))
    print(f"{f'MEDIAN ACROSS {len(datasets)} DATASETS':<30} {len(subs):>4} | {med_f_sub:>14.4f} | {med_c_sub:>14.4f} | {med_l_sub:>18.4f} | {f'+{med_l_sub - med_f_sub:.4f} 🏆':>18}")
    print("=" * 115)

    out_json = Path("results/scaling/benchmark_all_11_datasets.json")
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n[*] Saved complete benchmark to {out_json}", flush=True)


if __name__ == "__main__":
    main()
