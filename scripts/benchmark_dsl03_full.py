#!/usr/bin/env python3
"""Comprehensive dsL03_pursuit Benchmark: DeepMReye 1.0 vs DeepMReye 2.0.

Evaluates all 24 participants of dsL03_pursuit across:
- Target Resolutions: Sub-TR (10 samples/TR), 1-TR mean, 5-TR bin mean
- Temporal Context: Instantaneous (1 TR), Multi-lag (+-1 TR), Multi-lag (+-2 TRs)
- Protocols: Within-Subject (5-fold CV) and LODO Cross-Dataset
- Filtering: 100% Full Cohort (N=24) and Top-80% Reliable Cohort (N=19, DME1 paper protocol)
- Metrics: r_x, r_y, mean r, Euclidean error (deg)
"""
import os
import sys
import json
import subprocess
import tempfile
import warnings
from pathlib import Path
import h5py
import numpy as np
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = Path.home() / ".cache/deepmreye/dsL03_pursuit"
BASIS_PATH = REPO / "results/scaling/basis_n1039.npz"
DME1_WEIGHTS_WITHIN = REPO / "results/dme1/dataset3_pursuit.h5"
DME1_WEIGHTS_LODO = REPO / "results/dme1/datasets_1to5.h5"
DME1_CACHE = REPO / "results/dme1_dsl03_cached_preds.npz"
OUT_JSON = REPO / "results/dsl03_full_benchmark.json"

V1_FILES = ["deepmreye/architecture.py", "deepmreye/util/util.py", "deepmreye/util/model_opts.py"]

def vendor_v1():
    root = Path(tempfile.mkdtemp(prefix="dme1_"))
    (root / "deepmreye" / "util").mkdir(parents=True)
    (root / "deepmreye" / "__init__.py").write_text("")
    (root / "deepmreye" / "util" / "__init__.py").write_text("")
    for rel in V1_FILES:
        out = subprocess.run(["git", "-C", str(REPO), "show", f"main:{rel}"],
                             capture_output=True, check=True)
        (root / rel).write_bytes(out.stdout)
    sys.path.insert(0, str(root))
    return root

def build_dme1_model(spatial_shape=(47, 29, 18, 1), inner_timesteps=10):
    from deepmreye import architecture
    from deepmreye.util import model_opts
    from tensorflow.keras.optimizers import Adam
    architecture.get_adam_optimizer = lambda lr: Adam(learning_rate=lr)

    opts = model_opts.get_opts()
    opts["mc_dropout"] = False
    opts["gaussian_noise"] = 0
    opts["inner_timesteps"] = inner_timesteps
    _, model_inference = architecture.create_standard_model(spatial_shape, opts)
    return model_inference

def load_basis(path):
    b = np.load(path)
    mask = b["mask"]
    bases = {
        "lr-cca": {
            "mean": b["lr-cca/mean"],
            "left_index": b["lr-cca/left_index"],
            "right_index": b["lr-cca/right_index"],
            "left_weights": b["lr-cca/left_weights"],
            "right_weights": b["lr-cca/right_weights"],
        },
        "gev-fast": {
            "mean": b["gev-fast/mean"],
            "components": b["gev-fast/components"],
        },
        "corpus-pca": {
            "mean": b["corpus-pca/mean"],
            "components": b["corpus-pca/components"],
        }
    }
    return mask, bases

def project_basis(basis_name, basis_dict, x, k=32):
    if basis_name == "lr-cca":
        mu = basis_dict["mean"]
        li, ri = basis_dict["left_index"], basis_dict["right_index"]
        wl, wr = basis_dict["left_weights"][:, :k], basis_dict["right_weights"][:, :k]
        xc = x - mu
        return 0.5 * (xc[:, li] @ wl + xc[:, ri] @ wr)
    else:
        mu = basis_dict["mean"]
        comp = basis_dict["components"][:, :k] # [14236, k]
        return (x - mu) @ comp

def load_dsl03_data():
    files = sorted(DATA_DIR.glob("*.h5"))
    print(f"[*] Loading all {len(files)} participants from {DATA_DIR}...", flush=True)
    data = []
    for p in files:
        with h5py.File(p, "r") as f:
            block = f["eye_block"][:] # [47, 29, 18, T]
            labels = f["labels"][:]   # [T, 10, 2]
            tr = float(f.attrs.get("repetition_time", 1.02))
            data.append({
                "sub": p.stem,
                "block": block,
                "labels": labels,
                "tr": tr
            })
    return data

def calc_r(p, t):
    ok = np.isfinite(p) & np.isfinite(t)
    if ok.sum() < 20 or np.std(t[ok]) < 1e-6 or np.std(p[ok]) < 1e-6:
        return np.nan
    return float(np.corrcoef(p[ok], t[ok])[0, 1])

def calc_error(p, t):
    ok = np.isfinite(p).all(axis=-1) & np.isfinite(t).all(axis=-1)
    if ok.sum() < 20:
        return np.nan
    return float(np.mean(np.linalg.norm(p[ok] - t[ok], axis=-1)))

def make_lag_features(z, lags=2):
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

def run_benchmark():
    data = load_dsl03_data()
    mask, bases = load_basis(BASIS_PATH)
    benchmark_results = []
    
    # Helper to evaluate subject predictions
    def evaluate_predictions(preds_subtr, labels, name, res_name, proto_name, temporal_ctx):
        n = min(len(preds_subtr), len(labels))
        p_subtr = preds_subtr[:n]
        t_subtr = labels[:n]
        
        # Sub-TR
        p_flat = p_subtr.reshape(-1, 2)
        t_flat = t_subtr.reshape(-1, 2)
        rx_sub = calc_r(p_flat[:, 0], t_flat[:, 0])
        ry_sub = calc_r(p_flat[:, 1], t_flat[:, 1])
        err_sub = calc_error(p_flat, t_flat)
        
        # 1-TR mean
        p_tr = np.nanmean(p_subtr, axis=1)
        t_tr = np.nanmean(t_subtr, axis=1)
        rx_tr = calc_r(p_tr[:, 0], t_tr[:, 0])
        ry_tr = calc_r(p_tr[:, 1], t_tr[:, 1])
        err_tr = calc_error(p_tr, t_tr)
        
        # 5-TR bin mean
        n_bins = n // 5
        p_bin5 = np.nanmean(p_subtr[:n_bins*5].reshape(n_bins, 50, 2), axis=1)
        t_bin5 = np.nanmean(t_subtr[:n_bins*5].reshape(n_bins, 50, 2), axis=1)
        rx_b5 = calc_r(p_bin5[:, 0], t_bin5[:, 0])
        ry_b5 = calc_r(p_bin5[:, 1], t_bin5[:, 1])
        err_b5 = calc_error(p_bin5, t_bin5)
        
        return {
            "subtr": (rx_sub, ry_sub, err_sub),
            "1tr": (rx_tr, ry_tr, err_tr),
            "5tr": (rx_b5, ry_b5, err_b5),
        }

    # 1. DeepMReye 1.0 (with caching for speed)
    if DME1_CACHE.exists():
        print(f"[*] Loading cached DME1 predictions from {DME1_CACHE.name}...", flush=True)
        cached = np.load(DME1_CACHE, allow_pickle=True)
        dme1_preds_within = [cached[f"within_{i}"] for i in range(len(data))]
        dme1_preds_lodo = [cached[f"lodo_{i}"] for i in range(len(data))]
    else:
        vendor_v1()
        print("[*] Loading DeepMReye 1.0 models and computing predictions...", flush=True)
        dme1_within = build_dme1_model()
        dme1_within.load_weights(str(DME1_WEIGHTS_WITHIN))
        
        dme1_lodo = build_dme1_model()
        dme1_lodo.load_weights(str(DME1_WEIGHTS_LODO))
        
        dme1_preds_within = []
        dme1_preds_lodo = []
        for i, d in enumerate(data):
            x_in = np.moveaxis(d["block"], -1, 0)[..., None].astype(np.float32)
            n = min(len(x_in), len(d["labels"]))
            p_w = dme1_within.predict(x_in[:n], batch_size=64, verbose=0)[0]
            p_l = dme1_lodo.predict(x_in[:n], batch_size=64, verbose=0)[0]
            dme1_preds_within.append(p_w)
            dme1_preds_lodo.append(p_l)
            print(f"  Processed DME1 sub {i+1}/{len(data)}", flush=True)
            
        save_dict = {}
        for i in range(len(data)):
            save_dict[f"within_{i}"] = dme1_preds_within[i]
            save_dict[f"lodo_{i}"] = dme1_preds_lodo[i]
        np.savez_compressed(DME1_CACHE, **save_dict)
        print(f"[+] Saved cached DME1 predictions to {DME1_CACHE}", flush=True)

    for model_name, preds_list, proto in [
        ("DeepMReye 1.0 (3D-CNN Within)", dme1_preds_within, "Within-Dataset"),
        ("DeepMReye 1.0 (3D-CNN LODO)", dme1_preds_lodo, "LODO Cross-Dataset")
    ]:
        print(f"[*] Scoring {model_name}...", flush=True)
        res_by_res = {"Sub-TR (10 pts/TR)": [], "1-TR mean": [], "5-TR bin mean": []}
        for i, d in enumerate(data):
            pred = preds_list[i]
            scores = evaluate_predictions(pred, d["labels"], model_name, "", proto, "3-frame 3D-CNN")
            res_by_res["Sub-TR (10 pts/TR)"].append(scores["subtr"])
            res_by_res["1-TR mean"].append(scores["1tr"])
            res_by_res["5-TR bin mean"].append(scores["5tr"])
            
        for res_label, scores_list in res_by_res.items():
            rx = np.array([s[0] for s in scores_list])
            ry = np.array([s[1] for s in scores_list])
            err = np.array([s[2] for s in scores_list])
            r_mean = 0.5 * (rx + ry)
            
            valid = np.isfinite(r_mean)
            r_val = r_mean[valid]
            rx_val = rx[valid]
            ry_val = ry[valid]
            err_val = err[valid]
            top80_idx = np.argsort(r_val)[int(len(r_val) * 0.2):]
            
            benchmark_results.append({
                "model": model_name,
                "resolution": res_label,
                "protocol": proto,
                "temporal_context": "3D-CNN (3 frames)",
                "all_rx": float(np.nanmedian(rx)),
                "all_ry": float(np.nanmedian(ry)),
                "all_r": float(np.nanmedian(r_mean)),
                "all_err": float(np.nanmedian(err)),
                "top80_rx": float(np.median(rx_val[top80_idx])),
                "top80_ry": float(np.median(ry_val[top80_idx])),
                "top80_r": float(np.median(r_val[top80_idx])),
                "top80_err": float(np.median(err_val[top80_idx])),
            })

    # 2. Evaluate DeepMReye 2.0 Linear Bases & Temporal Multi-Lags
    v2_configs = [
        ("DeepMReye 2.0 (lr-cca:32)", "lr-cca", 32, 0, "Instantaneous (1 TR)"),
        ("DeepMReye 2.0 (lr-cca:32 + lags+-1)", "lr-cca", 32, 1, "Multi-lag (+-1 TR)"),
        ("DeepMReye 2.0 (lr-cca:32 + lags+-2)", "lr-cca", 32, 2, "Multi-lag (+-2 TRs)"),
        ("DeepMReye 2.0 (fold-pca:64)", "fold-pca", 64, 0, "Instantaneous (1 TR)"),
        ("DeepMReye 2.0 (fold-pca:64 + lags+-2)", "fold-pca", 64, 2, "Multi-lag (+-2 TRs)"),
        ("DeepMReye 2.0 (gev-fast:32)", "gev-fast", 32, 0, "Instantaneous (1 TR)"),
        ("DeepMReye 2.0 (corpus-pca:64)", "corpus-pca", 64, 0, "Instantaneous (1 TR)"),
    ]
    
    for model_name, basis_name, k, lags, ctx_str in v2_configs:
        print(f"[*] Running {model_name}...", flush=True)
        res_by_res = {"Sub-TR (10 pts/TR)": [], "1-TR mean": [], "5-TR bin mean": []}
        
        for d in data:
            block = d["block"]
            labels = d["labels"]
            T = block.shape[-1]
            vox = block[mask].T
            
            if basis_name == "fold-pca":
                pca = PCA(n_components=k)
                z = pca.fit_transform(vox)
            else:
                z = project_basis(basis_name, bases[basis_name], vox, k)
                
            X = make_lag_features(z, lags=lags)
            
            # Predict Sub-TR: [T, 20] from X
            y_sub = labels.reshape(T, 20)
            valid_sub = ~np.isnan(y_sub).any(axis=1)
            X_sv, y_sv = X[valid_sub], y_sub[valid_sub]
            
            if len(X_sv) >= 20:
                kf = KFold(n_splits=5, shuffle=False)
                preds_sub = np.zeros_like(y_sv)
                for tr_idx, te_idx in kf.split(X_sv):
                    reg = Ridge(alpha=10.0)
                    reg.fit(X_sv[tr_idx], y_sv[tr_idx])
                    preds_sub[te_idx] = reg.predict(X_sv[te_idx])
                
                preds_full_subtr = np.full((T, 10, 2), np.nan)
                preds_full_subtr[valid_sub] = preds_sub.reshape(-1, 10, 2)
                
                scores = evaluate_predictions(preds_full_subtr, labels, model_name, "", "Within (5-CV)", ctx_str)
                res_by_res["Sub-TR (10 pts/TR)"].append(scores["subtr"])
                res_by_res["1-TR mean"].append(scores["1tr"])
                res_by_res["5-TR bin mean"].append(scores["5tr"])

        for res_label, scores_list in res_by_res.items():
            rx = np.array([s[0] for s in scores_list])
            ry = np.array([s[1] for s in scores_list])
            err = np.array([s[2] for s in scores_list])
            r_mean = 0.5 * (rx + ry)
            
            valid = np.isfinite(r_mean)
            r_val = r_mean[valid]
            rx_val = rx[valid]
            ry_val = ry[valid]
            err_val = err[valid]
            top80_idx = np.argsort(r_val)[int(len(r_val) * 0.2):]
            
            benchmark_results.append({
                "model": model_name,
                "resolution": res_label,
                "protocol": "Within (5-CV)",
                "temporal_context": ctx_str,
                "all_rx": float(np.nanmedian(rx)),
                "all_ry": float(np.nanmedian(ry)),
                "all_r": float(np.nanmedian(r_mean)),
                "all_err": float(np.nanmedian(err)),
                "top80_rx": float(np.median(rx_val[top80_idx])),
                "top80_ry": float(np.median(ry_val[top80_idx])),
                "top80_r": float(np.median(r_val[top80_idx])),
                "top80_err": float(np.median(err_val[top80_idx])),
            })

    # Save JSON results
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\n[+] Saved full results to {OUT_JSON}", flush=True)
    
    # Print formatted markdown table
    print("\n" + "="*115, flush=True)
    print(f"{'Model & Configuration':<36} {'Resolution':<18} {'Protocol':<14} {'100% r_x':<9} {'100% r_y':<9} {'100% r':<9} {'Top-80% r':<10} {'Error (deg)':<10}", flush=True)
    print("="*115, flush=True)
    for r in benchmark_results:
        print(f"{r['model']:<36} {r['resolution']:<18} {r['protocol']:<14} {r['all_rx']:>+8.3f} {r['all_ry']:>+8.3f} {r['all_r']:>+8.3f} {r['top80_r']:>+9.3f} {r['all_err']:>9.2f}", flush=True)
    print("="*115, flush=True)

if __name__ == "__main__":
    run_benchmark()
