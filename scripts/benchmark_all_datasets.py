#!/usr/bin/env python3
"""Cross-Dataset Benchmark: DeepMReye 1.0 vs DeepMReye 2.0.

Evaluates all available labeled datasets (dsL01 to dsL07, dsL11) at:
- 1-TR Mean Resolution
- Sub-TR Continuous Trajectory (10 samples/TR)

Comparing:
- DeepMReye 1.0 (Published Supervised 3D-CNN checkpoints)
- DeepMReye 2.0 (Linear Basis lr-cca:32 instantaneous)
- DeepMReye 2.0 (Linear Basis lr-cca:32 + +-2 temporal lags)
- DeepMReye 2.0 (Fold-PCA:64)
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
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = Path.home() / ".cache/deepmreye"
BASIS_PATH = REPO / "results/scaling/basis_n1039.npz"
DME1_DIR = REPO / "results/dme1"
CACHE_DIR = REPO / "results/cache_benchmark"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = REPO / "results/all_datasets_benchmark.json"

V1_FILES = ["deepmreye/architecture.py", "deepmreye/util/util.py", "deepmreye/util/model_opts.py"]

DATASET_CONFIGS = [
    {
        "id": "dsL01_guided_fixations",
        "name": "dsL01 (Fixations)",
        "dme1_within": DME1_DIR / "dataset1_guided_fixations.h5",
        "max_subs": 20,
    },
    {
        "id": "dsL02_pursuit",
        "name": "dsL02 (Pursuit)",
        "dme1_within": DME1_DIR / "dataset2_pursuit.h5",
        "max_subs": 9,
    },
    {
        "id": "dsL03_pursuit",
        "name": "dsL03 (Pursuit)",
        "dme1_within": DME1_DIR / "dataset3_pursuit.h5",
        "max_subs": 24,
    },
    {
        "id": "dsL04_pursuit",
        "name": "dsL04 (Pursuit)",
        "dme1_within": DME1_DIR / "dataset4_pursuit.h5",
        "max_subs": 15,
    },
    {
        "id": "dsL05_free_viewing",
        "name": "dsL05 (Free Viewing)",
        "dme1_within": DME1_DIR / "dataset5_free_viewing.h5",
        "max_subs": 15,
    },
    {
        "id": "dsL06_sequences",
        "name": "dsL06 (Sequences)",
        "dme1_within": DME1_DIR / "dataset6_openclosed.h5",
        "max_subs": 6,
    },
    {
        "id": "dsL07_deepmreye_calib",
        "name": "dsL07 (Calibration)",
        "dme1_within": DME1_DIR / "datasets_1to5.h5",
        "max_subs": 15,
    },
    {
        "id": "dsL11_backtothefuture",
        "name": "dsL11 (Movie)",
        "dme1_within": DME1_DIR / "datasets_1to5.h5",
        "max_subs": 4,
    },
]

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
        comp = basis_dict["components"][:, :k]
        return (x - mu) @ comp

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

def evaluate_subtr_and_1tr(preds_subtr, labels):
    n = min(len(preds_subtr), len(labels))
    p_subtr = preds_subtr[:n]
    t_subtr = labels[:n]
    
    # Sub-TR
    p_flat = p_subtr.reshape(-1, 2)
    t_flat = t_subtr.reshape(-1, 2)
    rx_sub = calc_r(p_flat[:, 0], t_flat[:, 0])
    ry_sub = calc_r(p_flat[:, 1], t_flat[:, 1])
    err_sub = calc_error(p_flat, t_flat)
    r_sub = 0.5 * (rx_sub + ry_sub) if np.isfinite(rx_sub) and np.isfinite(ry_sub) else np.nan
    
    # 1-TR mean
    p_tr = np.nanmean(p_subtr, axis=1)
    t_tr = np.nanmean(t_subtr, axis=1)
    rx_tr = calc_r(p_tr[:, 0], t_tr[:, 0])
    ry_tr = calc_r(p_tr[:, 1], t_tr[:, 1])
    err_tr = calc_error(p_tr, t_tr)
    r_tr = 0.5 * (rx_tr + ry_tr) if np.isfinite(rx_tr) and np.isfinite(ry_tr) else np.nan
    
    return {
        "subtr": (rx_sub, ry_sub, r_sub, err_sub),
        "1tr": (rx_tr, ry_tr, r_tr, err_tr),
    }

def run_all_benchmarks():
    mask, bases = load_basis(BASIS_PATH)
    vendor_v1()
    
    all_dataset_results = []
    
    for cfg in DATASET_CONFIGS:
        ds_id = cfg["id"]
        ds_name = cfg["name"]
        ds_dir = DATA_ROOT / ds_id
        if not ds_dir.exists():
            print(f"[-] Skipping {ds_id}, not found in cache", flush=True)
            continue
            
        files = sorted(ds_dir.glob("*.h5"))[:cfg["max_subs"]]
        print(f"\n==================================================", flush=True)
        print(f"[*] Processing {ds_name} ({len(files)} subjects)...", flush=True)
        print(f"==================================================", flush=True)
        
        # Load dataset files
        data = []
        for p in files:
            with h5py.File(p, "r") as f:
                block = f["eye_block"][:] # [47, 29, 18, T]
                labels = f["labels"][:]   # [T, 10, 2]
                data.append({"sub": p.stem, "block": block, "labels": labels})
                
        # 1. DeepMReye 1.0
        cache_file = CACHE_DIR / f"dme1_{ds_id}.npz"
        if cache_file.exists():
            print(f"[*] Loading cached DME1 predictions for {ds_id}...", flush=True)
            c = np.load(cache_file, allow_pickle=True)
            dme1_preds = [c[f"pred_{i}"] for i in range(len(data))]
        else:
            print(f"[*] Computing DME1 predictions with {cfg['dme1_within'].name}...", flush=True)
            dme1_model = build_dme1_model()
            dme1_model.load_weights(str(cfg["dme1_within"]))
            dme1_preds = []
            for i, d in enumerate(data):
                x_in = np.moveaxis(d["block"], -1, 0)[..., None].astype(np.float32)
                n = min(len(x_in), len(d["labels"]))
                p_w = dme1_model.predict(x_in[:n], batch_size=64, verbose=0)[0]
                dme1_preds.append(p_w)
            save_dict = {f"pred_{i}": dme1_preds[i] for i in range(len(data))}
            np.savez_compressed(cache_file, **save_dict)
            
        # Score DME 1.0
        dme1_scores_1tr = []
        dme1_scores_subtr = []
        for i, d in enumerate(data):
            res = evaluate_subtr_and_1tr(dme1_preds[i], d["labels"])
            dme1_scores_subtr.append(res["subtr"])
            dme1_scores_1tr.append(res["1tr"])
            
        dme1_sub_r = np.nanmedian([s[2] for s in dme1_scores_subtr])
        dme1_sub_err = np.nanmedian([s[3] for s in dme1_scores_subtr])
        dme1_1tr_rx = np.nanmedian([s[0] for s in dme1_scores_1tr])
        dme1_1tr_ry = np.nanmedian([s[1] for s in dme1_scores_1tr])
        dme1_1tr_r = np.nanmedian([s[2] for s in dme1_scores_1tr])
        dme1_1tr_err = np.nanmedian([s[3] for s in dme1_scores_1tr])
        
        print(f"  -> DME 1.0: 1-TR r = {dme1_1tr_r:+.3f} (err: {dme1_1tr_err:.2f} deg) | Sub-TR r = {dme1_sub_r:+.3f} (err: {dme1_sub_err:.2f} deg)", flush=True)

        # 2. DeepMReye 2.0 Models
        v2_variants = [
            ("lr-cca:32", "lr-cca", 32, 0),
            ("lr-cca:32 + lags+-2", "lr-cca", 32, 2),
            ("fold-pca:64", "fold-pca", 64, 0),
            ("corpus-pca:64", "corpus-pca", 64, 0),
        ]
        
        variant_results = {}
        for var_name, b_name, k, lags in v2_variants:
            v2_scores_1tr = []
            v2_scores_subtr = []
            
            for d in data:
                block = d["block"]
                labels = d["labels"]
                T = block.shape[-1]
                vox = block[mask].T
                
                if b_name == "fold-pca":
                    pca = PCA(n_components=k)
                    z = pca.fit_transform(vox)
                else:
                    z = project_basis(b_name, bases[b_name], vox, k)
                    
                X = make_lag_features(z, lags=lags)
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
                    
                    preds_full = np.full((T, 10, 2), np.nan)
                    preds_full[valid_sub] = preds_sub.reshape(-1, 10, 2)
                    res = evaluate_subtr_and_1tr(preds_full, labels)
                    v2_scores_subtr.append(res["subtr"])
                    v2_scores_1tr.append(res["1tr"])
                    
            v2_sub_r = np.nanmedian([s[2] for s in v2_scores_subtr])
            v2_sub_err = np.nanmedian([s[3] for s in v2_scores_subtr])
            v2_1tr_rx = np.nanmedian([s[0] for s in v2_scores_1tr])
            v2_1tr_ry = np.nanmedian([s[1] for s in v2_scores_1tr])
            v2_1tr_r = np.nanmedian([s[2] for s in v2_scores_1tr])
            v2_1tr_err = np.nanmedian([s[3] for s in v2_scores_1tr])
            
            variant_results[var_name] = {
                "1tr_rx": float(v2_1tr_rx),
                "1tr_ry": float(v2_1tr_ry),
                "1tr_r": float(v2_1tr_r),
                "1tr_err": float(v2_1tr_err),
                "subtr_r": float(v2_sub_r),
                "subtr_err": float(v2_sub_err),
            }
            print(f"  -> DME 2.0 ({var_name}): 1-TR r = {v2_1tr_r:+.3f} (err: {v2_1tr_err:.2f} deg) | Sub-TR r = {v2_sub_r:+.3f} (err: {v2_sub_err:.2f} deg)", flush=True)

        all_dataset_results.append({
            "dataset_id": ds_id,
            "dataset_name": ds_name,
            "n_subjects": len(data),
            "dme1": {
                "1tr_rx": float(dme1_1tr_rx),
                "1tr_ry": float(dme1_1tr_ry),
                "1tr_r": float(dme1_1tr_r),
                "1tr_err": float(dme1_1tr_err),
                "subtr_r": float(dme1_sub_r),
                "subtr_err": float(dme1_sub_err),
            },
            "dme2": variant_results
        })

    with open(OUT_JSON, "w") as f:
        json.dump(all_dataset_results, f, indent=2)
    print(f"\n[+] Saved cross-dataset benchmark results to {OUT_JSON}", flush=True)
    
    # Print formatted comparison table
    print("\n" + "="*120, flush=True)
    print(f"{'Dataset':<22} | {'DME 1.0 (1-TR)':<15} {'DME 1.0 (Sub)':<15} | {'DME 2.0 lr-cca (1-TR)':<20} {'DME 2.0 +lags (1-TR)':<20} {'DME 2.0 +lags (Sub)':<20}", flush=True)
    print("="*120, flush=True)
    for r in all_dataset_results:
        d1 = r["dme1"]
        d2_cca = r["dme2"]["lr-cca:32"]
        d2_lags = r["dme2"]["lr-cca:32 + lags+-2"]
        print(f"{r['dataset_name']:<22} | r={d1['1tr_r']:+.3f} ({d1['1tr_err']:.1f}°)  r={d1['subtr_r']:+.3f} ({d1['subtr_err']:.1f}°) | r={d2_cca['1tr_r']:+.3f} ({d2_cca['1tr_err']:.1f}°)    r={d2_lags['1tr_r']:+.3f} ({d2_lags['1tr_err']:.1f}°)    r={d2_lags['subtr_r']:+.3f} ({d2_lags['subtr_err']:.1f}°)", flush=True)
    print("="*120, flush=True)

if __name__ == "__main__":
    run_all_benchmarks()
