#!/usr/bin/env python3
"""Sweep Corpus-PCA bottleneck dimensionality K in {16, 32, 64, 96, 128, 192, 256} across ALL 8 labeled datasets to find the optimal component budget."""
import json
import subprocess
import time
from pathlib import Path

import numpy as np

LABELED_DATASETS = [
    "dsL01_guided_fixations",
    "dsL02_pursuit",
    "dsL03_pursuit",
    "dsL04_pursuit",
    "dsL05_free_viewing",
    "dsL06_sequences",
    "dsL07_deepmreye_calib",
    "dsL11_backtothefuture",
]

KS = [16, 32, 64, 96, 128, 192, 256]


def run_cmd(cmd):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[!] Error: {res.stderr}")
    return res.stdout


def main():
    out_dir = Path("results/corpus_pca_k_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Sweeping Corpus-PCA component budgets K={KS} across ALL 8 datasets...", flush=True)

    all_results = {}

    for i, fold_ds in enumerate(LABELED_DATASETS, 1):
        t0 = time.time()
        print(f"\n[{i}/8] --- Evaluating K Sweep on Fold: {fold_ds} ---", flush=True)

        eval_path = out_dir / f"eval_{fold_ds}.json"
        features_arg = " ".join([f"corpus-pca:{k}" for k in KS] + ["fold-pca:64"])
        cmd = (
            f"uv run python scripts/eval_probe.py --protocol dataset "
            f"--features {features_arg} "
            f"--fold-name {fold_ds} --readouts ridge-cv "
            f"--standardize-targets dataset --out {eval_path}"
        )
        run_cmd(cmd)

        if eval_path.exists():
            data = json.loads(eval_path.read_text())
            fold_data = data.get(fold_ds, {})

            fold_k_res = {}
            for k in KS:
                feat_key = f"corpus-pca:{k}/ridge-cv"
                by_sub = fold_data.get(feat_key, {}).get("by_subject", {})
                rx = by_sub.get("pearson_r_x", 0.0)
                ry = by_sub.get("pearson_r_y", 0.0)
                rm = 0.5 * (rx + ry)
                fold_k_res[k] = {"r_x": rx, "r_y": ry, "r_mean": rm}

            # Also get fold-pca:64 reference
            fp_sub = fold_data.get("fold-pca:64/ridge-cv", {}).get("by_subject", {})
            fp_rx = fp_sub.get("pearson_r_x", 0.0)
            fp_ry = fp_sub.get("pearson_r_y", 0.0)
            fp_rm = 0.5 * (fp_rx + fp_ry)
            fold_k_res["fold_pca_64"] = {"r_x": fp_rx, "r_y": fp_ry, "r_mean": fp_rm}

            all_results[fold_ds] = fold_k_res

            best_k = max(KS, key=lambda k: fold_k_res[k]["r_mean"])
            best_r = fold_k_res[best_k]["r_mean"]

            k_strs = [f"K={k}:{fold_k_res[k]['r_mean']:.3f}" for k in KS]
            print(f"  [+] {fold_ds} [{time.time() - t0:.1f}s]: Best K={best_k} (r={best_r:.3f}) | Fold-PCA:64={fp_rm:.3f} | { ' '.join(k_strs) }", flush=True)

    # Summary Table across all K
    print("\n" + "=" * 125, flush=True)
    header = f"{'Held-Out Dataset':<24} " + " ".join([f"K={k:<6}" for k in KS]) + f" {'Fold-PCA:64':<12} {'Optimal K':<10}"
    print(header, flush=True)
    print("-" * 125, flush=True)

    k_means = {k: [] for k in KS}
    fp_means = []

    for ds in LABELED_DATASETS:
        r = all_results.get(ds, {})
        row_str = f"{ds:<24} "
        for k in KS:
            rm = r.get(k, {}).get("r_mean", 0.0)
            k_means[k].append(rm)
            row_str += f"{rm:<8.3f} "
        fp_rm = r.get("fold_pca_64", {}).get("r_mean", 0.0)
        fp_means.append(fp_rm)

        best_k = max(KS, key=lambda k: r.get(k, {}).get("r_mean", 0.0)) if r else 64
        row_str += f"{fp_rm:<12.3f} K={best_k}"
        print(row_str, flush=True)

    print("-" * 125, flush=True)
    med_row = f"{'MEDIAN ACROSS 8 FOLDOUTS':<24} " + " ".join([f"{np.median(k_means[k]):<8.3f}" for k in KS]) + f" {np.median(fp_means):<12.3f} K={max(KS, key=lambda k: np.median(k_means[k]))}"
    mean_row = f"{'MEAN ACROSS 8 FOLDOUTS':<24} " + " ".join([f"{np.mean(k_means[k]):<8.3f}" for k in KS]) + f" {np.mean(fp_means):<12.3f} K={max(KS, key=lambda k: np.mean(k_means[k]))}"
    print(med_row, flush=True)
    print(mean_row, flush=True)
    print("=" * 125, flush=True)

    (out_dir / "corpus_pca_k_sweep_summary.json").write_text(json.dumps(all_results, indent=2))
    print(f"\n[*] Summary saved to {out_dir / 'corpus_pca_k_sweep_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
