#!/usr/bin/env python3
"""Run corpus-pca:64 vs fold-pca:64 side-by-side across ALL 8 labeled datasets under strict zero-shot out-of-distribution cross-validation."""
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


def run_cmd(cmd):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[!] Error: {res.stderr}")
    return res.stdout


def main():
    out_dir = Path("results/corpus_vs_fold_pca")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Evaluating corpus-pca:64 vs fold-pca:64 across ALL 8 datasets...", flush=True)

    all_results = {}

    for i, fold_ds in enumerate(LABELED_DATASETS, 1):
        t0 = time.time()
        print(f"\n[{i}/8] --- Evaluating Fold: {fold_ds} ---", flush=True)

        eval_path = out_dir / f"eval_{fold_ds}.json"
        cmd = (
            f"uv run python scripts/eval_probe.py --protocol dataset "
            f"--features fold-pca:64 corpus-pca:64 "
            f"--fold-name {fold_ds} --readouts ridge-cv "
            f"--standardize-targets dataset --out {eval_path}"
        )
        run_cmd(cmd)

        if eval_path.exists():
            data = json.loads(eval_path.read_text())
            fold_data = data.get(fold_ds, {})
            fold_pca = fold_data.get("fold-pca:64/ridge-cv", {}).get("by_subject", {})
            corpus_pca = fold_data.get("corpus-pca:64/ridge-cv", {}).get("by_subject", {})

            res = {
                "fold_pca": {
                    "r_x": fold_pca.get("pearson_r_x", 0.0),
                    "r_y": fold_pca.get("pearson_r_y", 0.0),
                    "r_mean": 0.5 * (fold_pca.get("pearson_r_x", 0.0) + fold_pca.get("pearson_r_y", 0.0))
                },
                "corpus_pca": {
                    "r_x": corpus_pca.get("pearson_r_x", 0.0),
                    "r_y": corpus_pca.get("pearson_r_y", 0.0),
                    "r_mean": 0.5 * (corpus_pca.get("pearson_r_x", 0.0) + corpus_pca.get("pearson_r_y", 0.0))
                }
            }
            all_results[fold_ds] = res

            diff = res["corpus_pca"]["r_mean"] - res["fold_pca"]["r_mean"]
            diff_x = res["corpus_pca"]["r_x"] - res["fold_pca"]["r_x"]
            diff_y = res["corpus_pca"]["r_y"] - res["fold_pca"]["r_y"]

            print(f"  [+] {fold_ds} [{time.time() - t0:.1f}s]: "
                  f"Corpus-PCA={res['corpus_pca']['r_mean']:.3f} (rx={res['corpus_pca']['r_x']:.3f}, ry={res['corpus_pca']['r_y']:.3f}) | "
                  f"Fold-PCA={res['fold_pca']['r_mean']:.3f} (rx={res['fold_pca']['r_x']:.3f}, ry={res['fold_pca']['r_y']:.3f}) | "
                  f"Diff={diff:+.3f} (rx={diff_x:+.3f}, ry={diff_y:+.3f})", flush=True)

    # Summary Table
    print("\n" + "=" * 115, flush=True)
    print(f"{'Held-Out Dataset':<25} {'Fold-PCA:64 (r_x / r_y / r_mean)':<35} {'Corpus-PCA:64 (r_x / r_y / r_mean)':<35} {'Diff vs Fold':<15}", flush=True)
    print("-" * 115, flush=True)

    fold_means, corpus_means = [], []
    fold_rx_list, corpus_rx_list = [], []
    fold_ry_list, corpus_ry_list = [], []

    for ds in LABELED_DATASETS:
        r = all_results.get(ds, {})
        fp = r.get("fold_pca", {})
        cp = r.get("corpus_pca", {})

        fp_m, cp_m = fp.get("r_mean", 0.0), cp.get("r_mean", 0.0)
        fold_means.append(fp_m)
        corpus_means.append(cp_m)

        fold_rx_list.append(fp.get("r_x", 0.0))
        corpus_rx_list.append(cp.get("r_x", 0.0))
        fold_ry_list.append(fp.get("r_y", 0.0))
        corpus_ry_list.append(cp.get("r_y", 0.0))

        fp_str = f"{fp.get('r_x',0):.3f} / {fp.get('r_y',0):.3f} / {fp_m:.3f}"
        cp_str = f"{cp.get('r_x',0):.3f} / {cp.get('r_y',0):.3f} / {cp_m:.3f}"

        diff = cp_m - fp_m
        print(f"{ds:<25} {fp_str:<35} {cp_str:<35} {diff:+15.3f}", flush=True)

    print("-" * 115, flush=True)
    print(f"{'MEDIAN ACROSS 8 FOLDOUTS':<25} {np.median(fold_rx_list):.3f} / {np.median(fold_ry_list):.3f} / {np.median(fold_means):<15.3f} {np.median(corpus_rx_list):.3f} / {np.median(corpus_ry_list):.3f} / {np.median(corpus_means):<15.3f} {np.median(corpus_means) - np.median(fold_means):+15.3f}", flush=True)
    print(f"{'MEAN ACROSS 8 FOLDOUTS':<25} {np.mean(fold_rx_list):.3f} / {np.mean(fold_ry_list):.3f} / {np.mean(fold_means):<15.3f} {np.mean(corpus_rx_list):.3f} / {np.mean(corpus_ry_list):.3f} / {np.mean(corpus_means):<15.3f} {np.mean(corpus_means) - np.mean(fold_means):+15.3f}", flush=True)
    print("=" * 115, flush=True)

    (out_dir / "corpus_vs_fold_pca_summary.json").write_text(json.dumps(all_results, indent=2))
    print(f"\n[*] Summary saved to {out_dir / 'corpus_vs_fold_pca_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
