#!/usr/bin/env python3
"""
Low-sample data efficiency benchmark: Corpus-PCA:64 vs Fold-PCA:64 across ALL 8 datasets.
Sweeps training budgets N in {100, 250, 500, 1000, 2500, 5000, All}.
"""
import os
import sys
import json
import time
import subprocess
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

SAMPLE_BUDGETS = [100, 250, 500, 1000, 2500, 5000, None]  # None = Full training set

def run_cmd(cmd):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[!] Command failed: {res.stderr[:500]}", flush=True)
    return res.stdout

def main():
    print("=" * 80)
    print("[*] LOW-SAMPLE DATA EFFICIENCY BENCHMARK: CORPUS-PCA:64 VS FOLD-PCA:64")
    print("=" * 80)
    
    out_dir = Path("results/low_sample_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_summary = {}

    for i, fold_ds in enumerate(LABELED_DATASETS, 1):
        print(f"\n[{i}/8] --- Sweeping Low-Sample Budgets on Fold: {fold_ds} ---", flush=True)
        fold_summary = {}
        
        for n_win in SAMPLE_BUDGETS:
            budget_key = "All" if n_win is None else str(n_win)
            eval_path = out_dir / f"eval_{fold_ds}_N{budget_key}.json"
            
            max_win_arg = f"--max-train-windows {n_win}" if n_win is not None else ""
            
            cmd = (
                f"uv run python scripts/eval_probe.py --protocol dataset "
                f"--features corpus-pca:64 fold-pca:64 "
                f"--fold-name {fold_ds} --readouts ridge-cv "
                f"--standardize-targets dataset {max_win_arg} --out {eval_path}"
            )
            
            run_cmd(cmd)
            
            if eval_path.exists():
                data = json.loads(eval_path.read_text())
                res = data.get(fold_ds, {})
                
                c_key = "corpus-pca:64/ridge-cv"
                f_key = "fold-pca:64/ridge-cv"
                
                c_sub = res.get(c_key, {}).get("by_subject", {})
                f_sub = res.get(f_key, {}).get("by_subject", {})
                
                c_rx = c_sub.get("pearson_r_x", np.nan)
                c_ry = c_sub.get("pearson_r_y", np.nan)
                c_r = 0.5 * (c_rx + c_ry)
                
                f_rx = f_sub.get("pearson_r_x", np.nan)
                f_ry = f_sub.get("pearson_r_y", np.nan)
                f_r = 0.5 * (f_rx + f_ry)
                
                diff = c_r - f_r
                
                winner = "CORPUS-PCA 🏆" if diff > 0.001 else ("FOLD-PCA" if diff < -0.001 else "TIE")
                
                print(f"  [+] N={budget_key:<6}: Corpus-PCA={c_r:.3f} (rx={c_rx:.3f}, ry={c_ry:.3f}) | Fold-PCA={f_r:.3f} (rx={f_rx:.3f}, ry={f_ry:.3f}) | Diff={diff:+.3f} | {winner}", flush=True)
                
                fold_summary[budget_key] = {
                    "budget": budget_key,
                    "corpus_pca_r_mean": float(c_r),
                    "corpus_pca_r_x": float(c_rx),
                    "corpus_pca_r_y": float(c_ry),
                    "fold_pca_r_mean": float(f_r),
                    "fold_pca_r_x": float(f_rx),
                    "fold_pca_r_y": float(f_ry),
                    "diff": float(diff),
                    "winner": winner
                }
        
        all_summary[fold_ds] = fold_summary

    summary_file = out_dir / "corpus_vs_fold_pca_low_sample_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_summary, f, indent=2)
        
    print("\n" + "=" * 80)
    print(f"[*] BENCHMARK COMPLETE! Summary saved to {summary_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
