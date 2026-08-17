#!/usr/bin/env python3
"""Corpus Benefits Benchmark Suite: Evaluates 5 corpus-leveraged methods across all 8 labeled datasets under zero-shot cross-validation.

Methods Evaluated:
1. Low-Sample Data Efficiency (10%, 25%, 50%, 100% training window budgets)
2. Left-Right Binocular Conjugate CCA (`lr-cca:64`)
3. Temporal Motion Differencing (`diff-pca:64`)
4. Multi-View Basis Concatenation (`corpus-pca:32+diff-pca:32+lr-cca:32`)
5. Self-Supervised Neural Sequence Model (`ar-gru`)
"""
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

TRAIN_BUDGETS = [500, 1000, 2500, 5000, None]  # Low-sample regimes to 100%
MULTIVIEW_SPEC = "corpus-pca:32+diff-pca:32+lr-cca:32"
METHODS = [
    "fold-pca:64",
    "corpus-pca:64",
    "lr-cca:64",
    "diff-pca:64",
    MULTIVIEW_SPEC,
    "ar-gru",
]


def run_cmd(cmd):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[!] Error: {res.stderr}")
    return res.stdout


def main():
    out_dir = Path("results/corpus_benefits_suite")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================================", flush=True)
    print("[*] STARTING CORPUS BENEFITS BENCHMARK SUITE ACROSS ALL 8 DATASETS", flush=True)
    print("=================================================================================", flush=True)

    suite_results = {
        "methods_benchmark": {},
        "data_efficiency_benchmark": {}
    }

    # ---------------------------------------------------------------------------------
    # Part 1: Full Methods Benchmark (Methods 2, 3, 4, 5 vs Baselines)
    # ---------------------------------------------------------------------------------
    print("\n--- PART 1: 5-Method Full Benchmark across ALL 8 Datasets ---", flush=True)
    feat_args = " ".join([f"'{m}'" for m in METHODS])

    full_results = {}
    for i, fold_ds in enumerate(LABELED_DATASETS, 1):
        t0 = time.time()
        print(f"\n[{i}/8] --- Evaluating Fold: {fold_ds} ---", flush=True)

        eval_path = out_dir / f"eval_methods_{fold_ds}.json"
        cmd = (
            f"uv run python scripts/eval_probe.py --protocol dataset "
            f"--features {feat_args} "
            f"--fold-name {fold_ds} --readouts ridge-cv "
            f"--standardize-targets dataset --out {eval_path}"
        )
        run_cmd(cmd)

        if eval_path.exists():
            data = json.loads(eval_path.read_text())
            fold_data = data.get(fold_ds, {})

            method_res = {}
            for m in METHODS:
                feat_key = f"{m}/ridge-cv"
                by_sub = fold_data.get(feat_key, {}).get("by_subject", {})
                rx = by_sub.get("pearson_r_x", 0.0)
                ry = by_sub.get("pearson_r_y", 0.0)
                rm = 0.5 * (rx + ry)
                method_res[m] = {"r_x": rx, "r_y": ry, "r_mean": rm}

            full_results[fold_ds] = method_res

            best_m = max(METHODS, key=lambda m: method_res[m]["r_mean"])
            best_r = method_res[best_m]["r_mean"]
            m_strs = [f"{m.split(':')[0]}:{method_res[m]['r_mean']:.3f}" for m in METHODS]

            print(f"  [+] {fold_ds} [{time.time() - t0:.1f}s]: Winner={best_m} ({best_r:.3f}) | { ' | '.join(m_strs) }", flush=True)

    suite_results["methods_benchmark"] = full_results

    # ---------------------------------------------------------------------------------
    # Part 2: Data Efficiency Benchmark (Method 1: Corpus-PCA vs Fold-PCA in low-sample regimes)
    # ---------------------------------------------------------------------------------
    print("\n\n--- PART 2: Low-Sample Data Efficiency Benchmark (10% to 100% Training Budget) ---", flush=True)

    eff_results = {}
    for i, fold_ds in enumerate(LABELED_DATASETS, 1):
        print(f"\n[{i}/8] --- Sweeping Training Budgets on Fold: {fold_ds} ---", flush=True)
        fold_eff = {}

        for budget in TRAIN_BUDGETS:
            budget_str = f"N={budget}" if budget else "N=All"
            eval_path = out_dir / f"eval_eff_{fold_ds}_{budget_str}.json"

            cmd = (
                f"uv run python scripts/eval_probe.py --protocol dataset "
                f"--features fold-pca:64 corpus-pca:64 {MULTIVIEW_SPEC} "
                f"--fold-name {fold_ds} --readouts ridge-cv "
                f"--standardize-targets dataset "
                + (f"--max-train-windows {budget} " if budget else "")
                + f"--out {eval_path}"
            )
            run_cmd(cmd)

            if eval_path.exists():
                data = json.loads(eval_path.read_text())
                fold_data = data.get(fold_ds, {})

                fp_sub = fold_data.get("fold-pca:64/ridge-cv", {}).get("by_subject", {})
                cp_sub = fold_data.get("corpus-pca:64/ridge-cv", {}).get("by_subject", {})
                mv_sub = fold_data.get(f"{MULTIVIEW_SPEC}/ridge-cv", {}).get("by_subject", {})

                fp_rm = 0.5 * (fp_sub.get("pearson_r_x", 0.0) + fp_sub.get("pearson_r_y", 0.0))
                cp_rm = 0.5 * (cp_sub.get("pearson_r_x", 0.0) + cp_sub.get("pearson_r_y", 0.0))
                mv_rm = 0.5 * (mv_sub.get("pearson_r_x", 0.0) + mv_sub.get("pearson_r_y", 0.0))

                fold_eff[budget_str] = {
                    "fold_pca": fp_rm,
                    "corpus_pca": cp_rm,
                    "multiview": mv_rm,
                    "diff_corpus_vs_fold": cp_rm - fp_rm
                }

                print(f"  [+] {fold_ds} ({budget_str:<6}): Corpus-PCA={cp_rm:.3f} | MultiView={mv_rm:.3f} | Fold-PCA={fp_rm:.3f} | Diff={cp_rm - fp_rm:+.3f}", flush=True)

        eff_results[fold_ds] = fold_eff

    suite_results["data_efficiency_benchmark"] = eff_results

    # Save final suite results JSON
    (out_dir / "corpus_benefits_suite_summary.json").write_text(json.dumps(suite_results, indent=2))
    print(f"\n[*] Full Suite Summary saved to {out_dir / 'corpus_benefits_suite_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
