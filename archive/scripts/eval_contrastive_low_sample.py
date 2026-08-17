#!/usr/bin/env python3
"""Evaluates PyTorch ContrastiveNet:64 vs Corpus-PCA:64 vs Fold-PCA:64 across low-sample regimes."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SYS_PYTHON = sys.executable


def run_probe_eval(fold_dataset: str, feature_spec: str, max_train_windows: str = None, contrastive_ckpt: str = "models/contrastive_net_64.pt") -> dict:
    """Invokes eval_probe.py for a single dataset fold and feature source."""
    temp_json = f"results/temp_{fold_dataset}_{feature_spec.replace(':', '_').replace('/', '_')}_{max_train_windows}.json"
    cmd = [
        SYS_PYTHON,
        "scripts/eval_probe.py",
        "--protocol",
        "dataset",
        "--features",
        feature_spec,
        "--readouts",
        "ridge-cv",
        "--fold-name",
        fold_dataset,
        "--out",
        temp_json,
    ]

    if feature_spec == "contrastive-net":
        cmd.extend(["--contrastive-checkpoint", contrastive_ckpt])
    if max_train_windows and max_train_windows != "All":
        cmd.extend(["--max-train-windows", str(max_train_windows)])

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        with open(temp_json, "r") as f:
            data = json.load(f)
        if os.path.exists(temp_json):
            os.remove(temp_json)

        fold_key = list(data.keys())[0]
        arm_key = list(data[fold_key].keys())[0]
        sub_metrics = data[fold_key][arm_key]["by_subject"]
        rx = sub_metrics["pearson_r_x"]
        ry = sub_metrics["pearson_r_y"]
        r_mean = 0.5 * (rx + ry)
        return {"r_mean": float(r_mean), "rx": float(rx), "ry": float(ry)}
    except Exception as e:
        print(f"Error evaluating {fold_dataset} with {feature_spec} (N={max_train_windows}): {e}")
        if os.path.exists(temp_json):
            os.remove(temp_json)
        return {"r_mean": float("nan"), "rx": float("nan"), "ry": float("nan")}





def main():
    parser = argparse.ArgumentParser(description="Evaluate ContrastiveNet vs Corpus-PCA vs Fold-PCA.")
    parser.add_argument("--contrastive-ckpt", type=str, default="models/contrastive_net_64.pt")
    parser.add_argument("--output-json", type=str, default="results/contrastive_low_sample_summary.json")
    args = parser.parse_args()

    folds = [
        "dsL01_guided_fixations",
        "dsL02_pursuit",
        "dsL03_pursuit",
        "dsL04_pursuit",
        "dsL05_free_viewing",
        "dsL06_sequences",
        "dsL07_deepmreye_calib",
        "dsL11_backtothefuture",
    ]
    budgets = ["100", "250", "500", "1000", "2500", "5000", "All"]

    results = {}

    print("=" * 80)
    print(f"[*] BENCHMARKING PYTORCH CONTRASTIVENET VS CORPUS-PCA:64 VS FOLD-PCA:64")
    print("=" * 80)

    for i, fold in enumerate(folds, 1):
        print(f"\n[{i}/{len(folds)}] --- Evaluating Fold: {fold} ---")
        results[fold] = {}

        for b in budgets:
            cnet_res = run_probe_eval(fold, "contrastive-net", b, args.contrastive_ckpt)

            cpca_res = run_probe_eval(fold, "corpus-pca:64", b)
            fpca_res = run_probe_eval(fold, "fold-pca:64", b)

            cnet_r = cnet_res["r_mean"]
            cpca_r = cpca_res["r_mean"]
            fpca_r = fpca_res["r_mean"]

            winner = "CONTRASTIVE 🏆" if cnet_r > max(cpca_r, fpca_r) else ("CORPUS-PCA" if cpca_r >= fpca_r else "FOLD-PCA")

            print(f"  [+] N={b:<6}: Contrastive={cnet_r:.3f} | Corpus-PCA={cpca_r:.3f} | Fold-PCA={fpca_r:.3f} | Winner={winner}", flush=True)


            results[fold][b] = {
                "contrastive": cnet_res,
                "corpus_pca": cpca_res,
                "fold_pca": fpca_res,
                "winner": winner,
            }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[*] BENCHMARK COMPLETE! Results saved to {args.output_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
