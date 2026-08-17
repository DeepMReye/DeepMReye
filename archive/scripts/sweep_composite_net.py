#!/usr/bin/env python3
"""Systematic Sweep of PyTorch CompositeNet Configurations.

Trains CompositeNet across:
- Short (10 epochs), Medium (30 epochs), Long (60 epochs) training durations
- Various Bottleneck dimensions K (64, 96, 128)
- Various Reconstruction weights alpha (0.01, 0.05, 0.1, 0.5)

Evaluates each model on zero-shot holdout dsL11 and logs results.
"""
import subprocess
import json
from pathlib import Path
import numpy as np


def run_cmd(cmd):
    print(f"[*] Running: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[!] Error: {res.stderr}")
    return res.stdout


def main():
    out_dir = Path("results/composite_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        # Short Schedule (10 Epochs)
        {"epochs": 10, "bottleneck": 64,  "alpha": 0.1,  "lr": 1e-3, "label": "Short-K64-a0.1"},
        {"epochs": 10, "bottleneck": 96,  "alpha": 0.05, "lr": 1e-3, "label": "Short-K96-a0.05"},
        {"epochs": 10, "bottleneck": 96,  "alpha": 0.1,  "lr": 1e-3, "label": "Short-K96-a0.1"},
        {"epochs": 10, "bottleneck": 128, "alpha": 0.1,  "lr": 1e-3, "label": "Short-K128-a0.1"},

        # Medium Schedule (30 Epochs)
        {"epochs": 30, "bottleneck": 64,  "alpha": 0.05, "lr": 5e-4, "label": "Med-K64-a0.05"},
        {"epochs": 30, "bottleneck": 96,  "alpha": 0.01, "lr": 5e-4, "label": "Med-K96-a0.01"},
        {"epochs": 30, "bottleneck": 96,  "alpha": 0.05, "lr": 5e-4, "label": "Med-K96-a0.05"},
        {"epochs": 30, "bottleneck": 96,  "alpha": 0.1,  "lr": 5e-4, "label": "Med-K96-a0.1"},
        {"epochs": 30, "bottleneck": 128, "alpha": 0.05, "lr": 5e-4, "label": "Med-K128-a0.05"},

        # Long Schedule (60 Epochs)
        {"epochs": 60, "bottleneck": 96,  "alpha": 0.01, "lr": 3e-4, "label": "Long-K96-a0.01"},
        {"epochs": 60, "bottleneck": 96,  "alpha": 0.05, "lr": 3e-4, "label": "Long-K96-a0.05"},
        {"epochs": 60, "bottleneck": 96,  "alpha": 0.1,  "lr": 3e-4, "label": "Long-K96-a0.1"},
        {"epochs": 60, "bottleneck": 128, "alpha": 0.05, "lr": 3e-4, "label": "Long-K128-a0.05"},
    ]

    results_table = []

    for cfg in configs:
        ckpt_path = out_dir / f"model_{cfg['label']}.pt"
        eval_out = out_dir / f"eval_{cfg['label']}.json"

        # 1. Train model
        train_cmd = (
            f"uv run python scripts/train_composite_net.py "
            f"--epochs {cfg['epochs']} --bottleneck {cfg['bottleneck']} "
            f"--alpha {cfg['alpha']} --lr {cfg['lr']} --out {ckpt_path}"
        )
        run_cmd(train_cmd)

        # 2. Evaluate model on dsL11 holdout
        eval_cmd = (
            f"uv run python scripts/eval_probe.py --protocol dataset "
            f"--features composite-net fold-pca:64 fold-pca:64+fold-pls:32 "
            f"--composite-checkpoint {ckpt_path} --fold-name dsL11 "
            f"--standardize-targets dataset --out {eval_out}"
        )
        run_cmd(eval_cmd)

        if eval_out.exists():
            data = json.loads(eval_out.read_text())
            dsl11 = data.get("dsL11_backtothefuture", {})

            comp_res = dsl11.get("composite-net", {}).get("by_subject", {})
            pca_res = dsl11.get("fold-pca:64", {}).get("by_subject", {})

            rx = comp_res.get("pearson_r_x", 0.0)
            ry = comp_res.get("pearson_r_y", 0.0)
            r_mean = 0.5 * (rx + ry)

            pca_rx = pca_res.get("pearson_r_x", 0.0)
            pca_ry = pca_res.get("pearson_r_y", 0.0)
            pca_mean = 0.5 * (pca_rx + pca_ry)

            results_table.append({
                "label": cfg["label"],
                "epochs": cfg["epochs"],
                "bottleneck": cfg["bottleneck"],
                "alpha": cfg["alpha"],
                "r_x": rx,
                "r_y": ry,
                "r_mean": r_mean,
                "pca_r_mean": pca_mean,
                "diff_vs_pca": r_mean - pca_mean
            })

    # Print summary leaderboard
    print("\n" + "=" * 90)
    print(f"{'Config Label':<20} {'Epochs':<8} {'K':<6} {'Alpha':<8} {'r_x':<8} {'r_y':<8} {'Mean r':<8} {'vs PCA':<8}")
    print("-" * 90)
    for r in sorted(results_table, key=lambda x: -x["r_mean"]):
        print(f"{r['label']:<20} {r['epochs']:<8} {r['bottleneck']:<6} {r['alpha']:<8.2f} "
              f"{r['r_x']:<8.3f} {r['r_y']:<8.3f} {r['r_mean']:<8.3f} {r['diff_vs_pca']:+8.3f}")
    print("=" * 90)

    (out_dir / "sweep_results.json").write_text(json.dumps(results_table, indent=2))
    print(f"[*] Saved sweep results to {out_dir / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
