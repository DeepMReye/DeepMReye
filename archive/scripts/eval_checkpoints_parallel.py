#!/usr/bin/env python3
"""Parallel evaluation of all saved CompositeNet epoch checkpoints on dsL11."""
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def eval_one(epoch):
    ckpt_path = Path(f"results/epoch_saturation/checkpoint_epoch_{epoch:02d}.pt")
    eval_path = Path(f"results/epoch_saturation/eval_epoch_{epoch:02d}.json")
    if not ckpt_path.exists():
        return epoch, None

    cmd = (
        f"uv run python scripts/eval_probe.py --protocol dataset "
        f"--features composite-net fold-pca:64 "
        f"--composite-checkpoint {ckpt_path} --fold-name dsL11 "
        f"--readouts ridge-cv --standardize-targets dataset --out {eval_path}"
    )
    t0 = time.time()
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[!] Error evaluating epoch {epoch}: {res.stderr}")
        return epoch, None

    if eval_path.exists():
        data = json.loads(eval_path.read_text())
        dsl11 = data.get("dsL11_backtothefuture", {})
        comp = dsl11.get("composite-net/ridge-cv", {}).get("by_subject", {})
        pca = dsl11.get("fold-pca:64/ridge-cv", {}).get("by_subject", {})

        rx = comp.get("pearson_r_x", 0.0)
        ry = comp.get("pearson_r_y", 0.0)
        r_mean = 0.5 * (rx + ry)

        pca_rx = pca.get("pearson_r_x", 0.0)
        pca_ry = pca.get("pearson_r_y", 0.0)
        pca_mean = 0.5 * (pca_rx + pca_ry)

        print(f"[*] Checkpoint Epoch {epoch:02d} [{time.time() - t0:.1f}s]: CompositeNet mean r={r_mean:.3f} (r_x={rx:.3f}, r_y={ry:.3f}) vs PCA={pca_mean:.3f} (diff={r_mean - pca_mean:+.3f})", flush=True)

        return epoch, {
            "epoch": epoch,
            "r_x": rx,
            "r_y": ry,
            "r_mean": r_mean,
            "pca_r_mean": pca_mean,
            "diff_vs_pca": r_mean - pca_mean
        }
    return epoch, None


def main():
    epochs = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    print(f"[*] Evaluating {len(epochs)} checkpoints in parallel (4 workers)...", flush=True)
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(eval_one, epochs))

    trajectory = []
    for ep, res in sorted(results, key=lambda x: x[0]):
        if res is not None:
            trajectory.append(res)

    print("\n" + "=" * 80, flush=True)
    print(f"{'Epoch':<8} {'r_x':<10} {'r_y':<10} {'Mean r':<10} {'PCA r':<10} {'vs PCA':<10}", flush=True)
    print("-" * 80, flush=True)
    for r in trajectory:
        print(f"{r['epoch']:<8d} {r['r_x']:<10.3f} {r['r_y']:<10.3f} "
              f"{r['r_mean']:<10.3f} {r['pca_r_mean']:<10.3f} {r['diff_vs_pca']:+10.3f}", flush=True)
    print("=" * 80, flush=True)
    print(f"\n[*] All evaluations finished in {time.time() - t_start:.1f}s", flush=True)

    Path("results/epoch_saturation/trajectory_table.json").write_text(json.dumps(trajectory, indent=2))


if __name__ == "__main__":
    main()
