#!/usr/bin/env python3
"""Fast evaluation of all saved epoch checkpoints on zero-shot holdout dsL11."""
import json
import subprocess
import time
from pathlib import Path

out_dir = Path("results/epoch_saturation")
epochs = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

print(f"[*] Fast-evaluating {len(epochs)} CompositeNet checkpoints on dsL11...", flush=True)

results = []
for ep in epochs:
    ckpt_path = out_dir / f"checkpoint_epoch_{ep:02d}.pt"
    eval_path = out_dir / f"eval_epoch_{ep:02d}.json"

    if not ckpt_path.exists():
        print(f"[-] Checkpoint {ckpt_path} missing, skipping...")
        continue

    cmd = (
        f"uv run python scripts/eval_probe.py --protocol dataset "
        f"--features composite-net --composite-checkpoint {ckpt_path} "
        f"--fold-name dsL11 --readouts ridge-cv --standardize-targets dataset "
        f"--out {eval_path}"
    )

    t0 = time.time()
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[!] Error evaluating epoch {ep}: {res.stderr}")
        continue

    if eval_path.exists():
        data = json.loads(eval_path.read_text())
        comp = data.get("dsL11_backtothefuture", {}).get("composite-net/ridge-cv", {}).get("by_subject", {})
        rx = comp.get("pearson_r_x", 0.0)
        ry = comp.get("pearson_r_y", 0.0)
        r_mean = 0.5 * (rx + ry)

        print(f"Epoch {ep:02d} [{time.time() - t0:.1f}s]: r_x = {rx:.3f} | r_y = {ry:.3f} | Mean r = {r_mean:.3f}", flush=True)
        results.append({"epoch": ep, "r_x": rx, "r_y": ry, "mean_r": r_mean})

print("\n" + "=" * 65, flush=True)
print(f"{'Epoch':<8} {'r_x (Horiz)':<14} {'r_y (Vert)':<14} {'Mean r':<12} {'vs PCA (0.776)':<14}", flush=True)
print("-" * 65, flush=True)
for r in results:
    diff = r['mean_r'] - 0.776
    print(f"{r['epoch']:<8d} {r['r_x']:<14.3f} {r['r_y']:<14.3f} {r['mean_r']:<12.3f} {diff:+14.3f}", flush=True)
print("=" * 65, flush=True)

(out_dir / "saturation_clean_trajectory.json").write_text(json.dumps(results, indent=2))
print(f"\n[*] Clean trajectory saved to {out_dir / 'saturation_clean_trajectory.json'}", flush=True)
