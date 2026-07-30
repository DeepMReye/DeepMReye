#!/bin/bash
#SBATCH --job-name=eval-sweep
#SBATCH --output=logs/eval_sweep_%A_%a.out
#SBATCH --error=logs/eval_sweep_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail
cd /leonardo_work/EUHPC_D21_101/mfrey/dme/DeepMReye

SCRATCH_DIR="/leonardo_scratch/fast/AIFAC_S07_154/mfrey/dme"
DATA_DIR="$SCRATCH_DIR/data"
SWEEP_NAME="${SWEEP_NAME:-sweep_spatial_d32}"

mapfile -t RUN_DIRS < <(ls -d "$SCRATCH_DIR/runs/jepa/$SWEEP_NAME"/*/ | sort)
RUN_DIR="${RUN_DIRS[$SLURM_ARRAY_TASK_ID]}"
CKPT="$RUN_DIR/last.pt"
NAME="$(basename "$RUN_DIR")"
OUT_JSON="results/eval_${SWEEP_NAME}_${NAME}.json"

mkdir -p results logs

echo "Evaluating $NAME from $CKPT..."
srun .venv/bin/python scripts/eval_probe.py \
    --data-dir "$DATA_DIR" \
    --protocol dataset \
    --arms trained \
    --readouts ridge-cv \
    --spatial-pool 6x4x3 \
    --checkpoint "$CKPT" \
    --out "$OUT_JSON"
