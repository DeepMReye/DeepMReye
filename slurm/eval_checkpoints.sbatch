#!/bin/bash
#SBATCH --job-name=dme-ckpt-eval
#SBATCH --output=logs/ckpteval_%A_%a.out
#SBATCH --error=logs/ckpteval_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# One full eval_probe run per training checkpoint, as an array. This is the
# overfitting curve: the same folds, arms and readouts as the baseline table, so
# a point on it can be read straight against `voxels`.
#
#   CKPT_DIR=$SCRATCH/runs/jepa/long/base PROTOCOL=dataset \
#     sbatch -A AIFAC_S07_154 -p boost_usr_prod --array=0-6 slurm/eval_checkpoints.sbatch
#
# The in-training probe in train_jepa.py is NOT this. That one is ridge-cv only,
# on a subject split, over 400 sub-sampled windows -- fine for "is it learning",
# useless for "does it beat raw voxels", because it shares neither protocol nor
# readout set with the baseline. Train with --probe-every 0 and use this instead
# when the numbers have to be comparable.
#
# Array tasks share the feature cache. That is safe and is the point: `voxels`
# and `random` features are keyed on arm and architecture only, so they are
# computed once and reused by every task, while the `trained` arm is keyed on
# checkpoint path and mtime, so no two tasks collide on a write. Warm the cache
# with a voxels+random run first and each task only pays for its own encoder.

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
if [[ ! -f "$REPO_DIR/scripts/eval_probe.py" ]]; then
    echo "REPO_DIR=$REPO_DIR does not look like the repo; submit from the repo root" >&2
    exit 1
fi
cd "$REPO_DIR"

SCRATCH_DIR="${SCRATCH_DIR:-/leonardo_scratch/fast/AIFAC_S07_154/mfrey/dme}"
DATA_DIR="${DATA_DIR:-$SCRATCH_DIR/data}"
CKPT_DIR="${CKPT_DIR:?set CKPT_DIR to the run directory holding epoch*.pt}"

PROTOCOL="${PROTOCOL:-dataset}"
ARMS="${ARMS:-voxels random trained}"
# Identical to the baseline run. No gbt: it is slow, and CLAUDE.md already
# records that it is a coin flip against ridge-cv on this feature source.
READOUTS="${READOUTS:-mean linear ridge-cv pca-ridge pls}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/results}"

# Checkpoints in epoch order; the array index selects one.
mapfile -t CKPTS < <(ls -1 "$CKPT_DIR"/epoch*.pt 2>/dev/null | sort -V)
if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "no epoch*.pt in $CKPT_DIR" >&2
    exit 1
fi
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "$IDX" -ge ${#CKPTS[@]} ]]; then
    echo "array index $IDX past the ${#CKPTS[@]} checkpoints in $CKPT_DIR" >&2
    exit 1
fi
CKPT="${CKPTS[$IDX]}"
RUN_NAME="$(basename "$CKPT_DIR")"
EPOCH_TAG="$(basename "$CKPT" .pt)"
OUT_JSON="$RESULTS_DIR/ckpt_${RUN_NAME}_${PROTOCOL}_${EPOCH_TAG}.json"

mkdir -p logs "$RESULTS_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

echo "job ${SLURM_JOB_ID:-manual}[$IDX] on $(hostname)"
echo "run $RUN_NAME | checkpoint $CKPT | protocol $PROTOCOL"

# WINDOW_SIZE must match what the checkpoint was trained at; train_jepa.py
# stores it, so read it back rather than trusting the eval default.
WINDOW_SIZE="${WINDOW_SIZE:-$(.venv/bin/python -c "
import torch, sys
print(torch.load(sys.argv[1], map_location='cpu', weights_only=False).get('window_size', 100))
" "$CKPT")}"
echo "window size $WINDOW_SIZE"

srun .venv/bin/python scripts/eval_probe.py \
    --data-dir "$DATA_DIR" \
    --protocol "$PROTOCOL" \
    --arms $ARMS \
    --readouts $READOUTS \
    --checkpoint "$CKPT" \
    --window-size "$WINDOW_SIZE" \
    --num-workers "$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))" \
    --out "$OUT_JSON" \
    ${EXTRA_ARGS:-}

echo "checkpoint $EPOCH_TAG done; results in $OUT_JSON"
