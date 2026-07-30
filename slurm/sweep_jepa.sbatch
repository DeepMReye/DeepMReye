#!/bin/bash
#SBATCH --job-name=dme-sweep
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# One short JEPA run per array task, configs read from slurm/sweep_configs.tsv.
# Submit from the repo root, sizing the array from the config file:
#
#   N=$(grep -cvE '^\s*(#|$)' slurm/sweep_configs.tsv)
#   sbatch -A AIFAC_S07_154 -p boost_usr_prod --array=0-$((N-1)) slurm/sweep_jepa.sbatch
#
# Deliberately short (EPOCHS=30, ~90 min including startup). This is a search
# for which direction is worth a long run, not the run that gets reported: the
# winning config is retrained before its number goes anywhere near the paper.
#
# Note what selecting on these runs costs, and note that it got *sharper*, not
# looser, now that the in-training probe computes the real leave-one-dataset-out
# number: `probe/dataset/mean_r` here is the same quantity eval_probe.py
# reports, so picking the best config by it is picking on the test folds
# directly. That is why CLAUDE.md calls the probe a monitor and not a model
# selection criterion. Two ways out, both fine, but pick one deliberately:
# select on `train/jepa_loss` (unsupervised, leaks nothing), or select on the
# probe and report the headline from a retrained model.
#
# Metrics go to wandb (offline; see the sync line the task prints when it ends)
# as well as to logs/sweep_%A_%a.out.

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
if [[ ! -f "$REPO_DIR/scripts/train_jepa.py" ]]; then
    echo "REPO_DIR=$REPO_DIR does not look like the repo; submit from the repo root" >&2
    exit 1
fi
cd "$REPO_DIR"

SCRATCH_DIR="${SCRATCH_DIR:-/leonardo_scratch/fast/AIFAC_S07_154/mfrey/dme}"
DATA_DIR="${DATA_DIR:-$SCRATCH_DIR/data}"
CONFIG_FILE="${CONFIG_FILE:-slurm/sweep_configs.tsv}"
SWEEP_NAME="${SWEEP_NAME:-sweep}"

EPOCHS="${EPOCHS:-30}"
PROBE_EVERY="${PROBE_EVERY:-5}"
SAVE_EVERY="${SAVE_EVERY:-10}"
SPATIAL_POOL="${SPATIAL_POOL:-6x4x3}"
PROBE_PROTOCOLS="${PROBE_PROTOCOLS:-dataset subject}"
WANDB_PROJECT="${WANDB_PROJECT:-deepmreye-jepa}"

# Pick this task's line, skipping comments and blanks. awk rather than sed so
# the comment filter and the line selection cannot disagree about numbering.
LINE=$(awk -v want="$((SLURM_ARRAY_TASK_ID + 1))" \
    '!/^[[:space:]]*(#|$)/ { if (++n == want) { print; exit } }' "$CONFIG_FILE")
if [[ -z "$LINE" ]]; then
    echo "no config at index $SLURM_ARRAY_TASK_ID in $CONFIG_FILE" >&2
    exit 1
fi
NAME=${LINE%%$'\t'*}
CONFIG_ARGS=${LINE#*$'\t'}
OUT_DIR="$SCRATCH_DIR/runs/jepa/$SWEEP_NAME/$NAME"

mkdir -p logs "$OUT_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

# Offline: compute nodes have no network. One directory per task under its own
# OUT_DIR, so the whole sweep syncs with a single glob from a login node.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="$OUT_DIR"
export WANDB_SILENT="${WANDB_SILENT:-true}"

echo "sweep task $SLURM_ARRAY_TASK_ID = '$NAME' on $(hostname)"
echo "config args: $CONFIG_ARGS"
echo "out         $OUT_DIR"

# CONFIG_ARGS and PROBE_PROTOCOLS are deliberately unquoted word lists.
# shellcheck disable=SC2086
srun .venv/bin/python scripts/train_jepa.py \
    --data-dir "$DATA_DIR" \
    --out "$OUT_DIR" \
    --epochs "$EPOCHS" \
    --probe-every "$PROBE_EVERY" \
    --probe-protocols $PROBE_PROTOCOLS \
    --spatial-pool "$SPATIAL_POOL" \
    --save-every "$SAVE_EVERY" \
    --num-workers "$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-name "$SWEEP_NAME/$NAME" \
    $CONFIG_ARGS

echo "sweep task $SLURM_ARRAY_TASK_ID ('$NAME') finished"
if [[ "$WANDB_MODE" == "offline" ]]; then
    echo "upload with: cd $REPO_DIR && .venv/bin/wandb sync" \
         "$SCRATCH_DIR/runs/jepa/$SWEEP_NAME/*/wandb/offline-run-*"
fi
