#!/bin/bash
#SBATCH --job-name=dme-jepa
#SBATCH --output=logs/jepa_%j.out
#SBATCH --error=logs/jepa_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#
# JEPA pretraining on the eye-block corpus. One A100, one process.
#
# Partition and account are NOT set here, for the same reason as
# extract_array.sbatch: a wrong value fails at submit time with a message that
# blames the account when it is often the partition. Pass them explicitly:
#
#   sbatch -A AIFAC_S07_154 -p boost_usr_prod slurm/train_jepa.sbatch
#
# Everything the run reads is already on scratch, so this needs no network --
# which is just as well, because compute nodes have none. wandb therefore runs
# in OFFLINE mode: the run is written to $OUT_DIR/wandb and uploaded afterwards
# from a login node, which does have network:
#
#   .venv/bin/wandb sync <OUT_DIR>/wandb/offline-run-*
#
# The job prints that exact command when it finishes. WANDB_MODE=online will
# hang and then drop metrics here; do not set it.
#
# --mem is 128G rather than 64G because the in-training probe now fits its
# readout on every labeled window at full spatial resolution, and sklearn fits
# in float64: rows x features x 8 bytes, a few times over. At EMBED_DIM=32
# (2,304 features) that is a few GB; at EMBED_DIM=256 (18,432) it is tens, so
# cap it there with EXTRA_ARGS="--probe-windows 2000" or probe less often.
#
# Single GPU is deliberate. The corpus is ~8k windows of 100 TRs (~250 steps per
# epoch at batch 32), so the run is bound by reading 47x29x18x100 blocks off
# Lustre rather than by the GPU. Data loader workers buy more here than a second
# device would; DDP is not wired up in train_jepa.py.
#
# Checkpoints go to scratch, not the repo: /leonardo_work is at 95% of quota,
# and runs/ is regenerable output rather than something to keep next to code.

set -euo pipefail

# Slurm copies the batch script into its spool directory before running it, so
# BASH_SOURCE points at /var/spool/... and not at the repo -- deriving the root
# from it lands somewhere unwritable. SLURM_SUBMIT_DIR is the directory sbatch
# was called from, which is the repo root (the --output path above assumes the
# same thing).
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
if [[ ! -f "$REPO_DIR/scripts/train_jepa.py" ]]; then
    echo "REPO_DIR=$REPO_DIR does not look like the repo; submit from the repo root" >&2
    exit 1
fi
SCRATCH_DIR="${SCRATCH_DIR:-/leonardo_scratch/fast/AIFAC_S07_154/mfrey/dme}"
DATA_DIR="${DATA_DIR:-$SCRATCH_DIR/data}"
OUT_DIR="${OUT_DIR:-$SCRATCH_DIR/runs/jepa/${SLURM_JOB_ID:-manual}}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-2e-4}"
WINDOW_SIZE="${WINDOW_SIZE:-100}"
EMBED_DIM="${EMBED_DIM:-256}"
ENCODER_DEPTH="${ENCODER_DEPTH:-6}"
# Masking. Temporal beat spatial at every ratio in both 18-config sweeps, and
# t=0.6 topped them (mean r 0.655); s=0 keeps the spatial axis unmasked. Both at
# zero masks nothing and train_jepa.py refuses to start.
S_RATIO="${S_RATIO:-0.0}"
T_RATIO="${T_RATIO:-0.6}"
PROBE_EVERY="${PROBE_EVERY:-5}"
SAVE_EVERY="${SAVE_EVERY:-10}"
# Full spatial resolution. Mean-pooling the 6x4x3 token grid averages away the
# across-orbit contrast gaze lives in and costs about half the recoverable
# correlation, so a curve logged under 'mean' cannot be compared with anything.
SPATIAL_POOL="${SPATIAL_POOL:-6x4x3}"
PROBE_PROTOCOLS="${PROBE_PROTOCOLS:-dataset subject}"
WANDB_PROJECT="${WANDB_PROJECT:-deepmreye-jepa}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-jepa-${SLURM_JOB_ID:-manual}}"

cd "$REPO_DIR"
mkdir -p logs "$OUT_DIR"

# Match the CPU allocation. Left unset, torch's intra-op pool grabs every core
# on the node and then fights the data loader workers for them.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Every worker opens the same read-only HDF5 files from Lustre. Without this,
# HDF5 tries to take POSIX locks the file system does not serve and opens fail
# with "unable to lock file" partway into an epoch.
export HDF5_USE_FILE_LOCKING=FALSE

# Python block-buffers stdout when it is a file rather than a terminal, so
# without this a 12-hour run shows nothing in the log until it ends (or shows
# nothing at all if it is killed at the wall clock).
export PYTHONUNBUFFERED=1

# Compute nodes have no outbound network. Offline, wandb writes the run to disk
# and never opens a socket; `wandb sync` uploads it later from a login node.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="$OUT_DIR"
export WANDB_SILENT="${WANDB_SILENT:-true}"

echo "job ${SLURM_JOB_ID:-manual} on $(hostname), ${SLURM_CPUS_PER_TASK:-8} cpus"
echo "data    $DATA_DIR"
echo "out     $OUT_DIR"
echo "wandb   $WANDB_PROJECT / $WANDB_RUN_NAME (mode $WANDB_MODE)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# PROBE_PROTOCOLS is a deliberately unquoted word list ("dataset subject").
# shellcheck disable=SC2086
srun .venv/bin/python scripts/train_jepa.py \
    --data-dir "$DATA_DIR" \
    --out "$OUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --window-size "$WINDOW_SIZE" \
    --embed-dim "$EMBED_DIM" \
    --encoder-depth "$ENCODER_DEPTH" \
    --s-ratio "$S_RATIO" \
    --t-ratio "$T_RATIO" \
    --probe-every "$PROBE_EVERY" \
    --probe-protocols $PROBE_PROTOCOLS \
    --spatial-pool "$SPATIAL_POOL" \
    --save-every "$SAVE_EVERY" \
    --num-workers "$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-name "$WANDB_RUN_NAME" \
    ${EXTRA_ARGS:-}

echo "job ${SLURM_JOB_ID:-manual} finished"
if [[ "$WANDB_MODE" == "offline" ]]; then
    echo "upload the metrics from a login node with:"
    echo "  cd $REPO_DIR && .venv/bin/wandb sync $OUT_DIR/wandb/offline-run-*"
fi
