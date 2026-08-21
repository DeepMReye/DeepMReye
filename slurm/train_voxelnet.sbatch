#!/bin/bash
#SBATCH --job-name=dme-voxelnet
#SBATCH --output=logs/voxelnet_%A_%a.out
#SBATCH --error=logs/voxelnet_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#
# From-scratch voxel -> gaze network. GPU, offline, no network needed at run time.
#
# Partition and account are NOT set here (a wrong value fails at submit with a
# confusing "invalid account or expired budget" that blames the account when the
# partition is the problem). Pass them:
#
#   sbatch -A AIFAC_S07_154 -p boost_usr_prod slurm/train_voxelnet.sbatch \
#          --note "shift2 long" --shift 2 --epochs 150 --cosine \
#          --out results/subtr/trial07.json
#
# Every argument after the script name is forwarded to train_voxelnet_scratch.py.
#
# --mem=120G: the labeled cache is an 11 GB fp16 memmap. It is *read* through the
# page cache rather than loaded, so RSS stays modest, but scoring materialises
# 512-row float32 slices and the OS will happily cache the whole file. 120 G on a
# 490 G boost node costs nothing and removes a class of failure.
#
# --time=12:00:00: one fold at 150 epochs took ~2 h on an M-series laptop GPU.
# A full 9-fold run is the array form below, one fold per task, not one job.
#
#   sbatch --array=0-8 -A AIFAC_S07_154 -p boost_usr_prod \
#          slurm/train_voxelnet.sbatch --note "shift2 long" --shift 2 --epochs 150 --cosine
#
# In array mode the script supplies --fold-index and a per-task --out itself (see below), so
# do NOT pass either: nine tasks sharing one --out would each rewrite the same file and the
# last writer would win, leaving one fold's results where nine were expected.
# Merge afterwards with: python scripts/summarize_voxelnet_trials.py
#
# ITK/OMP thread caps: nothing here calls ANTs, but numpy/torch will otherwise
# grab every core on the node and fight concurrent array tasks for them.

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${SLURM_CPUS_PER_TASK:-8}
# Deterministic row sampling across tasks; PYTHONHASHSEED drift is worth ~0.01 r,
# which is half the noise floor this project resolves at.
export PYTHONHASHSEED=0

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results/subtr/logs

PY=${PY:-.venv/bin/python}
DATA=${DEEPMREYE_DATA:-$PWD/data}

echo "[*] host $(hostname)  job ${SLURM_JOB_ID:-none}  task ${SLURM_ARRAY_TASK_ID:-none}"
echo "[*] data $DATA"
"$PY" -c "import torch; print('[*] torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    # One fold per task, each writing its own JSON. The summarizer globs them back together.
    EXTRA=(--fold-index "$SLURM_ARRAY_TASK_ID"
           --out "results/subtr/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_f${SLURM_ARRAY_TASK_ID}.json")
else
    EXTRA=()
fi

DEEPMREYE_DATA="$DATA" "$PY" scripts/train_voxelnet_scratch.py --device cuda "${EXTRA[@]}" "$@"
