#!/bin/bash
#SBATCH --job-name=dme-extract
#SBATCH --output=logs/extract_%A_%a.out
#SBATCH --error=logs/extract_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#
# Partition and account are NOT set here: they are site- and project-specific
# and a wrong value fails at submit time with a confusing message. Pass them on
# the command line, or let slurm/submit_extraction.sh read SLURM_PARTITION and
# SLURM_ACCOUNT from the environment:
#
#   sbatch -A EUHPC_D21_101 -p dcgp_usr_prod --array=0-63 slurm/extract_array.sbatch
#
# --mem is deliberate, not inherited. ANTs SyNAggro diverges unpredictably on
# some acquisitions -- a 0.10 GB volume exceeded 32 GB while a 1.16 GB one was
# fine -- so this is sized for the bad case, not the median (~3.4 GB). At 32 G
# roughly 20% of subjects were killed by the per-subject watchdog; boost nodes
# carry ~490 GB, so 120 G costs nothing and recovers almost all of them.
# extract_staged.py caps each child below this (--mem-limit-gb), so a runaway
# costs one subject instead of the whole task.
#
# Coregistration + extraction of staged BOLD files (no network required).
#
# Compute nodes have no outbound network, so run slurm/stage_downloads.py on a
# login node first: it writes the .nii.gz files and the manifest this reads.
#
#   python slurm/stage_downloads.py --data-dir $DATA_DIR --staging-dir $STAGING
#   sbatch --array=0-63 slurm/extract_array.sbatch
#
# Each array task takes every Nth manifest line, so the array size sets the
# parallelism and no shared queue is needed. Tasks skip subjects already on disk
# and own their output files outright, so resubmitting after a timeout or
# preemption is safe and picks up where it left off.

set -euo pipefail

# Derive the repo root from this script so a clone anywhere works.
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${DATA_DIR:-/leonardo_scratch/fast/AIFAC_S07_154/mfrey/dme/data}"
STAGING_DIR="${STAGING_DIR:-/leonardo_scratch/fast/AIFAC_S07_154/mfrey/dme/staging}"
MANIFEST="${MANIFEST:-$STAGING_DIR/manifest.jsonl}"

cd "$REPO_DIR"

# ANTs threads must match the CPU allocation: left unset it grabs every core on
# the node, and concurrent array tasks then fight each other for the same cores.
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# ANTs writes large intermediates; keep them on node-local disk when available.
export TMPDIR="${TMPDIR:-/tmp}/dme-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

echo "task $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT on $(hostname), ${SLURM_CPUS_PER_TASK:-8} cpus"

srun .venv/bin/python slurm/extract_staged.py \
    --manifest "$MANIFEST" \
    --data-dir "$DATA_DIR" \
    --mem-limit-gb "${MEM_LIMIT_GB:-100}" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --stride "$SLURM_ARRAY_TASK_COUNT"

echo "task $SLURM_ARRAY_TASK_ID finished"
