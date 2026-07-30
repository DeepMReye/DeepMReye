#!/bin/bash
#SBATCH --job-name=dme-probe
#SBATCH --output=logs/probe_%j.out
#SBATCH --error=logs/probe_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#
# The baseline table (scripts/eval_probe.py), as a batch job.
#
# Partition and account are not set here; pass them explicitly, as with the
# other scripts in this directory:
#
#   PROTOCOL=dataset ARMS="voxels random" \
#       sbatch -A AIFAC_S07_154 -p boost_usr_prod slurm/eval_probe.sbatch
#
#   PROTOCOL=dataset ARMS=trained CHECKPOINT=$SCRATCH/runs/jepa/main/epoch150.pt \
#       sbatch -A AIFAC_S07_154 -p boost_usr_prod slurm/eval_probe.sbatch
#
# A GPU is requested because the `random` and `trained` arms push every probe
# window through a ViT. The `voxels` arm never touches it and the readouts are
# all sklearn on CPU, so a voxels-only run does not need one.
#
# The feature cache is keyed on arm, architecture and checkpoint mtime, so runs
# that share an arm reuse each other's embeddings rather than recomputing them.

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
if [[ ! -f "$REPO_DIR/scripts/eval_probe.py" ]]; then
    echo "REPO_DIR=$REPO_DIR does not look like the repo; submit from the repo root" >&2
    exit 1
fi

SCRATCH_DIR="${SCRATCH_DIR:-/leonardo_scratch/fast/AIFAC_S07_154/mfrey/dme}"
DATA_DIR="${DATA_DIR:-$SCRATCH_DIR/data}"

PROTOCOL="${PROTOCOL:-dataset}"
ARMS="${ARMS:-voxels random}"
READOUTS="${READOUTS:-mean linear ridge-cv pca-ridge pls}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/results}"
OUT_JSON="${OUT_JSON:-$RESULTS_DIR/probe_${PROTOCOL}_${SLURM_JOB_ID:-manual}.json}"

cd "$REPO_DIR"
mkdir -p logs "$RESULTS_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

echo "job ${SLURM_JOB_ID:-manual} on $(hostname)"
echo "protocol $PROTOCOL | arms $ARMS | checkpoint ${CHECKPOINT:-none}"

srun .venv/bin/python scripts/eval_probe.py \
    --data-dir "$DATA_DIR" \
    --protocol "$PROTOCOL" \
    --arms $ARMS \
    --readouts $READOUTS \
    --num-workers "$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))" \
    --out "$OUT_JSON" \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
    ${EXTRA_ARGS:-}

echo "job ${SLURM_JOB_ID:-manual} finished; results in $OUT_JSON"
