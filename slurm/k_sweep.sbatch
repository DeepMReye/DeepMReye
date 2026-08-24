#!/bin/bash
#SBATCH --job-name=dme-ksweep
#SBATCH --output=logs/ksweep_%j.out
#SBATCH --error=logs/ksweep_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=04:00:00
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONHASHSEED=0
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DEEPMREYE_DATA=${DEEPMREYE_DATA:-/leonardo_scratch/fast/AIFAC_S07_154/mfrey/dme/data} \
  .venv/bin/python "$@"
