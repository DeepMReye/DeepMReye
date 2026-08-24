#!/bin/bash
#SBATCH --job-name=dme
#SBATCH --account=AIFAC_S07_154
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#
# Generic CPU wrapper: everything in this package except extraction is one
# python command, so there is one sbatch file rather than one per command.
#
#   sbatch slurm/run.sbatch -m deepmreye evaluate --calibrate
#   sbatch slurm/run.sbatch -m deepmreye fit-basis --k 256
#
# Compute nodes have no outbound network, so the corpus must already be on
# disk; DEEPMREYE_DATA points at it.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export DEEPMREYE_DATA=${DEEPMREYE_DATA:-/leonardo_scratch/fast/AIFAC_S07_154/mfrey/dme/data}
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "[*] $(date)  node $(hostname)  data $DEEPMREYE_DATA"
srun .venv/bin/python "$@"
