#!/bin/bash
#SBATCH --job-name=dme-scaleext
#SBATCH --output=logs/scaleext_%j.out
#SBATCH --error=logs/scaleext_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=440G
#SBATCH --time=24:00:00
#
# Extend the corpus-scaling curve past N=1039 to the full 2009-participant corpus.
#
# ONE incremental pass: `Moments` is additive, so a snapshot at each checkpoint costs the
# largest fit rather than the sum. Shuffled subject order with --seed 0, which is the order
# `basis_n2000.npz` was fitted under (its `_meta` records `shuffled_seed: 0`, 2000 subjects,
# 96000 TRs = 48 TRs/subject) -- so the N=2000 snapshot here must reproduce that file. It is
# written to a SEPARATE directory precisely so that check is possible: pointing this at
# results/scaling would overwrite the artifact it is supposed to be validated against.
#
# --mem=440G: two 14236^2 float64 accumulators are 3.2 GB, but the CCA reduction and the
# 512-direction band pool allocate several more of that order, and an OOM 20 hours in costs
# the whole pass.
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export PYTHONHASHSEED=0
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DEEPMREYE_DATA=${DEEPMREYE_DATA:-/leonardo_scratch/fast/AIFAC_S07_154/mfrey/dme/data} \
  .venv/bin/python scripts/sweep_corpus_scaling.py \
    --out-dir results/scaling_ext \
    --checkpoints 100 200 400 800 1039 1400 1700 2000 \
    --seed 0 --trs-per-subject 48 --k 256 "$@"
