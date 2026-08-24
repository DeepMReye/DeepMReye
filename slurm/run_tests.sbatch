#!/bin/bash
#SBATCH --job-name=pytest
#SBATCH --account=AIFAC_S07_154
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err
set -e
module purge
cd /leonardo_work/EUHPC_D21_101/mfrey/dme/DeepMReye
export DEEPMREYE_DATA=/leonardo_scratch/fast/AIFAC_S07_154/mfrey/dme/data
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
echo "START $(date)"
.venv/bin/python -m pytest deepmreye/tests/ -q -p no:cacheprovider
echo "EXIT=$?"
echo "END $(date)"
