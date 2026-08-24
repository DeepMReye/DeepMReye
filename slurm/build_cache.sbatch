#!/bin/bash
#SBATCH --job-name=voxcache
#SBATCH --account=AIFAC_S07_154
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err
set -x -e
module purge
REPO=/leonardo_work/EUHPC_D21_101/mfrey/dme/DeepMReye
cd "$REPO"
export DEEPMREYE_DATA=/leonardo_scratch/fast/AIFAC_S07_154/mfrey/dme/data
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=8
echo "START $(date)"
.venv/bin/python scripts/build_voxel_cache.py --out results/subtr/voxels
echo "END $(date)"
ls -la results/subtr/voxels/
