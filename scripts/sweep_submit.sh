#!/bin/bash
# Submit one voxelnet hyperparameter configuration across the screening folds x seeds.
#
#   ./scripts/sweep_submit.sh <tag> [extra trainer args...]
#
# Selection is on `best_val` -- validation datasets are drawn from each fold's TRAINING pool,
# so nothing here sees the held-out dataset and the final 9-fold test number stays clean.
# TTA is off during screening: it is inference-only, roughly config-independent, and doubles
# every run. It goes back on for the finalists.
set -euo pipefail
TAG=$1; shift
REPO=/leonardo_work/EUHPC_D21_101/mfrey/dme/DeepMReye
cd "$REPO"
export DEEPMREYE_DATA=/leonardo_scratch/fast/AIFAC_S07_154/mfrey/dme/data
FOLDS=${SWEEP_FOLDS:-"dsL03_pursuit dsL05_free_viewing dsL07_deepmreye_calib"}
SEEDS=${SWEEP_SEEDS:-"0 1"}
BASE="--shift 1 --mirror 0.5 --epochs 150 --patience 40 --cosine --lr 1e-3 --weight-decay 1e-1"
n=0
for fd in $FOLDS; do
  for sd in $SEEDS; do
    out="results/subtr/sweep/${TAG}__${fd}__s${sd}.json"
    [ -f "$out" ] && continue
    sbatch -A AIFAC_S07_154 -p boost_usr_prod --job-name="sw_${TAG}" \
      slurm/train_voxelnet.sbatch --note "sweep ${TAG} ${fd} seed${sd}" \
      --folds "$fd" --seed "$sd" $BASE "$@" --out "$out" >/dev/null
    n=$((n+1))
  done
done
echo "[$TAG] submitted $n"
