#!/bin/bash
# Promote one configuration to the full 9-fold LODO, several seeds, TTA on.
#
#   ./scripts/promote_submit.sh <tag> [extra trainer args...]
#
# One job per (fold, seed): a single network per cell, never an ensemble. The screening
# rounds ran with TTA off (inference-only, roughly configuration-independent, doubles the
# cost of every run); it goes back on here.
#
# THREE of the nine folds -- dsL03, dsL05, dsL07 -- were the screening folds, so a
# configuration chosen on them is optimistically biased on those three. Report the median
# over the SIX unseen folds alongside the 9-fold median; `promote_report.py` prints both.
set -euo pipefail
TAG=$1; shift
REPO=/leonardo_work/EUHPC_D21_101/mfrey/dme/DeepMReye
cd "$REPO"
export DEEPMREYE_DATA=/leonardo_scratch/fast/AIFAC_S07_154/mfrey/dme/data
FOLDS="dsL01_guided_fixations dsL02_pursuit dsL03_pursuit dsL04_pursuit dsL05_free_viewing
       dsL06_sequences dsL07_deepmreye_calib dsL08_studyforrest_movie dsL11_backtothefuture"
SEEDS=${PROMOTE_SEEDS:-"0 1 2"}
BASE="--shift 1 --mirror 0.5 --tta-mirror --epochs 150 --patience 40 --cosine --lr 1e-3 --weight-decay 1e-1"
mkdir -p results/subtr/promote
n=0
for fd in $FOLDS; do
  for sd in $SEEDS; do
    out="results/subtr/promote/${TAG}__${fd}__s${sd}.json"
    [ -f "$out" ] && continue
    sbatch -A AIFAC_S07_154 -p boost_usr_prod --job-name="pr_${TAG}" \
      slurm/train_voxelnet.sbatch --note "promote ${TAG} ${fd} seed${sd}" \
      --folds "$fd" --seed "$sd" $BASE "$@" --out "$out" >/dev/null
    n=$((n+1))
  done
done
echo "[$TAG] submitted $n"
