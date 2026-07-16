#!/bin/bash
# LR sweep at 1b to pick the recipe before spending A100 hours on 4b+.
#
# The report publishes no LR for this stage; its {1e-7, 5e-7, 1e-6} are for
# downstream task adaptation of an already-tuned MedGemma and underfit badly
# here (docs/RECIPE.md). So we sweep conventional SFT values instead.
#
#   bash scripts/sweep_lr.sh [data_dir]
set -euo pipefail

REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
DATA_DIR=${1:-${REPO_ROOT}/data/mix_default}
SIZE=${SIZE:-1b}
LRS=${LRS:-"5e-6 1e-5 2e-5"}

for lr in ${LRS}; do
    jid=$(sbatch --parsable \
        -p oermannlab --gres=gpu:a100:2 -t 12:00:00 \
        -J "sweep_${SIZE}_lr${lr}" \
        --export=ALL,SIZE=${SIZE},LR=${lr},DATA_DIR=${DATA_DIR},VARIANT=sweep \
        "${REPO_ROOT}/scripts/train_med.sbatch")
    echo "submitted lr=${lr} -> job ${jid}"
done
