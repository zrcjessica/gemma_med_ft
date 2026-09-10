#!/bin/bash
# 4b red-flag pilot: t=0, mid-trajectory, and final, on the banked lenses.
#
# Three checkpoints, chosen to bracket the finetune: step 0 (the untuned
# gemma-3-4b-it text base -- the probe's origin, and the point the lab's own
# gemma-3-4b-it redflag run is comparable to), step 512 (mid, the step the
# decode + judge A/B probes already characterize), step 4882 (final).
#
# Costs a read-out, not a fit: every one of these lenses is already banked
# (band_out/4b_it_fanout/ck<N>/ck<N>/lens.pt), so this is ~150 forward passes
# plus 150 greedy generations per checkpoint.
#
#   scripts/redflag/run_pilot_4b.sh            # submit the pilot
#   SMOKE=1 scripts/redflag/run_pilot_4b.sh    # 2 vignettes, 1 task, first
set -euo pipefail

REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
OUT_ROOT=${OUT_ROOT:-/gpfs/data/oermannlab/users/zhouj14/redflag_out}
RUN_DIR=${RUN_DIR:-outputs/4b/full_lr1e-5_25911616}
ARM=${ARM:-4b_it}
STEPS=${STEPS:-"0 512 4882"}
N_SHARDS=${N_SHARDS:-2}
SMOKE=${SMOKE:-0}

# Partition routing. superpod by default, same as the band fanout: oermannlab
# was estimating a ~2-day start on 2026-08-14, and a100_short is closed to this
# association (see the sbatch header). ALWAYS pass --qos=qos_superpod with
# --partition=superpod -- without it the job runs at QOS `normal` (priority 0)
# behind everyone's qos_superpod work and never starts. The gres name differs
# per partition too: superpod is H100, oermannlab is A100.
PARTITION=${PARTITION:-superpod}
if [[ "${PARTITION}" == "superpod" ]]; then
    QOS=${QOS:-qos_superpod}; GRES=${GRES:-gpu:h100:1}
else
    QOS=${QOS:-}; GRES=${GRES:-gpu:a100:1}
fi
SB_FLAGS=(--partition="${PARTITION}" --gres="${GRES}")
[[ -n "${QOS}" ]] && SB_FLAGS+=(--qos="${QOS}")

cd "${REPO_ROOT}"
source .venv-jlens/bin/activate

echo "=== configs ==="
python scripts/redflag/make_configs.py --arm "${ARM}" --run-dir "${RUN_DIR}" --steps ${STEPS}

mkdir -p "${OUT_ROOT}" slurm/logs
CONFIG_LIST=${OUT_ROOT}/configs_${ARM}.txt
: > "${CONFIG_LIST}"
for s in ${STEPS}; do
    echo "${REPO_ROOT}/scripts/redflag/configs/${ARM}_ck${s}.yaml" >> "${CONFIG_LIST}"
done
N_CFG=$(wc -l < "${CONFIG_LIST}")

EXPORTS="ALL,CONFIG_LIST=${CONFIG_LIST},OUT_ROOT=${OUT_ROOT},N_SHARDS=${N_SHARDS}"
if [[ "${SMOKE}" == "1" ]]; then
    # One task, two vignettes: proves the checkpoint loads, the banked lens
    # matches its d_model, the chat template round-trips, and the npz lands --
    # before committing 6 GPU-hours to the full grid.
    ARRAY="0-0"; EXPORTS="${EXPORTS},LIMIT=2"
    echo "SMOKE: 1 task, 2 vignettes"
else
    ARRAY="0-$(( N_CFG * N_SHARDS - 1 ))"
fi

echo "=== submitting array ${ARRAY} (${N_CFG} checkpoints x ${N_SHARDS} shards) on ${PARTITION} ==="
sbatch "${SB_FLAGS[@]}" --array="${ARRAY}" --export="${EXPORTS}" scripts/redflag/probe_redflag.sbatch
