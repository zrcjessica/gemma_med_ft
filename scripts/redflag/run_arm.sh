#!/bin/bash
# Red-flag read-out for one arm at the pilot's three steps, on banked lenses.
#
# Generalizes run_pilot_4b.sh across sizes. Same three steps everywhere --
# 0 (untuned base, the trajectory origin), 512 (mid, the step the decode and
# judge A/B probes already characterize), 4882 (final) -- because a cross-size
# panel is only legible if every arm is read at the same clock positions.
#
# Costs a read-out, not a fit: every lens is already banked by the band grid, so
# this is ~150 forward passes plus 150 greedy generations per checkpoint.
# Refitting instead would be 0.35 h at 270m and 35 h at 27b, per checkpoint.
#
#   scripts/redflag/run_arm.sh 270m_it     # one arm
#   SMOKE=1 scripts/redflag/run_arm.sh 27b_it
#   scripts/redflag/run_arm.sh             # every arm not yet run
set -euo pipefail

REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
OUT_ROOT=${OUT_ROOT:-/gpfs/data/oermannlab/users/zhouj14/redflag_out}
STEPS=${STEPS:-"0 512 4882"}
N_SHARDS=${N_SHARDS:-2}
SMOKE=${SMOKE:-0}

# The training run each arm's checkpoints come from. 27b has three run dirs and
# only 25970073 carries checkpoints; the other two died before the first save.
run_dir_for() {
    case "$1" in
        270m_it) echo outputs/270m/full_lr1e-5_25799673 ;;
        1b_it)   echo outputs/1b/full_lr1e-5_25716223 ;;
        4b_it)   echo outputs/4b/full_lr1e-5_25911616 ;;
        12b_it)  echo outputs/12b/full_lr1e-5_26041764 ;;
        27b_it)  echo outputs/27b/full_lr1e-5_25970073 ;;
        *) echo "unknown arm: $1" >&2; return 1 ;;
    esac
}

# Host RAM per arm, not 27b's request everywhere: torch.load pulls the whole
# fp32 lens through CPU (7.0 GB at 27b, 2.8 GB at 12b) before the weights.
mem_for() {
    case "$1" in
        270m_it|1b_it|4b_it) echo 64G ;;
        12b_it)              echo 96G ;;
        27b_it)              echo 160G ;;
    esac
}

# Partition routing. superpod by default, same as the band fanout: oermannlab
# was estimating a ~2-day start on 2026-08-14, and a100_short is closed to this
# association (see the sbatch header). ALWAYS pass --qos=qos_superpod with
# --partition=superpod -- without it the job runs at QOS `normal` (priority 0)
# behind everyone's qos_superpod work and never starts. The gres name differs
# per partition too: superpod is H100, oermannlab is A100.
#
# 27b needs the H100: bf16 weights (54 GB) plus the fp32 lens (7 GB) is 61 GB on
# one device, which fits 80 GB but not 40 GB. On A100s it needs device_map=auto
# across 2 GPUs -- pass DEVICE_MAP=auto GRES=gpu:a100:2 and read
# medlens/loading.py's MAX_MEM_HEADROOM_GIB note first.
PARTITION=${PARTITION:-superpod}
if [[ "${PARTITION}" == "superpod" ]]; then
    QOS=${QOS:-qos_superpod}; GRES=${GRES:-gpu:h100:1}
else
    QOS=${QOS:-}; GRES=${GRES:-gpu:a100:1}
fi
DEVICE_MAP=${DEVICE_MAP:-cuda}

# Node pin. Every job of this study runs on sp-0006 so the whole cross-size
# panel is measured on one machine -- same GPUs, same driver, same host. That
# matters here because the read-out is compared ACROSS arms.
#
# The cost is real and is the same one the band fanout hit: with --nodelist,
# tasks beyond the node's 8 GPUs queue at `Reason=ReqNodeNotAvail, May be
# reserved for other job`, which looks like a cluster reservation problem but is
# self-inflicted serialization. That is acceptable for 24 short read-out tasks;
# it was not for a 13-way fanout of multi-hour fits. Pass NODE= (empty) to spread.
NODE=${NODE-sp-0006}

ARMS=("$@")
[[ ${#ARMS[@]} -eq 0 ]] && ARMS=(270m_it 1b_it 12b_it 27b_it)

cd "${REPO_ROOT}"
source .venv-jlens/bin/activate
mkdir -p "${OUT_ROOT}" slurm/logs

for ARM in "${ARMS[@]}"; do
    RUN_DIR=$(run_dir_for "${ARM}")
    echo "=== configs: ${ARM} (${RUN_DIR}) ==="
    python scripts/redflag/make_configs.py --arm "${ARM}" --run-dir "${RUN_DIR}" \
        --steps ${STEPS} --device-map "${DEVICE_MAP}"

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
        # before committing GPU-hours to the full grid.
        ARRAY="0-0"; EXPORTS="${EXPORTS},LIMIT=2"
        echo "SMOKE: 1 task, 2 vignettes"
    else
        ARRAY="0-$(( N_CFG * N_SHARDS - 1 ))"
    fi

    SB_FLAGS=(--partition="${PARTITION}" --gres="${GRES}" --mem="$(mem_for "${ARM}")"
              --job-name="rf_${ARM}")
    [[ -n "${QOS}" ]] && SB_FLAGS+=(--qos="${QOS}")
    [[ -n "${NODE}" ]] && SB_FLAGS+=(--nodelist="${NODE}")

    echo "=== ${ARM}: array ${ARRAY} (${N_CFG} ckpts x ${N_SHARDS} shards) on ${PARTITION} ==="
    sbatch "${SB_FLAGS[@]}" --array="${ARRAY}" --export="${EXPORTS}" \
        scripts/redflag/probe_redflag.sbatch
done
