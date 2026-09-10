#!/bin/bash
# Submit a whole band arm as ONE Slurm array job -- the array sibling of
# fanout_band.sh, which submits N independent jobs instead.
#
# Why an array: N independent jobs each carry their own priority and can be
# scheduled (or starved) separately, which is how the 12b arm ended up with 12
# checkpoints done and ck512 stranded through 10 resubmits. An array is a single
# queue entity with a throttle (%N), so the arm advances as a unit and the
# throttle -- not a hand-rolled watcher -- is what limits how much of a node we
# take.
#
# Cost is the LENS FIT, ~(d_model x params): measured 3.5 h/ckpt at 4b, ~13 h at
# 12b, ~47 h at 27b. Every task runs SAVE_LENS=1 so the refit is paid once.
#
# Prerequisites, in order:
#   1. (4b+ only) scripts/make_text_base.sbatch -> data/text_bases/<size>_<kind>
#   2. probe_band.sbatch BASE_ONLY=1 OUT=$SRC  -> $SRC/band.jsonl step-0 row.
#      Pass LENS=<jlens_out>/<size>_<kind>/.base_lens.pt to reuse the t=0 lens
#      the j-lens fanout already paid for. UNLIKE fanout_band.sh this script does
#      not require that row to exist yet -- tasks seed themselves at run time --
#      so pass BASE_JOB=<jobid> to chain the array afterok on it.
#   3. this script
#
#   SIZE=27b RUN_DIR=... NODE=sp-0006 QOS=qos_superpod scripts/array_band.sh
set -euo pipefail

REPO=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
: "${SIZE:?SIZE must be set (270m|1b|4b|12b|27b)}"
: "${RUN_DIR:?RUN_DIR must be set (dir containing checkpoint-*/)}"
KIND=${KIND:-it}
BAND_OUT=${BAND_OUT:-/gpfs/data/oermannlab/users/zhouj14/band_out}
SRC=${SRC:-$BAND_OUT/${SIZE}_${KIND}}
DEST=${DEST:-$BAND_OUT/${SIZE}_${KIND}_array}
PARTITION=${PARTITION:-superpod}
# Never omit this on superpod: the default QOS `normal` is priority 0 against
# everyone else's qos_superpod at 50000, so the job sits at Reason=Priority
# forever while H100s idle. Cost the 12b arm 10 attempts over 11 h.
QOS=${QOS:-qos_superpod}
GRES=${GRES:-gpu:h100:1}
SAVE_LENS=${SAVE_LENS:-1}
QUIET_WORKERS=${QUIET_WORKERS:-1}
[[ $QUIET_WORKERS == 0 ]] && QUIET_WORKERS=""
BASE_JOB=${BASE_JOB:-}
DRY_RUN=${DRY_RUN:-0}

# Concurrent tasks. Defaults to the 8 GPUs of one node; it is the ONLY thing
# bounding how much of a shared node the arm takes, so raising it past the node's
# GPU count just queues tasks that cannot start.
THROTTLE=${THROTTLE:-8}

# Same per-size walltimes as fanout_band.sh, and explicit for the same reason:
# superpod's default is UNLIMITED and an unlimited job cannot be backfilled --
# the scheduler cannot prove it finishes before a higher-priority job needs the
# node, so it leaves the node IDLE at Reason=Priority.
if [[ -z ${WORKER_TIME:-} ]]; then
    case "$SIZE" in
    270m) WORKER_TIME=06:00:00 ;;
    1b)   WORKER_TIME=12:00:00 ;;
    4b)   WORKER_TIME=1-00:00:00 ;;
    12b)  WORKER_TIME=3-00:00:00 ;;
    27b)  WORKER_TIME=7-00:00:00 ;;
    *)    WORKER_TIME=1-00:00:00 ;;
    esac
fi
if [[ -z ${WORKER_MEM:-} ]]; then
    case "$SIZE" in
    270m) WORKER_MEM=24G ;;
    1b)   WORKER_MEM=32G ;;
    4b)   WORKER_MEM=48G ;;
    12b)  WORKER_MEM=80G ;;
    27b)  WORKER_MEM=140G ;;
    *)    WORKER_MEM=96G ;;
    esac
fi
if [[ -z ${WORKER_CPUS:-} ]]; then
    case "$SIZE" in
    270m|1b) WORKER_CPUS=4 ;;
    *)       WORKER_CPUS=8 ;;
    esac
fi
# 27b needs both knobs, same values as run_band_grid.sh's arm spec so the array
# stays on one instrument with the rest of the grid: dim_batch 8 or the fit OOMs
# on top of 55 GB of weights, and the CKA is n_layers^2/2 matmuls at d=5376, so
# halve the token draw to keep it in tens of minutes rather than hours.
if [[ $SIZE == 27b ]]; then
    DIM_BATCH=${DIM_BATCH:-8}
    CKA_TOKENS=${CKA_TOKENS:-4096}
fi

# 4b+ must run against the pre-converted text-only base: transformers 5.14.1
# does NOT remap the multimodal base's nested keys onto Gemma3ForCausalLM and
# silently random-initializes instead, which would make t=0 meaningless.
if [[ -z "${BASE_MODEL:-}" && "$SIZE" != "270m" && "$SIZE" != "1b" ]]; then
    BASE_MODEL=$REPO/data/text_bases/${SIZE}_${KIND}
    [[ -d $BASE_MODEL ]] || {
        echo "FATAL: no text base at $BASE_MODEL (run make_text_base.sbatch)" >&2; exit 1; }
fi

mkdir -p "$DEST"
CKPT_LIST=$DEST/ckpt_list.txt
: > "$CKPT_LIST"
n=0
for ck in $(ls -d "$RUN_DIR"/checkpoint-* | sed 's/.*checkpoint-//' | sort -n | sed "s|^|$RUN_DIR/checkpoint-|"); do
    [[ -d $ck ]] || continue
    s=$(basename "$ck"); s=${s#checkpoint-}
    # Only undone checkpoints go in the list, so array indices stay contiguous
    # and a resubmit after a partial arm is just a shorter array.
    if [[ -s $DEST/ck$s/band.jsonl ]] && grep -q "\"step\": $s," "$DEST/ck$s/band.jsonl" 2>/dev/null; then
        echo "skip ck$s (already probed)"
        continue
    fi
    echo "$ck" >> "$CKPT_LIST"
    n=$((n + 1))
done
[[ $n -gt 0 ]] || { echo "nothing to do: every checkpoint already has a band row"; exit 0; }
echo "array of $n tasks (throttle %$THROTTLE) -> $DEST"
echo "  list: $CKPT_LIST"

if [[ $DRY_RUN == 1 ]]; then
    echo "DRY_RUN: would submit --array=0-$((n - 1))%$THROTTLE on $PARTITION/${QOS}${NODE:+ (node $NODE)}"
    echo "         $WORKER_TIME, $WORKER_MEM, $WORKER_CPUS cpu, ${DIM_BATCH:+dim_batch=$DIM_BATCH}"
    cat "$CKPT_LIST"
    exit 0
fi

aid=$(sbatch --parsable --partition="$PARTITION" --qos="$QOS" \
    ${NODE:+--nodelist="$NODE"} --gres="$GRES" \
    --array="0-$((n - 1))%$THROTTLE" \
    --time="$WORKER_TIME" --mem="$WORKER_MEM" --cpus-per-task="$WORKER_CPUS" \
    ${BASE_JOB:+--dependency=afterok:"$BASE_JOB"} \
    --job-name="bd${SIZE}arr" \
    --output="$DEST/bd${SIZE}_%A_%a.out" --error="$DEST/bd${SIZE}_%A_%a.err" \
    --export=ALL,SIZE="$SIZE",KIND="$KIND"${BASE_MODEL:+,BASE_MODEL="$BASE_MODEL"},CKPT_LIST="$CKPT_LIST",OUT_ROOT="$DEST",SEED_SRC="$SRC",ARM="${SIZE}_${KIND}",SAVE_LENS="$SAVE_LENS",REQUEUE_ON_GPU_FAIL=0${QUIET_WORKERS:+,QUIET_SLACK=1}${DIM_BATCH:+,DIM_BATCH=$DIM_BATCH}${CKA_TOKENS:+,CKA_TOKENS=$CKA_TOKENS} \
    "$REPO/scripts/probe_band.sbatch")
echo "array job: $aid"

# afterany on the array id waits for EVERY task, and fires even if some failed --
# a half-finished arm is exactly the case worth being told about.
MERGED=$REPO/eval/band/${SIZE}_${KIND}/band.jsonl
notify_id=$(sbatch --parsable --dependency=afterany:"$aid" \
    --job-name="bdnotify_${SIZE}" \
    --export=ALL,ARM="${SIZE}_${KIND}",FANOUT_DIR="$DEST",RUN_DIR="$RUN_DIR",OUT="$MERGED",JOBS="$aid" \
    "$REPO/scripts/notify_band.sbatch")
echo "merge + Slack on arm completion: job $notify_id (afterany on array $aid)"
