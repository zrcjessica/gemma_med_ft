#!/bin/bash
# Drive the remaining 12b band fanout by hand, bypassing superpod's scheduler.
#
# SUPERSEDED (2026-08-13): the premise below was a misdiagnosis. We were at
# priority 0 because we never asked for the QOS -- `--qos=qos_superpod` (priority
# 50000) is opt-in, and our association has it. Submit with that flag and jobs
# schedule normally; there is no gridlock to bypass and no reason to poll a node
# by hand. Kept for the ck512 archaeology only. Don't build on it.
#
# Why this existed: superpod gridlocks. Our jobs run under QOS=normal (priority
# factor 0, total ~11,982) while everyone else is at ~50,400, so Slurm holds
# idle H100s in reserve for higher-priority pending work rather than giving them
# to us -- observed with 5 free GPUs on sp-0006 and 10 of our jobs stuck at
# Reason=Priority. Queue position is therefore useless to us; free-GPU count is
# the only signal that means anything.
#
# So: poll the NODE directly, submit ONE worker when a GPU is genuinely free,
# and -- the part that actually prevents gridlock -- verify it reaches RUNNING
# within START_GRACE. A pinned job that does not start is a job sitting in the
# queue making things worse, so it gets cancelled and retried later rather than
# left to jam.
#
#   sbatch scripts/watch_band_12b.sbatch                    # the normal way
#   DRY_RUN=1 scripts/watch_band_12b.sh                     # print, submit nothing
#   QUEUE_GATE=off scripts/watch_band_12b.sh                # don't wait for an empty queue
#   ONESHOT=1 scripts/watch_band_12b.sh                     # one cycle, then exit
set -uo pipefail

REPO=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
BAND_OUT=/gpfs/data/oermannlab/users/zhouj14/band_out
RUN_DIR=$REPO/outputs/12b/full_lr1e-5_26041764
BASE_MODEL=$REPO/data/text_bases/12b_it
DEST=$BAND_OUT/12b_it_fanout
MERGED=$REPO/eval/band/12b_it/band.jsonl
STATE=$DEST/.watcher_state

NODE=${NODE:-sp-0006}
PARTITION=${PARTITION:-superpod}
GRES=${GRES:-gpu:h100:1}
WORKER_TIME=${WORKER_TIME:-3-00:00:00}
WORKER_MEM=${WORKER_MEM:-80G}
WORKER_CPUS=${WORKER_CPUS:-8}
# Wait for other users' superpod queue to drain before submitting anything.
# 'empty' = no PENDING jobs from anyone else (what you asked for, since rsteel
# is cancelling); 'off' = submit as soon as a GPU is free. The start-verify
# below protects against gridlock either way, so 'off' is not reckless.
QUEUE_GATE=${QUEUE_GATE:-empty}
POLL=${POLL:-120}            # seconds between cycles
START_GRACE=${START_GRACE:-300}   # a submitted job must be RUNNING within this
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}   # per step, before giving up on it
MAX_INFLIGHT=${MAX_INFLIGHT:-8}   # never exceed the node's GPU count
DRY_RUN=${DRY_RUN:-0}
ONESHOT=${ONESHOT:-0}

# Workers already running from the original fanout; folded into the final
# notifier's dependency so the arm still merges exactly once.
SEED_JOBS=${SEED_JOBS:-"26297614 26297615 26297616"}

mkdir -p "$DEST"
touch "$STATE"

log() { echo "[$(date '+%F %T')] $*"; }

slack() {
    [[ -z "${SLACK_WEBHOOK_URL:-}" ]] && return 0
    local text=$1 color=${2:-good}
    curl -s -X POST "$SLACK_WEBHOOK_URL" -H 'Content-Type: application/json' -d @- <<JSON >/dev/null || true
{"attachments":[{"color":"${color}","title":"band 12b watcher","text":"${text}"}]}
JSON
    return 0
}

# --- state: attempts per step, and every job id we have submitted ------------
attempts_of() { grep -E "^attempt $1 " "$STATE" 2>/dev/null | tail -1 | awk '{print $3}'; }
bump_attempt() {
    local n=$1 c
    c=$(attempts_of "$n"); c=${c:-0}
    echo "attempt $n $(( c + 1 ))" >> "$STATE"
}
record_job() { echo "job $1" >> "$STATE"; }
all_jobs()   { grep -E '^job ' "$STATE" 2>/dev/null | awk '{print $2}' | sort -un | tr '\n' ' '; }

# --- cluster reads -----------------------------------------------------------
free_gpus() {
    local out cfg alloc st
    out=$(scontrol show node "$NODE" 2>/dev/null) || { echo 0; return; }
    st=$(grep -oP 'State=\K\S+' <<<"$out")
    case "$st" in
        *DOWN*|*DRAIN*|*FAIL*|*INVAL*) echo 0; return ;;
    esac
    cfg=$(grep -oP 'CfgTRES=\S*' <<<"$out" | grep -oP 'gres/gpu=\K[0-9]+' | head -1)
    alloc=$(grep -oP 'AllocTRES=\S*' <<<"$out" | grep -oP 'gres/gpu=\K[0-9]+' | head -1)
    echo $(( ${cfg:-0} - ${alloc:-0} ))
}

others_pending() { squeue -p "$PARTITION" -h -t PENDING -o '%u' 2>/dev/null | grep -cv "^${USER}$"; }

# our bd12b_* jobs, as "name state jobid elapsed_seconds"
my_jobs() {
    squeue -u "$USER" -h -o '%j %T %i %M' 2>/dev/null | awk '$1 ~ /^bd12b_/'
}

step_done()     { [[ -s $DEST/ck$1/band.jsonl ]] && grep -q "\"step\": $1," "$DEST/ck$1/band.jsonl" 2>/dev/null; }
step_inflight() { my_jobs | awk -v n="bd12b_$1" '$1==n' | grep -q .; }

submit_step() {
    # One `local` per variable, deliberately. Bash expands every word of a
    # `local` command BEFORE applying any of its assignments, so the one-liner
    # `local n=$1 ck=$RUN_DIR/checkpoint-$n` reads the *enclosing* n, not the
    # argument -- and the classify loop above leaves n=8 every cycle, because
    # checkpoint-8 sorts last under the glob. That silently sent every worker to
    # checkpoint-8 with a correct --job-name (which expands after the local
    # completes), so the logs said ck32 while the job refit ck8: five 12.8h
    # H100 fits of the same checkpoint, and nine steps left with no row.
    local n=$1
    local ck=$RUN_DIR/checkpoint-$n
    local o=$DEST/ck$n
    grep -q '"step": 0,' "$o/band.jsonl" 2>/dev/null || {
        log "  SKIP ck$n: no step-0 seed row in $o/band.jsonl"; return 1; }
    # Belt and braces: the export and the job name must name the same step.
    if [[ $(basename "$ck") != "checkpoint-$n" || $(basename "$o") != "ck$n" ]]; then
        log "  BUG: step $n resolved to $ck / $o -- refusing to submit"; return 1
    fi
    if [[ $DRY_RUN == 1 ]]; then
        log "  DRY_RUN: would submit ck$n -> $NODE (ckpt=$ck out=$o)"; return 0
    fi
    local id
    id=$(sbatch --parsable --partition="$PARTITION" --nodelist="$NODE" --gres="$GRES" \
        --time="$WORKER_TIME" --mem="$WORKER_MEM" --cpus-per-task="$WORKER_CPUS" \
        --job-name="bd12b_$n" \
        --output="$DEST/bd12b_${n}_%j.out" --error="$DEST/bd12b_${n}_%j.err" \
        --export=ALL,SIZE=12b,KIND=it,BASE_MODEL="$BASE_MODEL",CKPTS="$ck",OUT="$o",ARM=12b_it,SAVE_LENS=1,REQUEUE_ON_GPU_FAIL=0,QUIET_SLACK=1 \
        "$REPO/scripts/probe_band.sbatch") || { log "  sbatch FAILED for ck$n"; return 1; }
    record_job "$id"
    bump_attempt "$n"
    log "  submitted ck$n -> job $id (attempt $(attempts_of "$n")/$MAX_ATTEMPTS)"
    slack "submitted ck$n as job $id on $NODE"
    return 0
}

finish() {
    local ids dep
    ids=$(echo "$SEED_JOBS $(all_jobs)" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -un | tr '\n' ' ')
    log "all steps accounted for; wiring notifier over: $ids"
    if [[ $DRY_RUN == 1 ]]; then log "DRY_RUN: would submit notifier"; return 0; fi
    dep=$(echo "$ids" | tr -s ' ' | sed 's/ $//' | tr ' ' ':')
    local nid
    nid=$(sbatch --parsable --dependency=afterany:"$dep" \
        --job-name=bdnotify_12b \
        --export=ALL,ARM=12b_it,FANOUT_DIR="$DEST",RUN_DIR="$RUN_DIR",OUT="$MERGED",JOBS="$ids" \
        "$REPO/scripts/notify_band.sbatch") \
        && log "notifier: job $nid" \
        || log "notifier submit FAILED -- merge by hand with merge_band_fanout.py"
    slack "12b arm complete: every step has a row or exhausted its retries. Notifier $nid will merge."
}

log "watcher start: node=$NODE partition=$PARTITION gate=$QUEUE_GATE poll=${POLL}s dry_run=$DRY_RUN"
slack "watcher started on $NODE (gate=$QUEUE_GATE) -- will submit 12b steps one at a time as H100s free"

while :; do
    # 1. classify every checkpoint
    todo=() inflight=0 done_n=0 stuck=()
    for ck in "$RUN_DIR"/checkpoint-*; do
        [[ -d $ck ]] || continue
        n=$(basename "$ck"); n=${n#checkpoint-}
        if step_done "$n"; then done_n=$((done_n + 1)); continue; fi
        if step_inflight "$n"; then inflight=$((inflight + 1)); continue; fi
        a=$(attempts_of "$n"); a=${a:-0}
        if (( a >= MAX_ATTEMPTS )); then stuck+=("$n"); continue; fi
        todo+=("$n")
    done
    log "done=$done_n inflight=$inflight todo=${#todo[@]} stuck=${#stuck[@]} free_gpus=$(free_gpus) others_pending=$(others_pending)"
    [[ ${#stuck[@]} -gt 0 ]] && log "  giving up on steps: ${stuck[*]} (hit MAX_ATTEMPTS=$MAX_ATTEMPTS)"

    # 2. anything left?
    if [[ ${#todo[@]} -eq 0 && $inflight -eq 0 ]]; then finish; exit 0; fi

    # 3. gridlock guard -- cancel our own PENDING jobs that never started.
    #    This is the whole point: a pinned job that has not started is not
    #    "waiting its turn", it is clogging the queue we are trying to bypass.
    pending_now=0
    while read -r name state jid tm; do
        [[ -z ${state:-} ]] && continue
        [[ $state == PENDING ]] || continue
        # squeue %M for a PENDING job is 0:00, so age it from our own record
        sub=$(grep -E "^submitted $jid " "$STATE" 2>/dev/null | tail -1 | awk '{print $3}')
        if [[ -z ${sub:-} ]]; then echo "submitted $jid $(date +%s)" >> "$STATE"; sub=$(date +%s); fi
        age=$(( $(date +%s) - sub ))
        if (( age > START_GRACE )); then
            log "  GRIDLOCK: $name ($jid) pending ${age}s > ${START_GRACE}s -- cancelling"
            [[ $DRY_RUN == 1 ]] || scancel "$jid" 2>/dev/null
            slack "cancelled $name ($jid): did not start within ${START_GRACE}s, would have jammed the queue" warning
        else
            pending_now=$((pending_now + 1))
        fi
    done < <(my_jobs)

    # 4. one at a time: never submit while a previous submit is still proving itself
    if (( pending_now > 0 )); then
        log "  holding: $pending_now of our jobs still pending inside the start grace"
    elif [[ $QUEUE_GATE == empty ]] && (( $(others_pending) > 0 )); then
        log "  holding: QUEUE_GATE=empty and $(others_pending) other-user jobs pending"
    elif (( inflight >= MAX_INFLIGHT )); then
        log "  holding: $inflight in flight (MAX_INFLIGHT=$MAX_INFLIGHT)"
    elif (( $(free_gpus) < 1 )); then
        log "  holding: no free H100 on $NODE"
    elif [[ ${#todo[@]} -gt 0 ]]; then
        submit_step "${todo[0]}"
    fi

    [[ $ONESHOT == 1 ]] && { log "ONESHOT -- exiting"; exit 0; }
    sleep "$POLL"
done
