#!/bin/bash
# Fan a workspace-band probe out to one Slurm job per checkpoint. Size-agnostic;
# the band sibling of fanout_jlens.sh, and deliberately the same shape.
#
# Serial cost is the LENS FIT, ~(d_model x params) -- measured 3.5 h/ckpt for 4b
# on an H100, ~15 h for 12b, ~47 h for 27b. The band metrics on top are minutes
# (tens of minutes for 27b's CKA). Every worker runs with SAVE_LENS=1 so this
# refit is paid exactly once: the lens lands in $DEST/ck<n>/lens.pt and any later
# lens-based analysis can point --lens at it instead of fitting again.
#
# Each worker gets its own OUT dir, because concurrent appends to one band.jsonl
# would interleave. One thing must be seeded there:
#   band.jsonl -- containing the arm's step-0 row, because probe_band.py's
#                 discover_checkpoints() ALWAYS prepends the base as step 0, so
#                 without a done_steps hit every worker would refit t=0.
#
# Prerequisites, in order:
#   1. (4b+ only) scripts/make_text_base.sbatch -> data/text_bases/<size>_<kind>
#   2. probe_band.sbatch with BASE_ONLY=1, OUT=$SRC  -> $SRC/band.jsonl step-0 row
#      (pass LENS=<jlens_out>/<size>_<kind>/.base_lens.pt to reuse the t=0 lens
#       the j-lens fanout already paid for -- that is free at 4b/12b/27b)
#   3. this script
#
#   SIZE=12b RUN_DIR=... SRC=... DEST=... scripts/fanout_band.sh
set -euo pipefail

REPO=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
: "${SIZE:?SIZE must be set (270m|1b|4b|12b|27b)}"
: "${RUN_DIR:?RUN_DIR must be set (dir containing checkpoint-*/)}"
KIND=${KIND:-it}
# On the project fileset, not /gpfs/home: a 27b arm's saved lenses are ~50 GB
# and the home quota will not hold them.
BAND_OUT=${BAND_OUT:-/gpfs/data/oermannlab/users/zhouj14/band_out}
SRC=${SRC:-$BAND_OUT/${SIZE}_${KIND}}
DEST=${DEST:-$BAND_OUT/${SIZE}_${KIND}_fanout}
PARTITION=${PARTITION:-superpod}
GRES=${GRES:-gpu:h100:1}
DRY_RUN=${DRY_RUN:-0}
SAVE_LENS=${SAVE_LENS:-1}
# Workers stay silent on Slack by default (66 of them across the grid); the
# per-arm notifier speaks instead. QUIET_WORKERS=0 restores the per-job pings.
QUIET_WORKERS=${QUIET_WORKERS:-1}
[[ $QUIET_WORKERS == 0 ]] && QUIET_WORKERS=""

# 4b+ must run against the pre-converted text-only base: transformers 5.14.1
# does NOT remap the multimodal base's nested keys onto Gemma3ForCausalLM and
# silently random-initializes instead, which would make t=0 meaningless.
if [[ -z "${BASE_MODEL:-}" && "$SIZE" != "270m" && "$SIZE" != "1b" ]]; then
    BASE_MODEL=$REPO/data/text_bases/${SIZE}_${KIND}
    [[ -d $BASE_MODEL ]] || {
        echo "FATAL: no text base at $BASE_MODEL (run make_text_base.sbatch)" >&2; exit 1; }
fi

BASE_ROW=$(grep -m1 '"step": 0,' "$SRC/band.jsonl") \
    || { echo "FATAL: no step-0 row in $SRC/band.jsonl (run BASE_ONLY=1 first)" >&2; exit 1; }

mkdir -p "$DEST"
n_sub=0
worker_ids=()
for ck in "$RUN_DIR"/checkpoint-*; do
    [[ -d $ck ]] || continue
    n=$(basename "$ck"); n=${n#checkpoint-}
    o=$DEST/ck$n
    if [[ -s $o/band.jsonl ]] && grep -q "\"step\": $n," "$o/band.jsonl" 2>/dev/null; then
        echo "skip ck$n (already probed)"
        continue
    fi
    if [[ $DRY_RUN == 1 ]]; then
        echo "would submit ck$n -> $o"
        n_sub=$((n_sub + 1))
        continue
    fi
    mkdir -p "$o"
    printf '%s\n' "$BASE_ROW" > "$o/band.jsonl"
    # QUIET_WORKERS: the per-job Slack ping is useful for a one-off probe and is
    # pure noise across a 66-worker grid, so workers stay silent by default and
    # notify_band.sbatch sends one message per ARM instead.
    id=$(sbatch --parsable --partition="$PARTITION" ${NODE:+--nodelist="$NODE"} --gres="$GRES" \
        --job-name="bd${SIZE}_$n" \
        --output="$DEST/bd${SIZE}_${n}_%j.out" --error="$DEST/bd${SIZE}_${n}_%j.err" \
        --export=ALL,SIZE="$SIZE",KIND="$KIND"${BASE_MODEL:+,BASE_MODEL="$BASE_MODEL"},CKPTS="$ck",OUT="$o",ARM="${SIZE}_${KIND}",SAVE_LENS="$SAVE_LENS",REQUEUE_ON_GPU_FAIL=0${QUIET_WORKERS:+,SLACK_WEBHOOK_URL=}${DIM_BATCH:+,DIM_BATCH=$DIM_BATCH}${CKA_TOKENS:+,CKA_TOKENS=$CKA_TOKENS} \
        "$REPO/scripts/probe_band.sbatch")
    worker_ids+=("$id")
    n_sub=$((n_sub + 1))
done
echo "submitted $n_sub workers -> $DEST"

MERGED=$REPO/eval/band/${SIZE}_${KIND}/band.jsonl
if [[ ${#worker_ids[@]} -gt 0 && $DRY_RUN != 1 ]]; then
    # afterany, not afterok: an arm that half-failed is exactly the case worth
    # being told about, and the notifier reports per-job states from sacct.
    dep=$(IFS=:; echo "${worker_ids[*]}")
    notify_id=$(sbatch --parsable --dependency=afterany:"$dep" \
        --job-name="bdnotify_${SIZE}" \
        --export=ALL,ARM="${SIZE}_${KIND}",FANOUT_DIR="$DEST",RUN_DIR="$RUN_DIR",OUT="$MERGED",JOBS="${worker_ids[*]}" \
        "$REPO/scripts/notify_band.sbatch")
    echo "merge + Slack on arm completion: job $notify_id (afterany on $n_sub workers)"
else
    echo "merge when done:  python scripts/merge_band_fanout.py --fanout-dir $DEST \\"
    echo "                      --expect-run-dir $RUN_DIR --out $MERGED"
fi
