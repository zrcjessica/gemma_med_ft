#!/bin/bash
# Fan a jlens trajectory probe out to one Slurm job per checkpoint, all reusing a
# t=0 base lens fitted once by a prior BASE_ONLY run. Size-agnostic
# generalization of fanout_jlens_4b.sh (kept for provenance of the 4b run).
#
# Serial cost is ~(d_model x params) -- measured 3.5h/ckpt for 4b on an H100,
# which scales to ~15h for 12b and ~47h for 27b. Fanning 13 checkpoints over
# sp-xxxx's 8 H100s is ~2 waves.
#
# Each worker gets its own OUT dir, because concurrent appends to one
# metrics.jsonl would interleave. Two things must be seeded there:
#   .base_lens.pt  -- symlinked to the shared fp32 lens (drift baseline)
#   metrics.jsonl  -- containing the step-0 row, for two reasons: probe_jlens.py
#                     discover_checkpoints() ALWAYS prepends the base as step 0,
#                     so without a done_steps hit every worker refits it; and a
#                     missing metrics.jsonl triggers the
#                     `base_lens_path.unlink()` branch, deleting the baseline.
#
# Prerequisites, in order:
#   1. scripts/make_text_base.sbatch      -> data/text_bases/<size>_<kind>
#   2. probe_jlens.sbatch with BASE_ONLY=1, OUT=$SRC   -> $SRC/.base_lens.pt
#   3. this script
#
#   SIZE=12b RUN_DIR=... SRC=... DEST=... scripts/fanout_jlens.sh
set -euo pipefail

REPO=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
: "${SIZE:?SIZE must be set (4b|12b|27b)}"
: "${RUN_DIR:?RUN_DIR must be set (dir containing checkpoint-*/)}"
KIND=${KIND:-it}
# On the project fileset, not /gpfs/home: a 27b base lens is 7 GB and the
# fanout keeps one per size, which the home quota will not hold.
JLENS_OUT=${JLENS_OUT:-/gpfs/data/oermannlab/users/zhouj14/jlens_out}
SRC=${SRC:-$JLENS_OUT/${SIZE}_${KIND}}
DEST=${DEST:-$JLENS_OUT/${SIZE}_${KIND}_fanout}
BASE=${BASE:-$REPO/data/text_bases/${SIZE}_${KIND}}
PARTITION=${PARTITION:-superpod}
GRES=${GRES:-gpu:h100:1}
DRY_RUN=${DRY_RUN:-0}

# Never submit without a walltime. probe_jlens.sbatch's own --time is sized for
# a direct small-arm submit; the fanout knows the size, so it sizes per arm from
# the measured fit rate (~3.5h/ckpt at 4b, ~15h at 12b, ~47h at 27b) with
# headroom. Same table as fanout_band.sh. Without this the site's job_submit lua
# stamps 365 days, which is un-backfillable on superpod and rejected outright
# (Reason=PartitionTimeLimit) on a capped partition like a100_short.
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

[[ -d $BASE ]] || { echo "FATAL: no text base at $BASE (run make_text_base.sbatch)" >&2; exit 1; }
BASE_LENS=$SRC/.base_lens.pt
[[ -f $BASE_LENS ]] || { echo "FATAL: no base lens at $BASE_LENS (run BASE_ONLY=1 first)" >&2; exit 1; }
STEP0=$(grep -m1 '"step": 0,' "$SRC/metrics.jsonl") \
    || { echo "FATAL: no step-0 row in $SRC/metrics.jsonl" >&2; exit 1; }

mkdir -p "$DEST"
n_sub=0
for ck in "$RUN_DIR"/checkpoint-*; do
    [[ -d $ck ]] || continue
    n=$(basename "$ck"); n=${n#checkpoint-}
    o=$DEST/ck$n
    if [[ -s $o/metrics.jsonl ]] && grep -q "\"step\": $n," "$o/metrics.jsonl" 2>/dev/null; then
        echo "skip ck$n (already probed)"
        continue
    fi
    if [[ $DRY_RUN == 1 ]]; then
        echo "would submit ck$n -> $o"
        n_sub=$((n_sub + 1))
        continue
    fi
    mkdir -p "$o"
    ln -sf "$BASE_LENS" "$o/.base_lens.pt"
    printf '%s\n' "$STEP0" > "$o/metrics.jsonl"
    sbatch --partition="$PARTITION" ${NODE:+--nodelist="$NODE"} --gres="$GRES" \
        --time="$WORKER_TIME" \
        --job-name="jl${SIZE}_$n" \
        --output="$DEST/jl${SIZE}_${n}_%j.out" --error="$DEST/jl${SIZE}_${n}_%j.err" \
        --export=ALL,SIZE="$SIZE",KIND="$KIND",BASE_MODEL="$BASE",CKPTS="$ck",OUT="$o",WANDB_DIR="$o",REQUEUE_ON_GPU_FAIL=0${DIM_BATCH:+,DIM_BATCH=$DIM_BATCH} \
        "$REPO/scripts/probe_jlens.sbatch"
    n_sub=$((n_sub + 1))
done
echo "submitted $n_sub workers -> $DEST"
echo "merge when done:  python scripts/merge_jlens_fanout.py --fanout-dir $DEST --out $REPO/eval/jlens/${SIZE}_${KIND}_full/metrics.jsonl"
