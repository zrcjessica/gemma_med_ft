#!/bin/bash
# Fan the 4b jlens probe out to one Slurm job per checkpoint, all reusing the
# t=0 base lens fitted by the serial run. Serial is ~3.5h/checkpoint x 14 = ~50h;
# 13 workers over sp-0006's 8 H100s is ~2 waves, ~7h.
#
# Each worker gets its own OUT dir, because concurrent appends to one
# metrics.jsonl would interleave. Two things must be seeded there:
#   .base_lens.pt  -- symlinked to the shared fp32 lens (drift baseline)
#   metrics.jsonl  -- containing the step-0 row, for two reasons: probe_jlens.py
#                     discover_checkpoints() ALWAYS prepends the base as step 0,
#                     so without a done_steps hit every worker refits it (+3.5h
#                     each); and a missing metrics.jsonl triggers the
#                     `base_lens_path.unlink()` branch, deleting the baseline.
set -euo pipefail

REPO=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
SRC=${SRC:-/gpfs/home/zhouj14/jlens_out/4b_it}
DEST=${DEST:-/gpfs/home/zhouj14/jlens_out/4b_it_fanout}
RUN_DIR=${RUN_DIR:-$REPO/outputs/4b/full_lr1e-5_25911616}
BASE=${BASE:-$REPO/data/text_bases/4b_it}
NODE=${NODE:-sp-0006}

BASE_LENS=$SRC/.base_lens.pt
[[ -f $BASE_LENS ]] || { echo "FATAL: no base lens at $BASE_LENS" >&2; exit 1; }
STEP0=$(grep -m1 '"step": 0,' "$SRC/metrics.jsonl") \
    || { echo "FATAL: no step-0 row in $SRC/metrics.jsonl" >&2; exit 1; }

mkdir -p "$DEST"
n_sub=0
for ck in "$RUN_DIR"/checkpoint-*; do
    [[ -d $ck ]] || continue
    n=$(basename "$ck"); n=${n#checkpoint-}
    o=$DEST/ck$n
    mkdir -p "$o"
    ln -sf "$BASE_LENS" "$o/.base_lens.pt"
    printf '%s\n' "$STEP0" > "$o/metrics.jsonl"
    sbatch --partition=superpod --nodelist="$NODE" --gres=gpu:h100:1 \
        --job-name="jl4b_$n" \
        --output="$DEST/jl4b_${n}_%j.out" --error="$DEST/jl4b_${n}_%j.err" \
        --export=ALL,SIZE=4b,KIND=it,BASE_MODEL="$BASE",CKPTS="$ck",OUT="$o",WANDB_DIR="$o",REQUEUE_ON_GPU_FAIL=0 \
        "$REPO/scripts/probe_jlens.sbatch"
    n_sub=$((n_sub + 1))
done
echo "submitted $n_sub workers -> $DEST"
