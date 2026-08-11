#!/bin/bash
# Launch the whole workspace-band grid: 5 sizes x every checkpoint, plus t=0.
#
# Per arm this submits two things:
#   1. a BASE_ONLY probe of t=0, which every fanout worker is later seeded from;
#   2. a dependent CPU job (afterok) that runs fanout_band.sh once (1) lands.
# The dependency replaces the poll-until-the-step-0-row-appears waiter the jlens
# fanout used -- same gate, no job burning a slot to watch a file.
#
# t=0 at 4b/12b/27b REUSES the base lens the jlens fanout already fitted
# (jlens_out/<size>_it/.base_lens.pt). Those were fitted on fit_corpus_v2.txt
# (verified: the base run logged `fit_corpus=1000 probes=193`, and v2 is 1000/193
# lines while v1 is 40/30), which is what every refit here uses -- so t=0 and the
# later steps stay on one instrument. If that ever stops being true, drop the
# LENS= and pay for the base fit instead; do NOT mix corpora on one trajectory.
#
# Cost is the lens fit, ~(d_model x params): ~3.5 h/ckpt at 4b, ~15 h at 12b,
# ~47 h at 27b. The full grid is ~860 GPU-h, dominated by 27b. Every worker runs
# SAVE_LENS=1, so the ~77 GB of saved lenses means this is paid exactly once.
#
#   scripts/run_band_grid.sh            # submit everything
#   DRY_RUN=1 scripts/run_band_grid.sh  # print what would be submitted
#   ARMS="270m 1b" scripts/run_band_grid.sh   # a subset
#
# PARTITION/GRES/NODE steer placement, and are passed through to the fanout
# workers so an arm's base probe and its 13 workers land in the same place:
#   # big arms on one dedicated 8-GPU node
#   ARMS="4b 12b 27b" NODE=sp-0006 scripts/run_band_grid.sh
#   # small arms wherever a GPU frees up first -- any partition, any vendor
#   ARMS="270m 1b" PARTITION=superpod,oermannlab,a100_short,a100_long \
#       GRES=gpu:1 scripts/run_band_grid.sh
set -euo pipefail

REPO=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
JLENS_OUT=/gpfs/data/oermannlab/users/zhouj14/jlens_out
BAND_OUT=${BAND_OUT:-/gpfs/data/oermannlab/users/zhouj14/band_out}
PARTITION=${PARTITION:-superpod}
GRES=${GRES:-gpu:h100:1}
DRY_RUN=${DRY_RUN:-0}
ARMS=${ARMS:-"270m 1b 4b 12b 27b"}
# Always give the base probe a finite walltime. superpod defaults to UNLIMITED,
# and an unlimited job is un-backfillable: the scheduler holds the node IDLE
# rather than start it (Reason=Priority), and a capped partition rejects it
# outright (Reason=PartitionTimeLimit). Base probes are metrics-only whenever a
# t=0 lens is reused, so this is generous. See fanout_band.sh for the worker
# side of the same rule.
TIME=${TIME:-08:00:00}

# size : run dir : base model ("" = resolve from the HF hub) : reusable t=0 lens
# : dim_batch (lens-fit memory knob) : cka tokens
arm_spec() {
    case "$1" in
    270m) echo "outputs/270m/full_lr1e-5_25799673||||" ;;
    1b)   echo "outputs/1b/full_lr1e-5_25716223||||" ;;
    4b)   echo "outputs/4b/full_lr1e-5_25911616|$REPO/data/text_bases/4b_it|$JLENS_OUT/4b_it/.base_lens.pt||" ;;
    12b)  echo "outputs/12b/full_lr1e-5_26041764|$REPO/data/text_bases/12b_it|$JLENS_OUT/12b_it/.base_lens.pt||" ;;
    # 27b: dim_batch 8 or the fit OOMs on top of 55 GB of weights; the CKA is
    # n_layers^2/2 matmuls at d=5376, so halve the token draw to keep it in
    # tens of minutes rather than hours.
    27b)  echo "outputs/27b/full_lr1e-5_25970073|$REPO/data/text_bases/27b_it|$JLENS_OUT/27b_it/.base_lens.pt|8|4096" ;;
    *)    echo "" ;;
    esac
}

# Lookup mode, so the merge command below is copy-pasteable.
if [[ "${1:-}" == "--print-run-dir" ]]; then
    spec=$(arm_spec "${2:?usage: --print-run-dir <size>}")
    [[ -n $spec ]] || { echo "unknown arm ${2}" >&2; exit 1; }
    IFS='|' read -r rd _ <<< "$spec"
    echo "$REPO/$rd"
    exit 0
fi

mkdir -p "$BAND_OUT"
for SIZE in $ARMS; do
    spec=$(arm_spec "$SIZE")
    [[ -n $spec ]] || { echo "FATAL: unknown arm $SIZE" >&2; exit 1; }
    IFS='|' read -r RUN_DIR BASE_MODEL LENS DIM_BATCH CKA_TOKENS <<< "$spec"
    RUN_DIR=$REPO/$RUN_DIR
    SRC=$BAND_OUT/${SIZE}_it
    n_ck=$(ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null | wc -l)

    [[ -d $RUN_DIR ]] || { echo "FATAL: no run dir $RUN_DIR" >&2; exit 1; }
    [[ -z $BASE_MODEL || -d $BASE_MODEL ]] || {
        echo "FATAL: no text base $BASE_MODEL (run make_text_base.sbatch)" >&2; exit 1; }
    [[ -z $LENS || -f $LENS ]] || { echo "FATAL: no base lens $LENS" >&2; exit 1; }

    echo "=== ${SIZE}_it: ${n_ck} checkpoints + t=0 ==="
    echo "    run_dir=$RUN_DIR"
    echo "    base=${BASE_MODEL:-<resolve from hub>}  t0_lens=${LENS:-<fit>}"
    if [[ $DRY_RUN == 1 ]]; then
        echo "    would submit: base probe -> $SRC, then fanout of $n_ck workers"
        continue
    fi

    base_export="ALL,SIZE=$SIZE,KIND=it,BASE_ONLY=1,OUT=$SRC,ARM=${SIZE}_it,SAVE_LENS=1,REQUEUE_ON_GPU_FAIL=0"
    [[ -n $BASE_MODEL ]] && base_export+=",BASE_MODEL=$BASE_MODEL"
    [[ -n $LENS ]] && base_export+=",LENS=$LENS"
    [[ -n $DIM_BATCH ]] && base_export+=",DIM_BATCH=$DIM_BATCH"
    [[ -n $CKA_TOKENS ]] && base_export+=",CKA_TOKENS=$CKA_TOKENS"

    base_id=$(sbatch --parsable --partition="$PARTITION" --gres="$GRES" \
        ${NODE:+--nodelist="$NODE"} ${TIME:+--time="$TIME"} \
        --job-name="bd${SIZE}_base" \
        --output="$BAND_OUT/bd${SIZE}_base_%j.out" \
        --error="$BAND_OUT/bd${SIZE}_base_%j.err" \
        --export="$base_export" "$REPO/scripts/probe_band.sbatch")
    echo "    base probe: job $base_id"

    fan_env="SIZE=$SIZE KIND=it RUN_DIR=$RUN_DIR SRC=$SRC PARTITION=$PARTITION GRES=$GRES"
    [[ -n ${NODE:-} ]] && fan_env+=" NODE=$NODE"
    [[ -n $BASE_MODEL ]] && fan_env+=" BASE_MODEL=$BASE_MODEL"
    [[ -n $DIM_BATCH ]] && fan_env+=" DIM_BATCH=$DIM_BATCH"
    [[ -n $CKA_TOKENS ]] && fan_env+=" CKA_TOKENS=$CKA_TOKENS"
    fan_id=$(sbatch --parsable --dependency=afterok:"$base_id" --kill-on-invalid-dep=yes \
        --partition=cpu_short --time=00:20:00 --cpus-per-task=1 --mem=2G \
        --job-name="bdfan_${SIZE}" \
        --output="$BAND_OUT/bdfan_${SIZE}_%j.out" \
        --error="$BAND_OUT/bdfan_${SIZE}_%j.err" \
        --wrap="env $fan_env $REPO/scripts/fanout_band.sh")
    echo "    fanout (after $base_id): job $fan_id -> $n_ck workers"
done

cat <<EOF

Submitted. Watch with:  squeue -u zhouj14 -o "%.10i %.14j %.8T %.10M %R"
Merge each arm when its workers finish:
  for s in $ARMS; do
    python scripts/merge_band_fanout.py \\
      --fanout-dir $BAND_OUT/\${s}_it_fanout \\
      --expect-run-dir \$(scripts/run_band_grid.sh --print-run-dir \$s) \\
      --out $REPO/eval/band/\${s}_it/band.jsonl
  done
Then plot:
  python scripts/plot_band.py --arm 270m --arm 1b --arm 4b --arm 12b --arm 27b \\
    $REPO/eval/band/{270m,1b,4b,12b,27b}_it/band.jsonl --out $REPO/figures/band
EOF
