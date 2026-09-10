#!/bin/bash
# Two holes that block a clean cross-size red-flag panel. Both are cheap.
#
# 1. 12b_it has a banked lens at step 512 but NO band row for it -- the fanout
#    dropped that step. Without the row, medlens's analysis falls back to the
#    paper-relative 38-92% of the layer axis, which is a DIFFERENT band from the
#    one every other arm is analysed in, at the one step the whole panel is
#    aligned on. Reusing the banked lens makes this a band-metrics pass, not a
#    12.8 h refit.
#
# 2. The 270m_it and 1b_it t=0 lenses were banked in fp16; every other lens in
#    the study is fp32. That precision split sits exactly at the trajectory
#    origin, which is the reference point for "does medical SFT move the
#    dissociation" -- so the two smallest arms would compare an fp16 t=0 against
#    fp32 finetuned steps. Refitting t=0 in fp32 costs 0.35 h (270m) and 0.96 h
#    (1b), which is cheaper than defending the caveat.
#
#    Refits land in a STAGING dir (eval/band/<arm>_fp32base), never on top of
#    eval/band/<arm>/band.jsonl: if the fp32 band disagrees with the fp16 one,
#    that is a finding about existing figures, not something to overwrite in
#    place. Compare, then adopt deliberately.
#
#   scripts/redflag/fill_band_gaps.sh          # submit all three
#   scripts/redflag/fill_band_gaps.sh 12b      # just the 12b band row
set -euo pipefail

REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
BAND_OUT=${BAND_OUT:-/gpfs/data/oermannlab/users/zhouj14/band_out}

# probe_band.sbatch bakes in --partition=a100_short, which this association may
# not use at all (no qos_a100_short) -- so ALWAYS override the partition here.
PARTITION=${PARTITION:-superpod}
if [[ "${PARTITION}" == "superpod" ]]; then
    QOS=${QOS:-qos_superpod}; GPU=${GPU:-h100}
else
    QOS=${QOS:-}; GPU=${GPU:-a100}
fi
# Same node as the red-flag read-outs (see run_arm.sh): the whole study is
# measured on sp-0006. Pass NODE= (empty) to let Slurm place them.
NODE=${NODE-sp-0006}

_sb() {  # _sb <gres-count> <mem> <time> <jobname> <exports...>
    local n=$1 mem=$2 time=$3 name=$4; shift 4
    local flags=(--partition="${PARTITION}" --gres="gpu:${GPU}:${n}" --mem="${mem}"
                 --time="${time}" --job-name="${name}")
    [[ -n "${QOS}" ]] && flags+=(--qos="${QOS}")
    [[ -n "${NODE}" ]] && flags+=(--nodelist="${NODE}")
    sbatch "${flags[@]}" --export="$1" "${REPO_ROOT}/scripts/probe_band.sbatch"
}

cd "${REPO_ROOT}"
WHICH=("$@"); [[ ${#WHICH[@]} -eq 0 ]] && WHICH=(12b 270m 1b)

for w in "${WHICH[@]}"; do
case "${w}" in

12b)
    OUT=${REPO_ROOT}/eval/band/12b_it_ck512
    LENS=${BAND_OUT}/12b_it_fanout/ck512/ck512/lens.pt
    [[ -f "${LENS}" ]] || { echo "ERROR: no banked lens at ${LENS}" >&2; exit 1; }
    mkdir -p "${OUT}"
    # discover_checkpoints() ALWAYS prepends the base as step 0, and --lens names
    # the lens for the SINGLE checkpoint being probed -- so without seeding the
    # step-0 row this job would refit t=0 at 12b (12.8 h) and might pair step
    # 512's lens with the base. Seed it from the arm's own band.jsonl.
    if ! grep -q '"step": 0,' "${OUT}/band.jsonl" 2>/dev/null; then
        row=$(grep -m1 '"step": 0,' "${REPO_ROOT}/eval/band/12b_it/band.jsonl") \
            || { echo "ERROR: no step-0 row to seed from" >&2; exit 1; }
        printf '%s\n' "${row}" > "${OUT}/band.jsonl"
    fi
    echo "=== 12b_it step 512 band row (lens reused, no fit) ==="
    _sb 1 96G 08:00:00 bd12b_ck512 \
        "ALL,SIZE=12b,KIND=it,CKPTS=${REPO_ROOT}/outputs/12b/full_lr1e-5_26041764/checkpoint-512,LENS=${LENS},OUT=${OUT}"
    ;;

270m|1b)
    OUT=${REPO_ROOT}/eval/band/${w}_it_fp32base
    mkdir -p "${OUT}"
    # BASE_ONLY=1 probes t=0 and nothing else; SAVE_LENS=1 banks the fit
    # (probe_band.py's --lens-dtype already defaults to float32). Fresh OUT so
    # the existing step-0 row does not make it a no-op.
    case "${w}" in 270m) T=06:00:00; M=64G ;; 1b) T=10:00:00; M=64G ;; esac
    echo "=== ${w}_it t=0 lens refit in fp32 -> ${OUT} ==="
    _sb 1 "${M}" "${T}" "bd${w}_fp32base" \
        "ALL,SIZE=${w},KIND=it,BASE_ONLY=1,SAVE_LENS=1,OUT=${OUT}"
    ;;

*) echo "unknown gap: ${w}" >&2; exit 1 ;;
esac
done
