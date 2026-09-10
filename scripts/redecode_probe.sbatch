#!/bin/bash
#SBATCH --job-name=gemma_med_redecode
#SBATCH --partition=oermannlab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --output=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%A_%a.err
#
# Is the fine-tuned models' answer-rate shortfall a *decoding* artefact?
#
# docs/BEHAVIORAL_EVAL.md §8: ~95% of unanswered rows are verbatim repetition
# loops that run to the token cap, not honest long CoT that got truncated. Loops
# are the classic pathology of greedy decoding, and our eval is greedy with no
# repetition penalty. This re-decodes ONE checkpoint on a fixed slice under four
# decode settings, holding model / prompt / items / judge fixed, so any change in
# answer rate is attributable to decoding alone.
#
#   sbatch --array=0-3 scripts/redecode_probe.sbatch
#   sbatch --array=0-3 --export=ALL,CKPT=<abs ckpt>,TAG=4b_step4882 scripts/redecode_probe.sbatch
#
# Then judge on olab1 (small: 4 variants x 2 benches x LIMIT items):
#   .venv-judge/bin/python -m gemma_med.judge --root eval/decode_probe/<TAG> --mode sync
#   .venv-probe/bin/python scripts/analyze_unanswered.py eval/decode_probe/<TAG>/*
#
# The control arm (idx 0) is not redundant: it re-runs the decode settings the
# corpus was produced under on the same slice, so the other three are read
# against a number produced by this job, not against the full-set trajectory row.
#
# NB (2026-08-05): that control, `t0.0_rp1.00`, is no longer the frozen config --
# this probe is what moved it to rp=1.1. Arm idx 2 (`t0.0_rp1.10`) is now the
# frozen setting and idx 0 is the historical one, kept because the pre-2026-08-05
# corpus was decoded under it. Adjust which arm you read as "control" accordingly;
# the arm list is deliberately unchanged so old and new probe runs stay comparable.

set -euo pipefail

REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
export HF_HOME=${HF_HOME:-/gpfs/data/oermannlab/users/zhouj14/hf}

# step-512 of the 4b-it full FT: the worst point of the trajectory
# (21.3% of all MedMCQA rows end in a loop at the cap, answer rate 89.2%).
CKPT=${CKPT:-${REPO_ROOT}/outputs/4b/full_lr1e-5_25911616/checkpoint-512}
[[ "${CKPT}" = /* ]] || CKPT="${REPO_ROOT}/${CKPT}"
TAG=${TAG:-4b_step512}
OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/eval/decode_probe/${TAG}}
# MedQA + MedMCQA only: PubMedQA answers ~99.5% under every arm, so it has no
# headroom to detect a change.
BENCHMARKS=${BENCHMARKS:-medqa medmcqa}
LIMIT=${LIMIT:-300}
TP=${TP:-1}

# name                temperature  repetition_penalty
VARIANTS=(
    "t0.0_rp1.00      0.0          1.00"
    "t0.0_rp1.05      0.0          1.05"
    "t0.0_rp1.10      0.0          1.10"
    "t0.7_rp1.00      0.7          1.00"
)

IDX=${SLURM_ARRAY_TASK_ID:-0}
if (( IDX >= ${#VARIANTS[@]} )); then
    echo "task ${IDX} >= ${#VARIANTS[@]} variants -- nothing to do"
    exit 0
fi
read -r VNAME TEMP RP <<<"${VARIANTS[$IDX]}"
LABEL="${TAG}-${VNAME}"
OUT_DIR="${OUT_ROOT}/${VNAME}"

_notify_slack() {
    [[ -z "${SLACK_WEBHOOK_URL:-}" ]] && return 0
    local code=$1 elapsed status color summary
    elapsed=$(( SECONDS / 60 ))
    if [[ $code -eq 0 ]]; then status="SUCCESS"; color="good"
    else                       status="FAILED (exit ${code})"; color="danger"; fi
    summary=""
    if [[ -f "${OUT_DIR}/summary.json" ]]; then
        # Unjudged at this point, so report the one thing this job is about:
        # how many generations ran to the token cap.
        summary=$(python -c "
import json
r=json.load(open('${OUT_DIR}/summary.json'))['results']
print(' | '.join(f\"{k}: {v['n_at_token_cap']}/{v['n']} at cap\" for k,v in r.items()))
" 2>/dev/null || echo "")
    fi
    curl -s -X POST "${SLACK_WEBHOOK_URL}" \
        -H 'Content-Type: application/json' \
        -d @- <<JSON || true
{"attachments":[{"color":"${color}","title":"Slurm: redecode ${LABEL} — ${status}","fields":[{"title":"Job ID","value":"${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${IDX}","short":true},{"title":"Partition","value":"${SLURM_JOB_PARTITION:-?}","short":true},{"title":"Node","value":"$(hostname)","short":true},{"title":"Elapsed","value":"${elapsed}m","short":true},{"title":"At cap","value":"${summary:-n/a}","short":false}]}]}
JSON
}
trap '_notify_slack $?' EXIT

echo "=== redecode [${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task ${IDX}] ==="
echo "Ckpt:    ${CKPT}"
echo "Variant: ${VNAME}  (temperature=${TEMP}, repetition_penalty=${RP})"
echo "Items:   ${LIMIT} per benchmark of [${BENCHMARKS}]"
echo "Out:     ${OUT_DIR}"

if [[ -f "${OUT_DIR}/summary.json" && -z "${FORCE:-}" ]]; then
    echo "already done (${OUT_DIR}/summary.json) -- skipping; set FORCE=1 to redo"
    exit 0
fi

cd "${REPO_ROOT}"
source .venv-eval/bin/activate
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Array tasks land on the same node, and vLLM's torch-compile cache is a single
# shared directory: concurrent tasks race writing it and all but one die with
# "vLLM failed to compile the model ... corrupted compilation artifact". Killed
# 3 of 4 on the first submission. One cache root per task.
export VLLM_CACHE_ROOT="${HOME}/.cache/vllm/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${IDX}"
export TORCHINDUCTOR_CACHE_DIR="${VLLM_CACHE_ROOT}/inductor"

python -m gemma_med.evaluate \
    --model-path "${CKPT}" \
    --out "${OUT_DIR}" \
    --label "${LABEL}" \
    --tensor-parallel-size "${TP}" \
    --temperature "${TEMP}" \
    --repetition-penalty "${RP}" \
    --seed 0 \
    --limit "${LIMIT}" \
    --benchmarks ${BENCHMARKS}

echo "=== Finished: $(date -Is) ==="
