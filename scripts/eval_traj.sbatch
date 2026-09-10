#!/bin/bash
#SBATCH --job-name=gemma_med_eval_traj
#SBATCH --partition=oermannlab
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%A_%a.err
#
# Behavioral counterpart to scripts/probe_jlens.sbatch: benchmark every
# checkpoint of a trajectory so the lens signal can be correlated against what
# the model actually does. One array task per checkpoint.
#
#   sbatch --array=0-13%4 --export=ALL,RUN_DIR=outputs/1b/full_lr1e-5_25716223,TAG=1b_it_full_v3 \
#       scripts/eval_traj.sbatch
#
# The partition/gres/node headers below are defaults; override them on the
# command line for the big arms, which need H100s:
#
#   sbatch --partition=superpod --nodelist=sp-0006 --gres=gpu:h100:2 --array=0-12%1 \
#       --export=ALL,RUN_DIR=outputs/27b/full_lr1e-5_25970073,TAG=27b_it_full_rp11,TP=2 \
#       scripts/eval_traj.sbatch
#
# t=0 is NOT included -- it is the base model; eval it once with
# `sbatch --export=ALL,SIZE=1b,BASELINE=1 scripts/eval_med.sbatch`.
# `final/` is skipped too: it duplicates the last-step checkpoint.

set -euo pipefail

REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
export HF_HOME=${HF_HOME:-/gpfs/data/oermannlab/users/zhouj14/hf}

: "${RUN_DIR:?RUN_DIR required (abs, or relative to repo root)}"
[[ "${RUN_DIR}" = /* ]] || RUN_DIR="${REPO_ROOT}/${RUN_DIR}"
TAG=${TAG:-$(basename "${RUN_DIR}")}
OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/eval/traj/${TAG}}
TP=${TP:-1}
# Unset -> evaluate.py's frozen default (rp=1.1 as of 2026-08-05). Set it only
# to run a deliberately off-instrument trajectory, and give that TAG a suffix:
# two repetition penalties are two instruments and may not share a figure.
RP=${RP:-}
# Gemma 3 declares a 131072 window; 12b/27b cannot fit its KV cache next to the
# weights on one 80GB card. Unset -> vLLM's default (what the small arms used).
MAX_LEN=${MAX_LEN:-}

# Checkpoints sorted by step, same ordering the probe uses, so array index N
# here and trajectory point N there refer to the same weights.
mapfile -t CKPTS < <(ls -d "${RUN_DIR}"/checkpoint-* | sed -E 's/.*checkpoint-([0-9]+)$/\1 &/' | sort -n | cut -d' ' -f2)
N=${#CKPTS[@]}
IDX=${SLURM_ARRAY_TASK_ID:-0}
if (( IDX >= N )); then
    echo "array index ${IDX} >= ${N} checkpoints -- nothing to do" >&2
    exit 0
fi
MODEL=${CKPTS[$IDX]}
STEP=$(basename "${MODEL}" | sed 's/checkpoint-//')
LABEL="${TAG}-step${STEP}"
OUT_DIR="${OUT_ROOT}/step-${STEP}"

_notify_slack() {
    [[ -z "${SLACK_WEBHOOK_URL:-}" ]] && return 0
    local code=$1 elapsed status color summary
    elapsed=$(( SECONDS / 60 ))
    if [[ $code -eq 0 ]]; then status="SUCCESS"; color="good"
    else                       status="FAILED (exit ${code})"; color="danger"; fi
    summary=""
    if [[ -f "${OUT_DIR}/summary.json" ]]; then
        summary=$(python -c "
import json
# evaluate.py only generates now; scoring is gemma_med.judge, which writes
# judged_summary.json later. Report whichever exists.
import os
p='${OUT_DIR}/judged_summary.json'
if os.path.exists(p):
    r=json.load(open(p))['results']
    print(' | '.join(f\"{k}: {v['acc_raw']} ({v['answer_rate']}% answered)\" for k,v in r.items()))
else:
    r=json.load(open('${OUT_DIR}/summary.json'))['results']
    print(' | '.join(f\"{k}: n={v['n']}\" for k,v in r.items())+' (unscored)')
" 2>/dev/null || echo "")
    fi
    curl -s -X POST "${SLACK_WEBHOOK_URL}" \
        -H 'Content-Type: application/json' \
        -d @- <<JSON || true
{"attachments":[{"color":"${color}","title":"Slurm: eval-traj ${LABEL} — ${status}","fields":[{"title":"Job ID","value":"${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${IDX}","short":true},{"title":"Partition","value":"${SLURM_JOB_PARTITION:-?}","short":true},{"title":"Node","value":"$(hostname)","short":true},{"title":"Elapsed","value":"${elapsed}m","short":true},{"title":"Scores","value":"${summary:-n/a}","short":false}]}]}
JSON
}
trap '_notify_slack $?' EXIT

echo "=== eval-traj [${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task ${IDX}/${N}] ==="
echo "Model: ${MODEL}"
echo "Step:  ${STEP}"
echo "Out:   ${OUT_DIR}"
echo "Decode: rp=${RP:-default(1.1)} tp=${TP}"

# Idempotent: a completed task is skipped so a partially-failed array can just
# be resubmitted.
if [[ -f "${OUT_DIR}/summary.json" && -z "${FORCE:-}" ]]; then
    echo "already done (${OUT_DIR}/summary.json) -- skipping; set FORCE=1 to redo"
    exit 0
fi

cd "${REPO_ROOT}"
source .venv-eval/bin/activate
source scripts/_cc_guard.sh   # superpod is Ubuntu; the gcc module there can't link
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Per-task torch-compile cache: co-scheduled array tasks otherwise race on the
# one shared cache dir and die with "corrupted compilation artifact".
export VLLM_CACHE_ROOT="${HOME}/.cache/vllm/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${IDX}"
export TORCHINDUCTOR_CACHE_DIR="${VLLM_CACHE_ROOT}/inductor"
# Benchmarks are pulled from the hub, so we do NOT set HF_HUB_OFFLINE here.

python -m gemma_med.evaluate \
    --model-path "${MODEL}" \
    --out "${OUT_DIR}" \
    --label "${LABEL}" \
    --tensor-parallel-size "${TP}" \
    ${RP:+--repetition-penalty ${RP}} \
    ${MAX_LEN:+--max-model-len ${MAX_LEN}} \
    ${LIMIT:+--limit ${LIMIT}} \
    ${BENCHMARKS:+--benchmarks ${BENCHMARKS}}

echo "=== Finished: $(date -Is) ==="
