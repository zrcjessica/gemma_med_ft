#!/bin/bash
#SBATCH --job-name=gemma_med_eval_traj
#SBATCH --partition=oermannlab
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
import json;r=json.load(open('${OUT_DIR}/summary.json'))['results']
print(' | '.join(f\"{k}: {v['accuracy']}\" for k,v in r.items()))
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

# Idempotent: a completed task is skipped so a partially-failed array can just
# be resubmitted.
if [[ -f "${OUT_DIR}/summary.json" && -z "${FORCE:-}" ]]; then
    echo "already done (${OUT_DIR}/summary.json) -- skipping; set FORCE=1 to redo"
    exit 0
fi

cd "${REPO_ROOT}"
source .venv-eval/bin/activate
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Benchmarks are pulled from the hub, so we do NOT set HF_HUB_OFFLINE here.

python -m gemma_med.evaluate \
    --model-path "${MODEL}" \
    --out "${OUT_DIR}" \
    --label "${LABEL}" \
    --tensor-parallel-size "${TP}" \
    ${LIMIT:+--limit ${LIMIT}} \
    ${BENCHMARKS:+--benchmarks ${BENCHMARKS}}

echo "=== Finished: $(date -Is) ==="
