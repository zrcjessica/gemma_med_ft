#!/bin/bash
#SBATCH --job-name=gemma_med_eval
#SBATCH --partition=oermannlab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --output=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.out
#SBATCH --error=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.err
#
# Benchmark a checkpoint (or a base model) on the MedGemma text benchmarks.
#   sbatch --gres=gpu:a100:1 --export=ALL,MODEL=/path/to/final,LABEL=gemma3-1b-med scripts/eval_med.sbatch
#   sbatch --gres=gpu:a100:1 --export=ALL,SIZE=1b,BASELINE=1 scripts/eval_med.sbatch
#
# Always eval the matching base gemma-3-*-it: that's the number to beat.

set -euo pipefail

REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
export HF_HOME=${HF_HOME:-/gpfs/data/oermannlab/users/zhouj14/hf}

source "${REPO_ROOT}/scripts/_resolve_model.sh"
if [[ -n "${BASELINE:-}" ]]; then
    : "${SIZE:?SIZE required when BASELINE=1}"
    MODEL=$(resolve_gemma "${SIZE}" "${KIND:-it}") || {
        echo "ERROR: no gemma-3-${SIZE}-${KIND:-it} under: ${GEMMA_MODEL_ROOTS}" >&2
        exit 1
    }
    LABEL=${LABEL:-gemma-3-${SIZE}-it-baseline}
fi
: "${MODEL:?MODEL must be set (or BASELINE=1 with SIZE)}"
LABEL=${LABEL:-$(basename "$(dirname "${MODEL}")")}
OUT_DIR=${OUT_DIR:-${REPO_ROOT}/eval/${LABEL}}
TP=${TP:-1}
# Unset -> evaluate.py's frozen default (rp=1.1). Set it only to match an
# off-instrument trajectory, and suffix LABEL to match.
RP=${RP:-}

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
{"attachments":[{"color":"${color}","title":"Slurm: eval ${LABEL} — ${status}","fields":[{"title":"Job ID","value":"${SLURM_JOB_ID}","short":true},{"title":"Node","value":"$(hostname)","short":true},{"title":"Elapsed","value":"${elapsed}m","short":true},{"title":"Scores","value":"${summary:-n/a}","short":false}]}]}
JSON
}
trap '_notify_slack $?' EXIT

echo "=== eval [job ${SLURM_JOB_ID}] ==="
echo "Model: ${MODEL}"
echo "Label: ${LABEL}"
echo "Out:   ${OUT_DIR}"

cd "${REPO_ROOT}"
source .venv-eval/bin/activate
source scripts/_cc_guard.sh   # superpod is Ubuntu; the gcc module there can't link
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Benchmarks are pulled from the hub, so we do NOT set HF_HUB_OFFLINE here.

python -m gemma_med.evaluate \
    --model-path "${MODEL}" \
    --out "${OUT_DIR}" \
    --label "${LABEL}" \
    --tensor-parallel-size "${TP}" \
    ${RP:+--repetition-penalty ${RP}} \
    ${LIMIT:+--limit ${LIMIT}} \
    ${BENCHMARKS:+--benchmarks ${BENCHMARKS}}

echo "=== Finished: $(date -Is) ==="
