#!/bin/bash
#SBATCH --job-name=gemma_med_judge
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.out
#SBATCH --error=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.err
#
# Score a trajectory with the lab's self-hosted Kimi (gemma_med.judge).
#
# CPU-only on purpose: the judging is network-bound against a GPU server someone
# else is already paying for. It is on a *compute* node because nothing outside
# the cluster can reach that server -- not the login node, not olab1.
#
#   sbatch --export=ALL,TAG=1b_it_full_v3 scripts/judge_traj.sbatch
#   sbatch --export=ALL,TAG=1b_it_full_v3,LIMIT=50 scripts/judge_traj.sbatch   # smoke test
#
# Idempotent: judged items are keyed by benchmark index, so resubmitting after a
# timeout resumes rather than re-judging.

set -euo pipefail

REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
cd "${REPO_ROOT}"

: "${TAG:?TAG required (eval/traj/<TAG>/)}"
TRAJ=${TRAJ:-${REPO_ROOT}/eval/traj/${TAG}}
CONC=${CONC:-16}

_notify_slack() {
    [[ -z "${SLACK_WEBHOOK_URL:-}" ]] && return 0
    local code=$1 elapsed status color
    elapsed=$(( SECONDS / 60 ))
    if [[ $code -eq 0 ]]; then status="SUCCESS"; color="good"
    else                       status="FAILED (exit ${code})"; color="danger"; fi
    curl -s -X POST "${SLACK_WEBHOOK_URL}" \
        -H 'Content-Type: application/json' \
        -d @- <<JSON || true
{"attachments":[{"color":"${color}","title":"Slurm: judge ${TAG} — ${status}","fields":[{"title":"Job ID","value":"${SLURM_JOB_ID}","short":true},{"title":"Partition","value":"${SLURM_JOB_PARTITION:-?}","short":true},{"title":"Node","value":"$(hostname)","short":true},{"title":"Elapsed","value":"${elapsed}m","short":true},{"title":"Judge","value":"${JUDGE_MODEL:-?} @ ${JUDGE_BASE_URL:-?}","short":false}]}]}
JSON
}
trap '_notify_slack $?' EXIT

# Discover, never hardcode: the server's node changes every time its job
# restarts, and a stale address fails as thousands of connection errors.
if [[ -z "${JUDGE_BASE_URL:-}" ]]; then
    JUDGE_BASE_URL=$(bash scripts/kimi_url.sh 2>/dev/null)
    [[ -z "${JUDGE_BASE_URL}" ]] && { echo "no live Kimi server found" >&2; exit 1; }
fi
export JUDGE_BASE_URL
# The served model id comes from --served-model-name and is NOT the HF name.
JUDGE_MODEL=${JUDGE_MODEL:-$(curl -s --max-time 10 "${JUDGE_BASE_URL}/models" \
    -H 'Authorization: Bearer dummy' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')}

echo "=== judge ${TAG} ==="
echo "endpoint: ${JUDGE_BASE_URL}"
echo "model:    ${JUDGE_MODEL}"
echo "traj:     ${TRAJ}"

source .venv-judge/bin/activate
export PYTHONUNBUFFERED=1

python -m gemma_med.judge \
    --root "${TRAJ}" \
    --provider local \
    --model "${JUDGE_MODEL}" \
    --mode sync \
    --concurrency "${CONC}" \
    ${LIMIT:+--limit ${LIMIT}} \
    ${BENCHMARKS:+--benchmarks ${BENCHMARKS}}

echo "=== Finished: $(date -Is) ==="
