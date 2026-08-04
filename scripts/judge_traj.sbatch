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
# Score a trajectory (gemma_med.judge). PROVIDER picks the judge:
#
#   sbatch --export=ALL,TAG=1b_it_full_v3 scripts/judge_traj.sbatch                    # local Kimi
#   sbatch --export=ALL,TAG=1b_it_full_v3,PROVIDER=anthropic scripts/judge_traj.sbatch # Claude, batched
#   sbatch --export=ALL,TAG=1b_it_full_v3,LIMIT=50 scripts/judge_traj.sbatch           # smoke test
#
# CPU-only on purpose: the judging is network-bound, either against a GPU server
# someone else is already paying for or against an API.
#
# PROVIDER=local must run on a *compute* node -- nothing outside the cluster can
# reach that server, not the login node, not olab1. PROVIDER=anthropic has no
# such constraint and is usually run on olab1 instead (scripts/judge.sh); it
# works here too, but needs .env, which is gitignored and does not arrive with a
# git pull.
#
# The two judges' scores are NOT comparable, and judging is resumable -- so
# re-running an already-judged trajectory under the other PROVIDER is refused
# rather than silently filling in the gaps. Judge a trajectory end to end with one.
#
# Idempotent: judged items are keyed by benchmark index, so resubmitting after a
# timeout resumes rather than re-judging.

set -euo pipefail

REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
cd "${REPO_ROOT}"

: "${TAG:?TAG required (eval/traj/<TAG>/)}"
TRAJ=${TRAJ:-${REPO_ROOT}/eval/traj/${TAG}}

_notify_slack() {
    [[ -z "${SLACK_WEBHOOK_URL:-}" ]] && return 0
    local code=$1 elapsed status color
    elapsed=$(( SECONDS / 60 ))
    if [[ $code -eq 0 ]]; then status="SUCCESS"; color="good"
    else                       status="FAILED (exit ${code})"; color="danger"; fi
    curl -s -X POST "${SLACK_WEBHOOK_URL}" \
        -H 'Content-Type: application/json' \
        -d @- <<JSON || true
{"attachments":[{"color":"${color}","title":"Slurm: judge ${TAG} (${PROVIDER:-?}) — ${status}","fields":[{"title":"Job ID","value":"${SLURM_JOB_ID}","short":true},{"title":"Partition","value":"${SLURM_JOB_PARTITION:-?}","short":true},{"title":"Node","value":"$(hostname)","short":true},{"title":"Elapsed","value":"${elapsed}m","short":true},{"title":"Judge","value":"${PROVIDER:-?}/${JUDGE_MODEL:-?} @ ${JUDGE_BASE_URL:-?}","short":false}]}]}
JSON
}
trap '_notify_slack $?' EXIT

source scripts/_judge_provider.sh

echo "=== judge ${TAG} ==="
echo "provider: ${PROVIDER} (${MODE})"
echo "endpoint: ${JUDGE_BASE_URL}"
echo "model:    ${JUDGE_MODEL}"
echo "traj:     ${TRAJ}"

source .venv-judge/bin/activate
export PYTHONUNBUFFERED=1

judge_run --root "${TRAJ}"

echo "=== Finished: $(date -Is) ==="
