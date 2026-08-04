#!/bin/bash
#SBATCH --job-name=gemma_med_judge_dirs
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.out
#SBATCH --error=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.err
#
# judge_traj.sbatch's flat-directory sibling. That one takes a trajectory root
# and judges every step-*/ under it; this one takes an explicit list of eval
# dirs that are NOT trajectories -- baselines, single checkpoints, reference
# models -- and judges each with --pred-dir.
#
# Same PROVIDER toggle, same idempotence: judged items are keyed by benchmark
# index, so resubmitting after a timeout resumes rather than re-judging. Keep the
# judge frozen across everything you intend to compare (see CLAUDE.md) -- that is
# the whole reason this shares _judge_provider.sh with judge_traj.sbatch instead
# of resolving its own endpoint. A baseline judged by Claude and a trajectory
# judged by Kimi do not belong in one figure, and nothing downstream can tell
# except by reading judged_summary.json.
#
#   sbatch --export=ALL,DIRS="eval/gemma-3-1b-it-baseline eval/medgemma-4b-it" \
#       scripts/judge_dirs.sbatch
#   sbatch --export=ALL,PROVIDER=anthropic,DIRS="eval/medgemma-4b-it" \
#       scripts/judge_dirs.sbatch
#
# Dirs are space-separated, absolute or relative to the repo root.

set -euo pipefail

REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
cd "${REPO_ROOT}"

# --- Barney endpoint config -------------------------------------------------
# Option A (recommended): edit barney.env and set your NetID (JUDGE_API_KEY).
# The block below sources it; set -a exports the vars so kimi_url.sh and the
# judge subprocess inherit them. No-op if barney.env is absent — behavior
# unchanged.
set -a
[ -f barney.env ] && source barney.env
set +a
# Option B: skip barney.env and hardcode here instead — uncomment + edit:
# export KIMI_PORT=8001               # gateway (tracked) instead of raw serve (8000)
# export JUDGE_API_KEY=your_netid     # your NetID = who usage logs under
# CONC=20                             # concurrency (requests in flight)

: "${DIRS:?DIRS required (space-separated eval dirs holding *_predictions.jsonl)}"

_notify_slack() {
    [[ -z "${SLACK_WEBHOOK_URL:-}" ]] && return 0
    local code=$1 elapsed status color
    elapsed=$(( SECONDS / 60 ))
    if [[ $code -eq 0 ]]; then status="SUCCESS"; color="good"
    else                       status="FAILED (exit ${code})"; color="danger"; fi
    curl -s -X POST "${SLACK_WEBHOOK_URL}" \
        -H 'Content-Type: application/json' \
        -d @- <<JSON || true
{"attachments":[{"color":"${color}","title":"Slurm: judge-dirs (${PROVIDER:-?}) — ${status}","fields":[{"title":"Job ID","value":"${SLURM_JOB_ID}","short":true},{"title":"Partition","value":"${SLURM_JOB_PARTITION:-?}","short":true},{"title":"Node","value":"$(hostname)","short":true},{"title":"Elapsed","value":"${elapsed}m","short":true},{"title":"Dirs","value":"${DIRS}","short":false},{"title":"Judge","value":"${PROVIDER:-?}/${JUDGE_MODEL:-?} @ ${JUDGE_BASE_URL:-?}","short":false}]}]}
JSON
}
trap '_notify_slack $?' EXIT

source scripts/_judge_provider.sh

echo "=== judge-dirs ==="
echo "provider: ${PROVIDER} (${MODE})"
echo "endpoint: ${JUDGE_BASE_URL}"
echo "model:    ${JUDGE_MODEL}"

source .venv-judge/bin/activate
export PYTHONUNBUFFERED=1

# One --pred-dir invocation per dir rather than one process over all of them:
# a dir that fails (missing predictions, a gold-join mismatch) then costs only
# itself, and the rest still get judged.
rc=0
for d in ${DIRS}; do
    [[ "${d}" = /* ]] || d="${REPO_ROOT}/${d}"
    echo "--- ${d}"
    if [[ ! -d "${d}" ]]; then
        echo "  no such dir -- skipping" >&2
        rc=1
        continue
    fi
    judge_run --pred-dir "${d}" || { echo "  FAILED: ${d}" >&2; rc=1; }
done

echo "=== Finished: $(date -Is) (rc=${rc}) ==="
exit ${rc}
