#!/bin/bash
#SBATCH --job-name=gemma_med_judge_ab
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.out
#SBATCH --error=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.err
#
# A/B the judge's own decode config on the self-hosted Kimi (scripts/ab_judge_decode.py):
# four passes over the same rows, two with no sampling params (the pre-2026-08-05
# wire) and two at temperature=0 + seed=0 (the frozen config).
#
#   sbatch scripts/ab_judge_decode.sbatch
#   sbatch --export=ALL,N=100 scripts/ab_judge_decode.sbatch     # quicker
#
# Compute node, not the login node and not olab1: the server is reachable only
# from inside BigPurple. CPU-only -- the GPUs are the server's, not ours.
#
# This measures the *judge of record*. The Claude arm of the same experiment runs
# on olab1 (PYTHONPATH=. .venv-judge/bin/python scripts/ab_judge_decode.py) and is
# only a cross-check on the fallback provider.
#
# Writes eval/decode_probe/judge_ab_local/. That tree is probe output and may not
# be mixed into trajectory figures -- same rule as the decode probe.

set -euo pipefail

REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
cd "${REPO_ROOT}"

# barney.env ships working defaults; kimi_url.sh and _judge_provider.sh consume
# these. Same precedence as judge_traj.sbatch.
set -a
[ -f barney.local.env ] && source barney.local.env
[ -f barney.env ]       && source barney.env
set +a

PROVIDER=${PROVIDER:-local}
N=${N:-300}
PRED_DIR=${PRED_DIR:-eval/traj/4b_it_full/step-512}

_notify_slack() {
    [[ -z "${SLACK_WEBHOOK_URL:-}" ]] && return 0
    local code=$1 elapsed status color
    elapsed=$(( SECONDS / 60 ))
    if [[ $code -eq 0 ]]; then status="SUCCESS"; color="good"
    else                       status="FAILED (exit ${code})"; color="danger"; fi
    curl -s -X POST "${SLACK_WEBHOOK_URL}" \
        -H 'Content-Type: application/json' \
        -d @- <<JSON || true
{"attachments":[{"color":"${color}","title":"Slurm: judge decode A/B (${PROVIDER}) — ${status}","fields":[{"title":"Job ID","value":"${SLURM_JOB_ID}","short":true},{"title":"Partition","value":"${SLURM_JOB_PARTITION:-?}","short":true},{"title":"Node","value":"$(hostname)","short":true},{"title":"Elapsed","value":"${elapsed}m","short":true},{"title":"Judge","value":"${PROVIDER}/${JUDGE_MODEL:-?} @ ${JUDGE_BASE_URL:-?}","short":false},{"title":"Rows","value":"${N}/bench x 4 passes","short":true}]}]}
JSON
}
trap '_notify_slack $?' EXIT

# Discovers the moving endpoint and the served model id; exits with a useful
# message if the server's job is gone.
source scripts/_judge_provider.sh

echo "=== judge decode A/B ==="
echo "provider: ${PROVIDER}"
echo "endpoint: ${JUDGE_BASE_URL}"
echo "model:    ${JUDGE_MODEL}"
echo "rows:     ${N} per benchmark, 4 passes"

source .venv-judge/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="${REPO_ROOT}"

python scripts/ab_judge_decode.py \
    --provider "${PROVIDER}" \
    --model "${JUDGE_MODEL}" \
    --pred-dir "${PRED_DIR}" \
    --n "${N}" \
    --concurrency "${CONC}"

echo "=== Finished: $(date -Is) ==="
