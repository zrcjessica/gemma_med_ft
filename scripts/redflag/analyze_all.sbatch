#!/bin/bash
#SBATCH --job-name=rf_analyze
#SBATCH --partition=cpu_short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --output=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.out
#SBATCH --error=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.err
#
# Batch wrapper for analyze_all.sh. No GPU -- this is arithmetic over the npz the
# probe already wrote -- but it is NOT a login-node job: the per-run npz are
# 1.9 GB at 12b and 2.4 GB at 27b, and medlens.analysis holds several arrays in
# memory at once. Run interactively on the login node it gets OOM-killed at
# exit 137, partway through the sweep, leaving some runs analysed and some not.
#
# Env vars (all optional, passed through to analyze_all.sh):
#   RUNS     space-separated run names; default = every run with a judged.jsonl
#   OUT_ROOT redflag run root
#
#   sbatch scripts/redflag/analyze_all.sbatch
#   sbatch --export=ALL,RUNS="27b_it_ck0 27b_it_ck512" scripts/redflag/analyze_all.sbatch

set -euo pipefail

REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft

_notify_slack() {
    [[ -n "${QUIET_SLACK:-}" ]] && return 0
    [[ -z "${SLACK_WEBHOOK_URL:-}" ]] && return 0
    local code=$1 elapsed status color n
    elapsed=$(( SECONDS / 60 ))
    if [[ $code -eq 0 ]]; then status="SUCCESS"; color="good"
    else                       status="FAILED (exit ${code})"; color="danger"; fi
    n=$(ls -d "${OUT_ROOT:-/gpfs/data/oermannlab/users/zhouj14/redflag_out}"/*_ck*/analysis_all.json 2>/dev/null | wc -l) || n=0
    curl -s -X POST "${SLACK_WEBHOOK_URL}" -H 'Content-Type: application/json' -d @- <<JSON || true
{"attachments":[{"color":"${color}","title":"Slurm: ${SLURM_JOB_NAME} [redflag cross-size analysis] — ${status}","fields":[{"title":"Runs analysed","value":"${n}","short":true},{"title":"Job ID","value":"${SLURM_JOB_ID}","short":true},{"title":"Partition","value":"${SLURM_JOB_PARTITION}","short":true},{"title":"Node","value":"$(hostname)","short":true},{"title":"Elapsed","value":"${elapsed}m","short":true}]}]}
JSON
    return 0
}
trap '_notify_slack $?' EXIT

echo "=== redflag analysis [job ${SLURM_JOB_ID}] on $(hostname) ==="
cd "${REPO_ROOT}"
exec scripts/redflag/analyze_all.sh ${RUNS:-}
