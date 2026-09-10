#!/bin/bash
#SBATCH --job-name=bdwatch_12b
#SBATCH --partition=cpu_long
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
#SBATCH --output=/gpfs/data/oermannlab/users/zhouj14/band_out/bdwatch_12b_%j.out
#SBATCH --error=/gpfs/data/oermannlab/users/zhouj14/band_out/bdwatch_12b_%j.err
#
# Long-lived host for watch_band_12b.sh. A Slurm job rather than a login-node
# background process for the usual reason (an ssh-tethered loop dies when the
# session goes away), and cpu_long because the 12b arm is 10 steps x ~15 h that
# only trickle out as GPUs free -- this can legitimately run for days.
#
#   sbatch scripts/watch_band_12b.sbatch
#   sbatch --export=ALL,QUEUE_GATE=off scripts/watch_band_12b.sbatch
#   scancel <jobid>        # stopping the watcher does NOT touch the workers
set -euo pipefail

REPO=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft

_notify_slack() {
    [[ -z "${SLACK_WEBHOOK_URL:-}" ]] && return 0
    local code=$1 elapsed status color
    elapsed=$(( SECONDS / 60 ))
    if [[ $code -eq 0 ]]; then status="SUCCESS"; color="good"
    else                       status="FAILED (exit ${code})"; color="danger"; fi
    curl -s -X POST "${SLACK_WEBHOOK_URL}" -H 'Content-Type: application/json' -d @- <<JSON >/dev/null || true
{"attachments":[{"color":"${color}","title":"Slurm: ${SLURM_JOB_NAME} [12b band watcher] — ${status}","fields":[{"title":"Scope","value":"watcher loop only — submitted workers keep running","short":false},{"title":"Job ID","value":"${SLURM_JOB_ID}","short":true},{"title":"Partition","value":"${SLURM_JOB_PARTITION}","short":true},{"title":"Node","value":"$(hostname)","short":true},{"title":"Elapsed","value":"${elapsed}m","short":true}]}]}
JSON
    # Never let the notifier decide the job's exit status.
    return 0
}
trap '_notify_slack $?' EXIT

echo "=== 12b band watcher [job ${SLURM_JOB_ID}] ==="
echo "Host:  $(hostname)"
echo "Node:  ${NODE:-sp-0006}   Gate: ${QUEUE_GATE:-empty}   Poll: ${POLL:-120}s"
echo "=============================================="

# WATCHER_SH lets the loop be run from a deploy dir outside the git working
# copy, so an untracked copy on BigPurple cannot collide with a later `git pull`
# of the same path. Once the script is committed and pulled, the default is right.
exec "${WATCHER_SH:-$REPO/scripts/watch_band_12b.sh}"
