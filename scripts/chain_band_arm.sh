#!/bin/bash
# Serialize one band arm behind another: submit $NEXT's base+fanout only once
# every worker of $AFTER has ended. For when the queue cannot hold the whole
# grid at once and arms have to take the node in turn.
#
# Why this needs a job of its own rather than one --dependency: fanout_band.sh
# returns as soon as it has SUBMITTED its workers, so anything hanging off
# bdfan_<arm> fires hours before that arm is actually done. The real
# "arm complete" gate is bdnotify_<arm> (afterany on every worker), whose job id
# does not exist until the fanout has run -- so this script runs right after the
# fanout, looks that id up, and wires the dependency then.
#
#   sbatch --dependency=afterok:<bdfan_4b id> --kill-on-invalid-dep=yes \
#       --partition=cpu_short --time=00:10:00 --mem=2G --cpus-per-task=1 \
#       --job-name=bdchain_12b \
#       --export=ALL,AFTER=4b,NEXT=12b,NODE=sp-0006 \
#       scripts/chain_band_arm.sh
#
# PARTITION/GRES/NODE/TIME pass straight through to run_band_grid.sh, so the
# next arm lands wherever you say. Chain further arms by submitting another of
# these with --dependency=afterok on the fanout this one creates.
set -euo pipefail

REPO=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
: "${AFTER:?AFTER must be set (arm to wait on, e.g. 4b)}"
: "${NEXT:?NEXT must be set (arm to submit, e.g. 12b)}"

gate=$(squeue -u "$USER" -h -n "bdnotify_${AFTER}" -o "%i" | head -1)
if [[ -n $gate ]]; then
    echo "gating ${NEXT} on bdnotify_${AFTER} (job $gate)"
    export DEPEND="afterany:$gate"
else
    # No notifier means fanout_band.sh submitted no workers -- every checkpoint
    # was already probed -- so there is nothing left to wait for.
    echo "no bdnotify_${AFTER} queued: ${AFTER} has nothing outstanding, submitting ${NEXT} now"
fi

export ARMS=$NEXT
exec "$REPO/scripts/run_band_grid.sh"
