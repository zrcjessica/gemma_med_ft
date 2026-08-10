#!/bin/bash
#SBATCH --job-name=gemma_band_notify
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.out
#SBATCH --error=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.err
#
# One Slack message when a whole band ARM is done -- every checkpoint's lens fit
# and band pass for one model size -- rather than the per-job pings
# probe_band.sbatch already fires (71 of them across the full grid). Also does
# the merge, so the arm lands as a single eval/band/<arm>/band.jsonl ready to
# plot rather than 13 scattered worker dirs.
#
# fanout_band.sh submits this automatically with afterany on every worker it
# launched. afterany (not afterok) is deliberate, same reasoning as
# notify_arm.sbatch: an arm that half-failed is exactly the case you want to be
# told about, and a notifier that only fires on success is indistinguishable
# from one that never fired.
#
#   sbatch --dependency=afterany:$W1:$W2:... \
#       --export=ALL,ARM=27b_it,FANOUT_DIR=...,RUN_DIR=...,JOBS="$W1 $W2 ..." \
#       scripts/notify_band.sbatch

set -euo pipefail

REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
cd "${REPO_ROOT}"

: "${ARM:?ARM required (e.g. 27b_it)}"
: "${FANOUT_DIR:?FANOUT_DIR required}"
JOBS=${JOBS:-}
RUN_DIR=${RUN_DIR:-}
OUT=${OUT:-${REPO_ROOT}/eval/band/${ARM}/band.jsonl}

# Merge first, so the Slack message reports the merged artifact rather than a
# state that still needs a manual step to become real.
MERGE_LOG=$(python3 scripts/merge_band_fanout.py --fanout-dir "${FANOUT_DIR}" \
    ${RUN_DIR:+--expect-run-dir "${RUN_DIR}"} --out "${OUT}" 2>&1) || true
echo "${MERGE_LOG}"

# sacct is the only record of jobs that have already left the queue.
export STATES_TEXT MERGE_LOG
STATES_TEXT=$(sacct -j "$(echo ${JOBS} | tr ' ' ',')" -X -o JobID%16,JobName%18,State,Elapsed \
    --noheader 2>/dev/null || echo "sacct unavailable")

python3 - "${ARM}" "${OUT}" "${RUN_DIR}" <<'PY'
import json, os, sys, urllib.request

arm, out, run_dir = sys.argv[1], sys.argv[2], sys.argv[3]
states = os.environ.get("STATES_TEXT", "")
merge_log = os.environ.get("MERGE_LOG", "")
hook = os.environ.get("SLACK_WEBHOOK_URL", "")

rows = []
if os.path.exists(out):
    rows = [json.loads(l) for l in open(out) if l.strip()]
rows.sort(key=lambda r: r["step"])

expected = 0
if run_dir and os.path.isdir(run_dir):
    expected = len([d for d in os.listdir(run_dir) if d.startswith("checkpoint-")]) + 1

ok = bool(rows) and (expected == 0 or len(rows) >= expected)
lines = [f"*{arm}* — {len(rows)}/{expected or '?'} checkpoints (incl. t=0)"]

def band_str(r):
    lo, hi = r["band_depth"]
    return (f"layers {r['band'][0]}-{r['band'][1]} "
            f"({lo:.0%}-{hi:.0%} of depth, width {hi - lo:.0%})")

if rows:
    base = next((r for r in rows if r["step"] == 0), None)
    last = rows[-1]
    if base is not None:
        lines.append(f"    t=0 (untuned base): {band_str(base)}")
    if last is not base:
        lines.append(f"    step {last['step']}: {band_str(last)}")
    if base is not None and last is not base:
        # The whole point of the arm: did medical SFT move the band, and which
        # edge moved. Reported in depth because that is what compares across
        # sizes at all.
        d_lo = last["band_depth"][0] - base["band_depth"][0]
        d_hi = last["band_depth"][1] - base["band_depth"][1]
        lines.append(f"    shift vs base: onset {d_lo:+.1%}, offset {d_hi:+.1%}")
    missing = [l for l in merge_log.splitlines() if "WARNING" in l]
    lines.extend(f"    {m}" for m in missing)
    if missing:
        ok = False
else:
    lines.append("    NO ROWS -- merge produced nothing")

payload = {"attachments": [{
    "color": "good" if ok else "danger",
    "title": f"Band arm {'COMPLETE' if ok else 'INCOMPLETE'}: gemma-3-{arm}",
    "fields": [
        {"title": "Workspace band", "value": "\n".join(lines), "short": False},
        {"title": "Merged to", "value": f"`{out}`", "short": False},
        {"title": "Jobs", "value": f"```{states.strip()[:2500]}```", "short": False},
    ],
}]}
print(json.dumps(payload, indent=2))
if not hook:
    print("SLACK_WEBHOOK_URL unset -- printed only", file=sys.stderr)
    sys.exit(0)
req = urllib.request.Request(
    hook, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=20) as r:
    print("slack:", r.status)
PY
