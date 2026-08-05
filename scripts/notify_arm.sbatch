#!/bin/bash
#SBATCH --job-name=gemma_med_notify_arm
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.out
#SBATCH --error=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.err
#
# One Slack message when a whole behavioral-eval arm is done -- every decode AND
# every judge for one model size -- rather than the per-job pings the eval/judge
# scripts already fire. Submit it held behind the arm with afterany on every job
# in it, so it reports whatever actually happened instead of only firing on
# success:
#
#   sbatch --dependency=afterany:$A1:$A2:$B1:$J1:$J2 \
#       --export=ALL,ARM=12b,JOBS=$A1,$A2,$B1,$J1,$J2,\
# TAGS="12b_it_full_rp11 12b_it_full_rp105",DIRS="eval/gemma-3-12b-it-baseline-rp11" \
#       scripts/notify_arm.sbatch
#
# afterany (not afterok) is deliberate: an arm that half-failed is exactly the
# case you want to be told about, and a notifier that only fires on success is
# indistinguishable from a notifier that never fired.

set -euo pipefail

REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
cd "${REPO_ROOT}"

: "${ARM:?ARM required (e.g. 12b)}"
: "${JOBS:?JOBS required (comma-separated job ids to report on)}"
TAGS=${TAGS:-}
DIRS=${DIRS:-}
EXPECT_STEPS=${EXPECT_STEPS:-13}

# sacct is the only record of jobs that have already left the queue.
export STATES_TEXT
STATES_TEXT=$(sacct -j "${JOBS}" -X -o JobID%16,JobName%22,State,Elapsed --noheader 2>/dev/null || echo "sacct unavailable")

python3 - "${ARM}" "${EXPECT_STEPS}" "${TAGS}" "${DIRS}" <<'PY'
import json, os, subprocess, sys, urllib.request

arm, expect, tags, dirs = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(), sys.argv[4].split()
states = os.environ.get("STATES_TEXT", "")
hook = os.environ.get("SLACK_WEBHOOK_URL", "")

def judged(path):
    """Headline numbers from a judged dir, or None if it was never judged."""
    p = os.path.join(path, "judged_summary.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p))["results"]
    return " | ".join(
        f"{k} {v['acc_raw']} ({v['answer_rate']}% ans)" for k, v in sorted(r.items())
    )

lines, ok = [], True

for tag in tags:
    root = os.path.join("eval/traj", tag)
    steps = sorted(
        (d for d in os.listdir(root) if d.startswith("step-")),
        key=lambda d: int(d.split("-")[1]),
    ) if os.path.isdir(root) else []
    decoded = [s for s in steps if os.path.exists(os.path.join(root, s, "summary.json"))]
    scored  = [s for s in steps if os.path.exists(os.path.join(root, s, "judged_summary.json"))]
    if len(decoded) < expect or len(scored) < expect:
        ok = False
    head = f"*{tag}* — decoded {len(decoded)}/{expect}, judged {len(scored)}/{expect}"
    lines.append(head)
    # First and last judged step: the two ends of the trajectory are what you
    # look at first, and printing all 13 would bury them.
    for s in ([scored[0], scored[-1]] if len(scored) > 1 else scored):
        lines.append(f"    {s}: {judged(os.path.join(root, s))}")

for d in dirs:
    j = judged(d)
    if j is None:
        ok = False
    lines.append(f"*{os.path.basename(d)}* (t=0) — {j or 'NOT JUDGED'}")

payload = {
    "attachments": [{
        "color": "good" if ok else "danger",
        "title": f"Behavioral eval arm COMPLETE: gemma-3-{arm} "
                 f"({'all decoded + judged' if ok else 'INCOMPLETE — see below'})",
        "fields": [
            {"title": "Results", "value": "\n".join(lines) or "n/a", "short": False},
            {"title": "Jobs", "value": f"```{states.strip()}```", "short": False},
        ],
    }]
}
print(json.dumps(payload, indent=2))
if not hook:
    print("SLACK_WEBHOOK_URL unset -- printed only", file=sys.stderr)
    sys.exit(0)
req = urllib.request.Request(
    hook, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req, timeout=20) as r:
    print("slack:", r.status)
PY
