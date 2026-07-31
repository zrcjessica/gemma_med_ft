#!/bin/bash
# Print the base URL of the lab's running Kimi server, e.g.
#     http://sp-0005:8000/v1
#
# The server is somebody's Slurm job, so its node is not stable: the address
# that worked last week points at a stranger's pretraining job today, and the
# only symptom is every judgment failing with a connection error. Always
# discover; never hardcode.
#
# Two further facts this encodes, both learned the hard way (2026-07-30):
#   - Only the *head* node of a multi-node server answers. A job on sp-[0005,0008]
#     serves on sp-0005; sp-0008 refuses the connection.
#   - Nothing outside the cluster can reach it -- not the login node, not olab1.
#     Callers must run on a compute node (cpu_short is enough).
#
#   ./scripts/kimi_url.sh                 # first live server
#   ./scripts/kimi_url.sh --all           # every live server, with its model id
set -uo pipefail

PORT=${KIMI_PORT:-8000}
PATTERN=${KIMI_JOB_PATTERN:-kimi}
ALL=${1:-}

hosts=$(squeue -a -h -t RUNNING -o "%j %N" | grep -i "${PATTERN}" | awk '{print $2}' \
    | xargs -r -n1 scontrol show hostnames 2>/dev/null | sort -u)

if [[ -z "${hosts}" ]]; then
    echo "no running job matching '${PATTERN}' -- is the server up? (squeue -a | grep -i kimi)" >&2
    exit 1
fi

found=1
for h in ${hosts}; do
    body=$(curl -s --max-time 8 "http://${h}:${PORT}/v1/models" -H 'Authorization: Bearer dummy' 2>/dev/null) || continue
    [[ -z "${body}" ]] && continue
    model=$(printf '%s' "${body}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null) || continue
    found=0
    if [[ "${ALL}" == "--all" ]]; then
        echo "http://${h}:${PORT}/v1  ${model}"
    else
        # The served model id is set by --served-model-name and is NOT the
        # HuggingFace name; print it so callers can pass --model.
        echo "http://${h}:${PORT}/v1"
        echo "model: ${model}" >&2
        exit 0
    fi
done

if (( found != 0 )); then
    echo "job(s) running on [${hosts}] but nothing answered on :${PORT}" >&2
    exit 1
fi
