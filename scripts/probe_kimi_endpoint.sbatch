#!/bin/bash
#SBATCH --job-name=kimi_probe
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.out
#SBATCH --error=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.err
#
# Finds the live Kimi server and checks it can do what gemma_med.judge needs.
#
# The server is a Slurm job, so its node changes every restart -- a hardcoded
# `sp-0003` goes stale silently and every judgment turns into a connection
# error. This discovers the node from squeue instead, then checks the one
# capability the judge actually depends on: schema-constrained decoding. A server
# that ignores `response_format` returns prose and the whole run is void.
#
#   sbatch --partition=cpu_short scripts/probe_kimi_endpoint.sbatch
#
# Override discovery with KIMI_HOSTS="sp-0005 sp-0009"; override the job-name
# pattern with KIMI_JOB_PATTERN.

set -uo pipefail
PORT=${KIMI_PORT:-8000}
PATTERN=${KIMI_JOB_PATTERN:-kimi}

echo "=== node=$(hostname) partition=${SLURM_JOB_PARTITION:-?} ==="

if [[ -z "${KIMI_HOSTS:-}" ]]; then
    # Every node of every running job whose name matches; multi-node servers
    # usually answer on the first (head) node, but check them all.
    KIMI_HOSTS=$(squeue -a -h -t RUNNING -o "%j %N" | grep -i "${PATTERN}" \
        | awk '{print $2}' | xargs -r -n1 scontrol show hostnames | sort -u | tr '\n' ' ')
fi
echo "candidate hosts: ${KIMI_HOSTS:-<none found>}"
[[ -z "${KIMI_HOSTS// /}" ]] && { echo "no running Kimi server found"; exit 1; }

LIVE=""
for h in ${KIMI_HOSTS}; do
    code=$(curl -s -o /tmp/models.$$ -w '%{http_code}' --max-time 10 \
        "http://${h}:${PORT}/v1/models" -H 'Authorization: Bearer dummy' 2>/dev/null)
    if [[ "${code}" == "200" ]]; then
        echo "  ${h}:${PORT}  HTTP 200  <- live"
        echo "    served: $(tr -d '\n' < /tmp/models.$$ | head -c 400)"
        [[ -z "${LIVE}" ]] && LIVE="${h}"
    else
        echo "  ${h}:${PORT}  unreachable (http=${code:-none})"
    fi
    rm -f /tmp/models.$$
done
[[ -z "${LIVE}" ]] && { echo "no host answered on :${PORT}"; exit 1; }

URL="http://${LIVE}:${PORT}/v1"
MODEL=${KIMI_MODEL:-$(curl -s --max-time 10 "${URL}/models" -H 'Authorization: Bearer dummy' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)}
echo
echo "=== using ${URL} model=${MODEL} ==="

echo "--- plain completion ---"
curl -s --max-time 120 "${URL}/chat/completions" \
    -H 'Content-Type: application/json' -H 'Authorization: Bearer dummy' \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: ok\"}],\"max_tokens\":16}" \
    | head -c 600
echo

# THE decisive test. Content must be exactly {"chosen":...,"verdict":...} --
# no prose, no ```json fence. Anything else means guided decoding is off.
echo "--- response_format json_schema (decisive) ---"
curl -s --max-time 120 "${URL}/chat/completions" \
    -H 'Content-Type: application/json' -H 'Authorization: Bearer dummy' \
    -d "{
  \"model\": \"${MODEL}\",
  \"messages\": [
    {\"role\": \"system\", \"content\": \"Grade the answer. Reply with the JSON object only.\"},
    {\"role\": \"user\", \"content\": \"Options: A. aspirin  B. heparin. Correct option: B. heparin. Model response: 'I would give heparin.' Which option did it choose, and is it correct?\"}
  ],
  \"max_tokens\": 64,
  \"response_format\": {
    \"type\": \"json_schema\",
    \"json_schema\": {
      \"name\": \"verdict\", \"strict\": true,
      \"schema\": {
        \"type\": \"object\",
        \"properties\": {
          \"chosen\": {\"type\": \"string\", \"enum\": [\"A\",\"B\",\"none\"]},
          \"verdict\": {\"type\": \"string\", \"enum\": [\"correct\",\"incorrect\",\"no_answer\"]}
        },
        \"required\": [\"chosen\",\"verdict\"], \"additionalProperties\": false
      }
    }
  }
}" | head -c 800
echo

echo "--- fallback: vLLM guided_json (only needed if the above ignored the schema) ---"
curl -s --max-time 120 "${URL}/chat/completions" \
    -H 'Content-Type: application/json' -H 'Authorization: Bearer dummy' \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Pick A or B.\"}],\"max_tokens\":32,\"guided_json\":{\"type\":\"object\",\"properties\":{\"chosen\":{\"type\":\"string\",\"enum\":[\"A\",\"B\"]}},\"required\":[\"chosen\"]}}" \
    | head -c 600
echo
echo "=== done $(date -Is) ==="
