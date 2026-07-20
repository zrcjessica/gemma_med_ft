#!/bin/bash
#SBATCH --job-name=check_load
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=1:00:00
#SBATCH --output=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.out
#SBATCH --error=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.err
#
# CPU-only sanity check: does AutoModelForCausalLM (train.py's load path) populate
# every LM weight for the 4b+ multimodal-wrapper checkpoints? No GPU needed.
#
#   sbatch scripts/check_load_path.sbatch                       # 1b(control) 4b 12b 27b, -it
#   sbatch --export=ALL,SIZES="4b 12b",KIND=pt scripts/check_load_path.sbatch

set -euo pipefail

REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
SIZES=${SIZES:-"1b 4b 12b 27b"}
KIND=${KIND:-it}

source "${REPO_ROOT}/scripts/_resolve_model.sh"

PATHS=()
LABELS=()
for size in ${SIZES}; do
    snap=$(resolve_gemma "${size}" "${KIND}" || true)
    if [[ -z "${snap}" || ! -d "${snap}" ]]; then
        echo "WARN: no gemma-3-${size}-${KIND} snapshot; skipping" >&2
        continue
    fi
    PATHS+=("${snap}")
    LABELS+=("gemma-3-${size}-${KIND}")
done

_notify_slack() {
    [[ -z "${SLACK_WEBHOOK_URL:-}" ]] && return 0
    local code=$1 elapsed status color
    elapsed=$(( SECONDS / 60 ))
    if [[ $code -eq 0 ]]; then status="ALL LM WEIGHTS OK"; color="good"
    else                       status="MISSING WEIGHTS / FAILED (exit ${code})"; color="danger"; fi
    curl -s -X POST "${SLACK_WEBHOOK_URL}" \
        -H 'Content-Type: application/json' \
        -d @- <<JSON || true
{"attachments":[{"color":"${color}","title":"Slurm: ${SLURM_JOB_NAME} [load-path check ${KIND}] — ${status}","fields":[{"title":"Job ID","value":"${SLURM_JOB_ID}","short":true},{"title":"Partition","value":"${SLURM_JOB_PARTITION}","short":true},{"title":"Node","value":"$(hostname)","short":true},{"title":"Sizes","value":"${SIZES}","short":true},{"title":"Elapsed","value":"${elapsed}m","short":true}]}]}
JSON
}
trap '_notify_slack $?' EXIT

if [[ ${#PATHS[@]} -eq 0 ]]; then
    echo "ERROR: no snapshots resolved for sizes '${SIZES}' kind '${KIND}'" >&2
    exit 1
fi

cd "${REPO_ROOT}"
source .venv/bin/activate
export HF_HOME=${HF_HOME:-/gpfs/data/oermannlab/users/zhouj14/hf}
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "=== load-path check [job ${SLURM_JOB_ID}] ==="
echo "Host:  $(hostname)"
echo "Sizes: ${SIZES}   Kind: ${KIND}"
echo "==========================================="

python scripts/check_load_path.py "${PATHS[@]}" --labels "${LABELS[@]}"
