#!/bin/bash
#SBATCH --job-name=gemma_jlens_spike
#SBATCH --partition=a100_short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.out
#SBATCH --error=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.err
#SBATCH --requeue
#
# Detached wrapper for scripts/spike_jlens_gemma.py -- the jlens<->Gemma3
# integration gate. Runs in .venv-jlens (transformers>=5.5). Submit e.g.:
#   sbatch --export=ALL,SPIKE_MODEL=google/gemma-3-4b-it scripts/spike_jlens.sbatch
#
# Env vars:
#   SPIKE_MODEL   HF id resolved from the offline cache (default gemma-3-1b-it)

set -euo pipefail

REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
export SPIKE_MODEL=${SPIKE_MODEL:-google/gemma-3-1b-it}

_notify_slack() {
    [[ -z "${SLACK_WEBHOOK_URL:-}" ]] && return 0
    local code=$1 elapsed status color
    elapsed=$(( SECONDS / 60 ))
    if [[ $code -eq 0 ]]; then status="SUCCESS"; color="good"
    else                       status="FAILED (exit ${code})"; color="danger"; fi
    curl -s -X POST "${SLACK_WEBHOOK_URL}" -H 'Content-Type: application/json' -d @- <<JSON || true
{"attachments":[{"color":"${color}","title":"Slurm: ${SLURM_JOB_NAME} [jlens spike ${SPIKE_MODEL}] — ${status}","fields":[{"title":"Job ID","value":"${SLURM_JOB_ID}","short":true},{"title":"Partition","value":"${SLURM_JOB_PARTITION}","short":true},{"title":"Node","value":"$(hostname)","short":true},{"title":"Elapsed","value":"${elapsed}m","short":true}]}]}
JSON
}
trap '_notify_slack $?' EXIT

echo "=== gemma_med jlens spike [job ${SLURM_JOB_ID}] ==="
echo "Host:  $(hostname)"
echo "Model: ${SPIKE_MODEL}"
echo "==========================================="

cd "${REPO_ROOT}"
source .venv-jlens/bin/activate

module load cuda/12.6 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME:-/gpfs/share/apps/cuda/12.6}
export PATH=${CUDA_HOME}/bin:${PATH}
export HF_HOME=${HF_HOME:-/gpfs/data/oermannlab/users/yeb04/hf}
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=index,name,compute_mode,memory.used --format=csv,noheader

# Preflight the GPU (some shared a100 nodes hand out a device that rejects every
# allocation); requeue onto a fresh node on failure -- same pattern as train/probe.
if ! python -c "import torch; torch.zeros(8, device='cuda:0').sum().item(); print('preflight OK')"; then
    echo "ERROR: GPU preflight failed on $(hostname)." >&2
    RESTARTS=${SLURM_RESTART_COUNT:-0}
    if [[ "${REQUEUE_ON_GPU_FAIL:-1}" == "1" && "${RESTARTS}" -lt 5 ]]; then
        PREV_EXC=$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | grep -oP 'ExcNodeList=\K[^ ]+' | grep -v '^(null)$' || true)
        NEW_EXC=$(hostname); [[ -n "${PREV_EXC}" ]] && NEW_EXC="${PREV_EXC},${NEW_EXC}"
        echo "Requeueing (restart ${RESTARTS}/5), excluding: ${NEW_EXC}" >&2
        scontrol update jobid="${SLURM_JOB_ID}" ExcNodeList="${NEW_EXC}" || true
        scontrol requeue "${SLURM_JOB_ID}"
        sleep 60
    fi
    exit 1
fi

python scripts/spike_jlens_gemma.py

echo "=== Finished: $(date -Is) ==="
