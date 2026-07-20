#!/bin/bash
#SBATCH --job-name=gemma_med
#SBATCH --partition=oermannlab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --output=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.out
#SBATCH --error=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.err
#SBATCH --requeue
#
# Medical SFT for one Gemma 3 size. GPU count is set at submit time, e.g.
#   sbatch --gres=gpu:a100:2 --export=ALL,SIZE=1b,LR=1e-5 scripts/train_med.sbatch
#   sbatch --gres=gpu:a100:4 --export=ALL,SIZE=4b,LR=1e-5 scripts/train_med.sbatch
#   sbatch --gres=gpu:a100:8 --export=ALL,SIZE=27b,LORA=1 scripts/train_med.sbatch
#
# Env vars:
#   SIZE      1b | 4b | 12b | 27b            (required)
#   LR        learning rate                   (default 1e-5)
#   LORA      1 to use LoRA instead of full FT (default: on for 12b/27b)
#   DATA_DIR  prebuilt mixture                (default: data/mix_default)
#   VARIANT   run-name suffix                 (default: full or lora)
#   EPOCHS, MAX_SEQ_LEN, PER_DEV_BS, GRAD_ACC, MAX_TRAIN_SAMPLES

set -euo pipefail

: "${SIZE:?SIZE must be set (1b|4b|12b|27b)}"
LR=${LR:-1e-5}
EPOCHS=${EPOCHS:-2}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-2048}
REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
DATA_DIR=${DATA_DIR:-${REPO_ROOT}/data/mix_default}

# LoRA by default at 12b/27b: full FT there needs multi-node FSDP.
if [[ -z "${LORA:-}" ]]; then
    case "${SIZE}" in 12b|27b) LORA=1 ;; *) LORA=0 ;; esac
fi
VARIANT=${VARIANT:-$([[ "${LORA}" == "1" ]] && echo lora || echo full)}

# Per-GPU batch size shrinks with model size; grad accum holds the effective
# batch near 128 sequences.
case "${SIZE}" in
    1b)  PER_DEV_BS=${PER_DEV_BS:-8}; GRAD_ACC=${GRAD_ACC:-8} ;;
    4b)  PER_DEV_BS=${PER_DEV_BS:-4}; GRAD_ACC=${GRAD_ACC:-8} ;;
    12b) PER_DEV_BS=${PER_DEV_BS:-2}; GRAD_ACC=${GRAD_ACC:-8} ;;
    27b) PER_DEV_BS=${PER_DEV_BS:-1}; GRAD_ACC=${GRAD_ACC:-16} ;;
    *) echo "unknown SIZE ${SIZE}" >&2; exit 1 ;;
esac

source "${REPO_ROOT}/scripts/_resolve_model.sh"
MODEL_PATH=${MODEL_PATH:-$(resolve_gemma "${SIZE}" "${KIND:-it}" || true)}
RUN_NAME="gemma3-${SIZE}-med-${VARIANT}-lr${LR}-${SLURM_JOB_ID}"
OUT_DIR=${REPO_ROOT}/outputs/${SIZE}/${VARIANT}_lr${LR}_${SLURM_JOB_ID}

_notify_slack() {
    [[ -z "${SLACK_WEBHOOK_URL:-}" ]] && return 0
    local code=$1 elapsed status color wandb_field
    elapsed=$(( SECONDS / 60 ))
    if [[ $code -eq 0 ]]; then status="SUCCESS"; color="good"
    else                       status="FAILED (exit ${code})"; color="danger"; fi
    wandb_url=""
    [[ -f "${OUT_DIR}/wandb_run_url.txt" ]] && wandb_url=$(tr -d '\n' < "${OUT_DIR}/wandb_run_url.txt")
    if [[ -n "${wandb_url}" ]]; then
        wandb_field=",{\"title\":\"W\\u0026B Run\",\"value\":\"${wandb_url}\",\"short\":false}"
    else
        wandb_field=""
    fi
    curl -s -X POST "${SLACK_WEBHOOK_URL}" \
        -H 'Content-Type: application/json' \
        -d @- <<JSON || true
{"attachments":[{"color":"${color}","title":"Slurm: ${SLURM_JOB_NAME} [gemma-3-${SIZE} ${VARIANT}] — ${status}","fields":[{"title":"Job ID","value":"${SLURM_JOB_ID}","short":true},{"title":"Partition","value":"${SLURM_JOB_PARTITION}","short":true},{"title":"Node","value":"$(hostname)","short":true},{"title":"Size","value":"gemma-3-${SIZE}","short":true},{"title":"Method","value":"${VARIANT}","short":true},{"title":"LR","value":"${LR}","short":true},{"title":"Elapsed","value":"${elapsed}m","short":true}${wandb_field}]}]}
JSON
}
trap '_notify_slack $?' EXIT

echo "=== gemma_med SFT [job ${SLURM_JOB_ID}] ==="
echo "Host:      $(hostname)"
echo "Start:     $(date -Is)"
echo "Size:      gemma-3-${SIZE}-it"
echo "Model:     ${MODEL_PATH}"
echo "Data:      ${DATA_DIR}"
echo "Method:    ${VARIANT}   LR=${LR}  epochs=${EPOCHS}"
echo "Batch:     per_dev=${PER_DEV_BS} grad_acc=${GRAD_ACC} seq=${MAX_SEQ_LEN}"
echo "Out:       ${OUT_DIR}"
echo "==========================================="

if [[ -z "${MODEL_PATH}" || ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: no gemma-3-${SIZE}-${KIND:-it} snapshot under: ${GEMMA_MODEL_ROOTS}" >&2
    echo "Set MODEL_PATH explicitly, or run scripts/fetch_models.sh to pull it." >&2
    exit 1
fi
echo "Model revision: $(basename "${MODEL_PATH}")"
if [[ ! -d "${DATA_DIR}/train" ]]; then
    echo "ERROR: no mixture at ${DATA_DIR}. Run scripts/build_data.py on a login node first." >&2
    exit 1
fi

cd "${REPO_ROOT}"
source .venv/bin/activate

# accelerate opportunistically imports DeepSpeed (even single-GPU, via
# unwrap_model), and DeepSpeed probes CUDA_HOME at import time -- raising
# MissingCUDAException if it can't find one. Only *some* a100 nodes have
# /usr/local/cuda (4012 and 4033 do, 4022 doesn't), so without this a run fails
# or survives depending purely on which node it landed on.
module load cuda/12.6 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME:-/gpfs/share/apps/cuda/12.6}
export PATH=${CUDA_HOME}/bin:${PATH}

export HF_HOME=${HF_HOME:-/gpfs/data/oermannlab/users/zhouj14/hf}
export HF_HUB_OFFLINE=1          # data + weights are prestaged; don't hit the hub mid-run
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_PROJECT=gemma-med-ft
export ACCELERATE_LOG_LEVEL=info

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Visible GPUs: ${NUM_GPUS}"
nvidia-smi --query-gpu=index,name,compute_mode,memory.used --format=csv,noheader

# Preflight: actually allocate on every visible GPU. Some shared nodes hand out
# a GPU that another process already holds exclusively, and without this the job
# loads the model for minutes before dying on an opaque
# "CUDA-capable device(s) is/are busy or unavailable". Fail in seconds instead,
# and print who's holding the device.
if ! python - <<'PY'
import sys, torch
for i in range(torch.cuda.device_count()):
    try:
        torch.zeros(8, device=f"cuda:{i}").sum().item()
        print(f"preflight: cuda:{i} OK")
    except Exception as e:
        print(f"preflight: cuda:{i} UNUSABLE -- {e}", file=sys.stderr)
        sys.exit(1)
PY
then
    echo "ERROR: GPU preflight failed on $(hostname)." >&2
    # Note: cgroup isolation hides other users' processes, so this often prints
    # nothing even when another job holds the device. Absence here is not proof
    # the GPU is free.
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2

    # Several a100_short nodes hand out a GPU that reports Default compute mode
    # and 0 MiB used but rejects every allocation. It's node-specific and
    # transient, so bounce the job back to the scheduler rather than burn the
    # submission -- it usually lands somewhere healthy.
    RESTARTS=${SLURM_RESTART_COUNT:-0}
    if [[ "${REQUEUE_ON_GPU_FAIL:-1}" == "1" && "${RESTARTS}" -lt 5 ]]; then
        # Requeueing alone isn't enough: the scheduler happily hands the job
        # straight back to the same bad node. Accumulate an exclude list so each
        # restart is forced somewhere new.
        PREV_EXC=$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null \
                   | grep -oP 'ExcNodeList=\K[^ ]+' | grep -v '^(null)$' || true)
        NEW_EXC=$(hostname)
        [[ -n "${PREV_EXC}" ]] && NEW_EXC="${PREV_EXC},${NEW_EXC}"
        echo "Requeueing (restart ${RESTARTS}/5), excluding: ${NEW_EXC}" >&2
        scontrol update jobid="${SLURM_JOB_ID}" ExcNodeList="${NEW_EXC}" || true
        scontrol requeue "${SLURM_JOB_ID}"
        sleep 60   # give the scheduler time to act before the trap fires
    fi
    echo "Giving up after ${RESTARTS} restarts. If this recurs across many nodes, escalate to BigPurple admins." >&2
    exit 1
fi

EXTRA=(--attn "${ATTN:-eager}")
[[ "${LORA}" == "1" ]] && EXTRA+=(--lora)
[[ -n "${MAX_TRAIN_SAMPLES:-}" ]] && EXTRA+=(--max-train-samples "${MAX_TRAIN_SAMPLES}")
[[ -n "${LOGGING_STEPS:-}" ]] && EXTRA+=(--logging-steps "${LOGGING_STEPS}")

LAUNCH=(python -m gemma_med.train)
if [[ "${NUM_GPUS}" -gt 1 ]]; then
    LAUNCH=(accelerate launch --num_processes "${NUM_GPUS}" --mixed_precision bf16
            --use_deepspeed --zero_stage 3 --gradient_clipping 1.0
            -m gemma_med.train)
fi

"${LAUNCH[@]}" \
    --model-path "${MODEL_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT_DIR}" \
    --run-name "${RUN_NAME}" \
    --lr "${LR}" \
    --epochs "${EPOCHS}" \
    --max-seq-len "${MAX_SEQ_LEN}" \
    --per-device-batch-size "${PER_DEV_BS}" \
    --grad-accum "${GRAD_ACC}" \
    "${EXTRA[@]}"

echo "=== Finished: $(date -Is) ==="
