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
# 27b FULL FT needs an 8x80GB node plus optimizer offload, so it also needs a
# much larger --mem and CPU count than the defaults above (see OFFLOAD_OPTIM):
#   sbatch --partition=superpod --gres=gpu:h100:8 --mem=900G --cpus-per-task=64 \
#       --export=ALL,SIZE=27b,KIND=it,LORA=0,OMP_NUM_THREADS=8 scripts/train_med.sbatch
#
# Env vars:
#   SIZE      270m | 1b | 4b | 12b | 27b     (required)
#   LR        learning rate                   (default 1e-5)
#   LORA      1 to use LoRA instead of full FT (default: on for 12b/27b)
#   OFFLOAD_OPTIM  1 to offload Adam state to host RAM (default: on for 27b full FT)
#   DATA_DIR  prebuilt mixture                (default: data/mix_default)
#   VARIANT   run-name suffix                 (default: full or lora)
#   EPOCHS, MAX_SEQ_LEN, PER_DEV_BS, GRAD_ACC, MAX_TRAIN_SAMPLES

set -euo pipefail

: "${SIZE:?SIZE must be set (270m|1b|4b|12b|27b)}"
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
# batch near 128 sequences. The defaults assume the GPU count each size is
# normally submitted with (270m: 1, 1b/4b: 4) -- the trajectory arms are only
# comparable if every one of them lands on the same effective batch, and so the
# same step clock (312335 train seqs / 128 = 2441 steps/epoch).
case "${SIZE}" in
    270m) PER_DEV_BS=${PER_DEV_BS:-4}; GRAD_ACC=${GRAD_ACC:-32} ;;
    1b)  PER_DEV_BS=${PER_DEV_BS:-8}; GRAD_ACC=${GRAD_ACC:-8} ;;
    # 4b at per_dev=4 OOMs on the lab's 40GB A100s (job 25799674): the fp32
    # logits upcast in the loss is 4 x 2048 x 262144 x 4B = 8.2GiB, and ZeRO-3
    # shards leave under 6GiB free. per_dev=1 cuts that to 2.1GiB; grad_acc
    # goes to 32 to hold the same effective-128 clock on 4 GPUs.
    4b)  PER_DEV_BS=${PER_DEV_BS:-1}; GRAD_ACC=${GRAD_ACC:-32} ;;
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
# Respect an explicit override: SLURM_CPUS_PER_TASK is the whole task's CPU
# allocation, but under accelerate every rank inherits this value, so on a
# multi-GPU node the naive setting oversubscribes by NUM_GPUS (64 CPUs x 8 ranks
# = 512 threads on 224 cores). Pass OMP_NUM_THREADS=<cpus/ranks> for those runs.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}
export TOKENIZERS_PARALLELISM=false
# Gemma 3's 262k vocab makes the fp32 logits upcast in the loss the single
# largest allocation (PER_DEV_BS x seq x 262144 x 4B -- 16GiB at bs=8/seq2048).
# expandable_segments cuts fragmentation around those big transient tensors.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
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
# Drop optimizer/scheduler state: the trajectory needs weights, not resumability,
# and full ZeRO-3 state is ~5x the size (44G vs ~8.6G per 4b checkpoint). Set
# SAVE_ONLY_MODEL=0 to keep it when you need to resume a long run.
[[ "${SAVE_ONLY_MODEL:-1}" == "1" ]] && EXTRA+=(--save-only-model)

LAUNCH=(python -m gemma_med.train)
if [[ "${NUM_GPUS}" -gt 1 ]]; then
    # 27b full FT OOMs on 8x80GB even under ZeRO-3 (job 25969532, OOM in the
    # first backward: 72.5GiB allocated of 79.11GiB). Sharded persistent state is
    # only ~55GB/GPU; what doesn't fit is what ZeRO-3 needs on TOP of it -- the
    # all-gathered lm_head (262144 x 5376 x 2B = 2.8GB, plus its grad) and the
    # fp32 logits upcast at Gemma's 262k vocab. Offloading Adam's fp32 states
    # (master + m + v, ~330GB total) to host RAM drops the persistent footprint
    # to ~14GB/GPU and leaves the transients room. The cost is amortized: the
    # optimizer step runs once per GRAD_ACC micro-batches, not per micro-batch.
    #
    # Requires a much larger --mem than the 200G default (the offloaded state
    # lives in the job's cgroup) and more CPUs for the optimizer step. Submit
    # 27b full FT with: --mem=900G --cpus-per-task=64 OMP_NUM_THREADS=8
    if [[ -z "${OFFLOAD_OPTIM:-}" ]]; then
        if [[ "${SIZE}" == "27b" && "${LORA}" == "0" ]]; then OFFLOAD_OPTIM=1
        else OFFLOAD_OPTIM=0; fi
    fi
    echo "DeepSpeed: zero3 offload_optimizer=${OFFLOAD_OPTIM}"

    if [[ "${OFFLOAD_OPTIM}" == "1" ]]; then
        # Offload goes through a JSON config rather than --offload_optimizer_device
        # because it needs one key the CLI cannot set: zero_force_ds_cpu_optimizer.
        # Left at its default (true), accelerate swaps the torch AdamW for
        # DeepSpeedCPUAdam -- a C++ extension JIT-compiled on first use, which is
        # impossible here. superpod compute nodes run a stripped image with no
        # glibc-devel: /usr/lib64/crti.o is absent, so `ld` cannot link any shared
        # object and the build dies (jobs 25969768, 25969837 -- the second even
        # after the 8-rank build race was removed, so it is the toolchain, not a
        # race). Setting it false keeps torch's own AdamW, which needs no
        # compilation. Slower per optimizer step, but that step runs once per
        # GRAD_ACC micro-batches.
        #
        # NOTE: a config file supersedes the --zero_stage / --gradient_clipping /
        # --zero3_save_16bit_model CLI flags, so the JSON must carry all of them.
        # In particular stage3_gather_16bit_weights_on_model_save is the config
        # spelling of --zero3_save_16bit_model, and is just as REQUIRED (see below).
        DS_CFG=${DS_CFG:-${REPO_ROOT}/configs/ds_zero3_offload.json}
        [[ -f "${DS_CFG}" ]] || { echo "ERROR: missing DeepSpeed config ${DS_CFG}" >&2; exit 1; }
        echo "DeepSpeed config: ${DS_CFG}"
        LAUNCH=(accelerate launch --num_processes "${NUM_GPUS}" --mixed_precision bf16
                --use_deepspeed --deepspeed_config_file "${DS_CFG}"
                -m gemma_med.train)
    else
        # --zero3_save_16bit_model is REQUIRED, not optional: without it ZeRO-3 leaves
        # each checkpoint as sharded DeepSpeed state under global_step*/ with no
        # model.safetensors at all, and the jlens probe (which loads HF format) can't
        # read them. Phase 0 only escaped this because 270m ran single-GPU.
        LAUNCH=(accelerate launch --num_processes "${NUM_GPUS}" --mixed_precision bf16
                --use_deepspeed --zero_stage 3 --gradient_clipping 1.0
                --zero3_save_16bit_model true
                -m gemma_med.train)
    fi
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
