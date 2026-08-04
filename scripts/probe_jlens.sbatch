#!/bin/bash
#SBATCH --job-name=gemma_jlens
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.out
#SBATCH --error=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.err
#SBATCH --requeue
#
# Jacobian-lens trajectory probe over one training run's checkpoints. Runs in the
# standalone .venv-jlens (transformers>=5.5). Submit e.g.:
#   sbatch --export=ALL,SIZE=270m,RUN_DIR=outputs/270m/full_lr1e-5_123 scripts/probe_jlens.sbatch
#
# Env vars:
#   SIZE        270m|1b|4b|12b|27b            (required; resolves the t=0 base)
#   KIND        it | pt                        (default it)
#   RUN_DIR     dir containing checkpoint-*/   (required unless CKPTS/BASE_ONLY set)
#   CKPTS       explicit space-sep ckpt dirs   (optional; overrides RUN_DIR)
#   BASE_ONLY   1 = fit t=0 only, no ckpts     (seeds .base_lens.pt for a fan-out;
#                                               implies --keep-base-lens, else the
#                                               end-of-run cleanup deletes it)
#   FIT_CORPUS  generic text, one seq per line (default data/jlens/fit_corpus.txt)
#   PROBE_SET   category<TAB>prompt tsv        (default data/jlens/probe_set.tsv)
#   OUT         output dir                     (default eval/jlens/<size>_<kind>)
#   WANDB_PROJECT  W&B project                 (optional; default gemma-med-jlens)

set -euo pipefail

: "${SIZE:?SIZE must be set (270m|1b|4b|12b|27b)}"
KIND=${KIND:-it}
REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
FIT_CORPUS=${FIT_CORPUS:-${REPO_ROOT}/data/jlens/fit_corpus_v2.txt}
PROBE_SET=${PROBE_SET:-${REPO_ROOT}/data/jlens/probe_set_v2.tsv}
OUT=${OUT:-${REPO_ROOT}/eval/jlens/${SIZE}_${KIND}}
WANDB_PROJECT=${WANDB_PROJECT:-gemma-med-jlens}

source "${REPO_ROOT}/scripts/_resolve_model.sh"
BASE_MODEL=${BASE_MODEL:-$(resolve_gemma "${SIZE}" "${KIND}" || true)}

_notify_slack() {
    [[ -z "${SLACK_WEBHOOK_URL:-}" ]] && return 0
    local code=$1 elapsed status color
    elapsed=$(( SECONDS / 60 ))
    if [[ $code -eq 0 ]]; then status="SUCCESS"; color="good"
    else                       status="FAILED (exit ${code})"; color="danger"; fi
    curl -s -X POST "${SLACK_WEBHOOK_URL}" -H 'Content-Type: application/json' -d @- <<JSON || true
{"attachments":[{"color":"${color}","title":"Slurm: ${SLURM_JOB_NAME} [jlens gemma-3-${SIZE}-${KIND}] — ${status}","fields":[{"title":"Job ID","value":"${SLURM_JOB_ID}","short":true},{"title":"Partition","value":"${SLURM_JOB_PARTITION}","short":true},{"title":"Node","value":"$(hostname)","short":true},{"title":"Out","value":"${OUT}","short":false},{"title":"Elapsed","value":"${elapsed}m","short":true}]}]}
JSON
}
trap '_notify_slack $?' EXIT

echo "=== gemma_med jlens probe [job ${SLURM_JOB_ID}] ==="
echo "Host:   $(hostname)"
echo "Base:   ${BASE_MODEL}"
echo "RunDir: ${RUN_DIR:-<none>}   Ckpts: ${CKPTS:-<none>}"
echo "Out:    ${OUT}"
echo "==========================================="

if [[ -z "${BASE_MODEL}" || ! -d "${BASE_MODEL}" ]]; then
    echo "ERROR: no gemma-3-${SIZE}-${KIND} snapshot found (set BASE_MODEL explicitly)." >&2
    exit 1
fi
if [[ -z "${RUN_DIR:-}" && -z "${CKPTS:-}" && "${BASE_ONLY:-0}" != "1" ]]; then
    echo "ERROR: set RUN_DIR, CKPTS, or BASE_ONLY=1." >&2
    exit 1
fi

cd "${REPO_ROOT}"
source .venv-jlens/bin/activate

module load cuda/12.6 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME:-/gpfs/share/apps/cuda/12.6}
export PATH=${CUDA_HOME}/bin:${PATH}
export HF_HOME=${HF_HOME:-/gpfs/data/oermannlab/users/yeb04/hf}
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# The fit retains a full-depth graph, so the allocator sees many
# different-sized blocks; expandable segments keep that from fragmenting.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

nvidia-smi --query-gpu=index,name,compute_mode,memory.used --format=csv,noheader

# Preflight the GPU (some shared a100 nodes hand out a device that rejects every
# allocation); requeue onto a fresh node on failure -- same pattern as train.
if ! python -c "import torch; torch.zeros(8, device='cuda:0').sum().item(); print('preflight OK')"; then
    echo "ERROR: GPU preflight failed on $(hostname)." >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
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

ARGS=(--base-model "${BASE_MODEL}" --fit-corpus "${FIT_CORPUS}" --probe-set "${PROBE_SET}"
      --out "${OUT}" --wandb-project "${WANDB_PROJECT}" --wandb-run "jlens-${SIZE}-${KIND}-${SLURM_JOB_ID}")
[[ -n "${RUN_DIR:-}" ]] && ARGS+=(--run-dir "${RUN_DIR}")
[[ -n "${CKPTS:-}" ]] && ARGS+=(--checkpoints ${CKPTS})
[[ -n "${SAVE_LENSES:-}" ]] && ARGS+=(--save-lenses)
[[ "${BASE_ONLY:-0}" == "1" || -n "${KEEP_BASE_LENS:-}" ]] && ARGS+=(--keep-base-lens)
# 27b retains a full-depth graph over dim_batch replicas on top of 55G of
# weights; drop DIM_BATCH if it OOMs (costs wall-clock, not accuracy).
[[ -n "${DIM_BATCH:-}" ]] && ARGS+=(--dim-batch "${DIM_BATCH}")

python scripts/probe_jlens.py "${ARGS[@]}"

echo "=== Finished: $(date -Is) ==="
