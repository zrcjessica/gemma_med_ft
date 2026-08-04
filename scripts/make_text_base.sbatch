#!/bin/bash
#SBATCH --job-name=gemma_textbase
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.out
#SBATCH --error=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.err
#
# Pre-convert a Gemma 3 4b+ multimodal base into a text-only Gemma3ForCausalLM
# base, so the jlens probe's t=0 origin matches the class and key layout of the
# checkpoints train.py writes. MUST run in the TRAINING env (.venv,
# transformers 4.53.2) -- the probe env (5.14.1) does not remap the nested
# language_model.model.* keys and would silently random-init the whole LM.
#
# CPU-only: this is a load + save_pretrained, no compute. 27b bf16 is ~55G
# resident, hence the 200G request.
#
#   sbatch --export=ALL,SIZE=12b,KIND=it scripts/make_text_base.sbatch

set -euo pipefail

: "${SIZE:?SIZE must be set (4b|12b|27b)}"
KIND=${KIND:-it}
REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
OUT=${OUT:-${REPO_ROOT}/data/text_bases/${SIZE}_${KIND}}

_notify_slack() {
    [[ -z "${SLACK_WEBHOOK_URL:-}" ]] && return 0
    local code=$1 elapsed status color
    elapsed=$(( SECONDS / 60 ))
    if [[ $code -eq 0 ]]; then status="SUCCESS"; color="good"
    else                       status="FAILED (exit ${code})"; color="danger"; fi
    curl -s -X POST "${SLACK_WEBHOOK_URL}" -H 'Content-Type: application/json' -d @- <<JSON || true
{"attachments":[{"color":"${color}","title":"Slurm: ${SLURM_JOB_NAME} [text base gemma-3-${SIZE}-${KIND}] — ${status}","fields":[{"title":"Job ID","value":"${SLURM_JOB_ID}","short":true},{"title":"Partition","value":"${SLURM_JOB_PARTITION}","short":true},{"title":"Node","value":"$(hostname)","short":true},{"title":"Out","value":"${OUT}","short":false},{"title":"Elapsed","value":"${elapsed}m","short":true}]}]}
JSON
}
trap '_notify_slack $?' EXIT

cd "${REPO_ROOT}"
source .venv/bin/activate

export HF_HOME=${HF_HOME:-/gpfs/data/oermannlab/users/yeb04/hf}
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

# No compute happens here, but save_pretrained() -> accelerate.unwrap_model()
# unconditionally imports deepspeed, whose op-builder aborts if CUDA_HOME is
# unset. The path is on shared gpfs, so it resolves on CPU-only nodes too.
module load cuda/12.6 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME:-/gpfs/share/apps/cuda/12.6}

echo "=== make_text_base [job ${SLURM_JOB_ID}] ==="
echo "Host: $(hostname)   Size: ${SIZE}-${KIND}   Out: ${OUT}"
python -c "import transformers; print('transformers', transformers.__version__)"
echo "==========================================="

python scripts/make_text_base.py --size "${SIZE}" --kind "${KIND}" --out "${OUT}"

echo "=== Finished: $(date -Is) ==="
