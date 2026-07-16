#!/bin/bash
# Build the project venv on BigPurple. Run on a compute node (needs a GPU-capable
# toolchain for the flash-attn build):
#   srun -p a100_dev --gres=gpu:a100:1 -c 8 --mem=64G -t 2:00:00 scripts/build_env.sh
set -euo pipefail

REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
export HF_HOME=/gpfs/data/oermannlab/users/zhouj14/hf
export UV_CACHE_DIR=/gpfs/data/oermannlab/users/zhouj14/.uv_cache

# flash-attn compiles CUDA kernels and needs nvcc + CUDA_HOME. BigPurple's
# newest CUDA modules are 12.6/12.9/13.0; 12.6 is the closest match to our
# cu124 torch (minor-version compatible).
module load cuda/12.6 || echo "WARN: module load cuda/12.6 failed; falling back to a fixed path"
export CUDA_HOME=${CUDA_HOME:-/gpfs/share/apps/cuda/12.6}
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}

if ! command -v nvcc >/dev/null; then
    echo "ERROR: nvcc not on PATH (CUDA_HOME=${CUDA_HOME}); flash-attn cannot build." >&2
    exit 1
fi
echo "CUDA_HOME=${CUDA_HOME}"
nvcc --version | tail -2

# A100 is sm_80; the 3090s on olab1 are sm_86. Building only what we run keeps
# the compile from taking hours.
export TORCH_CUDA_ARCH_LIST="8.0;8.6"

cd "${REPO_ROOT}"

uv venv --clear --python 3.11 .venv
uv sync
source .venv/bin/activate

# --no-build-isolation means flash-attn builds against THIS venv, so its build
# deps must already be here: uv venv seeds nothing, and flash-attn doesn't
# declare setuptools. torch must also be importable, hence the ordering.
uv pip install setuptools wheel packaging ninja

# MAX_JOBS caps parallel nvcc processes; the default OOMs the node.
MAX_JOBS=8 uv pip install flash-attn==2.7.4.post1 --no-build-isolation

python - <<'PY'
import torch, transformers, trl, peft, datasets
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| gpus", torch.cuda.device_count())
print("transformers", transformers.__version__, "| trl", trl.__version__, "| peft", peft.__version__, "| datasets", datasets.__version__)
from transformers import Gemma3ForConditionalGeneration, Gemma3ForCausalLM
print("Gemma3 classes import OK")
if torch.cuda.is_available():
    print("gpu0:", torch.cuda.get_device_name(0))
PY

echo BUILD_ENV_DONE
