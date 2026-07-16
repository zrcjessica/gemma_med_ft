#!/bin/bash
# Separate venv for vLLM-based eval. Kept apart from the training venv because
# vLLM hard-pins torch and would otherwise dictate the training stack's version.
set -euo pipefail

REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
export HF_HOME=/gpfs/data/oermannlab/users/zhouj14/hf
export UV_CACHE_DIR=/gpfs/data/oermannlab/users/zhouj14/.uv_cache

cd "${REPO_ROOT}"
uv venv --clear --python 3.11 .venv-eval
source .venv-eval/bin/activate
uv pip install "vllm==0.9.1" "transformers>=4.53" "datasets>=3.6" pandas rich

# Check versions via metadata, not `import vllm`: importing it runs platform
# detection and hard-fails on a CPU-only node, where this build usually runs.
python - <<'PY'
from importlib.metadata import version
import torch
print("vllm", version("vllm"), "| torch", torch.__version__)
PY

echo BUILD_EVAL_ENV_DONE
