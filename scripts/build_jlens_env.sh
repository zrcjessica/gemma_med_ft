#!/bin/bash
# Separate venv for the Jacobian-lens probe (scripts/probe_jlens.py). Kept apart
# from the training/eval venvs because jlens requires transformers>=5.5, which is
# hard-incompatible with the 4.53.2 pin the training + vLLM stacks share.
#
# jlens is pinned to a commit (the repo is an unmaintained reference impl, so a
# floating git install could silently drift). Local reference clone lives at the
# sibling ~/jacobian-lens on olab1 (same commit).
set -euo pipefail

JLENS_SHA=581d398613e5602a5af361e1c34d3a92ea82ba8e
REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
export UV_CACHE_DIR=/gpfs/data/oermannlab/users/zhouj14/.uv_cache

cd "${REPO_ROOT}"
uv venv --clear --python 3.11 .venv-jlens
source .venv-jlens/bin/activate

# torch from the cu124 index first, so the jlens install below finds it already
# satisfied and doesn't pull a CPU/other-CUDA wheel. cu124 pairs with the
# module-loaded cuda/12.6 on BigPurple's A100 nodes.
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# transformers is pinned (not just >=5.5) so rebuilds are reproducible; wandb for
# the trajectory logging convention; datasets so jlens.examples.load_wikitext_prompts
# works (scripts/build_fit_corpus.py) -- it is a lazy import inside that function.
uv pip install \
    "jlens @ git+https://github.com/anthropics/jacobian-lens@${JLENS_SHA}" \
    "transformers==5.14.1" \
    wandb \
    datasets

python - <<'PY'
from importlib.metadata import version
import torch, jlens  # noqa: F401
print("jlens", version("jlens"), "| transformers", version("transformers"),
      "| torch", torch.__version__, "| wandb", version("wandb"))
PY

echo BUILD_JLENS_ENV_DONE
