#!/bin/bash
# Fetch the gemma-3-*-it checkpoints into the user's own HF cache.
#
# The -pt bases at .../yeb04/hf/hub/models--google--gemma-3-*pt are raw
# pretrained checkpoints with no chat template. MedGemma starts from -pt but
# applies Google's full (unpublished) post-training recipe with medical data
# folded in; we can't reproduce that recipe, so we start from -it, which already
# embodies Gemma 3's post-training. That also gives a clean baseline:
# gemma-3-*-it vs our medical SFT vs published MedGemma.
#
# Run on a login node (has internet). Gemma is gated -- needs HF_TOKEN.
set -euo pipefail

export HF_HOME=${HF_HOME:-/gpfs/data/oermannlab/users/zhouj14/hf}
REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
SIZES=${1:-"1b 4b"}

cd "${REPO_ROOT}"
source .venv/bin/activate

for s in ${SIZES}; do
    echo "=== google/gemma-3-${s}-it ==="
    python - "$s" <<'PY'
import sys
from huggingface_hub import snapshot_download

size = sys.argv[1]
path = snapshot_download(
    f"google/gemma-3-{size}-it",
    allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
    ignore_patterns=["*.gguf", "*.pth"],
)
print("OK", path)
PY
done

echo FETCH_MODELS_DONE
