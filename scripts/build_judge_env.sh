#!/bin/bash
# Env for the LLM-judge scorer (gemma_med.judge). Deliberately tiny and separate
# from .venv / .venv-eval / .venv-jlens: judging is a network-bound post-process
# over eval/**/ *_predictions.jsonl, so it needs no torch, no vLLM, and no
# transformers -- which is exactly why it must not be bolted onto an env that
# pins them.
#
# `datasets` is needed only for trajectories evaluated before evaluate.py started
# saving question/options alongside each generation; judge.py re-derives those by
# index for those runs.
#
# This runs wherever the eval outputs live -- usually olab1, not BigPurple.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

uv venv --clear --python 3.11 .venv-judge
source .venv-judge/bin/activate
# >=0.115 is not cosmetic: `output_config` (structured outputs) landed later than
# the version a stale resolver will hand you, and without it the judge would fall
# back to free-text parsing -- the exact failure mode we are removing. Python
# 3.11 matters for the same reason: on 3.8 the resolver caps at anthropic 0.72.
uv pip install "anthropic>=0.115" "openai>=1.40" "datasets>=3.6"

python - <<'PY'
from importlib.metadata import version
print("anthropic", version("anthropic"), "| openai", version("openai"),
      "| datasets", version("datasets"))
PY

cat <<'EOF'
BUILD_JUDGE_ENV_DONE

--provider local (the default) is the lab's self-hosted Kimi and needs no key,
but runs only from inside BigPurple. For the other two, set a key first:
    export ANTHROPIC_API_KEY=...       # --provider anthropic, has batching
    export MOONSHOT_API_KEY=...        # --provider moonshot (Kimi), sync only
EOF
