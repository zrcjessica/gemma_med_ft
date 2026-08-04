#!/bin/bash
# Resolve $PROVIDER into a ready-to-run judge invocation. Sourced, not executed.
#
# The two judges live in different places and want different call patterns, and
# that difference is the only reason switching between them was ever more than
# one variable:
#
#   local      the lab's self-hosted Kimi. Free, but it is somebody's Slurm job:
#              the node moves, so the endpoint must be discovered, and it is
#              reachable only from inside BigPurple. Sync only.
#   anthropic  the Claude API. Costs money, needs a key, but has a Batches API --
#              half price, and the judge of record for this study.
#   moonshot   Kimi's hosted API. Sync only; here for completeness.
#
# Everything above is handled here so callers say only PROVIDER=<x>:
#
#   source scripts/_judge_provider.sh
#   judge_run --root eval/traj/4b_it_full
#   judge_run --pred-dir eval/medgemma-4b-it
#
# Sets PROVIDER, JUDGE_MODEL, JUDGE_BASE_URL (informational for the Slack
# notify), MODE, CONC. Honours LIMIT and BENCHMARKS if exported.

PROVIDER=${PROVIDER:-local}
CONC=${CONC:-16}
_JUDGE_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

case "${PROVIDER}" in
local)
    # Discover, never hardcode: the server's node changes every time its job
    # restarts, and a stale address fails as thousands of connection errors.
    if [[ -z "${JUDGE_BASE_URL:-}" ]]; then
        JUDGE_BASE_URL=$(bash "${_JUDGE_REPO_ROOT}/scripts/kimi_url.sh" 2>/dev/null)
        [[ -z "${JUDGE_BASE_URL}" ]] && { echo "no live Kimi server found" >&2; exit 1; }
    fi
    export JUDGE_BASE_URL
    # The served model id comes from --served-model-name and is NOT the HF name.
    JUDGE_MODEL=${JUDGE_MODEL:-$(curl -s --max-time 10 "${JUDGE_BASE_URL}/models" \
        -H 'Authorization: Bearer dummy' \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')}
    MODE=${MODE:-sync}
    ;;
anthropic)
    JUDGE_MODEL=${JUDGE_MODEL:-claude-haiku-4-5}
    JUDGE_BASE_URL=${JUDGE_BASE_URL:-api.anthropic.com}
    MODE=${MODE:-batch}
    # judge.py reads repo-root .env itself; fail here anyway, because the
    # alternative is discovering it after a job has waited in the queue.
    if [[ -z "${ANTHROPIC_API_KEY:-}${ANTHROPIC_AUTH_TOKEN:-}" ]] \
        && ! grep -qs '^ANTHROPIC_API_KEY=' "${_JUDGE_REPO_ROOT}/.env"; then
        echo "PROVIDER=anthropic needs ANTHROPIC_API_KEY (exported, or in ${_JUDGE_REPO_ROOT}/.env)." >&2
        echo ".env is gitignored, so it does NOT arrive with a git pull -- copy it over." >&2
        exit 1
    fi
    ;;
moonshot)
    JUDGE_MODEL=${JUDGE_MODEL:-kimi-k2.5}
    JUDGE_BASE_URL=${JUDGE_BASE_URL:-api.moonshot.ai}
    MODE=${MODE:-sync}
    [[ -z "${MOONSHOT_API_KEY:-}" ]] && { echo "PROVIDER=moonshot needs MOONSHOT_API_KEY." >&2; exit 1; }
    ;;
*)
    echo "unknown PROVIDER='${PROVIDER}' (local | anthropic | moonshot)" >&2
    exit 1
    ;;
esac

# Judge one source (--root <traj> or --pred-dir <dir>).
#
# Batch mode is three passes on purpose, and the third is not optional:
#   1. --no-wait submits every batch at once. Without it the judge polls each
#      batch to completion before creating the next, serialising ~42 of them.
#   2. the collect pass drains them.
#   3. the sync sweep picks up individually-failed items -- the judge exits 0
#      when a batch comes back with a few failures, so nothing else retries them.
judge_run() {
    local -a base=(--provider "${PROVIDER}" --model "${JUDGE_MODEL}" --concurrency "${CONC}")
    base+=(${LIMIT:+--limit ${LIMIT}} ${BENCHMARKS:+--benchmarks ${BENCHMARKS}})
    # Explicit `|| return`: a caller that invokes this as `judge_run ... || rc=1`
    # suppresses `set -e` inside the function, so a failed submit would otherwise
    # fall through to collecting batches that were never created.
    if [[ "${MODE}" == "batch" ]]; then
        python -m gemma_med.judge "$@" "${base[@]}" --mode batch --no-wait || return $?
        python -m gemma_med.judge "$@" "${base[@]}" --mode batch || return $?
        python -m gemma_med.judge "$@" "${base[@]}" --mode sync || return $?
    else
        python -m gemma_med.judge "$@" "${base[@]}" --mode sync || return $?
    fi
}
