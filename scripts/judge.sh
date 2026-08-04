#!/bin/bash
# Judge without Slurm, with the same PROVIDER toggle as judge_traj.sbatch.
#
# This is the olab1 path. PROVIDER=anthropic (the default here) needs only an
# API key, so there is nothing to submit and no reason to: a whole trajectory is
# ~11 minutes batched.
#
#   scripts/judge.sh --root eval/traj/4b_it_full            # Claude, batched
#   scripts/judge.sh --pred-dir eval/medgemma-4b-it
#   PROVIDER=local scripts/judge.sh --root eval/traj/<tag>   # only on a BigPurple
#                                                           # compute node
#
# Price it first -- --estimate goes straight to the module, which this does not
# wrap:
#   .venv-judge/bin/python -m gemma_med.judge --root eval/traj/<tag> --estimate
#
# LIMIT, BENCHMARKS, CONC, MODE, JUDGE_MODEL are read from the environment.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

[[ $# -gt 0 ]] || { echo "usage: PROVIDER=<local|anthropic> $0 --root <traj> | --pred-dir <dir>" >&2; exit 2; }

PROVIDER=${PROVIDER:-anthropic}
source scripts/_judge_provider.sh

source .venv-judge/bin/activate
export PYTHONUNBUFFERED=1

judge_run "$@"
