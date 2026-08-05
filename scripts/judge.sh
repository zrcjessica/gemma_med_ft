#!/bin/bash
# Judge without Slurm, with the same PROVIDER toggle as judge_traj.sbatch.
#
# PROVIDER defaults to `local` (the lab's self-hosted Kimi) everywhere, which
# means this script only works unmodified on a BigPurple compute node -- nothing
# outside the cluster can reach that server. On olab1 the default cannot connect
# and exits before judging anything; use PROVIDER=anthropic there, which needs
# only an API key and is ~11 minutes batched for a whole trajectory.
#
#   scripts/judge.sh --root eval/traj/4b_it_full             # Kimi, on a compute node
#   PROVIDER=anthropic scripts/judge.sh --root eval/traj/<tag>   # Claude, from olab1
#   PROVIDER=anthropic scripts/judge.sh --pred-dir eval/medgemma-4b-it
#
# For a full trajectory on the lab server, prefer `sbatch scripts/judge_traj.sbatch`
# over running this in a login shell: it is hours of wall-clock.
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

# No default of its own: _judge_provider.sh owns it, so this script and the two
# sbatch wrappers cannot drift into judging with different instruments.
source scripts/_judge_provider.sh

source .venv-judge/bin/activate
export PYTHONUNBUFFERED=1

judge_run "$@"
