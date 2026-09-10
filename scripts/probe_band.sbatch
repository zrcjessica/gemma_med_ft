#!/bin/bash
#SBATCH --job-name=gemma_band
#SBATCH --partition=a100_short
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.out
#SBATCH --error=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.err
#SBATCH --requeue
#
# Workspace-band probe (medlens Figs. 27-28) over one training run's checkpoints.
# Runs in the standalone .venv-jlens. Sibling of probe_jlens.sbatch and it shares
# that script's env, preflight, and requeue behaviour. Submit e.g.:
#   sbatch --export=ALL,SIZE=270m,RUN_DIR=outputs/270m/full_lr1e-5_25799673 scripts/probe_band.sbatch
#
# The 1-day --time above is a floor that keeps the job backfillable (a
# walltime-less submit gets 365 days from the site's job_submit lua). It does
# not cover a serial 12b/27b probe -- pass --time (or TIME= via run_band_grid.sh,
# WORKER_TIME= via fanout_band.sh/array_band.sh) for those.
#
# The lens fit dominates the cost (~3.5 h/ckpt at 4b, ~47 h at 27b), so pass
# LENS/LENS_DIR to reuse lenses that already exist, and SAVE_LENS=1 whenever you
# are paying for a fit so the next analysis is free.
#
# Env vars:
#   SIZE        270m|1b|4b|12b|27b            (required; resolves the t=0 base)
#   KIND        it | pt                        (default it)
#   RUN_DIR     dir containing checkpoint-*/   (required unless CKPTS/BASE_ONLY set)
#   CKPTS       explicit space-sep ckpt dirs   (optional; overrides RUN_DIR)
#   BASE_ONLY   1 = probe t=0 only, no ckpts   (seeds the step-0 row every
#                                               fanout worker is primed with)
#   LENS        pre-fit lens .pt (single ckpt) (optional)
#   LENS_DIR    dir of per-step lenses         (optional)
#   SAVE_LENS   1 = save the fitted lens       (optional)
#   FIT_CORPUS  generic text, one seq per line (default data/jlens/fit_corpus_v2.txt)
#   CKA_TOKENS  vocab subsample for CKA        (default 8192; halve at 27b if slow)
#   TEXT_PROMPTS  prompts for the text stats   (default 50)
#   DIM_BATCH   lens-fit memory knob           (optional; 8 at 27b)
#   OUT         output dir                     (default eval/band/<size>_<kind>)

set -euo pipefail

: "${SIZE:?SIZE must be set (270m|1b|4b|12b|27b)}"
KIND=${KIND:-it}
REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
FIT_CORPUS=${FIT_CORPUS:-${REPO_ROOT}/data/jlens/fit_corpus_v2.txt}
OUT=${OUT:-${REPO_ROOT}/eval/band/${SIZE}_${KIND}}

# --- array mode (scripts/array_band.sh) --------------------------------------
# One array task per checkpoint. CKPT_LIST is a file of checkpoint dirs written
# at submit time; task N takes line N+1. Each task needs its OWN OUT dir --
# concurrent appends to a shared band.jsonl interleave -- so OUT is derived from
# OUT_ROOT and the step, ignoring any inherited OUT.
if [[ -n "${CKPT_LIST:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    : "${OUT_ROOT:?OUT_ROOT required in array mode}"
    : "${SEED_SRC:?SEED_SRC required in array mode}"
    _line=$(( SLURM_ARRAY_TASK_ID + 1 ))
    CKPTS=$(sed -n "${_line}p" "${CKPT_LIST}")
    [[ -n "${CKPTS}" ]] || { echo "ERROR: no line ${_line} in ${CKPT_LIST}" >&2; exit 1; }
    _step=$(basename "${CKPTS}"); _step=${_step#checkpoint-}
    OUT=${OUT_ROOT}/ck${_step}
    mkdir -p "${OUT}"
    # Seed the arm's step-0 row. discover_checkpoints() ALWAYS prepends the base
    # as step 0, so without a done_steps hit every task refits t=0 -- 13 spurious
    # 47 h fits at 27b. Seeded HERE rather than at submit time so the array can
    # be chained afterok on the BASE_ONLY job that produces the row; at submit
    # time that row does not exist yet.
    if ! grep -q '"step": 0,' "${OUT}/band.jsonl" 2>/dev/null; then
        # Capture before redirecting: `grep > file || exit` truncates the file
        # first, so a failed grep would leave an empty band.jsonl behind.
        _row=$(grep -m1 '"step": 0,' "${SEED_SRC}/band.jsonl") \
            || { echo "ERROR: no step-0 row in ${SEED_SRC}/band.jsonl" >&2; exit 1; }
        printf '%s\n' "${_row}" > "${OUT}/band.jsonl"
    fi
    echo "array task ${SLURM_ARRAY_TASK_ID}: ckpt=${CKPTS} out=${OUT}"
fi

source "${REPO_ROOT}/scripts/_resolve_model.sh"
BASE_MODEL=${BASE_MODEL:-$(resolve_gemma "${SIZE}" "${KIND}" || true)}

_notify_slack() {
    # QUIET_SLACK silences the per-job ping so a fanout can send one message per
    # ARM instead of one per worker (66 of them across the full grid); the arm
    # notifier is notify_band.sbatch. An explicit flag rather than blanking
    # SLACK_WEBHOOK_URL through --export: `--export=ALL,VAR=` is ambiguous
    # across Slurm versions (set-empty vs propagate-from-submitter), and the
    # failure mode of guessing wrong is 66 Slack messages.
    [[ -n "${QUIET_SLACK:-}" ]] && return 0
    [[ -z "${SLACK_WEBHOOK_URL:-}" ]] && return 0
    local code=$1 elapsed status color scope
    elapsed=$(( SECONDS / 60 ))
    if [[ $code -eq 0 ]]; then status="SUCCESS"; color="good"
    else                       status="FAILED (exit ${code})"; color="danger"; fi
    # Name the SCOPE, not just the job. A bare "bd1b_base — SUCCESS" reads as
    # "the 1b arm is done" when it means "1 of ~15 checkpoints (t=0) is done" --
    # that misread actually happened. The arm-level message is the one from
    # notify_band.sbatch, and it says so.
    if [[ "${BASE_ONLY:-0}" == "1" ]]; then
        # `|| n_ck=0` is load-bearing: a BASE_ONLY probe has no RUN_DIR, so this
        # ls exits 2, `pipefail` hands that to the assignment, and `set -e` then
        # kills the shell INSIDE the EXIT trap -- silently, with no message and
        # without ever running curl. Slurm sees exit 2 and marks FAILED a job
        # whose work finished cleanly, killing everything chained afterok behind
        # it. Cost us bd4b_base 26290564, which had already written a complete
        # band.jsonl. Only jobs with the webhook set and QUIET_SLACK unset can
        # reach this line, which is why the quiet fanout workers never tripped it.
        local n_ck
        n_ck=$(ls -d "${RUN_DIR:-/nonexistent}"/checkpoint-* 2>/dev/null | wc -l) || n_ck=0
        scope="t=0 base ONLY (1 of $(( n_ck + 1 )) points) — arm NOT complete; fanout follows"
    else
        scope="checkpoint(s): ${CKPTS:-${RUN_DIR:-?}}"
    fi
    curl -s -X POST "${SLACK_WEBHOOK_URL}" -H 'Content-Type: application/json' -d @- <<JSON || true
{"attachments":[{"color":"${color}","title":"Slurm: ${SLURM_JOB_NAME} [band gemma-3-${SIZE}-${KIND}] — ${status}","fields":[{"title":"Scope","value":"${scope}","short":false},{"title":"Job ID","value":"${SLURM_JOB_ID}","short":true},{"title":"Partition","value":"${SLURM_JOB_PARTITION}","short":true},{"title":"Node","value":"$(hostname)","short":true},{"title":"Out","value":"${OUT}","short":false},{"title":"Elapsed","value":"${elapsed}m","short":true}]}]}
JSON
    # The notifier must never decide the job's exit status: anything that fails
    # in here would otherwise turn a finished run into a Slurm FAILED.
    return 0
}
trap '_notify_slack $?' EXIT

echo "=== gemma_med band probe [job ${SLURM_JOB_ID}] ==="
echo "Host:   $(hostname)"
echo "Base:   ${BASE_MODEL}"
echo "RunDir: ${RUN_DIR:-<none>}   Ckpts: ${CKPTS:-<none>}"
echo "Lens:   ${LENS:-<none>}      LensDir: ${LENS_DIR:-<none>}"
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
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

nvidia-smi --query-gpu=index,name,compute_mode,memory.used --format=csv,noheader

# Preflight the GPU (some shared a100 nodes hand out a device that rejects every
# allocation); requeue onto a fresh node on failure -- same pattern as the probe.
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

ARGS=(--base-model "${BASE_MODEL}" --fit-corpus "${FIT_CORPUS}" --out "${OUT}"
      --arm "${ARM:-${SIZE}_${KIND}}")
[[ -n "${RUN_DIR:-}" ]] && ARGS+=(--run-dir "${RUN_DIR}")
[[ -n "${CKPTS:-}" ]] && ARGS+=(--checkpoints ${CKPTS})
[[ -n "${LENS:-}" ]] && ARGS+=(--lens "${LENS}")
[[ -n "${LENS_DIR:-}" ]] && ARGS+=(--lens-dir "${LENS_DIR}")
[[ -n "${SAVE_LENS:-}" ]] && ARGS+=(--save-lens)
[[ -n "${CKA_TOKENS:-}" ]] && ARGS+=(--cka-tokens "${CKA_TOKENS}")
[[ -n "${TEXT_PROMPTS:-}" ]] && ARGS+=(--text-prompts "${TEXT_PROMPTS}")
[[ -n "${DIM_BATCH:-}" ]] && ARGS+=(--dim-batch "${DIM_BATCH}")
[[ -n "${FIT_PROMPTS:-}" ]] && ARGS+=(--fit-prompts "${FIT_PROMPTS}")

python scripts/probe_band.py "${ARGS[@]}"

echo "=== Finished: $(date -Is) ==="
