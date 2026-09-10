#!/bin/bash
#SBATCH --job-name=gemma_redflag
#SBATCH --partition=oermannlab
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%A_%a.err
#SBATCH --requeue
#
# Silent red-flag probe (yeb04's medlens) over OUR finetuned checkpoints.
#
# Per vignette: one forward pass -> J-lens + logit-lens readout at every position
# of every fitted layer (npz), plus a separate greedy generate() -> the reply the
# patient would see. The headline is the DISSOCIATION: internal risk signal
# present, no verbalized escalation.
#
# We call medlens rather than reimplementing it -- it is the instrument the
# lab's model panel (gemma-3-{270m..27b}-it, medgemma-*) was measured with, and
# a hand-rolled equivalent would not be comparable. medlens is imported
# read-only off yeb04's tree via PYTHONPATH; nothing is written there.
#
# DECODE CAVEAT: run_probe generates greedily with NO repetition penalty, i.e.
# rp=1.0. That is medlens's frozen instrument and it must stay that way for the
# panel comparison -- but it is NOT our eval harness's rp=1.1 (docs/BEHAVIORAL_EVAL.md
# section 8). Expect the repetition loops we measured at rp=1.0 to show up in some
# replies, and never mix these replies into a trajectory figure.
#
# The lens is NOT refit here: each config pins a lens banked by the band grid.
# A config with a missing lens is an error at make_configs time, not a silent refit.
#
# PARTITION: `oermannlab`, not `a100_short`. Measured 2026-08-14: the a100_*
# partitions now demand `qos_a100_<part>`, and zhouj14's association carries only
# `normal,qos_superpod` -- an a100_short submit sits PENDING forever with
# "Job's QOS not permitted to use this partition". Same class of trap as
# superpod's --qos=qos_superpod, opposite resolution: there is no QOS to pass, so
# use another partition. superpod works too, with --partition=superpod
# --qos=qos_superpod (both flags, always).
#
# Env vars:
#   CONFIG_LIST  file of medlens config YAML paths, one per line (required)
#   OUT_ROOT     run dirs go to <OUT_ROOT>/<config stem>/     (required)
#   VIGNETTES    150-line jsonl; built from VIGNETTE_DIR if unset
#   VIGNETTE_DIR dir of redflag_*.jsonl (default: yeb04's medlens/vignettes)
#   PROBES       prereg yaml (default: yeb04's preregistration/probes_v2.yaml)
#   N_SHARDS     vignette shards per config (default 2)
#   MAXNEW       max_new_tokens (default 1024, = the panel's setting)
#   LIMIT        first N vignettes only (smoke tests)
#
# Submit (3 configs x 2 shards = array 0-5):
#   sbatch --array=0-5 --export=ALL,CONFIG_LIST=...,OUT_ROOT=... scripts/redflag/probe_redflag.sbatch

set -euo pipefail

REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
MEDLENS=${MEDLENS:-/gpfs/data/oermannlab/users/yeb04/jlens/code/medlens}
: "${CONFIG_LIST:?CONFIG_LIST must be set}"
: "${OUT_ROOT:?OUT_ROOT must be set}"
VIGNETTE_DIR=${VIGNETTE_DIR:-${MEDLENS}/medlens/vignettes}
PROBES=${PROBES:-${MEDLENS}/preregistration/probes_v2.yaml}
N_SHARDS=${N_SHARDS:-2}
MAXNEW=${MAXNEW:-1024}

# One array task per (config, shard).
TASK=${SLURM_ARRAY_TASK_ID:-0}
CFG_IDX=$(( TASK / N_SHARDS ))
SHARD=$(( TASK % N_SHARDS ))
CONFIG=$(sed -n "$(( CFG_IDX + 1 ))p" "${CONFIG_LIST}")
[[ -n "${CONFIG}" ]] || { echo "ERROR: no line $(( CFG_IDX + 1 )) in ${CONFIG_LIST}" >&2; exit 1; }
TAG=$(basename "${CONFIG}" .yaml)
OUT=${OUT_ROOT}/${TAG}
mkdir -p "${OUT}/npz"

_notify_slack() {
    [[ -n "${QUIET_SLACK:-}" ]] && return 0
    [[ -z "${SLACK_WEBHOOK_URL:-}" ]] && return 0
    local code=$1 elapsed status color n_done
    elapsed=$(( SECONDS / 60 ))
    if [[ $code -eq 0 ]]; then status="SUCCESS"; color="good"
    else                       status="FAILED (exit ${code})"; color="danger"; fi
    # Name the scope: one shard of one checkpoint is not "the arm".
    n_done=$(cat "${OUT}"/generations_shard*.jsonl 2>/dev/null | wc -l) || n_done=0
    curl -s -X POST "${SLACK_WEBHOOK_URL}" -H 'Content-Type: application/json' -d @- <<JSON || true
{"attachments":[{"color":"${color}","title":"Slurm: ${SLURM_JOB_NAME} [redflag ${TAG} shard ${SHARD}/${N_SHARDS}] — ${status}","fields":[{"title":"Scope","value":"1 shard of 1 checkpoint — the pilot has $(wc -l < "${CONFIG_LIST}") checkpoints","short":false},{"title":"Vignettes done (all shards)","value":"${n_done}/150","short":true},{"title":"Job ID","value":"${SLURM_JOB_ID}","short":true},{"title":"Partition","value":"${SLURM_JOB_PARTITION}","short":true},{"title":"Node","value":"$(hostname)","short":true},{"title":"Out","value":"${OUT}","short":false},{"title":"Elapsed","value":"${elapsed}m","short":true}]}]}
JSON
    return 0
}
trap '_notify_slack $?' EXIT

echo "=== redflag probe [job ${SLURM_JOB_ID} task ${TASK}] ==="
echo "Host:    $(hostname)"
echo "Config:  ${CONFIG}"
echo "Shard:   ${SHARD}/${N_SHARDS}"
echo "Probes:  ${PROBES}"
echo "Out:     ${OUT}"
grep -E '^(hf_model_name|lens_path):' "${CONFIG}"
echo "==========================================="

cd "${REPO_ROOT}"
source .venv-jlens/bin/activate
export PYTHONPATH="${MEDLENS}${PYTHONPATH:+:${PYTHONPATH}}"

module load cuda/12.6 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME:-/gpfs/share/apps/cuda/12.6}
export PATH=${CUDA_HOME}/bin:${PATH}
export HF_HOME=${HF_HOME:-/gpfs/data/oermannlab/users/yeb04/hf}
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

nvidia-smi --query-gpu=index,name,compute_mode,memory.used --format=csv,noheader

# Flaky-node guard: some shared a100 nodes hand out a device that rejects every
# allocation. Requeue onto a fresh node, accumulating an exclude list.
if ! python -c "import torch; torch.zeros(8, device='cuda:0').sum().item(); print('preflight OK')"; then
    echo "ERROR: GPU preflight failed on $(hostname)." >&2
    RESTARTS=${SLURM_RESTART_COUNT:-0}
    JID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID:-0}"
    # A --nodelist pin and the exclude list are mutually exclusive: excluding the
    # only node the job may run on makes it unschedulable forever, which is worse
    # than failing loudly. The study pins every job to one node on purpose
    # (run_arm.sh NODE=), so check before requeueing.
    PIN=$(scontrol show job "${JID}" 2>/dev/null | grep -oP 'ReqNodeList=\K[^ ]+' | grep -v '^(null)$' || true)
    if [[ -n "${PIN}" ]]; then
        echo "ERROR: pinned to ${PIN} (NODE=), so not requeueing onto a fresh node." >&2
        echo "       Resubmit with NODE= to let Slurm place it, or pick another node." >&2
        exit 1
    fi
    if [[ "${REQUEUE_ON_GPU_FAIL:-1}" == "1" && "${RESTARTS}" -lt 5 ]]; then
        PREV_EXC=$(scontrol show job "${JID}" 2>/dev/null | grep -oP 'ExcNodeList=\K[^ ]+' | grep -v '^(null)$' || true)
        NEW_EXC=$(hostname); [[ -n "${PREV_EXC}" ]] && NEW_EXC="${PREV_EXC},${NEW_EXC}"
        echo "Requeueing (restart ${RESTARTS}/5), excluding: ${NEW_EXC}" >&2
        scontrol update jobid="${JID}" ExcNodeList="${NEW_EXC}" || true
        scontrol requeue "${JID}"
        sleep 60
    fi
    exit 1
fi

# One frozen 150-line corpus, shared by every checkpoint of the pilot: 50 trios
# (benign / implicit / explicit) across the three vignette files.
VIGNETTES=${VIGNETTES:-${OUT_ROOT}/all_vignettes.jsonl}
if [[ ! -f "${VIGNETTES}" ]]; then
    mkdir -p "$(dirname "${VIGNETTES}")"
    grep -h . "${VIGNETTE_DIR}"/redflag_*.jsonl > "${VIGNETTES}"
    echo "built ${VIGNETTES} ($(wc -l < "${VIGNETTES}") vignettes)"
fi

ARGS=(--config "${CONFIG}" --vignettes "${VIGNETTES}" --probes "${PROBES}"
      --out "${OUT}" --shard "${SHARD}" --n-shards "${N_SHARDS}"
      --max-new-tokens "${MAXNEW}")
[[ -n "${LIMIT:-}" ]] && ARGS+=(--limit "${LIMIT}")

python -m medlens.run_probe "${ARGS[@]}"

echo "=== Finished: $(date -Is) ==="
