#!/bin/bash
#SBATCH --job-name=jlens_smoke
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.out
#SBATCH --error=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft/slurm/logs/%x_%j.err
#SBATCH --requeue
#
# Verifies the resumable-probe machinery end-to-end on the small 270m run:
#   pass 1 -> fresh, probe base + checkpoint-1 + checkpoint-2 (tiny fit corpus)
#   pass 2 -> same command; must SKIP all three (resume path) and clean artifacts
set -euo pipefail
REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
cd "${REPO_ROOT}"
source .venv-jlens/bin/activate
module load cuda/12.6 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME:-/gpfs/share/apps/cuda/12.6}
export HF_HOME=${HF_HOME:-/gpfs/data/oermannlab/users/yeb04/hf}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

BASE=/gpfs/data/oermannlab/users/yeb04/hf/hub/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3
OUT=${REPO_ROOT}/eval/jlens/_smoke_resume
RUN=outputs/270m/phase0_verify
CK="${RUN}/checkpoint-1 ${RUN}/checkpoint-2"

python -c "import torch; torch.zeros(4, device='cuda:0').sum().item(); print('gpu ok')"
rm -rf "${OUT}"

COMMON=(--base-model "${BASE}" --checkpoints ${CK}
        --fit-corpus data/jlens/fit_corpus_v2.txt --fit-prompts 25 --fit-max-seq 64
        --probe-set data/jlens/probe_set_v2.tsv --out "${OUT}" --fit-ckpt-every 10)

echo "=========== PASS 1 (fresh) ==========="
python scripts/probe_jlens.py "${COMMON[@]}"
n1=$(wc -l < "${OUT}/metrics.jsonl")
cats=$(python -c "import json;rows=[json.loads(l) for l in open('${OUT}/metrics.jsonl')];print(sorted(rows[-1]['concordance']))")
echo "PASS1 rows=${n1} categories=${cats}"

echo "=========== PASS 2 (resume; must skip all) ==========="
python scripts/probe_jlens.py "${COMMON[@]}" 2>&1 | tee /tmp/smoke_pass2_$$.log
skips=$(grep -c "already in metrics" /tmp/smoke_pass2_$$.log || true)
n2=$(wc -l < "${OUT}/metrics.jsonl")

echo "=========== SMOKE RESULT ==========="
echo "pass1_rows=${n1} pass2_rows=${n2} pass2_skips=${skips}"
[[ -d "${OUT}/.fit_ckpt" ]] && echo "FAIL: .fit_ckpt not cleaned" || echo "ok: .fit_ckpt cleaned"
[[ -f "${OUT}/.base_lens.pt" ]] && echo "FAIL: .base_lens.pt not cleaned" || echo "ok: .base_lens.pt cleaned"
if [[ "${n1}" == "3" && "${n2}" == "3" && "${skips}" -ge 3 ]]; then
    echo "SMOKE PASS"
else
    echo "SMOKE FAIL"; exit 1
fi
