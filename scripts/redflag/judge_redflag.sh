#!/bin/bash
# Rule-based screening of the red-flag replies, one pass per run dir.
#
# This is medlens's OWN judge (regex rubric: escalated_rule / recognized_rule),
# NOT gemma_med.judge. The two are different instruments and their numbers may
# not share a figure: ours scores benchmark answers, this one scores whether a
# reply escalates. medlens calls the regex a first-pass signal pending clinician
# labels; we are not collecting those, so it is our measure of record. Hence no
# --blinded-csv: nobody is going to fill it in.
#
# Seconds per run dir (150 rows of regex), so it runs here rather than via sbatch.
#
#   scripts/redflag/judge_redflag.sh /gpfs/.../redflag_out/4b_it_ck*
set -euo pipefail

REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
MEDLENS=${MEDLENS:-/gpfs/data/oermannlab/users/yeb04/jlens/code/medlens}

[[ $# -gt 0 ]] || { echo "usage: $0 <run-dir> [run-dir ...]" >&2; exit 1; }

cd "${REPO_ROOT}"
source .venv-jlens/bin/activate
export PYTHONPATH="${MEDLENS}${PYTHONPATH:+:${PYTHONPATH}}"

for RUN in "$@"; do
    [[ -d "${RUN}" ]] || { echo "skip ${RUN}: not a dir" >&2; continue; }
    shopt -s nullglob
    gens=("${RUN}"/generations_shard*.jsonl)
    shopt -u nullglob
    if [[ ${#gens[@]} -eq 0 ]]; then
        echo "skip $(basename "${RUN}"): no generations yet" >&2
        continue
    fi
    n=$(cat "${gens[@]}" | wc -l)
    echo "=== $(basename "${RUN}") — ${n} replies"
    python -m medlens.judge \
        --generations "${RUN}/generations_shard*.jsonl" \
        --out "${RUN}/judged.jsonl"
done
