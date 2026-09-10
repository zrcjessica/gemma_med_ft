#!/bin/bash
# Run medlens's analysis over every finished red-flag run, each in ITS OWN band.
#
# The band is not a constant. It moves over the finetune (4b: [10,30] at t=0 ->
# [13,31] at 512 -> [9,30] at 4882) and it differs wildly across sizes (270m
# [10,16] of 18 layers, 27b [28,60] of 62). `--band` restricts the statistic to
# the workspace layers, so passing one arm's band to another -- or one step's to
# another step -- measures a different quantity under the same name. This script
# looks up the row for each exact (arm, step) and refuses the run if it is
# missing, rather than silently letting medlens fall back to 38-92% of the axis
# for that one cell of the panel.
#
# Cheap (seconds per run, CPU) -- it reads the npz the probe already wrote. Runs
# on BigPurple because that is where the npz live; only the JSON comes back.
#
#   scripts/redflag/analyze_all.sh                    # everything ready
#   scripts/redflag/analyze_all.sh 27b_it_ck512       # named runs only
set -euo pipefail

REPO_ROOT=/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
OUT_ROOT=${OUT_ROOT:-/gpfs/data/oermannlab/users/zhouj14/redflag_out}
MEDLENS=${MEDLENS:-/gpfs/data/oermannlab/users/yeb04/jlens/code/medlens}
BAND_ROOT=${BAND_ROOT:-${REPO_ROOT}/eval/band}

cd "${REPO_ROOT}"
source .venv-jlens/bin/activate
export PYTHONPATH="${MEDLENS}${PYTHONPATH:+:${PYTHONPATH}}"

RUNS=("$@")
if [[ ${#RUNS[@]} -eq 0 ]]; then
    mapfile -t RUNS < <(cd "${OUT_ROOT}" && for d in *_ck*/; do [[ -f "${d}judged.jsonl" ]] && echo "${d%/}"; done)
fi

ok=(); missing=()
for run in "${RUNS[@]}"; do
    arm=${run%_ck*}; step=${run##*_ck}
    # band.jsonl for a gap-filled step lives in its own dir (see fill_band_gaps.sh),
    # so check the arm's file first and the per-step staging dir second.
    band=$(python - "$arm" "$step" "$BAND_ROOT" <<'PY'
import json, sys
from pathlib import Path
arm, step, root = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
for bj in (root / arm / "band.jsonl", root / f"{arm}_ck{step}" / "band.jsonl"):
    if bj.exists():
        for line in bj.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("step") == step and r.get("band"):
                    print(f'{r["band"][0]},{r["band"][1]}'); raise SystemExit
PY
)
    if [[ -z "${band}" ]]; then
        echo "[skip] ${run}: no band row for ${arm} step ${step} -- fill it before analysing" >&2
        missing+=("${run}"); continue
    fi
    echo "=== ${run}  band=${band} ==="
    python -m medlens.analysis --run "${OUT_ROOT}/${run}" --band "${band}"
    # medlens writes analysis_all.json next to the run; record which band produced it.
    python - "${OUT_ROOT}/${run}" "${band}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "analysis_all.json"
d = json.loads(p.read_text())
d["band_source"] = f"eval/band per-step row, {sys.argv[2]}"
p.write_text(json.dumps(d, indent=2))
PY
    ok+=("${run}")
done

echo
echo "analysed ${#ok[@]}: ${ok[*]}"
[[ ${#missing[@]} -gt 0 ]] && echo "MISSING BAND ${#missing[@]}: ${missing[*]}" >&2
exit 0
