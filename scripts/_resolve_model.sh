# Resolve a Gemma 3 checkpoint snapshot dir. Sourced by the sbatch scripts.
#
# The lab already has every Gemma 3 size cached under yeb04's HF hub -- both -pt
# and -it, 270m/1b/4b/12b/27b -- so we read from there rather than re-downloading
# (27b-it alone is 52G). It's world-readable (-rw-rw-r--), and it's the same
# cache the -pt bases were pointed at.
#
# Roots are searched in order; override with GEMMA_MODEL_ROOTS to pin your own.
# Note: this is another user's directory. If yeb04 prunes their cache, jobs fail
# fast with a clear error rather than silently retraining a different revision --
# resolve_gemma prints the revision hash it picked, so runs stay auditable.

GEMMA_MODEL_ROOTS=${GEMMA_MODEL_ROOTS:-"/gpfs/data/oermannlab/users/yeb04/hf/hub /gpfs/data/oermannlab/users/zhouj14/hf/hub"}

# resolve_gemma <size> <kind>   e.g. resolve_gemma 27b it
# Echoes the snapshot dir; returns non-zero if not found anywhere.
resolve_gemma() {
    local size="$1" kind="$2" root snap
    for root in ${GEMMA_MODEL_ROOTS}; do
        snap=$(ls -d "${root}/models--google--gemma-3-${size}-${kind}"/snapshots/*/ 2>/dev/null | head -1)
        if [[ -n "${snap}" && -d "${snap}" ]]; then
            # A snapshot dir with no weights is a partial download, not a model.
            if ! ls "${snap}"*.safetensors >/dev/null 2>&1; then
                echo "WARN: ${snap} has no safetensors; skipping" >&2
                continue
            fi
            echo "${snap}"
            return 0
        fi
    done
    return 1
}
