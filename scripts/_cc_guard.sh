#!/bin/bash
# Pick a C compiler that can actually link, and export it as CC for Triton.
#
# Why this exists (2026-08-05): superpod nodes (sp-xxxx) run Ubuntu 22.04 while
# the rest of BigPurple does not, and the default environment auto-loads
# gcc/15.2.0 from /gpfs/share/apps, which is built for the non-Ubuntu nodes. On
# a superpod node that gcc cannot find crt1.o/crti.o, so it fails to link
# *anything* -- including the cuda_utils shim Triton JITs at vLLM engine init:
#
#   /usr/bin/ld: cannot find crt1.o: No such file or directory
#   torch._inductor.exc.InductorError: CalledProcessError: ... gcc ... status 1
#
# Under tensor parallelism the same failure surfaces one layer down, as a worker
# dying and taking the shm ring buffer with it ("'ShmRingBuffer' object has no
# attribute 'shared_memory'"), which looks nothing like a compiler problem.
#
# Probe rather than branch on hostname: which gcc works is a property of the
# node's libc, and both families should keep working without another edit.

_cc_works() {
    local cc=$1 tmp
    [[ -x "${cc}" ]] || return 1
    tmp=$(mktemp -d) || return 1
    printf 'int main(){return 0;}' > "${tmp}/probe.c"
    "${cc}" "${tmp}/probe.c" -o "${tmp}/probe" >/dev/null 2>&1
    local rc=$?
    rm -rf "${tmp}"
    return ${rc}
}

_select_cc() {
    local cc
    for cc in "${CC:-}" "$(command -v gcc 2>/dev/null)" /usr/bin/gcc /usr/bin/cc; do
        [[ -z "${cc}" ]] && continue
        if _cc_works "${cc}"; then
            export CC="${cc}"
            # Triton reads CC; older builds read TRITON_CC. Set both.
            export TRITON_CC="${cc}"
            echo "CC: ${cc}"
            return 0
        fi
        echo "CC: ${cc} cannot link -- trying next" >&2
    done
    echo "WARNING: no working C compiler found; Triton JIT will fail" >&2
    return 1
}

_select_cc || true
