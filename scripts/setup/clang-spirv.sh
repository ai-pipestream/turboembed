#!/bin/sh
# A clang for TURBO_CLANG that compiles the levelzero kernels to SPIR-V
# with LLVM's own SPIR-V backend (docs/setup/intel-gpu.md).
#
#   export TURBO_CLANG=$PWD/scripts/setup/clang-spirv.sh
#
# It runs TURBO_SPIRV_CLANG if set, else the first of clang-22, clang-21,
# clang-20 and clang on the PATH. clang 20 sends SPIR-V through the
# llvm-spirv translator unless told -fintegrated-objemitter, and the
# translator does not take the sub-group extension the kernels use, so for
# clang 20 the flag is added; later clangs use the backend already.
set -e
clang=${TURBO_SPIRV_CLANG:-}
if [ -z "$clang" ]; then
    for c in clang-22 clang-21 clang-20 clang; do
        if command -v "$c" >/dev/null 2>&1; then clang=$c; break; fi
    done
fi
if [ -z "$clang" ]; then
    echo "clang-spirv.sh: no clang on the PATH (set TURBO_SPIRV_CLANG)" >&2
    exit 127
fi
major=$("$clang" -dumpversion | cut -d. -f1)
if [ "$major" = 20 ]; then
    exec "$clang" -fintegrated-objemitter "$@"
fi
exec "$clang" "$@"
