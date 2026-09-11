#!/usr/bin/env bash
# rustc C-compiler-style linker for the *final* inferstream-intel binary.
# Do not set this as a workspace RUSTFLAGS linker — build scripts (clang-sys)
# cannot consume SYCL offload wrappers.
#
# icpx -fsycl extracts device images from ggml-sycl objects. rust-lld cannot.
# -fPIC keeps the offload wrapper objects relocatable for rustc's PIE binary.
# No python3.
set -euo pipefail
if ! command -v icpx >/dev/null 2>&1; then
    if [ -f /opt/intel/oneapi/setvars.sh ]; then
        set +u
        # shellcheck disable=SC1091
        source /opt/intel/oneapi/setvars.sh --force >/dev/null
        set -u
    fi
fi
command -v icpx >/dev/null || {
    echo "error: icpx not on PATH (source /opt/intel/oneapi/setvars.sh)" >&2
    exit 1
}
# Prefer bfd if present: rust-lld rejects R_X86_64_64 in SYCL offload wrappers.
FUSE_LD=()
if [ -x /usr/bin/ld.bfd ] || command -v ld.bfd >/dev/null 2>&1; then
    FUSE_LD=(-fuse-ld=bfd)
fi
exec icpx -fsycl -fPIC "${FUSE_LD[@]}" "$@"
