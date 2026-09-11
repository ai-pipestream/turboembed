#!/usr/bin/env bash
# rustc C-compiler-style linker for GGML_SYCL.
# icpx -fsycl extracts device images from ggml-sycl objects. rust-lld cannot.
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
exec icpx -fsycl "$@"
