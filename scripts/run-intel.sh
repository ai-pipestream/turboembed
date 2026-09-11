#!/usr/bin/env bash
# Launch inferstream-intel with the oneAPI / SYCL runtime on the loader path.
#
#   scripts/setup-llamacpp-sycl.sh
#   source /opt/intel/oneapi/setvars.sh
#   GGML_SYCL=ON CMAKE_C_COMPILER=icx CMAKE_CXX_COMPILER=icpx \
#     cargo build -p inferstream-arch-intel --release --features llamacpp-sycl
#   scripts/run-intel.sh --config config/intel.toml --listen 127.0.0.1:8471
#
# Does not start python. Does not talk to vlm-server :8085.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f /opt/intel/oneapi/setvars.sh ]; then
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force >/dev/null
fi

BIN="${INFERSTREAM_BIN:-}"
if [ -z "$BIN" ]; then
    for candidate in "$ROOT/target/release/inferstream-intel" \
                     "$ROOT/target/debug/inferstream-intel"; do
        if [ -x "$candidate" ]; then BIN="$candidate"; break; fi
    done
fi
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "error: inferstream-intel binary not found; build first:" >&2
    echo "  source /opt/intel/oneapi/setvars.sh" >&2
    echo "  GGML_SYCL=ON CMAKE_C_COMPILER=icx CMAKE_CXX_COMPILER=icpx \\" >&2
    echo "    cargo build -p inferstream-arch-intel --release --features llamacpp-sycl" >&2
    exit 1
fi

exec "$BIN" "$@"
