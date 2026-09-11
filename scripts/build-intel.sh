#!/usr/bin/env bash
# Build inferstream-intel with in-process llama.cpp SYCL (Level Zero).
#
#   scripts/build-intel.sh
#
# Sources oneAPI, injects ggml-sycl if needed, and uses icpx as the rustc
# linker *only for the final binary* so -fsycl can finish the device-image
# link without breaking crate build scripts. No python3.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ ! -f "$ROOT/vendor/llama-cpp-sys-2/llama.cpp/ggml/src/ggml-sycl/CMakeLists.txt" ]; then
    "$ROOT/scripts/setup-llamacpp-sycl.sh"
fi

if [ -f /opt/intel/oneapi/setvars.sh ]; then
    # setvars writes OCL_ICD_FILENAMES and similar; nounset cannot be on.
    set +u
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force >/dev/null
    set -u
fi

command -v icpx >/dev/null || {
    echo "error: icpx not on PATH; install oneAPI and source setvars.sh" >&2
    exit 1
}
command -v icx >/dev/null || {
    echo "error: icx not on PATH" >&2
    exit 1
}

export GGML_SYCL=ON
export CMAKE_C_COMPILER="${CMAKE_C_COMPILER:-icx}"
export CMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER:-icpx}"
export CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}"

LINKER="$ROOT/scripts/icpx-rust-linker.sh"
chmod +x "$LINKER"

echo "--- inferstream-intel SYCL build ---"
echo "    GGML_SYCL=$GGML_SYCL"
echo "    CMAKE_C_COMPILER=$CMAKE_C_COMPILER"
echo "    CMAKE_CXX_COMPILER=$CMAKE_CXX_COMPILER"
echo "    final-linker=$LINKER"
echo "    ONEAPI_ROOT=${ONEAPI_ROOT:-unset}"

# Deps and build scripts keep the default rustc linker. Only the intel
# binary is driven by icpx -fsycl (device-image extraction).
cargo rustc -p inferstream-arch-intel --release --features llamacpp-sycl \
    --bin inferstream-intel -- \
    -C "linker=${LINKER}"

ls -lh "$ROOT/target/release/inferstream-intel"
if ldd "$ROOT/target/release/inferstream-intel" | grep -Eiq 'libpython'; then
    echo "error: inferstream-intel linked libpython — forbidden on the intel hot path" >&2
    ldd "$ROOT/target/release/inferstream-intel" | grep -i python >&2
    exit 1
fi
if ! ldd "$ROOT/target/release/inferstream-intel" | grep -Eq 'libsycl\.so'; then
    echo "error: inferstream-intel did not link libsycl.so — not a SYCL build" >&2
    ldd "$ROOT/target/release/inferstream-intel" >&2
    exit 1
fi
echo "ok: libsycl linked, no libpython"
