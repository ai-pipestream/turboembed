#!/usr/bin/env bash
# run-nvidia.sh — launch inferstream-nvidia with the bundled CUDA 13 user-space
# libs on the loader path, so no manual LD_LIBRARY_PATH archaeology is needed.
#
#   scripts/fetch-runtime-libs.sh nvidia            # once per host
#   cargo build -p inferstream-arch-nvidia --release --features ort-cuda
#   scripts/run-nvidia.sh --config config/nvidia.toml
#
# Environment overrides:
#   INFERSTREAM_BIN     path to the binary (default: newest release/debug build)
#   INFERSTREAM_LIBS    lib dir (default: .libs/nvidia/lib from fetch-runtime-libs.sh)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBS="${INFERSTREAM_LIBS:-$ROOT/.libs/nvidia/lib}"

BIN="${INFERSTREAM_BIN:-}"
if [ -z "$BIN" ]; then
    for candidate in "$ROOT/target/release/inferstream-nvidia" "$ROOT/target/debug/inferstream-nvidia"; do
        if [ -x "$candidate" ]; then BIN="$candidate"; break; fi
    done
fi
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "error: inferstream-nvidia binary not found; build first:" >&2
    echo "  cargo build -p inferstream-arch-nvidia --release --features ort-cuda" >&2
    exit 1
fi

if [ ! -d "$LIBS" ]; then
    echo "warn: $LIBS not found — CUDA EP will only work if the host provides" >&2
    echo "      CUDA 13 user-space libs some other way. To bundle them, run:" >&2
    echo "  scripts/fetch-runtime-libs.sh nvidia" >&2
    LIBS=""
fi

export LD_LIBRARY_PATH="${LIBS:+$LIBS:}${LD_LIBRARY_PATH:-}"
exec "$BIN" "$@"
