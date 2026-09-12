#!/usr/bin/env bash
# Launch inferstream-intel with the oneAPI / OpenVINO / SYCL runtime on
# the loader path (GenAI embeddings + llama.cpp-SYCL generation).
#
#   scripts/build-intel.sh
#   scripts/run-intel.sh --config config/intel.toml --listen 127.0.0.1:8473
#
# Does not start python. Does not talk to vlm-server :8085.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f /opt/intel/oneapi/setvars.sh ]; then
    # setvars writes OCL_ICD_FILENAMES and similar; nounset cannot be on.
    set +u
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force >/dev/null
    set -u
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
    echo "  scripts/build-intel.sh" >&2
    exit 1
fi

exec "$BIN" "$@"
