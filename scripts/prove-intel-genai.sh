#!/usr/bin/env bash
# GPU + no-Python proof for in-process OpenVINO GenAI Embed.
# Talks to an already-running inferstream-intel. No python3.
#
# Usage:
#   scripts/prove-intel-genai.sh [host:port] [bearer] [pid] [model]
#
# Samples /proc/<pid>/maps for libopenvino / libopenvino_genai /
# libopenvino_tokenizers / libpython during Embed (not just before/after).
set -euo pipefail
cd "$(dirname "$0")/.."

ADDR="${1:-127.0.0.1:8473}"
TOKEN="${2:-change-me}"
PID="${3:-}"
MODEL="${4:-minilm}"

if [ -z "$PID" ]; then
    PID="$(pgrep -n -f 'inferstream-intel' || true)"
fi
if [ -z "$PID" ] || [ ! -d "/proc/$PID" ]; then
    echo "error: inferstream-intel not running (pass pid as \$3)" >&2
    exit 1
fi

echo "===== prove-intel-genai $(date -Iseconds) pid=$PID model=$MODEL ====="
ps -o pid,ppid,user,cmd -p "$PID"
echo "-- maps (openvino / python) --"
if [ -r "/proc/$PID/maps" ]; then
    grep -Ei 'libopenvino|libpython|libze_loader' "/proc/$PID/maps" || true
fi
if grep -Eiq 'libpython' "/proc/$PID/maps" 2>/dev/null; then
    echo "error: process maps include libpython — forbidden on the intel hot path" >&2
    exit 1
fi
if ! grep -Eq 'libopenvino\.so' "/proc/$PID/maps" 2>/dev/null; then
    echo "error: libopenvino.so not mapped — not an in-process GenAI binary?" >&2
    exit 1
fi
if ! grep -Eq 'libopenvino_genai' "/proc/$PID/maps" 2>/dev/null; then
    echo "error: libopenvino_genai not mapped" >&2
    exit 1
fi

PROTO="proto/inferstream_extension.proto"
AUTH=()
if [ -n "$TOKEN" ]; then
    AUTH=(-H "authorization: Bearer ${TOKEN}")
fi

echo "-- Embed $MODEL --"
grpcurl -plaintext -proto "$PROTO" "${AUTH[@]}" \
    -d "{\"model_name\":\"${MODEL}\",\"texts\":[\"hello world\"],\"normalize\":true}" \
    "$ADDR" inferstream.v1.InferstreamService/Embed

echo "ok: Embed returned; no libpython; OpenVINO GenAI mapped"
