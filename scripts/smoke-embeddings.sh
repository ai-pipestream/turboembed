#!/usr/bin/env bash
# Smoke-test the Embed surface for every model a running inferstream binary
# serves — catalog aliases and explicit entries alike. Works against any
# arch (nvidia / intel / apple / dev mock): the whole point of logical model
# names is that this script does not care which engine answers.
#
# Each model is embedded TWICE: the first call on a fresh server is "cold"
# (model load + possible HF download), the second is "warm" (model hot in
# the backend). Both latencies are reported alongside dims.
#
# Usage:
#   scripts/smoke-embeddings.sh [host:port] [bearer-token] [model ...]
#
#   host:port      default 127.0.0.1:8461
#   bearer-token   default "change-me" (pass "" for auth mode = none)
#   model ...      subset to test; default = every model from ListModels
#
# Examples:
#   scripts/smoke-embeddings.sh                                # local, all
#   scripts/smoke-embeddings.sh krick:8461 "$KEY"              # nvidia host
#   scripts/smoke-embeddings.sh krick-1:8461 "$KEY" minilm mpnet
#
# Needs grpcurl and jq. Exits nonzero if any tested model fails to embed.
set -euo pipefail
cd "$(dirname "$0")/.."

ADDR="${1:-127.0.0.1:8461}"
TOKEN="${2-change-me}"
shift $(( $# > 2 ? 2 : $# )) || true

EXT=(-proto crates/protocol/proto/inferstream_extension.proto)
AUTH=()
if [ -n "$TOKEN" ]; then
    AUTH=(-H "authorization: Bearer $TOKEN")
fi

command -v grpcurl >/dev/null || { echo "error: grpcurl not installed" >&2; exit 1; }
command -v jq >/dev/null || { echo "error: jq not installed" >&2; exit 1; }

echo "--- ListModels @ $ADDR ---"
LISTING=$(grpcurl -plaintext "${AUTH[@]}" "${EXT[@]}" "$ADDR" \
    inferstream.v1.InferstreamService/ListModels)
echo "$LISTING" | jq -r '.models[] | "\(.name)\tbackend=\(.backend)\tready=\(.ready // false)\tdim=\(.embeddingDim // 0)"'

if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    # No mapfile: macOS ships bash 3.2.
    MODELS=()
    while IFS= read -r name; do
        MODELS+=("$name")
    done < <(echo "$LISTING" | jq -r '.models[].name')
fi

# One Embed call: prints "<count> <dim> <norm> <latency_ms>" or fails.
embed_once() {
    local model="$1" t0 t1 out
    # E5-family models want a task prefix; harmless elsewhere in a smoke.
    local req
    req=$(jq -n --arg m "$model" \
        '{model_name:$m, texts:["query: hello embeddings","query: the quick brown fox"], normalize:true}')
    t0=$(python3 -c 'import time; print(time.time_ns())')
    out=$(grpcurl -plaintext "${AUTH[@]}" "${EXT[@]}" -d "$req" "$ADDR" \
        inferstream.v1.InferstreamService/Embed) || return 1
    t1=$(python3 -c 'import time; print(time.time_ns())')
    printf '%s' "$out" | python3 -c '
import json, math, sys
r = json.load(sys.stdin)
embs = r.get("embeddings", [])
v = embs[0]["values"] if embs else []
print(len(embs), len(v), f"{math.sqrt(sum(x*x for x in v)):.4f}", end=" ")'
    echo $(( (t1 - t0) / 1000000 ))
}

PASS=0
FAIL=0
FAILED_MODELS=()
printf '\n%-20s %6s %6s %8s %10s %10s\n' MODEL COUNT DIM NORM COLD_MS WARM_MS
for model in "${MODELS[@]}"; do
    if ! cold_run=$(embed_once "$model"); then
        printf '%-20s FAIL (Embed error; rerun grpcurl by hand for detail)\n' "$model"
        FAIL=$((FAIL + 1)); FAILED_MODELS+=("$model")
        continue
    fi
    warm_run=$(embed_once "$model")   # warm pass; count/dim must be stable
    cold_ms=${cold_run##* }
    set -- ${warm_run}                # -> count dim norm warm_ms
    if [ "$1" = "2" ] && [ "$2" -gt 0 ]; then
        printf '%-20s %6s %6s %8s %10s %10s\n' "$model" "$1" "$2" "$3" "$cold_ms" "$4"
        PASS=$((PASS + 1))
    else
        printf '%-20s FAIL: unexpected shape (count=%s dim=%s)\n' "$model" "$1" "$2"
        FAIL=$((FAIL + 1)); FAILED_MODELS+=("$model")
    fi
done

echo
echo "=== embed smoke: $PASS passed, $FAIL failed ==="
if [ "$FAIL" -gt 0 ]; then
    echo "failed models: ${FAILED_MODELS[*]}" >&2
    exit 1
fi
