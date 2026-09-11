#!/usr/bin/env bash
# Smoke-test the Embed surface for every model a running inferstream binary
# serves — catalog aliases and explicit entries alike. Works against any
# arch (nvidia / intel / apple / dev mock): the whole point of logical model
# names is that this script does not care which engine answers.
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
    mapfile -t MODELS < <(echo "$LISTING" | jq -r '.models[].name')
fi

PASS=0
FAIL=0
FAILED_MODELS=()
for model in "${MODELS[@]}"; do
    echo
    echo "--- Embed via $model ---"
    # E5-family models want a task prefix; harmless elsewhere in a smoke.
    REQUEST=$(jq -n --arg m "$model" \
        '{model_name:$m, texts:["query: hello embeddings","query: the quick brown fox"], normalize:true}')
    if RESPONSE=$(grpcurl -plaintext "${AUTH[@]}" "${EXT[@]}" -d "$REQUEST" "$ADDR" \
        inferstream.v1.InferstreamService/Embed 2>&1); then
        DIM=$(echo "$RESPONSE" | jq -r '.embeddings[0].values | length')
        COUNT=$(echo "$RESPONSE" | jq -r '.embeddings | length')
        if [ "$COUNT" = "2" ] && [ "$DIM" -gt 0 ]; then
            echo "ok: $COUNT vectors, $DIM dims"
            PASS=$((PASS + 1))
        else
            echo "FAIL: unexpected shape (count=$COUNT dim=$DIM)"
            FAIL=$((FAIL + 1)); FAILED_MODELS+=("$model")
        fi
    else
        echo "FAIL: $RESPONSE"
        FAIL=$((FAIL + 1)); FAILED_MODELS+=("$model")
    fi
done

echo
echo "=== embed smoke: $PASS passed, $FAIL failed ==="
if [ "$FAIL" -gt 0 ]; then
    echo "failed models: ${FAILED_MODELS[*]}" >&2
    exit 1
fi
