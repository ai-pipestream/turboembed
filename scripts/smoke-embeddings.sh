#!/usr/bin/env bash
# Smoke-test Embed for every model a running inferstream binary serves.
# Needs grpcurl and jq. No python3.
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

now_ms() {
    perl -MTime::HiRes=time -e 'printf("%d\n", time*1000)'
}

echo "--- ListModels @ $ADDR ---"
LISTING=$(grpcurl -plaintext "${AUTH[@]}" "${EXT[@]}" "$ADDR" \
    inferstream.v1.InferstreamService/ListModels)
echo "$LISTING" | jq -r '.models[] | "\(.name)\tbackend=\(.backend)\tready=\(.ready // false)\tdim=\(.embeddingDim // 0)"'

if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=()
    while IFS= read -r name; do
        MODELS+=("$name")
    done < <(echo "$LISTING" | jq -r '.models[].name')
fi

embed_once() {
    local model="$1"
    local req
    req=$(jq -n --arg m "$model" \
        '{model_name:$m, texts:["query: hello embeddings","query: the quick brown fox"], normalize:true}')
    local t0 t1 out
    t0=$(now_ms)
    out=$(grpcurl -plaintext "${AUTH[@]}" "${EXT[@]}" -d "$req" "$ADDR" \
        inferstream.v1.InferstreamService/Embed) || return 1
    t1=$(now_ms)
    printf '%s' "$out" | jq -r '
        .embeddings as $e
        | ($e|length) as $n
        | (($e[0].values // []) | length) as $d
        | (($e[0].values // []) | map(.*.) | add | sqrt) as $norm
        | "\($n) \($d) \($norm)"
    '
    echo " $((t1 - t0))"
}

PASS=0
FAIL=0
FAILED_MODELS=()
printf '\n%-20s %6s %6s %8s %10s %10s\n' MODEL COUNT DIM NORM COLD_MS WARM_MS
for model in "${MODELS[@]}"; do
    if ! cold_run=$(embed_once "$model"); then
        printf '%-20s FAIL (Embed error)\n' "$model"
        FAIL=$((FAIL + 1)); FAILED_MODELS+=("$model")
        continue
    fi
    warm_run=$(embed_once "$model")
    set -- ${cold_run}
    cold_ms=$4
    set -- ${warm_run}
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
