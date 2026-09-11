#!/usr/bin/env bash
# End-to-end gRPC smoke for the Apple arch binary on a macOS host.
#
# Prereqs (once): ./scripts/setup-mlx.sh   (tokenizers for Tokenize/Detokenize)
#                 brew install grpcurl jq
#
# Starts inferstream-apple with config/apple.toml, then exercises the full
# inferstream.v1 surface against the live MLX/Metal backend:
#   ListModels, Tokenize, Detokenize, Embed (MiniLM 4-bit)
# and streaming ModelStreamInfer against the `default-llm` catalog alias
# (mlx-lm Qwen2.5-0.5B 4-bit; first run downloads ~280 MB).
#
# Run from the repo root. First run downloads MiniLM (~25 MB) into the HF
# cache; the qwen stream test adds ~280 MB when enabled.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-config/apple.toml}"
ADDR="127.0.0.1:8461"
AUTH=(-H 'authorization: Bearer change-me')
EXT=(-proto crates/protocol/proto/inferstream_extension.proto)
OIP=(-proto crates/protocol/proto/open_inference_grpc.proto)

command -v grpcurl >/dev/null || { echo "error: grpcurl not installed" >&2; exit 1; }
command -v jq >/dev/null || { echo "error: jq not installed" >&2; exit 1; }

oip_text_b64() {
    local t="$1" n=${#t}
    {
        printf '%b' "$(printf '\\%03o\\%03o\\%03o\\%03o' \
            $((n & 255)) $(((n >> 8) & 255)) $(((n >> 16) & 255)) $(((n >> 24) & 255)))"
        printf '%s' "$t"
    } | base64 | tr -d '\n\r'
}

b64d() {
    if echo | base64 -d >/dev/null 2>&1; then
        base64 -d
    else
        base64 -D
    fi
}

stream_objs() {
    local raw="$1"
    if printf '%s' "$raw" | jq -e 'type == "array"' >/dev/null 2>&1; then
        printf '%s' "$raw"
    else
        printf '%s' "$raw" | jq -s '.'
    fi
}

cargo build -p inferstream-arch-apple

./target/debug/inferstream-apple --config "$CONFIG" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

echo "--- waiting for server on $ADDR ---"
for _ in $(seq 1 30); do
    if grpcurl -plaintext "${AUTH[@]}" "${OIP[@]}" "$ADDR" \
        inference.GRPCInferenceService/ServerLive >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

echo "--- ListModels ---"
grpcurl -plaintext "${AUTH[@]}" "${EXT[@]}" "$ADDR" \
    inferstream.v1.InferstreamService/ListModels

echo "--- Tokenize (local HF tokenizer) ---"
grpcurl -plaintext "${AUTH[@]}" "${EXT[@]}" \
    -d '{"model_name":"minilm-l6-v2","texts":["hello world"]}' \
    "$ADDR" inferstream.v1.InferstreamService/Tokenize

echo "--- Detokenize ---"
grpcurl -plaintext "${AUTH[@]}" "${EXT[@]}" \
    -d '{"model_name":"minilm-l6-v2","sequences":[{"ids":[101,7592,2088,102]}],"skip_special_tokens":true}' \
    "$ADDR" inferstream.v1.InferstreamService/Detokenize

echo "--- Embed (MiniLM 4-bit on Metal; expect dim=384, unit norm) ---"
grpcurl -plaintext "${AUTH[@]}" "${EXT[@]}" \
    -d '{"model_name":"minilm-l6-v2","texts":["gRPC inference on Apple Metal"],"normalize":true}' \
    "$ADDR" inferstream.v1.InferstreamService/Embed \
    | jq -r '
        .embeddings[0].values as $v
        | ($v | map(. * .) | add | sqrt) as $n
        | "dim=\($v|length) norm=\($n * 10000 | round / 10000)"
    '

# Streaming generation when the config serves an LLM alias (default-llm).
if grep -Eq 'default-llm|qwen-0.5b|qwen2.5-0.5b' "$CONFIG"; then
    echo "--- ModelStreamInfer (default-llm, one BYTES token chunk per token) ---"
    B64=$(oip_text_b64 "Say hello in five words or fewer.")
    OUT=$(grpcurl -plaintext "${AUTH[@]}" "${OIP[@]}" \
        -d '{"model_name":"default-llm","id":"smoke-gen","inputs":[{"name":"text","datatype":"BYTES","shape":[1]}],"raw_input_contents":["'"$B64"'"],"parameters":{"max_tokens":{"int64Param":"16"}}}' \
        "$ADDR" inference.GRPCInferenceService/ModelStreamInfer)
    ARR=$(stream_objs "$OUT")
    printf '%s' "$ARR" | jq -e '
        map(.inferResponse // .) as $objs
        | ([ $objs[] | .rawOutputContents[]? ] | length) as $n
        | ([$objs[] | .parameters.final.boolParam // false] | any) as $final
        | if ($final and $n > 0) then true else error("stream must yield tokens and end with a final chunk") end
    ' >/dev/null
    N=$(printf '%s' "$ARR" | jq '[.[] | (.inferResponse // .) | .rawOutputContents[]?] | length')
    FINAL=$(printf '%s' "$ARR" | jq '[.[] | (.inferResponse // .) | .parameters.final.boolParam // false] | any')
    echo "tokens=$N final=$FINAL"
    TEXT=""
    while IFS= read -r chunk; do
        [ -n "$chunk" ] || continue
        piece=$(printf '%s' "$chunk" | b64d | dd bs=1 skip=4 2>/dev/null || true)
        TEXT="${TEXT}${piece}"
    done < <(printf '%s' "$ARR" | jq -r '.[] | (.inferResponse // .) | .rawOutputContents[]? // empty')
    echo "text: $TEXT"
else
    echo "--- ModelStreamInfer skipped: no generation alias in $CONFIG ---"
fi

echo "SMOKE OK"
