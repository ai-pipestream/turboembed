#!/usr/bin/env bash
# End-to-end gRPC smoke for inferstream-apple on a macOS host.
#
# Prereqs:  cargo xtask fetch --mlx minilm qwen-0.5b
#           cargo xtask fetch --llms qwen-0.5b   # tokenizer.json
#           brew install grpcurl jq
#
# Starts inferstream-apple, then ListModels / Tokenize / Detokenize / Embed
# / ModelStreamInfer against native MLX (no Python).
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-config/apple.toml}"
ADDR="127.0.0.1:8461"
AUTH=(-H 'authorization: Bearer change-me')
EXT=(-proto crates/protocol/proto/inferstream_extension.proto)
OIP=(-proto crates/protocol/proto/open_inference_grpc.proto)

command -v grpcurl >/dev/null || { echo "error: grpcurl not installed" >&2; exit 1; }
command -v jq >/dev/null || { echo "error: jq not installed" >&2; exit 1; }

cargo build -p inferstream-arch-apple

./target/debug/inferstream-apple --config "$CONFIG" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

echo "--- waiting for server on $ADDR ---"
for _ in $(seq 1 60); do
    if grpcurl -plaintext "${AUTH[@]}" "${OIP[@]}" "$ADDR" \
        inference.GRPCInferenceService/ServerLive >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Prove the serve path has no Python interpreter.
if ps -o args= -p "$SERVER_PID" | grep -qi python; then
    echo "error: inferstream-apple command line mentions python" >&2
    exit 1
fi
if pgrep -P "$SERVER_PID" -l 2>/dev/null | grep -qi python; then
    echo "error: inferstream-apple spawned a python child" >&2
    exit 1
fi
echo "--- no python child of pid $SERVER_PID ---"

BIN=./target/debug/inferstream-apple
if command -v otool >/dev/null; then
    if otool -L "$BIN" | grep -qi python; then
        echo "error: inferstream-apple links a Python dylib" >&2
        otool -L "$BIN" >&2
        exit 1
    fi
    echo "--- otool -L (no Python) ---"
    otool -L "$BIN" | head -20
fi
if command -v vmmap >/dev/null; then
    if vmmap "$SERVER_PID" 2>/dev/null | grep -qiE 'Python\.framework|libpython|python[0-9]'; then
        echo "error: vmmap shows a Python image in inferstream-apple" >&2
        exit 1
    fi
    echo "--- vmmap: no Python image ---"
fi

echo "--- ListModels ---"
grpcurl -plaintext "${AUTH[@]}" "${EXT[@]}" "$ADDR" \
    inferstream.v1.InferstreamService/ListModels

echo "--- Tokenize (Rust tokenizers crate) ---"
grpcurl -plaintext "${AUTH[@]}" "${EXT[@]}" \
    -d '{"model_name":"minilm","texts":["hello world"]}' \
    "$ADDR" inferstream.v1.InferstreamService/Tokenize | jq .

echo "--- Detokenize ---"
grpcurl -plaintext "${AUTH[@]}" "${EXT[@]}" \
    -d '{"model_name":"minilm","sequences":[{"ids":[101,7592,2088,102]}],"skip_special_tokens":true}' \
    "$ADDR" inferstream.v1.InferstreamService/Detokenize | jq .

echo "--- Embed (native MLX MiniLM on Metal) ---"
grpcurl -plaintext "${AUTH[@]}" "${EXT[@]}" \
    -d '{"model_name":"minilm","texts":["gRPC inference on Apple Metal"],"normalize":true}' \
    "$ADDR" inferstream.v1.InferstreamService/Embed \
    | jq -r '.embeddings[0].values as $v | "dim=\($v|length) norm=\($v|map(.*.)|add|sqrt)"'

if grep -Eq 'default-llm|qwen-0.5b|qwen2.5-0.5b' "$CONFIG"; then
    echo "--- ModelStreamInfer (default-llm, native MLX) ---"
    TEXT="Say hello in five words or fewer."
    B64=$(printf '%s' "$TEXT" | perl -e 'undef $/; $t=<>; print pack("V", length($t)).$t' | base64)
    RAW=$(grpcurl -plaintext "${AUTH[@]}" "${OIP[@]}" \
        -d '{"model_name":"default-llm","id":"smoke-gen","inputs":[{"name":"text","datatype":"BYTES","shape":[1]}],"raw_input_contents":["'"$B64"'"],"parameters":{"max_tokens":{"int64Param":"32"}}}' \
        "$ADDR" inference.GRPCInferenceService/ModelStreamInfer)
    echo "$RAW" | jq -s '
        (map(.inferResponse // .) | map(select((.rawOutputContents // []) | length > 0))) as $c
        | (map(.inferResponse // .) | map(select(.parameters.final.boolParam == true)) | length) as $f
        | (map(.inferResponse // .) | map(.parameters.decode_tokens_per_second.doubleParam // empty) | .[0] // 0) as $tps
        | "tokens=\($c|length) final=\($f > 0) engine_decode_tps=\($tps)"
    '
else
    echo "--- ModelStreamInfer skipped: no generation alias in $CONFIG ---"
fi

echo "SMOKE OK"
