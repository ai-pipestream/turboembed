#!/usr/bin/env bash
# End-to-end gRPC smoke for the Apple arch binary on a macOS host.
#
# Prereqs (once): ./scripts/setup-mlx.sh   (MLX venv + MiniLM tokenizer)
#                 brew install grpcurl
#
# Starts inferstream-apple with config/apple.toml, then exercises the full
# inferstream.v1 surface against the live MLX/Metal bridge:
#   ListModels, Tokenize, Detokenize, Embed (MiniLM 4-bit)
# and, when a generation model is configured (uncomment qwen2.5-0.5b in the
# config), streaming ModelStreamInfer.
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

echo "--- Tokenize (local HF tokenizer, no Python round-trip) ---"
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
    | python3 -c 'import json,math,sys; r=json.load(sys.stdin); v=r["embeddings"][0]["values"]; print(f"dim={len(v)} norm={math.sqrt(sum(x*x for x in v)):.4f}")'

# Streaming generation only when the config serves a generation model.
if grep -Eq '^name = "qwen2.5-0.5b"' "$CONFIG"; then
    echo "--- ModelStreamInfer (qwen2.5-0.5b, one BYTES token chunk per token) ---"
    B64=$(python3 -c 'import base64,struct; t=b"Say hello in five words or fewer."; print(base64.b64encode(struct.pack("<I",len(t))+t).decode())')
    grpcurl -plaintext "${AUTH[@]}" "${OIP[@]}" \
        -d '{"model_name":"qwen2.5-0.5b","id":"smoke-gen","inputs":[{"name":"text","datatype":"BYTES","shape":[1]}],"raw_input_contents":["'"$B64"'"],"parameters":{"max_tokens":{"int64Param":"16"}}}' \
        "$ADDR" inference.GRPCInferenceService/ModelStreamInfer \
        | python3 -c '
import base64, json, sys
chunks, final = [], False
for obj in json.loads("[" + sys.stdin.read().replace("}\n{", "},{") + "]"):
    r = obj.get("inferResponse", obj)
    if r.get("parameters", {}).get("final", {}).get("boolParam"):
        final = True
    for raw in r.get("rawOutputContents", []):
        chunks.append(base64.b64decode(raw)[4:].decode())
print(f"tokens={len(chunks)} final={final}")
print("text:", "".join(chunks))
assert final and chunks, "stream must yield tokens and end with a final chunk"
'
else
    echo "--- ModelStreamInfer skipped: no generation model in $CONFIG (uncomment qwen2.5-0.5b) ---"
fi

echo "SMOKE OK"
