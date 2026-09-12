#!/usr/bin/env bash
# End-to-end gRPC smoke for the all-Swift inferstream-apple server.
#
# Prereqs:  cargo xtask fetch --mlx minilm qwen-0.5b
#           cargo xtask fetch --llms qwen-0.5b   # tokenizer.json
#           brew install grpcurl
#
# Starts the Swift gRPC server (mlx-swift in-process — no Rust process,
# no Python), then runs the canonical inferstream-e2e harness.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-config/apple.toml}"
ADDR="127.0.0.1:8461"
AUTH=(-H 'authorization: Bearer change-me')
OIP=(-proto proto/open_inference_grpc.proto)

command -v grpcurl >/dev/null || { echo "error: grpcurl not installed" >&2; exit 1; }

./scripts/sync-proto.sh --check
if [ ! -x swift/.build/release/inferstream-apple ]; then
    swift build --package-path swift -c release
fi
./scripts/build-apple-metallib.sh
BIN=swift/.build/release/inferstream-apple

"$BIN" --config "$CONFIG" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

echo "--- waiting for Swift server on $ADDR (pid $SERVER_PID) ---"
ready=0
for _ in $(seq 1 90); do
    if kill -0 "$SERVER_PID" 2>/dev/null \
        && grpcurl -plaintext "${AUTH[@]}" "${OIP[@]}" "$ADDR" \
            inference.GRPCInferenceService/ServerLive >/dev/null 2>&1; then
        ready=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "error: inferstream-apple (Swift) exited before becoming live" >&2
        exit 1
    fi
    sleep 1
done
if [ "$ready" != 1 ]; then
    echo "error: server never became live on $ADDR" >&2
    exit 1
fi

# Prove the serve path has no Python interpreter and is not the Rust binary.
if ps -o args= -p "$SERVER_PID" | grep -qi python; then
    echo "error: inferstream-apple command line mentions python" >&2
    exit 1
fi
if pgrep -P "$SERVER_PID" -l 2>/dev/null | grep -qi python; then
    echo "error: inferstream-apple spawned a python child" >&2
    exit 1
fi
echo "--- no python child of pid $SERVER_PID ---"

if ps -o args= -p "$SERVER_PID" | grep -q 'target/.*inferstream-apple'; then
    echo "error: smoke started the legacy Rust inferstream-apple" >&2
    ps -o args= -p "$SERVER_PID" >&2
    exit 1
fi

if command -v otool >/dev/null; then
    if otool -L "$BIN" | grep -qi python; then
        echo "error: inferstream-apple links a Python dylib" >&2
        otool -L "$BIN" >&2
        exit 1
    fi
    if otool -L "$BIN" | grep -q 'libMlxEngine'; then
        echo "error: Swift server still links legacy libMlxEngine.dylib" >&2
        otool -L "$BIN" >&2
        exit 1
    fi
    echo "--- otool -L (no Python, no libMlxEngine) ---"
    otool -L "$BIN" | head -20
fi
if command -v vmmap >/dev/null; then
    if vmmap "$SERVER_PID" 2>/dev/null | grep -qiE 'Python\.framework|libpython|python[0-9]'; then
        echo "error: vmmap shows a Python image in inferstream-apple" >&2
        exit 1
    fi
    echo "--- vmmap: no Python image ---"
fi

echo "--- inferstream-e2e (canonical Apple suite) ---"
INFERSTREAM_E2E_TARGET=apple cargo run -q -p inferstream-e2e -- \
    --target apple --addr "$ADDR" --token change-me

echo "SMOKE OK"
