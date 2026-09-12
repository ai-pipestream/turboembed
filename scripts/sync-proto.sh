#!/usr/bin/env bash
# Keep language-specific proto copies in lockstep with proto/ (source of truth).
set -euo pipefail
cd "$(dirname "$0")/.."

DEST=swift/Sources/InferstreamApple/Protos
mkdir -p "$DEST"
cp proto/open_inference_grpc.proto "$DEST/open_inference_grpc.proto"
cp proto/inferstream_extension.proto "$DEST/inferstream_extension.proto"

if [ "${1:-}" = "--check" ]; then
    diff -q proto/open_inference_grpc.proto "$DEST/open_inference_grpc.proto"
    diff -q proto/inferstream_extension.proto "$DEST/inferstream_extension.proto"
    echo "proto sync OK"
fi
