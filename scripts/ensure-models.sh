#!/usr/bin/env bash
# Ensure SHA-256-pinned artifacts for an e2e / serve target. No Python.
#
# Invokes inferstream-e2e --fetch-only, which calls inferstream-fetch
# (ONNX / GGUF / OpenVINO GenAI) or `cargo xtask fetch --mlx` (Apple).
# Files already on disk with a matching SHA-256 are skipped.
#
# Usage:
#   scripts/ensure-models.sh nvidia
#   scripts/ensure-models.sh intel minilm qwen-0.5b
#   scripts/ensure-models.sh apple --fetch-all
#   scripts/ensure-models.sh nvidia --fetch-serve
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -lt 1 ]; then
    echo "usage: $0 nvidia|intel|apple|mock [alias ...] [--fetch-all|--fetch-serve]" >&2
    exit 2
fi

TARGET="$1"
shift

ONLY=()
EXTRA=()
for arg in "$@"; do
    case "$arg" in
        --fetch-all|--fetch-serve) EXTRA+=("$arg") ;;
        -*)
            echo "error: unknown flag $arg" >&2
            exit 2
            ;;
        *) ONLY+=("$arg") ;;
    esac
done

ARGS=(--target "$TARGET" --fetch-only)
if [ ${#ONLY[@]} -gt 0 ]; then
    joined=$(printf '%s,' "${ONLY[@]}")
    ARGS+=(--only "${joined%,}")
fi
ARGS+=("${EXTRA[@]+"${EXTRA[@]}"}")

exec cargo run -q -p inferstream-e2e -- "${ARGS[@]}"
