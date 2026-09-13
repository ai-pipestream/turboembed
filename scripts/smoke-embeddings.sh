#!/usr/bin/env bash
# Thin wrapper around the canonical inferstream-e2e harness (crates/e2e).
#
# Usage:
#   scripts/smoke-embeddings.sh [host:port] [bearer-token] [model ...]
#
#   host:port      default 127.0.0.1:8461
#   bearer-token   default "change-me" (pass "" for auth mode = none)
#   model ...      subset to test; default = matrix embed aliases
#
# Target is INFERSTREAM_E2E_TARGET (nvidia / intel / apple for Machine A / B / C
# in the README lab chart). Localhost defaults to mock; other hosts need the
# env var or default to nvidia. Lab checkout paths are local to each Machine.
set -euo pipefail
cd "$(dirname "$0")/.."

ADDR="${1:-127.0.0.1:8461}"
TOKEN="${2-change-me}"
shift $(( $# > 2 ? 2 : $# )) || true

TARGET="${INFERSTREAM_E2E_TARGET:-}"
if [ -z "$TARGET" ]; then
    case "$ADDR" in
        127.0.0.1:*|localhost:*|'[::1]':*) TARGET=mock ;;
        *) TARGET=nvidia ;;
    esac
fi

ARGS=(--target "$TARGET" --addr "$ADDR" --suite embed --token "$TOKEN")
if [ $# -gt 0 ]; then
    ONLY=$(printf '%s,' "$@" | sed 's/,$//')
    ARGS+=(--only "$ONLY")
fi

exec cargo run -q -p inferstream-e2e -- "${ARGS[@]}"
