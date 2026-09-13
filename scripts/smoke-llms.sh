#!/usr/bin/env bash
# Thin wrapper around the canonical inferstream-e2e harness (crates/e2e).
#
# Usage:
#   scripts/smoke-llms.sh [host:port] [bearer-token] [model ...]
#
#   host:port      default 127.0.0.1:8461
#   bearer-token   default "change-me" (pass "" for auth mode = none)
#   model ...      subset to test; default = default-llm / qwen-0.5b / qwen-7b
#
# Live GPU is the acceptance path. This script talks to an already-running
# server — it does not start one, and it does not download weights.
set -euo pipefail
cd "$(dirname "$0")/.."

ADDR="${1:-127.0.0.1:8461}"
TOKEN="${2-change-me}"
shift $(( $# > 2 ? 2 : $# )) || true

# INFERSTREAM_E2E_TARGET selects nvidia / intel / apple (Machine A / B / C).
# Localhost defaults to mock. Lab checkout paths are local to each Machine.
TARGET="${INFERSTREAM_E2E_TARGET:-}"
if [ -z "$TARGET" ]; then
    case "$ADDR" in
        127.0.0.1:*|localhost:*|'[::1]':*) TARGET=mock ;;
        *) TARGET=nvidia ;;
    esac
fi

ARGS=(--target "$TARGET" --addr "$ADDR" --suite generate --token "$TOKEN")
if [ $# -gt 0 ]; then
    ONLY=$(printf '%s,' "$@" | sed 's/,$//')
    ARGS+=(--only "$ONLY")
fi

exec cargo run -q -p inferstream-e2e -- "${ARGS[@]}"
