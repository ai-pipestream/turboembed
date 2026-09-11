#!/usr/bin/env bash
# Thin wrapper around cargo xtask fetch --embeddings.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ $# -eq 0 ]; then
    exec cargo xtask fetch --embeddings --all
fi
exec cargo xtask fetch --embeddings "$@"
