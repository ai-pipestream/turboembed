#!/usr/bin/env bash
# Thin wrapper around cargo xtask fetch --llms.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ $# -eq 0 ]; then
    exec cargo xtask fetch --llms --all
fi
exec cargo xtask fetch --llms "$@"
