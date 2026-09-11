#!/usr/bin/env bash
# Thin wrapper — the real fetcher is scripts/fetch-llms.sh (curl + sha256sum,
# no python3). Downloads revision-pinned GGUF + tokenizer.json and verifies
# SHA-256 against models/manifests/llms.json.
#
# Usage:
#   scripts/fetch-llm-models.sh <alias> [<alias> ...]
#   scripts/fetch-llm-models.sh --all
#   scripts/fetch-llm-models.sh --list
#
# Or use make: `make fetch-llms [ALIASES=qwen-0.5b]`.
set -euo pipefail
exec "$(dirname "$0")/fetch-llms.sh" "$@"
