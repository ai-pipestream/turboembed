#!/usr/bin/env bash
# Thin wrapper — the real fetcher is scripts/fetch_models.py --llms,
# which downloads revision-pinned GGUF + tokenizer.json and verifies a
# SHA-256 hash for every file against models/manifests/llms.json.
#
# Usage:
#   scripts/fetch-llm-models.sh <alias> [<alias> ...]
#   scripts/fetch-llm-models.sh --all
#   scripts/fetch-llm-models.sh --list
#
# Or use make: `make fetch-llms [ALIASES=qwen-0.5b]`.
set -euo pipefail
exec python3 "$(dirname "$0")/fetch_models.py" --llms "$@"
