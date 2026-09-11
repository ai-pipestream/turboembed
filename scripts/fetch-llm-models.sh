#!/usr/bin/env bash
# Thin wrapper around inferstream-fetch --llms (Rust). Downloads
# revision-pinned GGUF + tokenizer.json and verifies SHA-256 against
# models/manifests/llms.json.
#
# Usage:
#   scripts/fetch-llm-models.sh <alias> [<alias> ...]
#   scripts/fetch-llm-models.sh --all
#   scripts/fetch-llm-models.sh --list
#
# Or: `make fetch-llms [ALIASES=qwen-0.5b]`.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec cargo run -q -p inferstream-fetch -- --llms "$@"
