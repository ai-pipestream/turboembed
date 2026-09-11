#!/usr/bin/env bash
# Thin wrapper around inferstream-fetch (Rust). Downloads revision-pinned
# ONNX + tokenizer artifacts and verifies SHA-256 against
# models/manifests/embeddings.json.
#
# Usage:
#   scripts/fetch-embedding-models.sh <alias> [<alias> ...]
#   scripts/fetch-embedding-models.sh --all
#   scripts/fetch-embedding-models.sh --list
#
# Or: `make fetch-embeddings [ALIASES=minilm,mpnet]`.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec cargo run -q -p inferstream-fetch -- "$@"
