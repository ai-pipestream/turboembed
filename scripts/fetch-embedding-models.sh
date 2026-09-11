#!/usr/bin/env bash
# Thin compatibility wrapper — the real fetcher is scripts/fetch_models.py,
# which downloads revision-pinned artifacts and verifies a SHA-256 hash for
# every file against the committed manifest models/manifests/embeddings.json.
#
# Usage (unchanged):
#   scripts/fetch-embedding-models.sh <alias> [<alias> ...]
#   scripts/fetch-embedding-models.sh --all
#   scripts/fetch-embedding-models.sh --list
#
# Or use make: `make fetch-embeddings [ALIASES=minilm,mpnet]`.
set -euo pipefail
exec python3 "$(dirname "$0")/fetch_models.py" "$@"
