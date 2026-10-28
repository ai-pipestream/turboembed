#!/usr/bin/env bash
# Generates the demo page's TypeScript protobuf and Connect-ES service
# code from server/proto into demo/web/src/gen with the buf CLI and
# protoc-gen-es pinned in demo/web/package-lock.json (issue #1374).
# With --check, fails when the committed output differs from what the
# protos generate. Cargo never runs this; it needs Node and npm.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

npm ci --prefix demo/web --ignore-scripts --no-audit --no-fund >/dev/null
BIN="$ROOT/demo/web/node_modules/.bin"
PATH="$BIN:$PATH" buf generate server/proto --template server/proto/buf.gen.yaml

if [[ "${1:-}" == "--check" ]]; then
  if [[ -n "$(git status --porcelain -- demo/web/src/gen)" ]]; then
    git status --short -- demo/web/src/gen
    git diff --stat -- demo/web/src/gen
    echo "error: demo/web/src/gen is not what server/proto generates; run scripts/generate-web-stubs.sh and commit the result" >&2
    exit 1
  fi
  echo "demo/web/src/gen matches server/proto"
fi
