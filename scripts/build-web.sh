#!/usr/bin/env bash
# Builds the demo page's script, demo/static/app.js, from demo/web/src
# (issue #1361): type checks with tsc, runs the unit tests under Node,
# and bundles with esbuild, all pinned in demo/web/package-lock.json.
# The bundle is committed so the cargo build needs no Node. With
# --check, fails when the committed bundle differs from what the sources
# build.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

npm ci --prefix demo/web --ignore-scripts --no-audit --no-fund >/dev/null
BIN="$ROOT/demo/web/node_modules/.bin"

"$BIN/tsc" -p demo/web

TESTS="$(mktemp -d)"
trap 'rm -rf "$TESTS"' EXIT
"$BIN/esbuild" demo/web/test/*.test.ts --bundle --platform=node --format=esm \
  --outdir="$TESTS" --out-extension:.js=.mjs --log-level=warning
node --test "$TESTS"/*.test.mjs

"$BIN/esbuild" demo/web/src/main.ts --bundle --format=esm --target=es2022 \
  --legal-comments=none --log-level=warning \
  --banner:js="// Built by scripts/build-web.sh from demo/web/src; do not edit." \
  --outfile=demo/static/app.js

if [[ "${1:-}" == "--check" ]]; then
  if [[ -n "$(git status --porcelain -- demo/static/app.js)" ]]; then
    git diff --stat -- demo/static/app.js
    echo "error: demo/static/app.js is not what demo/web/src builds; run scripts/build-web.sh and commit the result" >&2
    exit 1
  fi
  echo "demo/static/app.js matches demo/web/src"
fi
