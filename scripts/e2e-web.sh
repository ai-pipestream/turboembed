#!/usr/bin/env bash
# Runs the demo page in a real browser against the real server
# (docs/demo.md, Tests): builds turbo-kserve and turbo-kserve-demo, starts
# the server on the CPU with test bundles and the demo in front of it,
# and runs every test file in demo/web/e2e with Playwright. Nothing is stubbed.
#
#   scripts/e2e-web.sh [SCREENSHOT_DIR]
#
# With a directory, a full-page PNG of every tab at each width is written
# there (TURBO_E2E_SCALE sets the device scale factor, default 1).
#
# TURBO_E2E_DEVICE is the server's device index of the CPU (default 0).
# Needs Node, npm and a headless Chromium that the pinned Playwright can
# launch: after `npm ci --prefix demo/web`,
# `demo/web/node_modules/.bin/playwright install chromium-headless-shell`
# fetches one. The `proto` job of .github/workflows/ci.yml runs this.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SHOTS="${1:-}"
if [[ -n "$SHOTS" ]]; then
  mkdir -p "$SHOTS"
  SHOTS="$(cd "$SHOTS" && pwd)"
fi
DEVICE="${TURBO_E2E_DEVICE:-0}"

cargo build --locked -p turbo-kserve -p turbo-kserve-demo
# The binaries just built: Cargo's own target directory, which
# CARGO_TARGET_DIR or a Cargo config may put somewhere other than target/.
TARGET="$(cargo metadata --locked --no-deps --format-version 1 | node -e '
  let text = "";
  process.stdin.on("data", (d) => (text += d));
  process.stdin.on("end", () => console.log(JSON.parse(text).target_directory));')"
DEBUG="$TARGET/debug"
for exe in turbo-kserve turbo-kserve-demo; do
  if [[ ! -x "$DEBUG/$exe" ]]; then
    echo "error: $DEBUG/$exe is not there after the build" >&2
    exit 1
  fi
done
npm ci --prefix demo/web --ignore-scripts --no-audit --no-fund >/dev/null
BIN="$ROOT/demo/web/node_modules/.bin"

WORK="$(mktemp -d)"
cargo run --locked -p turbo-kserve --example inspect_layer_fixture -- "$WORK/layer-instances.graph"
PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  rm -rf "$WORK"
}
trap cleanup EXIT

# Two ports nothing listens on.
read -r SERVER_PORT DEMO_PORT < <(node -e '
  const net = require("node:net");
  const a = net.createServer().listen(0, "127.0.0.1", () => {
    const b = net.createServer().listen(0, "127.0.0.1", () => {
      console.log(a.address().port, b.address().port);
      a.close(); b.close();
    });
  });')

mkdir "$WORK/vocabulary" "$WORK/catalog" "$WORK/training"
BERT="bundle=testdata/tiny-bert-bundle,device=$DEVICE,sessions=1"
"$DEBUG/turbo-kserve" --listen "127.0.0.1:$SERVER_PORT" \
  --vocabulary-dir "$WORK/vocabulary" --catalog-dir "$WORK/catalog" --training-dir "$WORK/training" \
  --model "$BERT" \
  --model "name=tiny-bert-fastest,$BERT,precision=PRECISION_FASTEST" \
  --model "name=tiny-bert-exact,$BERT,precision=PRECISION_EXACT" \
  --model "bundle=testdata/tiny-modernbert-bundle,device=$DEVICE,sessions=1" >"$WORK/server.log" 2>&1 &
PIDS+=($!)
"$DEBUG/turbo-kserve-demo" --listen "127.0.0.1:$DEMO_PORT" \
  --server "http://127.0.0.1:$SERVER_PORT" >"$WORK/demo.log" 2>&1 &
PIDS+=($!)

for _ in $(seq 120); do
  if grep -q "ready" "$WORK/server.log" && grep -q "open http" "$WORK/demo.log"; then break; fi
  for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      cat "$WORK/server.log" "$WORK/demo.log" >&2
      echo "error: the server or the demo stopped before it was ready" >&2
      exit 1
    fi
  done
  sleep 0.5
done
if ! grep -q "ready" "$WORK/server.log"; then
  cat "$WORK/server.log" >&2
  echo "error: the server was not ready in 60 s" >&2
  exit 1
fi

# The tests are bundled beside node_modules so Node finds Playwright there.
OUT="$ROOT/demo/web/node_modules/.e2e"
rm -rf "$OUT"
"$BIN/esbuild" demo/web/e2e/*.test.ts --bundle --platform=node --format=esm \
  --packages=external --outdir="$OUT" --out-extension:.js=.mjs --log-level=warning
# One file at a time: each model is served with one session, which
# refuses a second call while one runs, so files run side by side would
# fail each other's calls.
TURBO_E2E_LAYER_GRAPH="$WORK/layer-instances.graph" TURBO_E2E_URL="http://127.0.0.1:$DEMO_PORT" TURBO_E2E_SHOTS="$SHOTS" node --test --test-concurrency=1 "$OUT"/*.test.mjs
