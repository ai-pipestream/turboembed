#!/usr/bin/env bash
# Takes the demo page's screenshots in docs/images/demo (docs/demo.md,
# Screenshots): builds turbo-kserve and turbo-kserve-demo in release,
# serves the embedding bundles given on the CPU with the demo in front,
# and runs demo/web/e2e/screenshots.ts in a real browser, once in the
# light scheme and once in the dark, each against a fresh server.
# Nothing is stubbed: the pictures show what the page draws from the
# server's own answers.
#
#   scripts/demo-screenshots.sh BUNDLE [BUNDLE...]
#
# The first bundle is the one Analyze, Search and the projection use.
# The pictures in the repository were taken with the static models
# minishlab/potion-base-8M and minishlab/potion-base-32M, made with
#   turbo-bundle make bundle/recipes/potion/potion-base-8M.json UPSTREAM BUNDLE
# TURBO_SHOTS_DIR sets where the PNGs go (default docs/images/demo).
# TURBO_SHOTS_MAX_SEQ serves every bundle with that max_seq: the chunker
# sizes chunks from the model's width, so a static model's 8192 tokens
# would put the sample document in one chunk (the pictures use 128).
# Needs Node, npm and a headless Chromium the pinned Playwright can
# launch (scripts/e2e-web.sh says how to fetch one).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 ]]; then
  echo "usage: scripts/demo-screenshots.sh BUNDLE [BUNDLE...]" >&2
  exit 2
fi
BUNDLES=()
for b in "$@"; do BUNDLES+=("$(cd "$b" && pwd)"); done
cd "$ROOT"
OUTDIR="${TURBO_SHOTS_DIR:-docs/images/demo}"
mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"

cargo build --locked --release -p turbo-kserve -p turbo-kserve-demo
TARGET="$(cargo metadata --locked --no-deps --format-version 1 | node -e '
  let text = "";
  process.stdin.on("data", (d) => (text += d));
  process.stdin.on("end", () => console.log(JSON.parse(text).target_directory));')"
RELEASE="$TARGET/release"
npm ci --prefix demo/web --ignore-scripts --no-audit --no-fund >/dev/null
BIN="$ROOT/demo/web/node_modules/.bin"
OUT="$ROOT/demo/web/node_modules/.shots"
rm -rf "$OUT"
"$BIN/esbuild" demo/web/e2e/screenshots.ts --bundle --platform=node --format=esm \
  --packages=external --outfile="$OUT/screenshots.mjs" --log-level=warning

# The first bundle's served name: its directory's name.
MODEL="$(basename "${BUNDLES[0]}")"
WORK="$(mktemp -d)"
PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  rm -rf "$WORK"
}
trap cleanup EXIT

for scheme in light dark; do
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  PIDS=()
  rm -rf "${WORK:?}/$scheme"
  mkdir -p "$WORK/$scheme/vocabulary" "$WORK/$scheme/catalog" "$WORK/$scheme/training"
  read -r SERVER_PORT DEMO_PORT < <(node -e '
    const net = require("node:net");
    const servers = [0, 1].map(() => net.createServer());
    let open = 0;
    for (const s of servers) s.listen(0, "127.0.0.1", () => {
      if (++open < servers.length) return;
      console.log(servers.map((x) => x.address().port).join(" "));
      for (const x of servers) x.close();
    });')
  MODELS=()
  for b in "${BUNDLES[@]}"; do
    MODELS+=(--model "bundle=$b,device=0,sessions=1${TURBO_SHOTS_MAX_SEQ:+,max_seq=$TURBO_SHOTS_MAX_SEQ}")
  done
  "$RELEASE/turbo-kserve" --listen "127.0.0.1:$SERVER_PORT" --admin \
    --vocabulary-dir "$WORK/$scheme/vocabulary" --catalog-dir "$WORK/$scheme/catalog" \
    --training-dir "$WORK/$scheme/training" "${MODELS[@]}" >"$WORK/$scheme/server.log" 2>&1 &
  PIDS+=($!)
  "$RELEASE/turbo-kserve-demo" --listen "127.0.0.1:$DEMO_PORT" --server "http://127.0.0.1:$SERVER_PORT" --user demo \
    --allow /turbo.search.v1.TurboSearchService/CreateIndex \
    --allow /turbo.search.v1.TurboSearchService/IndexDocuments \
    --allow /turbo.vocabulary.v1.TurboVocabularyService/LearnVocabulary >"$WORK/$scheme/demo.log" 2>&1 &
  PIDS+=($!)
  for _ in $(seq 120); do
    if grep -q "ready" "$WORK/$scheme/server.log" && grep -q "open http" "$WORK/$scheme/demo.log"; then break; fi
    for pid in "${PIDS[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        cat "$WORK/$scheme/server.log" "$WORK/$scheme/demo.log" >&2
        echo "error: the server or the demo stopped before it was ready" >&2
        exit 1
      fi
    done
    sleep 0.5
  done
  if ! grep -q "ready" "$WORK/$scheme/server.log"; then
    cat "$WORK/$scheme/server.log" >&2
    echo "error: the server was not ready in 60 s" >&2
    exit 1
  fi
  TURBO_E2E_URL="http://127.0.0.1:$DEMO_PORT" TURBO_SHOTS_DIR="$OUTDIR" TURBO_SHOTS_MODEL="$MODEL" \
    TURBO_SHOTS_SCHEME="$scheme" node "$OUT/screenshots.mjs"
done
echo "screenshots in $OUTDIR"
