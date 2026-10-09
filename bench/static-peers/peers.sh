#!/usr/bin/env bash
# turbo against other static-model implementations, on local copies of the
# potion models. Usage, from anywhere:
#   peers.sh setup      toolchains, builds, model copies, text sets
#   peers.sh accuracy   every tool's vectors for the golden texts, scored
#   peers.sh timing     texts/s per tool, model, text set and batch
# Inputs (directories with one subdirectory per model name):
#   MODELS_DIR    Model2Vec files: config.json, tokenizer.json, model.safetensors
#   GOLDEN_DIR    texts.json and golden-512.safetensors from bundle/model2vec-golden
#   TURBO_BUNDLES turbo bundles made by `turbo-bundle make`
# Everything is written under $W (default ~/static-peers); remove that
# directory to remove it all. Nothing is installed system-wide and no tool
# downloads a model: HF_HUB_OFFLINE is set, and each tool reads copies.
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$H/../.." && pwd)
W=${W:-$HOME/static-peers}
MODELS="potion-base-2M potion-base-4M potion-base-8M potion-base-32M potion-retrieval-32M potion-science-32M potion-code-16M potion-code-16M-v2 potion-multilingual-128M"
TIMED=${TIMED:-"potion-base-8M potion-multilingual-128M potion-code-16M-v2"}
BATCHES=${BATCHES:-1,32,256,1024}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GOTOOLCHAIN=local
export PATH=$W/go/bin:$W/zig:$PATH GOPATH=$W/gopath GOCACHE=$W/gocache GOMODCACHE=$W/gopath/pkg/mod
export ZIG_GLOBAL_CACHE_DIR=$W/zigcache
kind() { case $1 in
  potion-base-2M) echo BASE2M;; potion-base-4M) echo BASE4M;; potion-base-8M) echo BASE8M;;
  potion-base-32M) echo BASE32M;; potion-retrieval-32M) echo RETRIEVAL32M;; potion-science-32M) echo SCIENCE32M;;
  potion-code-16M) echo CODE16M;; potion-code-16M-v2) echo CODE16MV2;; potion-multilingual-128M) echo MULTILINGUAL128M;; esac; }
# model2vec-zig reads WordPiece F32 and I8 tables only.
zig_ok() { case $1 in potion-code-16M|potion-code-16M-v2|potion-multilingual-128M) return 1;; esac; }

setup() {
  : "${MODELS_DIR:?}" "${GOLDEN_DIR:?}"
  mkdir -p "$W"/src "$W"/models "$W"/gph "$W"/texts "$W"/out
  if [ ! -x "$W/go/bin/go" ]; then
    curl -fsSL https://go.dev/dl/go1.25.1.linux-amd64.tar.gz | tar -xz -C "$W"
  fi
  if [ ! -x "$W/zig/zig" ]; then
    mkdir -p "$W/zig"
    curl -fsSL https://ziglang.org/download/0.16.0/zig-x86_64-linux-0.16.0.tar.xz | tar -xJ -C "$W/zig" --strip-components=1 || {
      # The same build as published on PyPI (ziglang 0.16.0).
      python3 -m pip download -q --no-deps ziglang==0.16.0 -d "$W/zigwheel" &&
        (cd "$W/zigwheel" && python3 -m zipfile -e ziglang-*.whl x && cp -r x/ziglang/. "$W/zig/") && chmod +x "$W/zig/zig"; }
  fi
  go version; zig version
  # model2vec-zig at the commit build.zig.zon pins, put in Zig's cache from
  # a clone so the build fetches nothing.
  local z=$W/src/model2vec-zig
  [ -d "$z" ] || git clone -q https://github.com/PaytonWebber/model2vec-zig "$z"
  git -C "$z" checkout -q ed5b443910b833b710f7213ddebff4d064d94828
  zig fetch "$z"
  [ -d "$W/venv" ] || python3 -m venv "$W/venv"
  "$W/venv/bin/pip" install -q --extra-index-url https://download.pytorch.org/whl/cpu \
    numpy==2.5.3 torch==2.14.0+cpu sentence-transformers==6.1.0 \
    "model2vec @ git+https://github.com/MinishLab/model2vec@3ef2bf257d16429b9f822c6c5508a01b9b17b85f"
  (cd "$H/rs" && cargo build -q --release --target-dir "$W/target-rs" &&
    cargo build -q --release --features rayon --target-dir "$W/target-rs-rayon")
  (cd "$H/go" && go build -o "$W/go-potion-bench" .)
  (cd "$H/zig" && zig build --cache-dir "$W/zig-local-cache" --prefix "$W/zig-out")
  (cd "$REPO" && cargo test --release -q -p turbo --test static_speed --no-run)
  for m in $MODELS; do
    # Copies: statembed writes tokenizer.v2.json beside the tokenizer and
    # go-potion realigns the safetensors file in place.
    mkdir -p "$W/models/$m" "$W/gph/$(kind "$m")"
    for f in config.json tokenizer.json model.safetensors; do
      cp "$MODELS_DIR/$m/$f" "$W/models/$m/"
      cmp "$MODELS_DIR/$m/$f" "$W/models/$m/$f"
    done
    cp "$MODELS_DIR/$m/model.safetensors" "$MODELS_DIR/$m/tokenizer.json" "$W/gph/$(kind "$m")/"
  done
  "$W/venv/bin/python" -I "$H/texts.py" "$GOLDEN_DIR/potion-base-8M/texts.json" "$W/texts"
}

# run TOOL MODEL TEXTS BATCHES [OUT]: one tool on a copy of the model.
run() {
  local tool=$1 m=$2 texts=$3 batches=$4 out=${5:-}
  case $tool in
    statembed) "$W/target-rs/release/statembed" "$W/models/$m" "$texts" "$batches" $out;;
    statembed-rayon) "$W/target-rs-rayon/release/statembed" "$W/models/$m" "$texts" "$batches" $out;;
    model2vec-rs) "$W/target-rs/release/m2vrs" "$W/models/$m" "$texts" "$batches" $out;;
    go-potion) GO_POTION_HOME="$W/gph" "$W/go-potion-bench" "$(kind "$m")" "$texts" "$batches" $out;;
    model2vec-zig) zig_ok "$m" && "$W/zig-out/bin/m2vzig" "$W/models/$m" "$texts" "$batches" $out;;
    model2vec) "$W/venv/bin/python" -I "$H/py/bench.py" model2vec "$W/models/$m" "$texts" "$batches" $out;;
    st) "$W/venv/bin/python" -I "$H/py/bench.py" st "$W/models/$m" "$texts" "$batches" $out;;
  esac
}

accuracy() {
  : "${GOLDEN_DIR:?}"
  for m in $MODELS; do
    G=$GOLDEN_DIR/$m
    for tool in statembed model2vec-rs go-potion model2vec-zig model2vec st; do
      o=$W/out/$tool-$m.f32; rm -f "$o"
      if run $tool "$m" "$G/texts.json" - "$o" > "$W/out/$tool-$m.log" 2>&1 && [ -f "$o" ]; then
        "$W/venv/bin/python" -I "$H/score.py" "$G/golden-512.safetensors" "$o" "$tool $m"
      else
        echo "$tool $m: not run ($(tail -1 "$W/out/$tool-$m.log"))"
      fi
    done
  done
}

timing() {
  : "${TURBO_BUNDLES:?}"
  uptime
  for m in $TIMED; do
    for set in golden t60 w256; do
      echo "== $m $set"
      (cd "$REPO" && TURBO_SPEED_BUNDLE=$TURBO_BUNDLES/$m TURBO_SPEED_TEXTS=$W/texts/$set.json TURBO_SPEED_BATCHES=$BATCHES \
        cargo test --release -q -p turbo --test static_speed -- --nocapture 2>&1 | grep -E "loaded|turbo ")
      for tool in statembed statembed-rayon model2vec-rs go-potion model2vec-zig model2vec st; do
        run $tool "$m" "$W/texts/$set.json" "$BATCHES" 2>&1 | grep -E "loaded|texts/s" || echo "$tool: not run"
      done
    done
  done
  uptime
}

"$1"
