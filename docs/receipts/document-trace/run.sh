#!/usr/bin/env bash
set -euo pipefail
: "${TURBO_TRACE_CHECKOUT:?Set to a checkout containing PRs 495, 496 and 497}"
: "${TURBO_MASC_ROOT:?Set to the MASC-3.0.0/data directory}"
: "${TURBO_MINILM_BUNDLE:?Set to a verified all-MiniLM-L6-v2 bundle}"
checkout=$(realpath "$TURBO_TRACE_CHECKOUT")
case "$checkout" in
  *'"'*|*'\'*|*$'\n'*) echo 'Checkout path cannot contain quotes, backslashes or newlines' >&2; exit 2;;
esac
git -C "$checkout" diff --quiet HEAD -- analysis Cargo.toml Cargo.lock
export TURBO_TRACE_COMMIT
TURBO_TRACE_COMMIT=$(git -C "$checkout" rev-parse HEAD)
receipt_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output=$(realpath -m "${1:?Usage: run.sh OUTPUT_DIRECTORY}")
mkdir -p "$output"
build_dir=$(mktemp -d "$output/build.XXXXXX")
mkdir "$build_dir/src"
cp "$receipt_dir/main.rs" "$build_dir/src/main.rs"
cp "$receipt_dir/Cargo.lock" "$build_dir/Cargo.lock"
cat > "$build_dir/Cargo.toml" <<'TOML'
[package]
name = "turboembed-document-trace"
version = "0.1.0"
edition = "2024"
[workspace]
[dependencies]
serde_json = "1"
TOML
while read -r package directory; do
  printf '%s = { path = "%s/analysis/%s" }\n' "$package" "$checkout" "$directory" >> "$build_dir/Cargo.toml"
done <<'DEPS'
turbo-analysis core
turbo-analysis-layers layers
turbo-encoding encoding
turbo-encoding-layers encoding-layers
turbo-normalize normalize
turbo-sentdetect sentdetect
turbo-tokenize tokenize
turbo-structure structure
turbo-analysis-embed embed
turbo-chunker chunker
turbo-namefind namefind
turbo-maxent maxent
turbo-geo geo
turbo-numeric numeric
turbo-pii pii
turbo-stem stem
turbo-abbrev abbrev
DEPS
cargo run --locked --manifest-path "$build_dir/Cargo.toml" -- "$output"
