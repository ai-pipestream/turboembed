#!/usr/bin/env bash
# Build libturbo, compile the C smoke test against the public header only, and run it.
# Usage: scripts/c-smoke.sh [--release]
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
profile=debug
cargo_flags=()
if [[ "${1:-}" == "--release" ]]; then profile=release; cargo_flags=(--release); fi
cargo build -p turbo-shared "${cargo_flags[@]}"
libdir="$root/target/$profile"
out="$root/target/$profile/turbo-c-smoke"
cc="${CC:-cc}"
"$cc" -std=c11 -Wall -Wextra -Wpedantic -Werror -I"$root/include" \
    "$root/crates/turbo-conformance/c/smoke.c" -L"$libdir" -lturbo -lm -Wl,-rpath,"$libdir" -o "$out"
"$out" "$root/testdata/bundles/mock"
