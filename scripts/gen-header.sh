#!/usr/bin/env bash
# Regenerate include/turbo/turbo_types.h and include/turbo/turbo.h from the Rust ABI crates.
# Usage: scripts/gen-header.sh [--check]
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
command -v cbindgen >/dev/null || { echo "cbindgen is not installed: cargo install cbindgen --locked" >&2; exit 2; }
gen() { # crate dir, output
    # turbo-capi resolves its types from turbo_types.h, so cbindgen's "Can't find" notes are expected there.
    (cd "$root/crates/$1" && cbindgen --config cbindgen.toml --crate "$1" --output "$2" 2> >(grep -v "Can't find turbo_" >&2))
}
check="${1:-}"
status=0
for pair in "turbo-abi:turbo_types.h" "turbo-capi:turbo.h"; do
    crate="${pair%%:*}"; name="${pair##*:}"
    out="$root/include/turbo/$name"; tmp="$out.tmp"
    gen "$crate" "$tmp"
    if [[ "$check" == "--check" ]]; then
        if ! diff -u "$out" "$tmp"; then
            echo "include/turbo/$name is out of date; run scripts/gen-header.sh" >&2
            status=1
        fi
        rm -f "$tmp"
    else
        mv "$tmp" "$out"; echo "wrote $out"
    fi
done
[[ "$check" == "--check" && $status -eq 0 ]] && echo "headers are up to date"
exit $status
