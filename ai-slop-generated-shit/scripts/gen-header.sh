#!/usr/bin/env bash
# Regenerate include/turbo/turbo_types.h and include/turbo/turbo.h from the Rust ABI crates.
# Usage: scripts/gen-header.sh [--check]
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
command -v cbindgen >/dev/null || { echo "cbindgen is not installed: cargo install cbindgen --locked" >&2; exit 2; }
gen() { # crate dir, config, output
    # turbo-capi and the provider header resolve their types from turbo_types.h,
    # so cbindgen's "Can't find" notes are expected there. The string constant
    # TURBO_PROVIDER_ENTRY_SYMBOL is documented in the header text instead.
    (cd "$root/crates/$1" && cbindgen --config "$2" --crate "$1" --output "$3" \
        2> >(grep -v -e "Can't find turbo_" -e "Skip turbo-abi::TURBO_PROVIDER_ENTRY_SYMBOL" >&2))
}
check="${1:-}"
status=0
for triple in "turbo-abi:cbindgen.toml:turbo_types.h" "turbo-abi:cbindgen-provider.toml:turbo_provider.h" "turbo-capi:cbindgen.toml:turbo.h"; do
    IFS=: read -r crate config name <<< "$triple"
    out="$root/include/turbo/$name"; tmp="$out.tmp"
    gen "$crate" "$config" "$tmp"
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
