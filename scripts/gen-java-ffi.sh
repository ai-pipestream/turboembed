#!/usr/bin/env bash
# Regenerate the raw Java FFM layer (bindings/java/src/main/java/ai/pipestream/turbo/ffi)
# from include/turbo/turbo.h with jextract.
#
# Requires JEXTRACT (path to the jextract executable, jextract 22 or newer)
# and a JDK 22+ on JAVA_HOME. The generated sources are committed; run this
# after any header change and commit the result together with the header.
#
#   JEXTRACT=~/opt/jextract/jextract-22/bin/jextract scripts/gen-java-ffi.sh [--check]
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
out="$root/bindings/java/src/main/java"
pkg=ai.pipestream.turbo.ffi
: "${JEXTRACT:?set JEXTRACT to the jextract executable}"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Only turbo_* / TURBO_* symbols: the header pulls in stdint/stddef, whose
# declarations must not become part of the binding.
"$JEXTRACT" --dump-includes "$tmp/all.txt" -I "$root/include" "$root/include/turbo/turbo.h" >/dev/null 2>&1
grep -E 'turbo|TURBO' "$tmp/all.txt" > "$tmp/turbo.txt"
"$JEXTRACT" --output "$tmp/gen" --target-package "$pkg" --header-class-name TurboNative \
    "@$tmp/turbo.txt" -I "$root/include" "$root/include/turbo/turbo.h" 2>&1 \
    | grep -v 'Skipping turbo_' || true

# The library is located through the `turbo.library` system property (a path
# to libturbo) or the TURBO_LIBRARY environment variable, and only then
# through the loader's own search; jextract's default is the search alone.
python3 - "$tmp/gen/ai/pipestream/turbo/ffi/TurboNative.java" <<'EOF'
import sys, re
p = sys.argv[1]
s = open(p).read()
old = re.search(r'    static final SymbolLookup SYMBOL_LOOKUP = .*?;\n', s, re.S).group(0)
new = '''    static final SymbolLookup SYMBOL_LOOKUP = lookup();

    private static SymbolLookup lookup() {
        String explicit = System.getProperty("turbo.library", System.getenv("TURBO_LIBRARY"));
        if (explicit != null && !explicit.isEmpty()) {
            return SymbolLookup.libraryLookup(java.nio.file.Path.of(explicit), LIBRARY_ARENA)
                    .or(SymbolLookup.loaderLookup())
                    .or(Linker.nativeLinker().defaultLookup());
        }
        return SymbolLookup.libraryLookup(System.mapLibraryName("turbo"), LIBRARY_ARENA)
                .or(SymbolLookup.loaderLookup())
                .or(Linker.nativeLinker().defaultLookup());
    }
'''
s = s.replace(old, new, 1)
open(p, 'w').write(s)
EOF

if [ "${1:-}" = "--check" ]; then
    if ! diff -r "$tmp/gen/ai/pipestream/turbo/ffi" "$out/ai/pipestream/turbo/ffi" >/dev/null; then
        echo "bindings/java ffi sources are stale; run scripts/gen-java-ffi.sh" >&2
        exit 1
    fi
    echo "java ffi sources are up to date"
    exit 0
fi
rm -rf "$out/ai/pipestream/turbo/ffi"
mkdir -p "$out/ai/pipestream/turbo"
cp -r "$tmp/gen/ai/pipestream/turbo/ffi" "$out/ai/pipestream/turbo/ffi"
echo "wrote $(ls "$out/ai/pipestream/turbo/ffi" | wc -l) files to $out/ai/pipestream/turbo/ffi"
