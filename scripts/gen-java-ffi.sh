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
#
# SymbolLookup.libraryLookup throws IllegalArgumentException when the name or
# path does not identify a loadable library; it does not return an empty
# lookup, so `.or(...)` after it is never reached. The name-based branch
# therefore catches that and falls through to the loader, which is what a
# host that already called System.loadLibrary("turbo") needs. An explicit
# turbo.library / TURBO_LIBRARY path stays a hard failure: the caller named
# the library, so a library that cannot be loaded is an error, not a reason
# to use a different one.
python3 - "$tmp/gen/ai/pipestream/turbo/ffi/TurboNative.java" <<'EOF'
import sys, re
p = sys.argv[1]
s = open(p).read()
old = re.search(r'    static final SymbolLookup SYMBOL_LOOKUP = .*?;\n', s, re.S).group(0)
new = '''    static final SymbolLookup SYMBOL_LOOKUP = lookup();

    private static SymbolLookup lookup() {
        String explicit = System.getProperty("turbo.library", System.getenv("TURBO_LIBRARY"));
        if (explicit != null && !explicit.isEmpty()) {
            // Named by the caller: load that library or fail with what is wrong with it.
            return SymbolLookup.libraryLookup(java.nio.file.Path.of(explicit), LIBRARY_ARENA);
        }
        try {
            return SymbolLookup.libraryLookup(System.mapLibraryName("turbo"), LIBRARY_ARENA)
                    .or(SymbolLookup.loaderLookup())
                    .or(Linker.nativeLinker().defaultLookup());
        } catch (IllegalArgumentException notLoadable) {
            // libraryLookup throws rather than returning an empty lookup, so
            // this is the only way to reach a libturbo the host loaded itself.
            return SymbolLookup.loaderLookup().or(Linker.nativeLinker().defaultLookup());
        }
    }
'''
s = s.replace(old, new, 1)
open(p, 'w').write(s)
EOF

# jextract 22's indexed array accessors call the element var handle with a
# base offset of 0 instead of the field's offset, so on JDK 25 they read and
# write the wrong bytes (a `shape(desc, 0, n)` overwrote `struct_size`).
# Route them through the slice accessor, which is correct on every JDK.
python3 - "$tmp/gen/ai/pipestream/turbo/ffi" <<'EOF2'
import sys, re, pathlib
layouts = {'long': 'JAVA_LONG', 'int': 'JAVA_INT', 'short': 'JAVA_SHORT', 'byte': 'JAVA_BYTE',
           'float': 'JAVA_FLOAT', 'double': 'JAVA_DOUBLE', 'MemorySegment': 'ADDRESS'}
get = re.compile(r'return \((\w+)\)(\w+)\$ELEM_HANDLE\.get\(struct, 0L, index0\);')
put = re.compile(r'(\w+)\$ELEM_HANDLE\.set\(struct, 0L, index0, fieldValue\);')
for f in pathlib.Path(sys.argv[1]).glob('*.java'):
    s = f.read_text()
    def g(m):
        return f'return {m.group(2)}(struct).getAtIndex({layouts[m.group(1)]}, index0);'
    def p(m):
        # the setter's element type is the getter's, found on the line above
        t = re.search(r'public static (\w+) ' + m.group(1) + r'\(MemorySegment struct, long index0\)', s).group(1)
        return f'{m.group(1)}(struct).setAtIndex({layouts[t]}, index0, fieldValue);'
    s2 = put.sub(p, get.sub(g, s))
    if s2 != s:
        f.write_text(s2)
EOF2

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
