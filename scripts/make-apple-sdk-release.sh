#!/usr/bin/env bash
# Build the TurboEmbed Apple (MLX Metal) C ABI into a relocatable release
# archive: <output-dir>/turboembed-metal-sdk-<version>-macos-arm64.tar.gz plus
# a .sha256 sibling, staged in a clean prefix, never from a developer's
# working install. Apple analog of scripts/make-nvidia-sdk-release.sh.
#
# The archive contains lib/libTurboEmbed.dylib (the frozen turboembed.h ABI:
# Swift @_cdecl over mlx-swift on Metal, with the turbo_buffer Metal SHARED
# arena and native WordPiece statically linked), the MLX Metal kernel
# libraries (mlx.metallib / default.metallib, which must stay next to the
# dylib), include/turboembed.h, a CMake package, the external C consumer
# example, a documented Swift consumer example, a catalog template, SHA-256
# model pins for the qualified MiniLM MLX source, license texts, the
# exported-C-symbol record, and a hashed file manifest
# (share/turboembed/sdk-manifest.json).
#
# NOT packaged (the host provides them, like the NVIDIA package's driver and
# CUDA user-space): macOS 15+, the Metal framework/driver, and the Swift
# runtime libraries that ship with the OS. See the packaged
# README-runtime.md for the qualified versions.
#
# No model is downloaded and no inference runs. Must run on the Apple
# Silicon (Machine C) host — the produced dylib is arm64 Mach-O.
set -euo pipefail

usage() {
  echo "usage: $0 [--output-dir DIR]" >&2
  exit 2
}

OUTPUT_DIR="dist"
while [ $# -gt 0 ]; do
  case "$1" in
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) usage ;;
  esac
done

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "error: the Apple SDK is built on macOS arm64 (Machine C); this host is $(uname -sm)." >&2
  echo "Cross-building would ship a dylib that was never loaded — refusing." >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HEADER="$ROOT/include/turboembed.h"
VERSION="$(sed -n 's/^version = "\([0-9.]*\)"/\1/p' "$ROOT/Cargo.toml" | head -n1)"
[ -n "$VERSION" ] || { echo "error: could not read workspace version" >&2; exit 1; }
NAME="turboembed-metal-sdk-$VERSION-macos-arm64"

GIT_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ]; then
  GIT_COMMIT="$GIT_COMMIT-dirty"
fi

echo "== build libturbo_buffer_apple.a (Metal SHARED arena + WordPiece) =="
"$ROOT/scripts/build-turbo-buffer-apple.sh"

echo "== build libTurboEmbed.dylib (Swift @_cdecl over mlx-swift Metal) =="
swift build -c release --package-path "$ROOT/swift" --product TurboEmbed
DYLIB="$ROOT/swift/.build/release/libTurboEmbed.dylib"
[ -f "$DYLIB" ] || { echo "error: swift build did not emit $DYLIB" >&2; exit 1; }

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
PREFIX="$STAGE/$NAME"
mkdir -p "$PREFIX/lib/cmake/TurboEmbed" "$PREFIX/include" \
  "$PREFIX/share/turboembed/examples/swift" "$PREFIX/share/turboembed/model-pins" \
  "$PREFIX/share/licenses/turboembed"

cp "$DYLIB" "$PREFIX/lib/libTurboEmbed.dylib"
# Relocatable install name; re-sign after the edit (ad hoc — distribution
# signing is a release-management step, not a build step).
install_name_tool -id @rpath/libTurboEmbed.dylib "$PREFIX/lib/libTurboEmbed.dylib"
codesign --force --sign - "$PREFIX/lib/libTurboEmbed.dylib"

echo "== build the MLX Metal kernel libraries next to the dylib =="
# Cmlx is statically linked into the dylib; MLX resolves its metallib with
# dladdr, so mlx.metallib must live in the same directory as the dylib.
"$ROOT/scripts/build-apple-metallib.sh" "$PREFIX/lib"

cp "$HEADER" "$PREFIX/include/turboembed.h"
cp "$ROOT/native/turboembed/cabi/examples/embed.c" \
   "$ROOT/native/turboembed/cabi/examples/CMakeLists.txt" \
   "$PREFIX/share/turboembed/examples/"
cp "$ROOT/native/turboembed/cabi/examples/swift/main.swift" \
   "$PREFIX/share/turboembed/examples/swift/main.swift"
cp "$ROOT/LICENSE" "$PREFIX/share/licenses/turboembed/LICENSE"

cat > "$PREFIX/lib/cmake/TurboEmbed/TurboEmbedConfig.cmake" <<'EOF'
# CMake package for the installed turboembed.h ABI (Apple Metal SDK; the
# release script installs this as lib/cmake/TurboEmbed/TurboEmbedConfig.cmake).
get_filename_component(_te_prefix "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

add_library(TurboEmbed::turboembed SHARED IMPORTED)
set_target_properties(TurboEmbed::turboembed PROPERTIES
    IMPORTED_LOCATION "${_te_prefix}/lib/libTurboEmbed.dylib"
    INTERFACE_INCLUDE_DIRECTORIES "${_te_prefix}/include")
EOF

# Model pins for the qualified MiniLM MLX source (same revision the catalog
# and the Machine C contract tests qualify). Provisioning happens through
# `cargo xtask fetch --mlx minilm` or any tool that reproduces these hashes;
# the acceptance script re-verifies the files before running. The dylib's
# loader itself does not verify hashes at load time — that gap is recorded
# in the packaged README, same as the NVIDIA SDK.
python3 - "$ROOT/models/manifests/mlx.json" \
  "$PREFIX/share/turboembed/model-pins/minilm.json" <<'EOF'
import json, sys
manifest = json.load(open(sys.argv[1]))
entry = manifest["models"]["minilm"]
pins = {
    "alias": "minilm",
    "repo": entry["repo"],
    "revision": entry["revision"],
    "files": {f["path"]: {"sha256": f["sha256"], "size": f["size"]}
              for f in entry["files"]},
}
open(sys.argv[2], "w").write(json.dumps(pins, indent=2, sort_keys=True) + "\n")
EOF

cat > "$PREFIX/share/turboembed/catalog-template.toml" <<'EOF'
# Explicit model catalog for the packaged turboembed.h ABI (Apple / Metal).
# Copy this file, point `path` / `tokenizer_dir` at a provisioned MLX model
# directory whose files match share/turboembed/model-pins/minilm.json, and
# pass the copy's path to turboembed_engine_create. Nothing here depends on
# a developer's model cache.
[models.minilm]
description = "all-MiniLM-L6-v2 sentence embeddings (384 dims, mean pooling)"

[models.minilm.apple]
backend = "mlx"
path = "/absolute/path/to/minilm-mlx"
tokenizer_dir = "/absolute/path/to/minilm-mlx"
pooling = "mean"
normalize = true
max_seq_len = 256
max_batch_size = 32
EOF

SWIFT_VERSION="$(swift --version 2>/dev/null | head -n1)"
MACOS_VERSION="$(sw_vers -productVersion 2>/dev/null || echo unknown)"

cat > "$PREFIX/share/turboembed/README-runtime.md" <<EOF
# Host runtime requirements

Packaged: lib/libTurboEmbed.dylib (Swift @_cdecl implementation of the
frozen turboembed.h ABI over mlx-swift on Metal; the turbo_buffer Metal
SHARED arena and native WordPiece tokenizer are statically linked) and the
MLX Metal kernel libraries (mlx.metallib / default.metallib). The kernel
libraries are resolved relative to the dylib and must stay in the same
directory as lib/libTurboEmbed.dylib.

The host provides, and this package deliberately does not redistribute:

- macOS 15+ on Apple Silicon (built on $MACOS_VERSION)
- The Metal framework and GPU driver (part of macOS)
- The Swift runtime libraries that ship with macOS
- Xcode or the Command Line Tools only for building the consumer examples

Built with: $SWIFT_VERSION
mlx-swift / mlx-swift-lm pins: share/turboembed/swift-package-pins.json

Device policy: requesting METAL or AUTO without a Metal device fails
loudly; there is no CPU fallback for catalog models. Explicit CPU and MOCK
serve only the 8-d mock-embed ABI smoke alias — catalog aliases such as
minilm are refused on CPU with an error naming METAL/AUTO. CUDA / TensorRT
/ OpenVINO selections are refused by this dylib.

Model loading takes the catalog file passed to turboembed_engine_create;
verify provisioned files against share/turboembed/model-pins/ before use
(the acceptance flow does this). The loader does not re-hash files at load
time; that is the same recorded gap as the NVIDIA CUDA SDK.

Consumers: build share/turboembed/examples with CMake against this prefix
(C), or compile share/turboembed/examples/swift/main.swift with
\`swiftc main.swift -import-objc-header <prefix>/include/turboembed.h
-L <prefix>/lib -lTurboEmbed -Xlinker -rpath -Xlinker <prefix>/lib\`.
EOF

# Record the mlx-swift dependency pins the dylib was built from.
python3 - "$ROOT/swift/Package.resolved" \
  "$PREFIX/share/turboembed/swift-package-pins.json" <<'EOF'
import json, sys
resolved = json.load(open(sys.argv[1]))
pins = {}
for pin in resolved.get("pins", []):
    pins[pin["identity"]] = pin.get("state", {})
open(sys.argv[2], "w").write(json.dumps(pins, indent=2, sort_keys=True) + "\n")
EOF

echo "== gate the exported C symbol table =="
LIB="$PREFIX/lib/libTurboEmbed.dylib"
# Mach-O exports carry a leading underscore. A Swift dylib necessarily
# exports Swift runtime/metadata symbols too, so the gate here is: every
# turboembed.h symbol must be exported, and the recorded C surface is the
# turboembed_* / turbo_buffer_* / wordpiece_* families.
nm -gU "$LIB" | awk '$2 ~ /^[TS]$/ { sub(/^_/, "", $3); print $3 }' \
  | grep -E '^(turboembed|turbo_buffer|wordpiece)_' | sort -u \
  > "$STAGE/actual-symbols.txt"
grep -oE '^(turboembed_status|void|uint32_t|const char \*)[a-z_ *]*turboembed_[a-z_0-9]+' "$HEADER" \
  | grep -oE 'turboembed_[a-z_0-9]+$' | sort -u > "$STAGE/declared-symbols.txt"
while read -r sym; do
  if ! grep -q "^${sym}$" "$STAGE/actual-symbols.txt"; then
    echo "error: header symbol $sym is not exported by libTurboEmbed.dylib" >&2
    exit 1
  fi
done < "$STAGE/declared-symbols.txt"
cp "$STAGE/actual-symbols.txt" "$PREFIX/share/turboembed/exported-symbols.txt"

echo "== sanity-check the staged dylib identity =="
otool -D "$LIB" | tail -n1 | grep -q '^@rpath/libTurboEmbed.dylib$' || {
  echo "error: staged dylib install name is not @rpath/libTurboEmbed.dylib" >&2
  exit 1
}
lipo -archs "$LIB" | grep -q arm64 || {
  echo "error: staged dylib is not arm64" >&2
  exit 1
}

python3 - "$PREFIX" "$VERSION" "$GIT_COMMIT" "$SWIFT_VERSION" "$MACOS_VERSION" <<'EOF'
import hashlib, json, sys
from pathlib import Path

prefix = Path(sys.argv[1]).resolve()
version, git_commit, swift_version, macos_version = sys.argv[2:6]
files, symlinks = {}, {}
for path in sorted(prefix.rglob("*")):
    rel = str(path.relative_to(prefix))
    if path.is_symlink():
        symlinks[rel] = str(path.readlink())
    elif path.is_file():
        files[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {
    "schema_version": 1,
    "name": "turboembed-metal-sdk",
    "version": version,
    "target": "macos-arm64",
    "git_commit": git_commit,
    "swift_version": swift_version,
    "built_on_macos": macos_version,
    "files": files,
    "symlinks": symlinks,
}
out = prefix / "share/turboembed/sdk-manifest.json"
out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
EOF

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
TARBALL="$OUTPUT_DIR/$NAME.tar.gz"
tar -C "$STAGE" -czf "$TARBALL" "$NAME"
(cd "$OUTPUT_DIR" && shasum -a 256 "$NAME.tar.gz" > "$NAME.tar.gz.sha256")

echo "release: $TARBALL"
echo "sdk version: $VERSION ($GIT_COMMIT)"
echo "swift: $SWIFT_VERSION"
