#!/usr/bin/env bash
# Build the TurboEmbed prepared native SDK into a relocatable release archive.
#
# Produces <output-dir>/turboembed-prepared-sdk-<version>-linux-x86_64.tar.gz
# (plus a .sha256 sibling) from a clean staging prefix, never from a
# developer's working install. The archive contains the shared library,
# public header, provisioning tools, packaged OpenVINO/TBB runtime with
# licenses, the external consumer example, the exported-symbol list
# (share/turboembed/exported-symbols.txt), and a hashed file manifest
# (share/turboembed/sdk-manifest.json).
#
# Requires Linux x86_64, CMake 3.20+, a C++17 compiler, OpenCL development
# files, Python 3, and an extracted OpenVINO archive (--openvino-root).
# No model is downloaded and no inference runs.
set -euo pipefail

usage() {
  echo "usage: $0 --openvino-root DIR [--output-dir DIR] [--jobs N]" >&2
  exit 2
}

OPENVINO_ROOT="${OPENVINO_ROOT:-}"
OUTPUT_DIR="dist"
JOBS="$(nproc 2>/dev/null || echo 2)"
while [ $# -gt 0 ]; do
  case "$1" in
    --openvino-root) OPENVINO_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "$OPENVINO_ROOT" ] || usage
[ -f "$OPENVINO_ROOT/runtime/version.txt" ] || {
  echo "error: $OPENVINO_ROOT does not look like an extracted OpenVINO archive (missing runtime/version.txt)" >&2
  exit 1
}
# The OpenVINO CMake package needs setupvars.sh's environment so the linker
# resolves libopenvino's TBB dependency from the distribution.
set +u
# shellcheck disable=SC1091
source "$OPENVINO_ROOT/setupvars.sh" >/dev/null
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HEADER="$ROOT/include/turboembed_prepared.h"
VERSION="$(sed -n 's/^project(TurboEmbedPrepared VERSION \([0-9.]*\).*/\1/p' \
  "$ROOT/native/turboembed/sdk/CMakeLists.txt")"
[ -n "$VERSION" ] || { echo "error: could not read SDK version from CMakeLists.txt" >&2; exit 1; }
NAME="turboembed-prepared-sdk-$VERSION-linux-x86_64"

GIT_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ]; then
  GIT_COMMIT="$GIT_COMMIT-dirty"
fi
OPENVINO_BUILD="$(head -n1 "$OPENVINO_ROOT/runtime/version.txt")"

STAGE="$(mktemp -d)"
BUILD="$(mktemp -d)"
trap 'rm -rf "$STAGE" "$BUILD"' EXIT
PREFIX="$STAGE/$NAME"

cmake -S "$ROOT/native/turboembed/sdk" -B "$BUILD" \
  -DOpenVINO_DIR="$OPENVINO_ROOT/runtime/cmake" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib
cmake --build "$BUILD" --parallel "$JOBS"
cmake --install "$BUILD" --prefix "$PREFIX"
python3 "$ROOT/scripts/package-native-runtime.py" \
  --openvino-root "$OPENVINO_ROOT" --prefix "$PREFIX"

# The dynamic symbol table must contain exactly the versioned prepared API:
# every symbol the public header declares, and nothing else.
LIB="$(readlink -f "$PREFIX/lib/libturboembed_prepared.so.1")"
nm -D --defined-only "$LIB" \
  | awk '$2 ~ /^[TW]$/ { sub(/@.*/, "", $3); print $3 }' | sort -u \
  > "$STAGE/actual-symbols.txt"
grep -o 'turboembed_prepared_v1_[a-z_0-9]*' "$HEADER" | sort -u \
  > "$STAGE/declared-symbols.txt"
if ! diff -u "$STAGE/declared-symbols.txt" "$STAGE/actual-symbols.txt"; then
  echo "error: exported symbols do not match include/turboembed_prepared.h" >&2
  exit 1
fi
mkdir -p "$PREFIX/share/turboembed"
cp "$STAGE/actual-symbols.txt" "$PREFIX/share/turboembed/exported-symbols.txt"

# Hash every installed file; record symlinks by target. Written last so the
# manifest covers the complete artifact.
python3 - "$PREFIX" "$VERSION" "$GIT_COMMIT" "$OPENVINO_BUILD" <<'EOF'
import hashlib, json, sys
from pathlib import Path

prefix = Path(sys.argv[1]).resolve()
version, git_commit, openvino_build = sys.argv[2:5]
files, symlinks = {}, {}
for path in sorted(prefix.rglob("*")):
    rel = str(path.relative_to(prefix))
    if path.is_symlink():
        symlinks[rel] = str(path.readlink())
    elif path.is_file():
        files[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {
    "schema_version": 1,
    "name": "turboembed-prepared-sdk",
    "version": version,
    "target": "linux-x86_64",
    "git_commit": git_commit,
    "openvino_build": openvino_build,
    "files": files,
    "symlinks": symlinks,
}
out = prefix / "share/turboembed/sdk-manifest.json"
out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
EOF

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
TARBALL="$OUTPUT_DIR/$NAME.tar.gz"
# Deterministic member order/ownership; gzip -n omits the timestamp header.
# Content timestamps are the build's; the hashed sdk-manifest.json is the
# integrity reference for the extracted tree.
tar -C "$STAGE" --sort=name --owner=0 --group=0 --numeric-owner -cf - "$NAME" \
  | gzip -n > "$TARBALL"
(cd "$OUTPUT_DIR" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")

echo "release: $TARBALL"
echo "sdk version: $VERSION ($GIT_COMMIT)"
echo "openvino runtime: $OPENVINO_BUILD"
