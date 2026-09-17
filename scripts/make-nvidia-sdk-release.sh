#!/usr/bin/env bash
# Build the TurboEmbed NVIDIA (ORT CUDA) C ABI into a relocatable release
# archive: <output-dir>/turboembed-cuda-sdk-<version>-linux-x86_64.tar.gz plus
# a .sha256 sibling, staged in a clean prefix, never from a developer's
# working install.
#
# The archive contains lib/libturboembed.so.1 (the frozen turboembed.h ABI,
# built from crates/turboembed-cabi with --features ort-cuda; the ONNX Runtime
# core is statically linked), the dynamically loaded ORT CUDA / TensorRT
# provider plugins, include/turboembed.h, a CMake package, the external
# consumer example, a catalog template, SHA-256 model pins for the qualified
# MiniLM source, license texts, the exported-symbol record, and a hashed file
# manifest (share/turboembed/sdk-manifest.json).
#
# NOT packaged (the host provides them, like the Intel package's OpenCL ICD
# and GPU driver): the NVIDIA driver, libcudart, cuBLAS, and cuDNN. See the
# packaged README-runtime.md for the qualified versions.
#
# No model is downloaded and no inference runs.
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

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HEADER="$ROOT/include/turboembed.h"
VERSION="$(sed -n 's/^version = "\([0-9.]*\)"/\1/p' "$ROOT/Cargo.toml" | head -n1)"
[ -n "$VERSION" ] || { echo "error: could not read workspace version" >&2; exit 1; }
NAME="turboembed-cuda-sdk-$VERSION-linux-x86_64"

GIT_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ]; then
  GIT_COMMIT="$GIT_COMMIT-dirty"
fi

echo "== build libturboembed (ort-cuda) =="
cargo build --locked --release -p turboembed-cabi --features ort-cuda \
  --manifest-path "$ROOT/Cargo.toml"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
PREFIX="$STAGE/$NAME"
mkdir -p "$PREFIX/lib/cmake/TurboEmbed" "$PREFIX/include" \
  "$PREFIX/share/turboembed/examples" "$PREFIX/share/turboembed/model-pins" \
  "$PREFIX/share/licenses/turboembed"

cp "$ROOT/target/release/libturboembed.so" "$PREFIX/lib/libturboembed.so.1"
ln -s libturboembed.so.1 "$PREFIX/lib/libturboembed.so"
# ORT provider plugins are dlopened at session creation; dereference the
# target-dir symlinks into real files. TensorRT plugins are optional at
# runtime but packaged so TURBOEMBED_DEVICE_TENSORRT can resolve.
for plugin in libonnxruntime_providers_shared.so \
              libonnxruntime_providers_cuda.so \
              libonnxruntime_providers_tensorrt.so; do
  cp -L "$ROOT/target/release/$plugin" "$PREFIX/lib/$plugin"
done

cp "$HEADER" "$PREFIX/include/turboembed.h"
cp "$ROOT/native/turboembed/cabi/TurboEmbedConfig.cmake.in" \
  "$PREFIX/lib/cmake/TurboEmbed/TurboEmbedConfig.cmake"
cp "$ROOT/native/turboembed/cabi/examples/embed.c" \
   "$ROOT/native/turboembed/cabi/examples/CMakeLists.txt" \
   "$PREFIX/share/turboembed/examples/"
cp "$ROOT/LICENSE" "$PREFIX/share/licenses/turboembed/LICENSE"

# Model pins for the qualified MiniLM source (same revision the Intel
# prepared SDK qualifies). Provisioning happens through inferstream-fetch or
# any tool that reproduces these hashes; the acceptance script re-verifies
# the files before running. The turboembed.h loader itself does not verify
# hashes at load time — that gap is recorded in the packaged README.
python3 - "$ROOT/models/manifests/embeddings.json" \
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
# Explicit model catalog for the packaged turboembed.h ABI. Copy this file,
# point `path` / `tokenizer_dir` at a provisioned model directory whose files
# match share/turboembed/model-pins/minilm.json, and pass the copy's path to
# turboembed_engine_create. Nothing here depends on a developer's model cache.
[models.minilm]
description = "all-MiniLM-L6-v2 sentence embeddings (384 dims, mean pooling)"

[models.minilm.nvidia]
backend = "ort"
device = "cuda"
path = "/absolute/path/to/minilm/onnx/model.onnx"
tokenizer_dir = "/absolute/path/to/minilm"
pooling = "mean"
normalize = true
max_seq_len = 256
max_batch_size = 32
EOF

cat > "$PREFIX/share/turboembed/README-runtime.md" <<EOF
# Host runtime requirements

Packaged: libturboembed.so.1 (ONNX Runtime core statically linked) and the
ORT CUDA / TensorRT provider plugins in lib/.

The host provides, and this package deliberately does not redistribute:

- NVIDIA driver (qualified with 595.84 on Machine A)
- libcudart (the library links the system CUDA runtime)
- CUDA user-space libraries for the ORT CUDA EP: cuBLAS, cuDNN 9
  (for example the pip packages nvidia-cu13 / nvidia-cudnn-cu13, exported on
  LD_LIBRARY_PATH), matching docs/turboembed.md
- TensorRT 10 (libnvinfer.so.10) only when TURBOEMBED_DEVICE_TENSORRT is used

Device policy: requesting CUDA or TensorRT without working hardware/runtime
fails loudly; CPU is only ever an explicit selection.

Model loading takes the catalog file passed to turboembed_engine_create;
verify provisioned files against share/turboembed/model-pins/ before use
(scripts and the acceptance flow do this). The loader does not re-hash files
at load time; that is a recorded difference from the Intel prepared SDK
bundle loader.
EOF

echo "== gate the dynamic symbol table =="
LIB="$PREFIX/lib/libturboembed.so.1"
nm -D --defined-only "$LIB" \
  | awk '$2 ~ /^[TW]$/ { print $3 }' | sort -u > "$STAGE/actual-symbols.txt"
grep -oE '^(turboembed_status|void|uint32_t|const char \*)[a-z_ *]*turboembed_[a-z_0-9]+' "$HEADER" \
  | grep -oE 'turboembed_[a-z_0-9]+$' | sort -u > "$STAGE/declared-symbols.txt"
# Every header symbol must be exported at the versioned TURBOEMBED_1 node.
while read -r sym; do
  if ! grep -q "^${sym}@@TURBOEMBED_1$" "$STAGE/actual-symbols.txt"; then
    echo "error: $sym is not exported at TURBOEMBED_1" >&2
    exit 1
  fi
done < "$STAGE/declared-symbols.txt"
# Anything else must be an allowlisted internal hook (the Rust provider
# entry points stub.cpp calls; rustc force-exports #[no_mangle] items past
# the version script). No other leakage is allowed.
if grep -v '@@TURBOEMBED_1$' "$STAGE/actual-symbols.txt" \
  | grep -vE '^turboembed_ort_[a-z_0-9]+$' | grep -q .; then
  echo "error: unexpected exported symbols:" >&2
  grep -v '@@TURBOEMBED_1$' "$STAGE/actual-symbols.txt" \
    | grep -vE '^turboembed_ort_[a-z_0-9]+$' >&2
  exit 1
fi
cp "$STAGE/actual-symbols.txt" "$PREFIX/share/turboembed/exported-symbols.txt"

ORT_BUILD="$(strings "$PREFIX/lib/libturboembed.so.1" | grep 'git-branch=rel-' | head -n1)"
[ -n "$ORT_BUILD" ] || ORT_BUILD=unknown

python3 - "$PREFIX" "$VERSION" "$GIT_COMMIT" "$ORT_BUILD" <<'EOF'
import hashlib, json, sys
from pathlib import Path

prefix = Path(sys.argv[1]).resolve()
version, git_commit, ort_build = sys.argv[2:5]
files, symlinks = {}, {}
for path in sorted(prefix.rglob("*")):
    rel = str(path.relative_to(prefix))
    if path.is_symlink():
        symlinks[rel] = str(path.readlink())
    elif path.is_file():
        files[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {
    "schema_version": 1,
    "name": "turboembed-cuda-sdk",
    "version": version,
    "target": "linux-x86_64",
    "git_commit": git_commit,
    "ort_build": ort_build,
    "files": files,
    "symlinks": symlinks,
}
out = prefix / "share/turboembed/sdk-manifest.json"
out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
EOF

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
TARBALL="$OUTPUT_DIR/$NAME.tar.gz"
tar -C "$STAGE" --sort=name --owner=0 --group=0 --numeric-owner -cf - "$NAME" \
  | gzip -n > "$TARBALL"
(cd "$OUTPUT_DIR" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")

echo "release: $TARBALL"
echo "sdk version: $VERSION ($GIT_COMMIT)"
echo "ort build: $ORT_BUILD"
