#!/bin/sh
# Build mlx.metallib from mlx-swift's non-JIT Metal kernels.
#
# SwiftPM compiles Cmlx into libMlxEngine.dylib but does not emit the
# default metallib (PrepareMetalShaders was removed; Xcode apps get
# default.metallib from the app target). MLX's device.cpp looks next to
# the loaded image (dladdr → this dylib) for mlx.metallib / default.metallib.
#
# No Python. POSIX + xcrun only.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
OUT="${1:-$ROOT/.build/release}"
CHECKOUT="$ROOT/.build/checkouts/mlx-swift"

# `metal` / `metallib` live in the full Xcode toolchain. The Command Line
# Tools package (xcode-select default on many Macs) does not ship them.
if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app/Contents/Developer ]; then
    DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
    export DEVELOPER_DIR
fi
if ! xcrun --find metal >/dev/null 2>&1; then
    echo "error: xcrun cannot find 'metal'. Install Xcode (not just CLT) or set DEVELOPER_DIR." >&2
    exit 1
fi
MLX_ROOT="$CHECKOUT/Source/Cmlx/mlx"
KERNELS="$MLX_ROOT/mlx/backend/metal/kernels"

if [ ! -d "$KERNELS" ]; then
    echo "error: mlx-swift checkout missing at $KERNELS (run swift build first)" >&2
    exit 1
fi

# Kernels that stay in the default metallib when MLX_METAL_JIT is on
# (see mlx/backend/metal/kernels/CMakeLists.txt). The rest JIT from
# pre-generated C++ strings in mlx-generated/.
KERNELS_ALWAYS="
arg_reduce
conv
gemv
layer_norm
random
rms_norm
rope
scaled_dot_product_attention
fence
"

newest_src=$(find "$KERNELS" -name '*.metal' -o -name '*.h' | xargs stat -f '%m' 2>/dev/null | sort -n | tail -1)
if [ -f "$OUT/mlx.metallib" ] && [ -n "${newest_src:-}" ]; then
    lib_mtime=$(stat -f '%m' "$OUT/mlx.metallib")
    if [ "$lib_mtime" -ge "$newest_src" ]; then
        echo "mlx.metallib up to date at $OUT/mlx.metallib"
        exit 0
    fi
fi

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

air_files=""
failed=""
for k in $KERNELS_ALWAYS; do
    src="$KERNELS/$k.metal"
    if [ ! -f "$src" ]; then
        echo "skip $k (no $src)"
        continue
    fi
    stem=$(basename "$k")
    echo "metal $k"
    if ! xcrun -sdk macosx metal \
        -x metal \
        -Wall -Wextra \
        -fno-fast-math \
        -Wno-c++17-extensions \
        -Wno-c++20-extensions \
        -mmacosx-version-min=14.0 \
        -c "$src" \
        -I "$MLX_ROOT" \
        -o "$WORKDIR/$stem.air"; then
        echo "warning: metal compile failed for $k" >&2
        failed="$failed $k"
        continue
    fi
    air_files="$air_files $WORKDIR/$stem.air"
done

# shellcheck disable=SC2086
set -- $air_files
if [ $# -eq 0 ]; then
    echo "error: no Metal .air files produced$failed" >&2
    exit 1
fi

echo "metallib $# kernels -> mlx.metallib"
xcrun -sdk macosx metallib "$@" -o "$WORKDIR/mlx.metallib"

mkdir -p "$OUT/Resources"
cp "$WORKDIR/mlx.metallib" "$OUT/mlx.metallib"
cp "$WORKDIR/mlx.metallib" "$OUT/default.metallib"
cp "$WORKDIR/mlx.metallib" "$OUT/Resources/mlx.metallib"
cp "$WORKDIR/mlx.metallib" "$OUT/Resources/default.metallib"

bytes=$(wc -c < "$OUT/mlx.metallib" | tr -d ' ')
echo "wrote $OUT/mlx.metallib ($bytes bytes)${failed:+; skipped:$failed}"
