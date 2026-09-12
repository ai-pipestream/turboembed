#!/bin/sh
# Build mlx.metallib next to the Swift inferstream-apple binary.
#
# Cmlx is statically linked into the executable, so MLX's
# current_binary_dir() (dladdr) is swift/.build/release. Put mlx.metallib
# there. Reuses the kernel list from native/mlx-engine/build-metallib.sh.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
OUT="${1:-$ROOT/swift/.build/release}"
CHECKOUT="$ROOT/swift/.build/checkouts/mlx-swift"

if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app/Contents/Developer ]; then
    DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
    export DEVELOPER_DIR
fi
if ! xcrun --find metal >/dev/null 2>&1; then
    echo "error: xcrun cannot find 'metal'. Install Xcode or set DEVELOPER_DIR." >&2
    exit 1
fi
MLX_ROOT="$CHECKOUT/Source/Cmlx/mlx"
KERNELS="$MLX_ROOT/mlx/backend/metal/kernels"
if [ ! -d "$KERNELS" ]; then
    echo "error: mlx-swift checkout missing at $KERNELS (run swift build first)" >&2
    exit 1
fi

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
# SwiftPM also exposes the triple-specific dir.
TRIPLE_OUT="$ROOT/swift/.build/arm64-apple-macosx/release"
if [ -d "$TRIPLE_OUT" ] && [ "$TRIPLE_OUT" != "$OUT" ]; then
    mkdir -p "$TRIPLE_OUT/Resources"
    cp "$WORKDIR/mlx.metallib" "$TRIPLE_OUT/mlx.metallib"
    cp "$WORKDIR/mlx.metallib" "$TRIPLE_OUT/default.metallib"
    cp "$WORKDIR/mlx.metallib" "$TRIPLE_OUT/Resources/mlx.metallib"
    cp "$WORKDIR/mlx.metallib" "$TRIPLE_OUT/Resources/default.metallib"
fi

bytes=$(wc -c < "$OUT/mlx.metallib" | tr -d ' ')
echo "wrote $OUT/mlx.metallib ($bytes bytes)${failed:+; skipped:$failed}"
