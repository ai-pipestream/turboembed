#!/usr/bin/env bash
# Build native/turbo_buffer/build/libturbo_buffer_apple.a (Metal SHARED).
# Linked into libTurboEmbed.dylib so Swift/MLX embed rents arena MTLBuffers.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/native/turbo_buffer/build"
mkdir -p "$OUT"
CXX="${CXX:-c++}"
INC=(-I"$ROOT/include" -I"$ROOT/native/turbo_buffer/src")
FLAGS=(-std=c++17 -O2 -fPIC -DTURBO_BUFFER_METAL=1 -mmacosx-version-min=15.0)
for src in arena.cpp cuda.cpp ze.cpp metal.cpp; do
  "$CXX" "${FLAGS[@]}" "${INC[@]}" \
    -c "$ROOT/native/turbo_buffer/src/$src" \
    -o "$OUT/$src.o"
done
"$CXX" "${FLAGS[@]}" -fobjc-arc "${INC[@]}" \
  -c "$ROOT/native/turbo_buffer/src/metal.mm" \
  -o "$OUT/metal.mm.o"
ar rcs "$OUT/libturbo_buffer_apple.a" \
  "$OUT/arena.cpp.o" \
  "$OUT/cuda.cpp.o" \
  "$OUT/ze.cpp.o" \
  "$OUT/metal.cpp.o" \
  "$OUT/metal.mm.o"
nm -g "$OUT/libturbo_buffer_apple.a" | grep -q turbo_buffer_arena_rent
nm -g "$OUT/libturbo_buffer_apple.a" | grep -q turbo_buffer_metal_owns
nm -g "$OUT/libturbo_buffer_apple.a" | grep -q turbo_buffer_metal_lookup
echo "wrote $OUT/libturbo_buffer_apple.a (Metal SHARED)"
