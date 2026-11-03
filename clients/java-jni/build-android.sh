#!/usr/bin/env bash
# Builds libturbo.so for Android through the NDK, for the arm64-v8a and
# x86_64 ABIs, into a jniLibs layout an Android application module picks
# up as is (clients/java-jni/README.md, "Android").
#
#   ANDROID_NDK_HOME=/path/to/ndk clients/java-jni/build-android.sh [out-dir]
#
# out-dir defaults to clients/java-jni/jniLibs. The Rust targets are added
# with rustup when missing. The API level is 26, the lowest the binding
# supports; TURBO_ANDROID_API overrides it.
set -euo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
out="${1:-$repo/clients/java-jni/jniLibs}"
api="${TURBO_ANDROID_API:-26}"
ndk="${ANDROID_NDK_HOME:-${ANDROID_NDK_LATEST_HOME:-${ANDROID_NDK_ROOT:-}}}"
if [ -z "$ndk" ] || [ ! -d "$ndk/toolchains/llvm/prebuilt" ]; then
  echo "ANDROID_NDK_HOME must name an Android NDK with an llvm toolchain" >&2
  exit 2
fi

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) host=linux-x86_64 ;;
  Darwin-*) host=darwin-x86_64 ;;
  *) echo "no NDK host toolchain for $(uname -s)-$(uname -m)" >&2; exit 2 ;;
esac
bin="$ndk/toolchains/llvm/prebuilt/$host/bin"

build() {
  local target="$1" abi="$2" triple="$3"
  rustup target add "$target" >/dev/null
  local linker="$bin/${triple}${api}-clang"
  [ -x "$linker" ] || { echo "missing $linker" >&2; exit 2; }
  local var="CARGO_TARGET_$(echo "$target" | tr 'a-z-' 'A-Z_')_LINKER"
  env "$var=$linker" "CC_$target=$linker" "AR_$target=$bin/llvm-ar" \
    cargo build --locked --release -p turbo --target "$target" --manifest-path "$repo/Cargo.toml"
  mkdir -p "$out/$abi"
  cp "$repo/target/$target/release/libturbo.so" "$out/$abi/libturbo.so"
  echo "$out/$abi/libturbo.so"
}

build aarch64-linux-android arm64-v8a aarch64-linux-android
build x86_64-linux-android x86_64 x86_64-linux-android
