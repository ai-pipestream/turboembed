#!/usr/bin/env bash
# Builds libturbo.so for Android, stages sealed testdata into the instrumentation
# APK's assets, and runs AnalyzerConformance, SessionConformance, and the JNI
# bridge tests on a connected emulator or device through System.loadLibrary.
#
#   ANDROID_NDK_HOME=... ANDROID_HOME=... clients/java-jni/run-android-conformance.sh
#
# Environment:
#   TURBO_ANDROID_ABI   comma list of ABIs to build (default: x86_64,arm64-v8a)
#   TURBO_ANDROID_API   NDK API level (default: 26)
#   CARGO_TARGET_DIR    forwarded to cargo / build-android.sh
set -euo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
module="$repo/clients/java-jni/android-conformance"
app="$module/app"
jni_libs="$app/src/main/jniLibs"
assets_testdata="$app/src/main/assets/testdata"
abis_csv="${TURBO_ANDROID_ABI:-x86_64,arm64-v8a}"
api="${TURBO_ANDROID_API:-26}"

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 2
  fi
}

need cargo
need rustup

ndk="${ANDROID_NDK_HOME:-${ANDROID_NDK_LATEST_HOME:-${ANDROID_NDK_ROOT:-}}}"
if [ -z "$ndk" ] || [ ! -d "$ndk/toolchains/llvm/prebuilt" ]; then
  echo "ANDROID_NDK_HOME must name an Android NDK with an llvm toolchain" >&2
  exit 2
fi
export ANDROID_NDK_HOME="$ndk"

sdk="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}"
if [ -z "$sdk" ] || [ ! -d "$sdk" ]; then
  echo "ANDROID_HOME or ANDROID_SDK_ROOT must name an Android SDK" >&2
  exit 2
fi
export ANDROID_HOME="$sdk"
export ANDROID_SDK_ROOT="$sdk"

echo "staging sealed testdata into APK assets"
rm -rf "$assets_testdata"
mkdir -p "$assets_testdata"
cp -a "$repo/testdata/tiny-bert-bundle" "$assets_testdata/tiny-bert-bundle"
cp -a "$repo/testdata/tiny-bert-heads-bundle" "$assets_testdata/tiny-bert-heads-bundle"

echo "building libturbo.so for API $api ($abis_csv)"
rm -rf "$jni_libs"
mkdir -p "$jni_libs"
# build-android.sh always builds both ABIs; copy only the ones requested.
"$repo/clients/java-jni/build-android.sh" "$jni_libs"
IFS=',' read -r -a abis <<< "$abis_csv"
for abi in arm64-v8a x86_64; do
  keep=0
  for want in "${abis[@]}"; do
    if [ "$abi" = "$want" ]; then
      keep=1
      break
    fi
  done
  if [ "$keep" -eq 0 ]; then
    rm -rf "${jni_libs:?}/$abi"
  else
    test -f "$jni_libs/$abi/libturbo.so" || {
      echo "missing $jni_libs/$abi/libturbo.so" >&2
      exit 1
    }
  fi
done

cd "$module"
rm -rf "$app/build/outputs/androidTest-results/connected"
./gradlew --no-daemon :helpers:test :app:connectedDebugAndroidTest :helpers:verifyAndroidTestResults

echo "android conformance finished"
