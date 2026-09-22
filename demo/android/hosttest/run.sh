#!/usr/bin/env bash
# Device-free check of the Android JNI shim: compile jni/turbo_jni.c with the
# host compiler against the JDK's JNI headers, compile the Java classes the
# app uses, and run a small driver against the mock bundle through
# System.loadLibrary, exactly the call path the app takes. Needs a JDK
# (JAVA_HOME or java on PATH) and libturbo (cargo build -p turbo-shared).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
java_home="${JAVA_HOME:-$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")}"
lib_dir="${TURBO_LIB_DIR:-$root/target/debug}"
[[ -f "$lib_dir/libturbo.so" ]] || { echo "missing $lib_dir/libturbo.so; run: cargo build -p turbo-shared" >&2; exit 2; }
out="$here/build"
rm -rf "$out"; mkdir -p "$out/classes"
cc -std=c11 -Wall -Wextra -Werror -fPIC -shared -I"$root/include" -I"$java_home/include" -I"$java_home/include/linux" \
    "$root/demo/android/jni/turbo_jni.c" -L"$lib_dir" -lturbo -Wl,-rpath,"$lib_dir" -o "$out/libturbo_jni.so"
"$java_home/bin/javac" -d "$out/classes" \
    "$root/demo/android/app/src/main/java/ai/pipestream/turbo/android/TurboException.java" \
    "$root/demo/android/app/src/main/java/ai/pipestream/turbo/android/TurboJni.java" \
    "$root/demo/android/app/src/main/java/ai/pipestream/turbo/android/TurboEngine.java" \
    "$here/HostMain.java"
"$java_home/bin/java" --enable-native-access=ALL-UNNAMED -Djava.library.path="$out" -cp "$out/classes" ai.pipestream.turbo.android.HostMain \
    "${TURBO_DEMO_BUNDLE:-$root/testdata/bundles/mock/embedding}"
