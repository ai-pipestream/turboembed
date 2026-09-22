#!/usr/bin/env bash
# Build the Turbo Java binding and this app, then serve it.
#
#   demo/java-web-spring/run.sh                                   # mock bundle, port 8080
#   demo/java-web-spring/run.sh --turbo.bundle=/opt/bundles/minilm-onnx \
#       --turbo.provider-lib=build/openvino/libturbo_provider_openvino.so \
#       --turbo.provider=openvino --turbo.ordinal=1 --server.port=8080
#
# Needs JDK 25 (JAVA_HOME or PATH), Maven, and libturbo (cargo build -p turbo-shared).
# Provider runtimes (OpenVINO, CUDA) come from LD_LIBRARY_PATH as usual.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
lib="${TURBO_LIBRARY:-$root/target/debug/libturbo.so}"
[[ -f "$lib" ]] || { echo "missing $lib; run: cargo build -p turbo-shared" >&2; exit 2; }
bundle_arg=("--turbo.bundle=$root/testdata/bundles/mock/embedding")
for a in "$@"; do [[ "$a" == --turbo.bundle=* ]] && bundle_arg=(); done
if [[ "${TURBO_WEB_SKIP_BUILD:-0}" != "1" ]]; then
    mvn -q -B -f "$root/bindings/java/pom.xml" install -DskipTests
    mvn -q -B -f "$root/demo/java-web-spring/pom.xml" package -DskipTests
fi
jar="$(ls "$root"/demo/java-web-spring/target/turbo-demo-web-*.jar | grep -v original | head -1)"
exec "${JAVA_HOME:+$JAVA_HOME/bin/}java" --enable-native-access=ALL-UNNAMED -Dturbo.library="$lib" -jar "$jar" "${bundle_arg[@]}" "$@"
