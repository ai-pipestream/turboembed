#!/usr/bin/env bash
# Build the Turbo Java binding and this app, then serve it.
#
#   demo/java-web-spring/run.sh                       # mock embedding bundle, port 8080
#
#   # one real embedding model on an explicit device
#   demo/java-web-spring/run.sh --turbo.bundle=$HOME/opt/bundles/minilm-onnx \
#       --turbo.provider-lib=build/openvino/libturbo_provider_openvino.so \
#       --turbo.provider=openvino --turbo.ordinal=1 --server.port=8080
#
#   # several models at once: an embedder, a reranker and a generator
#   demo/java-web-spring/run.sh --turbo.bundle=$HOME/opt/bundles/minilm-onnx \
#       --turbo.provider-lib=target/debug/libturbo_provider_cuda.so --turbo.provider=cuda \
#       --turbo.models[0].name=rerank --turbo.models[0].bundle=$HOME/opt/bundles/rerank-onnx \
#       --turbo.models[0].provider=cuda \
#       --turbo.generate-bundle=$HOME/opt/bundles/qwen05-gguf \
#       --turbo.generate-provider-lib=target/debug/libturbo_provider_ggml.so --turbo.generate-provider=ggml
#
# The page is at /, the REST API under /api/v1, the KServe Open Inference
# Protocol v2 surface under /v2, and the API explorer at /swagger-ui.html.
#
# Needs JDK 25 (JAVA_HOME or PATH), Maven, and libturbo (cargo build -p turbo-shared).
# Provider runtimes (OpenVINO, CUDA) come from LD_LIBRARY_PATH as usual.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
lib="${TURBO_LIBRARY:-$root/target/debug/libturbo.so}"
[[ -f "$lib" ]] || { echo "missing $lib; run: cargo build -p turbo-shared" >&2; exit 2; }
bundle_arg=("--turbo.bundle=$root/testdata/bundles/mock/embedding")
for a in "$@"; do [[ "$a" == --turbo.bundle=* ]] && bundle_arg=(); done
# The Benchmarks panel reads the committed receipts; the default turbo.receipts
# is relative to the working directory, so name them absolutely from here.
receipts_arg=("--turbo.receipts=$root/testdata/receipts/turbo/bench")
for a in "$@"; do [[ "$a" == --turbo.receipts=* ]] && receipts_arg=(); done
if [[ "${TURBO_WEB_SKIP_BUILD:-0}" != "1" ]]; then
    mvn -q -B -f "$root/bindings/java/pom.xml" install -DskipTests
    mvn -q -B -f "$root/demo/java-web-spring/pom.xml" package -DskipTests
fi
jar="$(ls "$root"/demo/java-web-spring/target/turbo-demo-web-*.jar | grep -v original | head -1)"
exec "${JAVA_HOME:+$JAVA_HOME/bin/}java" --enable-native-access=ALL-UNNAMED -Dturbo.library="$lib" -jar "$jar" "${bundle_arg[@]}" "${receipts_arg[@]}" "$@"
