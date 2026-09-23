#!/usr/bin/env bash
# Run every demo this machine can run against the mock bundle. Each demo is
# a separate check; the first failure stops the script with its output.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
texts=("a brown dog runs through the grass" "a dog is running on the lawn" "the stock market closed higher")
bundle="$root/testdata/bundles/mock/embedding"
step() { printf '\n==> %s\n' "$*" >&2; }

step "libturbo"
cargo build -q -p turbo-shared

step "c (embed, and the streaming summarizer on the mock generative bundle)"
make -s -C demo/c test

step "python"
python3 demo/python/turbo_demo.py --bundle "$bundle" "${texts[@]}"

step "rust"
cargo run -q --manifest-path demo/rust/Cargo.toml -- --bundle "$bundle" "${texts[@]}"

if command -v mvn >/dev/null && command -v java >/dev/null; then
    step "java"
    demo/java/run.sh --bundle "$bundle" "${texts[@]}"
    step "android (host JNI check)"
    demo/android/hosttest/run.sh
    step "java-web-spring (HTTP contract)"
    mvn -q -B -f demo/java-web-spring/pom.xml test
else
    echo "skipping java, android, java-web-spring: no mvn/java on PATH" >&2
fi

if pkg-config --exists grpc++ protobuf 2>/dev/null; then
    step "grpc-c-server"
    demo/grpc-c-server/test.sh
else
    echo "skipping grpc-c-server: grpc++/protobuf not installed" >&2
fi

if [[ "$(uname)" == "Darwin" ]] && command -v swift >/dev/null; then
    step "swift"
    (cd demo/swift/TurboDemo && DYLD_LIBRARY_PATH="$root/target/debug" swift run -q turbo-demo --bundle "$bundle" "${texts[@]}")
else
    echo "skipping swift: needs macOS with the Swift toolchain" >&2
fi
printf '\ndemos: OK\n' >&2
