# Turbo demos

One small program per language, each doing the same thing: load a bundle
on the best device (`AUTO`, which never picks a CPU; pass `--provider` and
`--ordinal` to choose explicitly), embed a few sentences, and print their
cosine similarities. Every one checks every call and fails with the
library's status name and message; none falls back to anything.

All of them run against the committed mock bundle
(`testdata/bundles/mock/embedding`, deterministic 8-dimensional vectors)
with no hardware, which is what `demo/run-all.sh` does, and against a real
bundle with a provider library:

```sh
cargo build -p turbo-shared                     # target/debug/libturbo.so (.dylib on macOS)
demo/run-all.sh                                 # every demo this machine can run, on the mock bundle
demo/c/turbo-demo-c --provider-lib build/openvino/libturbo_provider_openvino.so \
    --provider openvino --ordinal 1 --bundle ~/opt/bundles/minilm-onnx "two dogs" "a pair of dogs" "a spreadsheet"
```

| directory | language and API | how it is checked |
|---|---|---|
| `c/` | C over `include/turbo/turbo.h` | `make -C demo/c test` |
| `python/` | Python 3 with `ctypes` over the C ABI (no extension module) | `python3 demo/python/turbo_demo.py --bundle ...` |
| `rust/` | the safe Rust API (`crates/turbo`), as an out-of-tree crate | `cargo run --manifest-path demo/rust/Cargo.toml -- --bundle ...` |
| `java/` | the JDK 25 FFM binding (`ai.pipestream:turbo`) | `demo/java/run.sh --bundle ...` |
| `swift/TurboDemo/` | the SwiftPM package `PipestreamTurbo` | `swift run turbo-demo --bundle ...` on macOS |
| `grpc-c-server/` | a gRPC C++ server over the C ABI, plus a client | `demo/grpc-c-server/test.sh` (starts, calls, checks a refusal, stops) |
| `android/` | an Android app over a JNI shim (`jni/turbo_jni.c`), since FFM is not on Android | `demo/android/hosttest/run.sh` builds the same shim on the host and drives it through the app's `TurboEngine` |
| `java-web-spring/` | a Spring Boot web app with a page that shows the similarity heat map | `mvn -f demo/java-web-spring/pom.xml test` (HTTP contract) and `demo/java-web-spring/e2e` (Playwright, browser) |

Provider libraries and bundles are the same ones the conformance suites
use; see `docs/providers.md` for what each machine can build and
`docs/bundles.md` for importing a model. The Android app itself needs the
Android SDK and NDK plus a cross-compiled `libturbo.so` in `jniLibs/<abi>/`
(`demo/android/README.md`); its JNI layer is exercised on the host without
either.
